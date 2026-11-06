import asyncio
import io

import pytest
from pydantic import ValidationError

from config import settings
from schemas.models import ImpactReportResponse
from services import agent_flow
from services.chunking import Chunk, chunk_pages, chunk_text, extract_identifiers
from services.ingestion_service import build_index, extend_index
from services.readers import UnsupportedFormatError, read_document
from services.vector_store import HybridIndex, tokenize

ARCH = ("Arquitectura de sistemas.\nEl motor de reservas se integra con la pasarela de pagos "
        "mediante API REST. El sistema de check-in depende del motor de reservas.")
POLICY = ("Política de gestión de cambios POL-GC-002.\nTodo cambio que afecte a clientes requiere "
          "aprobación del comité y aviso a atención al cliente con 15 días de anticipación.")


def run(coro):
    return asyncio.run(coro)


# ---------- Chunking ----------

def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"palabra{i}." for i in range(400))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_empty_text_produces_no_chunks():
    assert chunk_text(" \n\n ") == []


def test_identifiers_are_detected_by_shape_not_by_hardcoded_values():
    found = extract_identifiers("Según POL-GC-002, la Ley 18.331 e ISO 27001 aplican a XYZ-9.")
    assert {"POL-GC-002", "Ley 18.331", "ISO 27001", "XYZ-9"} <= set(found)


def test_tokenize_normalizes_accents_and_legal_numbers():
    assert "18331" in tokenize("Ley 18.331")
    assert "politica" in tokenize("Política")


# ---------- Readers ----------

def test_reads_txt_and_markdown_sections():
    assert read_document("a.txt", "hola mundo".encode())[0].text == "hola mundo"
    pages = read_document("b.md", b"# Alcance\nCambio X\n## Riesgos\nRiesgo Y")
    assert [(p.section, p.text.strip()) for p in pages] == [("Alcance", "Cambio X"), ("Riesgos", "Riesgo Y")]


def test_reads_docx_with_headings_and_tables():
    from docx import Document

    document = Document()
    document.add_heading("Sistemas afectados", level=1)
    document.add_paragraph("El motor de reservas cambia.")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Sistema", "Dueño"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Reservas", "TI"
    buffer = io.BytesIO()
    document.save(buffer)

    pages = read_document("cambio.docx", buffer.getvalue())
    assert pages[0].section == "Sistemas afectados"
    assert "Sistema: Reservas | Dueño: TI" in pages[-1].text


def test_reads_pdf(tmp_path):
    from pathlib import Path
    pdfs = sorted((Path(__file__).resolve().parents[1] / "data").glob("*.pdf"))
    if not pdfs:
        pytest.skip("No hay PDFs de ejemplo en data/")
    pages = read_document(pdfs[0].name, pdfs[0].read_bytes())
    assert pages and all(page.page for page in pages)


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("filename", ["cambio_word97.doc", "cambio_rtf.doc"])
def test_reads_legacy_doc_binary_and_rtf(filename):
    # Generados con Word: acentos, campo PAGE, hipervínculo, tabla y edición posterior.
    text = read_document(filename, (FIXTURES / filename).read_bytes())[0].text
    assert "Migración del módulo de check-in de AirBook a la nube. Año 2026, señalización y pingüinos." in text
    assert "Página actual: 1" in text            # resultado del campo, sin el código PAGE
    assert "Ver ticket" in text and "HYPERLINK" not in text
    assert "Sistema | Responsable\nAirBook | Operaciones" in text
    assert text.count("Párrafo final con texto extenso") == 40
    if filename == "cambio_word97.doc":          # texto agregado en una segunda edición
        assert text.startswith("Encabezado agregado después.") and text.endswith("Cierre agregado después.")


def test_doc_that_is_really_a_docx_is_read():
    from docx import Document

    document = Document()
    document.add_paragraph("Contenido real de un docx renombrado")
    buffer = io.BytesIO()
    document.save(buffer)
    assert "docx renombrado" in read_document("viejo.doc", buffer.getvalue())[0].text


def test_invalid_doc_raises_clear_error():
    from services.readers import DocFormatError

    with pytest.raises(DocFormatError):
        read_document("roto.doc", b"esto no es un documento de Word")


def test_unsupported_format_is_rejected():
    with pytest.raises(UnsupportedFormatError):
        read_document("planilla.xlsx", b"x")


# ---------- Índice híbrido ----------

def _index(fake_vectors=True):
    from tests.conftest import FakeEncoder
    from services.readers import Page
    chunks = (chunk_pages([Page(text=ARCH, source="arq.txt")], "specification")
              + chunk_pages([Page(text=POLICY, source="pol.txt")], "policy"))
    vectors = FakeEncoder().encode([c.context_text for c in chunks]).tolist()
    return HybridIndex(chunks, vectors), FakeEncoder()


def test_search_filters_by_document_type():
    index, encoder = _index()
    query = "aprobación de cambios"
    results = index.search(query, encoder.encode([query])[0], doc_types={"specification"})
    assert results and all(chunk.doc_type == "specification" for chunk in results)


def test_search_falls_back_to_all_documents_when_type_missing():
    index, encoder = _index()
    results = index.search("reservas", encoder.encode(["reservas"])[0], doc_types={"stakeholder"})
    assert results


def test_exact_identifier_ranks_first():
    index, encoder = _index()
    query = "POL-GC-002"
    assert index.search(query, encoder.encode([query])[0], top_k=1)[0].source == "pol.txt"


def test_merged_index_replaces_same_document_and_keeps_base():
    index, encoder = _index()
    new = [Chunk(id="pol.txt#s-1", text="nueva", context_text="nueva", source="pol.txt")]
    merged = index.merged_with(new, encoder.encode(["nueva"]).tolist())
    assert [c.text for c in merged.chunks if c.source == "pol.txt"] == ["nueva"]
    assert any(c.source == "arq.txt" for c in merged.chunks)
    assert len(index) != len(merged) or index.chunks[-1].text != "nueva"  # el base no muta


# ---------- Ingesta ----------

def test_build_index_classifies_documents_by_content(fake_llm):
    (settings.DATA_DIR / "doc1.txt").write_text(ARCH, encoding="utf-8")
    (settings.DATA_DIR / "doc2.md").write_text(POLICY, encoding="utf-8")
    index = run(build_index())
    assert index.sources == {"doc1.txt": "specification", "doc2.md": "policy"}

    # Segunda construcción: clasificación cacheada, sin nuevas llamadas al LLM.
    calls = len(fake_llm.calls)
    run(build_index())
    assert len(fake_llm.calls) == calls


def test_manifest_overrides_classification(fake_llm):
    (settings.DATA_DIR / "notas.txt").write_text(ARCH, encoding="utf-8")
    (settings.DATA_DIR / "manifest.json").write_text('{"notas.txt": {"doc_type": "stakeholder"}}')
    assert run(build_index()).sources == {"notas.txt": "stakeholder"}
    assert "classify_document" not in fake_llm.calls


def test_build_index_without_documents_returns_none():
    assert run(build_index()) is None


# ---------- Workflow del agente ----------

def _workflow_index(fake_llm):
    index, _ = run(extend_index(None, [("arq.txt", ARCH.encode()), ("pol.txt", POLICY.encode())]))
    return index


def test_workflow_runs_all_steps_and_returns_valid_report(fake_llm):
    index = _workflow_index(fake_llm)
    report = run(agent_flow.run_agent_workflow("Migrar el motor de reservas", index))
    validated = ImpactReportResponse.model_validate(report)

    assert validated.trace.steps == [
        "initialize", "retrieve_specifications_rag", "retrieve_stakeholder_notes_rag",
        "grade_context", "generate_impact_report"]
    assert validated.risk_level.value == "Alto"
    assert not validated.insufficient_information
    # Cada búsqueda consultó su propio tipo documental.
    by_type = {s.doc_type for s in validated.sources}
    assert {"specification", "policy"} <= by_type


def test_workflow_rewrites_at_most_twice_then_reports_insufficient(fake_llm):
    fake_llm.grade_sufficient = False
    index = _workflow_index(fake_llm)
    report = ImpactReportResponse.model_validate(
        run(agent_flow.run_agent_workflow("cambio vago", index)))

    assert fake_llm.calls.count("rewrite_query") == settings.MAX_QUERY_REWRITES == 2
    assert fake_llm.calls.count("grade_context") == 3
    assert report.trace.rewrites == 2 and len(report.trace.queries) == 3
    assert report.insufficient_information and report.confidence.value == "Bajo"
    assert report.missing_information == ["nombre del sistema"]


def test_workflow_without_documents_does_not_invent(fake_llm):
    report = ImpactReportResponse.model_validate(
        run(agent_flow.run_agent_workflow("Migrar reservas", None)))
    assert report.insufficient_information
    assert "generate" not in fake_llm.calls


def test_invalid_report_is_rejected():
    with pytest.raises(ValidationError):
        ImpactReportResponse(request_id="X", risk_level="Desconocido", summary_report="x")
