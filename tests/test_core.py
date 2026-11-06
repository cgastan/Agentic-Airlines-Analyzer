import asyncio
import io

import pytest
from pydantic import ValidationError

from config import settings
from schemas.models import ImpactReportResponse
from services import agent_flow
from services.chunking import Chunk, chunk_pages, chunk_text
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
    chunks = (chunk_pages([Page(text=ARCH, source="arq.txt")])
              + chunk_pages([Page(text=POLICY, source="pol.txt")]))
    vectors = FakeEncoder().encode([c.context_text for c in chunks]).tolist()
    return HybridIndex(chunks, vectors), FakeEncoder()


def test_search_excludes_already_retrieved_chunks():
    index, encoder = _index()
    first = index.search("reservas", encoder.encode(["reservas"])[0], top_k=1)
    second = index.search("reservas", encoder.encode(["reservas"])[0], top_k=1,
                          exclude_ids={first[0].id})
    assert second and second[0].id != first[0].id


def test_exact_code_ranks_first_via_bm25():
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

def test_build_index_processes_every_document_the_same_way(fake_llm):
    (settings.DATA_DIR / "doc1.txt").write_text(ARCH, encoding="utf-8")
    (settings.DATA_DIR / "doc2.md").write_text(POLICY, encoding="utf-8")
    (settings.DATA_DIR / "ignorado.xlsx").write_bytes(b"x")
    index = run(build_index())
    assert set(index.sources) == {"doc1.txt", "doc2.md"}
    assert fake_llm.calls == []  # indexar no requiere llamadas al LLM


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
    # Las dos búsquedas aportan fragmentos distintos (sin duplicados).
    ids = [s.chunk_id for s in validated.sources]
    assert len(ids) == len(set(ids)) and {"arq.txt", "pol.txt"} <= {s.document_id for s in validated.sources}


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


# ---------- target_system ----------

def test_system_name_is_cleaned_and_compared_case_insensitively():
    from schemas.models import ChatRequest, system_key

    assert ChatRequest(message="x", target_system="  Motor   de  Reservas ").target_system == "Motor de Reservas"
    assert ChatRequest(message="x", target_system="   ").target_system is None
    assert system_key("PAYHUB") == system_key("payhub") == system_key("  PayHub ")
    assert system_key("Reservación") == system_key("RESERVACION")


def test_canonical_spelling_uses_corpus_form():
    index, _ = _index()
    assert index.canonical_spelling("MOTOR DE   RESERVAS") == "motor de reservas"
    assert index.canonical_spelling("POLITICA") == "Política"      # recupera el acento del documento
    assert index.canonical_spelling("sistema inexistente") is None


def test_target_system_is_not_duplicated_when_case_differs(fake_llm):
    index = _workflow_index(fake_llm)
    captured = {}
    original = agent_flow.rag.search_specifications

    async def spy(idx, query):
        captured["query"] = query
        return await original(idx, query)

    agent_flow.rag.search_specifications = spy
    try:
        report = run(agent_flow.run_agent_workflow(
            "Migrar", index, target_system="RESERVAS"))
    finally:
        agent_flow.rag.search_specifications = original
    # El LLM falso extrae 'Reservas'; 'RESERVAS' no debe agregarse como sistema distinto.
    assert captured["query"].casefold().count("reservas") >= 1
    assert ImpactReportResponse.model_validate(report)


# ---------- Embeddings OpenAI ----------

def test_openai_is_the_default_embedding_provider():
    from config import Settings
    assert Settings(_env_file=None).EMBEDDING_PROVIDER == "openai"


def test_missing_openai_key_fails_fast_with_clear_message(monkeypatch):
    from services.embeddings import check_embedding_config
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        check_embedding_config()
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    check_embedding_config()


# ---------- Contexto del chat (sin estado) ----------

def test_history_keeps_recent_messages_and_summarizes_old_ones(fake_llm, monkeypatch):
    from config import CONTEXT_TOKEN_BUDGETS
    from schemas.models import ConversationMessage
    from services import context_service

    monkeypatch.setitem(CONTEXT_TOKEN_BUDGETS, "history", 120)
    monkeypatch.setattr(context_service, "MESSAGE_TOKEN_CAP", 40)
    history = [ConversationMessage(role="user" if i % 2 == 0 else "assistant",
                                   content=f"mensaje número {i} " * 10) for i in range(10)]

    context = run(context_service.build_context(history))

    assert "mensaje número 9" in context and "mensaje número 0" not in context
    assert "Resumen previo" in context            # lo antiguo se condensó con el LLM
    assert fake_llm.calls.count("other") == 1


def test_no_history_means_no_llm_calls(fake_llm):
    from services import context_service
    assert run(context_service.build_context([])) == ""
    assert run(context_service.standalone_query("hola", "")) == "hola"
    assert fake_llm.calls == []


def test_build_index_fails_loudly_when_every_document_fails_to_embed(monkeypatch):
    from services import ingestion_service

    async def no_credit(_texts):
        raise RuntimeError("insufficient_quota")

    monkeypatch.setattr(ingestion_service, "embed_texts", no_credit)
    (settings.DATA_DIR / "doc.txt").write_text(ARCH, encoding="utf-8")
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        run(build_index())
