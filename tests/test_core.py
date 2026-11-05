from pydantic import ValidationError

from schemas.models import ImpactReportResponse, RiskLevel
from services.agent_flow import normalize_generated_report, route_after_grade
from services.chunking import build_chunks_from_documents, chunk_text
from services.rag import reciprocal_rank_fusion


def test_chunk_text_without_overlap_returns_complete_content():
    text = "A" * 120

    chunks = chunk_text(text, chunk_size=50, overlap=0)

    assert len(chunks) >= 2
    assert "".join(chunks).replace("\n", "")


def test_empty_document_produces_no_chunks():
    assert chunk_text(" \n\n ") == []
    assert build_chunks_from_documents([{
        "id": "empty",
        "title": "Empty document",
        "text": "",
    }]) == []


def test_document_chunks_keep_source_metadata():
    chunks = build_chunks_from_documents([{
        "id": "ARQ-001.pdf",
        "title": "Arquitectura",
        "text": "AirBook utiliza reservas y PayHub procesa pagos.",
    }])

    assert chunks[0]["doc_id"] == "ARQ-001.pdf"
    assert chunks[0]["title"] == "Arquitectura"
    assert chunks[0]["chunk_id"] == "ARQ-001.pdf_chunk_1"


def test_rrf_combines_lexical_and_vector_rankings():
    fused = reciprocal_rank_fusion(
        lexical_ranking=[0, 1],
        vector_ranking=[1, 0],
    )

    assert set(fused) == {0, 1}
    assert fused[0] in {0, 1}


def test_valid_impact_report_passes_pydantic_validation():
    report = ImpactReportResponse(
        request_id="CHG-TEST",
        risk_level=RiskLevel.MEDIUM,
        summary_report="AirBook integra PayHub.",
        confidence="Medio",
    )

    assert report.request_id == "CHG-TEST"
    assert report.insufficient_information is False


def test_invalid_impact_report_is_rejected():
    try:
        ImpactReportResponse(
            request_id="CHG-TEST",
            risk_level="Riesgo desconocido",
            summary_report="Reporte incompleto.",
            confidence="Medio",
        )
        assert False, "El modelo debería rechazar un nivel de riesgo inválido"
    except ValidationError:
        pass


def test_rewrite_limit_routes_to_fallback():
    first_retry = {"context_grade": "no", "retry_count": 1}
    exhausted = {"context_grade": "no", "retry_count": 2}

    assert route_after_grade(first_retry) == "rewrite"
    assert route_after_grade(exhausted) == "fallback"


def test_report_without_retrieved_impact_is_marked_insufficient():
    report = {
        "technical_changes": [],
        "risk_level": "Medio",
        "affected_stakeholders": [],
        "required_mitigations": [],
        "dependencies": [],
        "summary_report": "Resultado preliminar",
        "confidence": "Alto",
        "insufficient_information": False,
    }

    normalized = normalize_generated_report(
        report, [{"text": "Contexto disponible pero no suficiente"}])

    assert normalized["insufficient_information"] is True
    assert normalized["confidence"] == "Bajo"
