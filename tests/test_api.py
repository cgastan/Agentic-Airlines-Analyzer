import json

from fastapi.testclient import TestClient

from main import app
from routers import analysis
from schemas.models import ImpactReportResponse
from services.context_service import session_store


client = TestClient(app)


def valid_report() -> dict:
    return {
        "request_id": "temporary",
        "technical_changes": [],
        "risk_level": "Medio",
        "affected_stakeholders": [],
        "required_mitigations": [],
        "dependencies": [],
        "summary_report": "AirBook integra PayHub para pagos.",
        "confidence": "Medio",
        "insufficient_information": False,
        "sources": [],
    }


def test_analyze_empty_document_returns_422():
    response = client.post(
        "/api/analyze",
        data={"analyst_prompt": "Analizar AirBook y PayHub"},
        files={"file": ("vacío.txt", b"", "text/plain")},
    )

    assert response.status_code == 422


def test_analyze_stream_finishes_with_final_event(monkeypatch):
    async def fake_process(_prompt, _file):
        return ImpactReportResponse.model_validate(valid_report())

    monkeypatch.setattr(analysis, "process_analysis", fake_process)

    with client.stream(
        "POST",
        "/api/analyze/stream",
        data={"analyst_prompt": "Analizar AirBook y PayHub"},
    ) as response:
        response.read()
        content = response.text

    assert response.status_code == 200
    assert "event: progress" in content
    assert "event: final" in content
    final_lines = [
        line for line in content.splitlines()
        if line.startswith("data: {") and '"request_id"' in line
    ]
    assert final_lines
    assert json.loads(final_lines[-1].removeprefix("data: "))["summary_report"]


def test_chat_stream_finishes_with_final_event(monkeypatch):
    session_store.clear()

    def fake_workflow(**_kwargs):
        return valid_report()

    monkeypatch.setattr(
        "routers.chat.run_agent_workflow", fake_workflow)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Analiza la integración de AirBook con PayHub"},
    ) as response:
        response.read()
        content = response.text

    assert response.status_code == 200
    assert "event: final" in content
    assert len(session_store) == 1
