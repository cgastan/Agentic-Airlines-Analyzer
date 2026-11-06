import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
from tests.test_core import ARCH, POLICY


@pytest.fixture
def client(fake_llm):
    (settings.DATA_DIR / "arq.txt").write_text(ARCH, encoding="utf-8")
    (settings.DATA_DIR / "pol.txt").write_text(POLICY, encoding="utf-8")
    with TestClient(app) as test_client:  # ejecuta el lifespan (build_index)
        yield test_client


def events(text: str) -> list[tuple[str, object]]:
    parsed = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((lines["event"], json.loads(lines["data"])))
    return parsed


def test_health_reports_indexed_documents(client):
    assert client.get("/health").json()["documents"] == 2


def test_analyze_rejects_empty_and_unsupported_files(client):
    data = {"analyst_prompt": "Migrar el motor de reservas"}
    assert client.post("/api/analyze", data=data, files={"file": ("v.txt", b"", "text/plain")}).status_code == 422
    assert client.post("/api/analyze", data=data, files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415


def test_analyze_uses_uploaded_change_request(client):
    response = client.post(
        "/api/analyze",
        data={"analyst_prompt": "Analizar el cambio adjunto"},
        files={"file": ("CR-100.md", "# Cambio\nMigrar el motor de reservas a la nube".encode(), "text/markdown")},
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["request_id"].startswith("CHG-")
    assert any(source["document_id"] == "CR-100.md" for source in report["sources"])
    # El upload es temporal: no queda en el índice global ni en disco.
    assert "CR-100.md" not in app.state.index.sources
    assert not (settings.DATA_DIR / "CR-100.md").exists()


def test_analyze_accepts_legacy_doc_upload(client):
    from tests.test_core import FIXTURES

    doc = (FIXTURES / "cambio_word97.doc").read_bytes()
    response = client.post("/api/analyze", data={"analyst_prompt": "Analizar el cambio adjunto"},
                           files={"file": ("CR-2026-099.doc", doc, "application/msword")})
    assert response.status_code == 200, response.text
    assert any(s["document_id"] == "CR-2026-099.doc" for s in response.json()["sources"])

    broken = client.post("/api/analyze", data={"analyst_prompt": "Analizar el cambio adjunto"},
                         files={"file": ("roto.doc", b"no es word", "application/msword")})
    assert broken.status_code == 422


def test_analyze_stream_emits_real_steps_then_final(client):
    with client.stream("POST", "/api/analyze/stream", data={"analyst_prompt": "Migrar reservas"}) as response:
        response.read()
    parsed = events(response.text)
    steps = [data["step"] for event, data in parsed if event == "progress"]
    assert steps[0] == "initialize" and steps[-1] == "generate_impact_report"
    assert parsed[-1][0] == "final" and parsed[-1][1]["summary_report"]


def test_documents_endpoint_adds_to_knowledge_base(client):
    response = client.post("/api/documents", files=[("files", ("notas.txt", b"Feedback de stakeholders de aeropuerto", "text/plain"))],
)
    assert response.status_code == 200, response.text
    assert app.state.index.sources["notas.txt"] >= 1
    assert (settings.DATA_DIR / "notas.txt").exists()


def test_chat_returns_readable_text_and_structured_report(client):
    body = client.post("/api/chat", json={"message": "¿Qué impacto tiene migrar reservas?"}).json()
    assert "Nivel de riesgo: Alto" in body["answer"]
    assert body["report"]["risk_level"] == "Alto" and body["sources"]


def test_chat_uses_client_history_and_keeps_no_server_state(client, fake_llm):
    first = client.post("/api/chat", json={"message": "¿Qué impacto tiene migrar reservas?"})
    assert fake_llm.calls.count("other") == 0          # sin historial no se reformula nada
    assert "set-cookie" not in first.headers           # sin sesión ni cookie

    history = [{"role": "user", "content": "¿Qué impacto tiene migrar reservas?"},
               {"role": "assistant", "content": first.json()["answer"]}]
    follow_up = client.post("/api/chat", json={"message": "¿y qué riesgo tiene eso?", "history": history})
    assert follow_up.status_code == 200, follow_up.text
    assert fake_llm.calls.count("other") == 1          # una reformulación de la consulta


def test_chat_rejects_invalid_history_role(client):
    bad = {"message": "x", "history": [{"role": "system", "content": "y"}]}
    assert client.post("/api/chat", json=bad).status_code == 422


def test_chat_stream_finishes_with_final_event(client):
    with client.stream("POST", "/api/chat/stream", json={"message": "Analiza migrar reservas"}) as response:
        response.read()
    parsed = events(response.text)
    assert {"progress", "text"} <= {event for event, _ in parsed}
    assert parsed[-1][0] == "final" and parsed[-1][1]["report"]["risk_level"] == "Alto"
