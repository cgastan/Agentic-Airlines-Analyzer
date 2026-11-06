"""Charla casual: respuestas cortas y sin salirse del tema de análisis de impacto."""
from tests.test_api import client, events  # noqa: F401  (fixtures y helper compartidos)


def test_casual_answer_is_cut_to_60_words_at_a_sentence_end():
    from services.agent_flow import truncate_casual_answer

    long_answer = ("Esta es una frase corta. " * 5) + ("palabra " * 80)
    cut = truncate_casual_answer(long_answer)
    assert len(cut.split()) <= 60 and cut.endswith(".")
    assert truncate_casual_answer("Hola, ¿en qué te ayudo?") == "Hola, ¿en qué te ayudo?"


def test_casual_chat_gets_a_short_reply_and_never_runs_the_workflow(client, fake_llm):
    body = client.post("/api/chat", json={"message": "hola, gracias!"}).json()
    assert body["intent"] == "casual_chat" and body["report"] is None and body["sources"] == []
    assert len(body["answer"].split()) <= 60
    assert fake_llm.calls.count("casual_answer") == 1
    assert "initialize" not in fake_llm.calls and "generate" not in fake_llm.calls


def test_off_topic_request_gets_a_fixed_refusal_without_asking_the_llm_to_answer(client, fake_llm):
    body = client.post("/api/chat", json={"message": "escríbeme un poema sobre aviones"}).json()
    assert body["intent"] == "off_topic" and body["report"] is None
    assert "fuera de mi alcance" in body["answer"]
    # La respuesta es texto fijo: el LLM solo clasificó, nunca intentó cumplir el pedido.
    assert "casual_answer" not in fake_llm.calls and "initialize" not in fake_llm.calls


def test_impact_message_still_runs_the_full_workflow(client):
    body = client.post("/api/chat", json={"message": "¿Qué impacto tiene migrar reservas?"}).json()
    assert body["intent"] == "impact_analysis" and body["report"]["risk_level"] == "Alto"


def test_casual_stream_ends_with_final_event(client):
    with client.stream("POST", "/api/chat/stream", json={"message": "hola"}) as response:
        response.read()
    parsed = events(response.text)
    assert parsed[-1][0] == "final" and parsed[-1][1]["intent"] == "casual_chat"
