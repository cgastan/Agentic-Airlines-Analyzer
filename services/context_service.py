import os

import tiktoken
from groq import Groq
from dotenv import load_dotenv

from config import CONTEXT_TOKEN_BUDGETS
from schemas.models import (
    ConversationMessage,
    ImpactReportResponse,
    SessionState,
)
from services.agent_flow import rewrite_query

load_dotenv(override=True)

encoding = tiktoken.get_encoding("o200k_base")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "openai/gpt-oss-20b")
session_store: dict[str, SessionState] = {}


def count_tokens(text: str) -> int:
    return len(encoding.encode(text or ""))


def truncate_by_tokens(text: str, budget: int) -> str:
    if count_tokens(text) <= budget:
        return text
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:budget]).rstrip() + "..."


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in session_store:
        session_store[session_id] = SessionState(session_id=session_id)
    return session_store[session_id]


def build_conversation_context(state: SessionState) -> str:
    return "\n".join(
        f"{message.role.upper()}: {message.content}" for message in state.messages
    )


def prepare_standalone_query(message: str, state: SessionState) -> str:
    if not state.messages and not state.summary:
        return message

    contextual_prompt = f"""
Convierte el último seguimiento en una consulta independiente para búsqueda documental.
Usa el resumen y el historial solo para resolver pronombres, sistemas y referencias implícitas.
Devuelve únicamente la consulta reformulada.

Resumen: {truncate_by_tokens(state.summary, CONTEXT_TOKEN_BUDGETS['summary'])}
Historial reciente:
{truncate_by_tokens(build_conversation_context(state), CONTEXT_TOKEN_BUDGETS['history'])}
Último mensaje: {message}
"""
    rewritten_query = rewrite_query(contextual_prompt).strip()
    if rewritten_query:
        return rewritten_query

    summary = truncate_by_tokens(state.summary, 300)
    if summary:
        return f"Contexto previo: {summary} Seguimiento: {message}"
    return message


def summarize_dropped_messages(previous_summary: str, messages: list[ConversationMessage]) -> str:
    text = "\n".join(
        f"{message.role}: {message.content}" for message in messages)
    prompt = f"""
Resume el contexto anterior conservando decisiones, sistemas afectados, riesgos,
stakeholders y preguntas abiertas. No incluyas fragmentos RAG ni texto innecesario.

Resumen anterior: {previous_summary}
Mensajes antiguos:
{text}

Devuelve un resumen compacto, no una lista exhaustiva.
"""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=350,
        reasoning_effort="low",
    )
    return truncate_by_tokens(
        (response.choices[0].message.content or "").strip(),
        CONTEXT_TOKEN_BUDGETS["summary"],
    )


def fit_recent_history(state: SessionState) -> None:
    budget = CONTEXT_TOKEN_BUDGETS["history"]
    while state.messages:
        if count_tokens(build_conversation_context(state)) <= budget:
            break
        dropped = state.messages.pop(0)
        state.summary = summarize_dropped_messages(state.summary, [dropped])


def compact_report(report: ImpactReportResponse) -> str:
    mitigations = ", ".join(
        report.required_mitigations[:3]) or "Sin mitigaciones confirmadas"
    return (
        f"Resumen ejecutivo: {report.summary_report}. "
        f"Riesgo: {report.risk_level.value}. "
        f"Mitigaciones clave: {mitigations}."
    )


def add_turn(state: SessionState, user_message: str, assistant_message: str) -> None:
    state.messages.append(ConversationMessage(
        role="user", content=user_message))
    state.messages.append(ConversationMessage(
        role="assistant", content=assistant_message))
    fit_recent_history(state)


def update_structured_state(state: SessionState, report: ImpactReportResponse) -> None:
    state.last_report = report

    if report.technical_changes:
        systems = [
            change.affected_system for change in report.technical_changes if change.affected_system]
        state.systems = list(dict.fromkeys(state.systems + systems))

    if report.affected_stakeholders:
        stakeholders = [
            item.stakeholder for item in report.affected_stakeholders]
        state.stakeholders = list(dict.fromkeys(
            state.stakeholders + stakeholders))

    state.risk_level = report.risk_level
    if not state.change_request_id:
        state.change_request_id = report.request_id


def prepare_context_message(state: SessionState, new_message: str) -> tuple[str, str, str]:
    standalone_query = prepare_standalone_query(new_message, state)
    recent_history = truncate_by_tokens(
        build_conversation_context(state),
        CONTEXT_TOKEN_BUDGETS["history"],
    )
    structured_state = "\n".join(filter(None, (
        f"Solicitud: {state.change_request_id}" if state.change_request_id else "",
        f"Sistemas: {', '.join(state.systems)}" if state.systems else "",
        f"Stakeholders: {', '.join(state.stakeholders)}" if state.stakeholders else "",
        f"Riesgo: {state.risk_level.value}" if state.risk_level else "",
    )))
    state_layers = "\n".join(filter(None, (
        f"Resumen previo: {truncate_by_tokens(state.summary, CONTEXT_TOKEN_BUDGETS['summary'])}" if state.summary else "",
        f"Estado estructurado:\n{structured_state}" if structured_state else "",
        f"Historia reciente:\n{recent_history}" if recent_history else "",
    )))
    return standalone_query, recent_history, state_layers


def record_report_turn(state: SessionState, user_message: str, report: ImpactReportResponse) -> None:
    update_structured_state(state, report)
    add_turn(state, user_message, compact_report(report))
