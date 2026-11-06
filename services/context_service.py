"""Memoria conversacional del chat: historial acotado por tokens, resumen y estado estructurado."""
from __future__ import annotations

import logging

import tiktoken

from config import CONTEXT_TOKEN_BUDGETS
from schemas.models import ConversationMessage, ImpactReportResponse, SessionState
from services import llm

logger = logging.getLogger(__name__)

encoding = tiktoken.get_encoding("o200k_base")
# Almacén en memoria: se pierde al reiniciar (suficiente para el alcance del proyecto).
session_store: dict[str, SessionState] = {}


def count_tokens(text: str) -> int:
    return len(encoding.encode(text or ""))


def truncate_by_tokens(text: str, budget: int) -> str:
    tokens = encoding.encode(text or "")
    if len(tokens) <= budget:
        return text or ""
    return encoding.decode(tokens[:budget]).rstrip() + "..."


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in session_store:
        session_store[session_id] = SessionState(session_id=session_id)
    return session_store[session_id]


def build_conversation_context(state: SessionState) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in state.messages)


async def prepare_standalone_query(message: str, state: SessionState) -> str:
    """Convierte un seguimiento ("¿y qué riesgo tiene eso?") en una consulta autocontenida."""
    if not state.messages and not state.summary:
        return message
    prompt = f"""Convierte el último mensaje en una consulta independiente para búsqueda documental.
Usa el resumen y el historial sólo para resolver pronombres, sistemas y referencias implícitas.
Devuelve únicamente la consulta reformulada, sin comillas.

Resumen: {truncate_by_tokens(state.summary, CONTEXT_TOKEN_BUDGETS['summary'])}
Historial reciente:
{truncate_by_tokens(build_conversation_context(state), CONTEXT_TOKEN_BUDGETS['history'])}
Último mensaje: {message}"""
    try:
        rewritten = await llm.complete(prompt, temperature=0, max_tokens=600)
    except Exception as error:
        logger.warning("No se pudo reformular el seguimiento: %s", error)
        rewritten = ""
    return rewritten or message


async def summarize_dropped_messages(previous_summary: str, messages: list[ConversationMessage]) -> str:
    text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    prompt = f"""Resume el contexto conservando decisiones, sistemas afectados, riesgos,
stakeholders y preguntas abiertas. Devuelve un resumen compacto.

Resumen anterior: {previous_summary}
Mensajes antiguos:
{text}"""
    summary = await llm.complete(prompt, temperature=0.1, max_tokens=900)
    return truncate_by_tokens(summary, CONTEXT_TOKEN_BUDGETS["summary"])


async def fit_recent_history(state: SessionState) -> None:
    """Saca mensajes antiguos del historial y los condensa en el resumen."""
    dropped: list[ConversationMessage] = []
    while state.messages and count_tokens(build_conversation_context(state)) > CONTEXT_TOKEN_BUDGETS["history"]:
        dropped.append(state.messages.pop(0))
    if dropped:
        try:
            state.summary = await summarize_dropped_messages(state.summary, dropped)
        except Exception as error:
            logger.warning("No se pudo resumir el historial: %s", error)


def compact_report(report: ImpactReportResponse) -> str:
    mitigations = ", ".join(report.required_mitigations[:3]) or "sin mitigaciones confirmadas"
    return (f"Resumen: {report.summary_report} Riesgo: {report.risk_level.value}. "
            f"Mitigaciones clave: {mitigations}.")


def build_session_context(state: SessionState) -> str:
    """Capas de contexto para el agente: resumen + estado estructurado + historial reciente."""
    structured = "\n".join(filter(None, (
        f"Solicitud previa: {state.change_request_id}" if state.change_request_id else "",
        f"Sistemas ya identificados: {', '.join(state.systems)}" if state.systems else "",
        f"Stakeholders ya identificados: {', '.join(state.stakeholders)}" if state.stakeholders else "",
        f"Último riesgo evaluado: {state.risk_level.value}" if state.risk_level else "",
    )))
    history = truncate_by_tokens(build_conversation_context(state), CONTEXT_TOKEN_BUDGETS["history"])
    return "\n".join(filter(None, (
        f"Resumen previo: {state.summary}" if state.summary else "",
        structured,
        f"Historial reciente:\n{history}" if history else "",
    )))


def update_structured_state(state: SessionState, report: ImpactReportResponse) -> None:
    state.last_report = report
    systems = [c.affected_system for c in report.technical_changes if c.affected_system]
    stakeholders = [s.stakeholder for s in report.affected_stakeholders]
    state.systems = list(dict.fromkeys(state.systems + systems))
    state.stakeholders = list(dict.fromkeys(state.stakeholders + stakeholders))
    state.risk_level = report.risk_level
    state.change_request_id = state.change_request_id or report.request_id


async def record_turn(state: SessionState, user_message: str, assistant_message: str) -> None:
    state.messages.append(ConversationMessage(role="user", content=user_message))
    state.messages.append(ConversationMessage(role="assistant", content=assistant_message))
    await fit_recent_history(state)


async def record_report_turn(state: SessionState, user_message: str, report: ImpactReportResponse) -> None:
    update_structured_state(state, report)
    await record_turn(state, user_message, compact_report(report))
