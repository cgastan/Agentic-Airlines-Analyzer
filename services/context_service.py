"""Contexto conversacional por sesión, guardado en memoria del servidor.

Cada conversación se identifica con un `session_id`. El historial se ajusta a un presupuesto de
tokens: se conservan los mensajes más recientes y los más antiguos se condensan en un resumen
generado por el LLM.
"""
from __future__ import annotations

import logging
import uuid
from collections import OrderedDict

import tiktoken

from config import CONTEXT_TOKEN_BUDGETS
from schemas.models import ConversationMessage
from services import llm

logger = logging.getLogger(__name__)
encoding = tiktoken.get_encoding("o200k_base")

# Un reporte previo puede ser largo: se limita cada mensaje para que quepan varios turnos.
MESSAGE_TOKEN_CAP = CONTEXT_TOKEN_BUDGETS["history"] // 3


MAX_SESSIONS = 200          # al superarse, se descarta la sesión menos usada
MAX_MESSAGES_PER_SESSION = 40
_sessions: OrderedDict[str, list[ConversationMessage]] = OrderedDict()


def get_history(session_id: str | None) -> tuple[str, list[ConversationMessage]]:
    """Devuelve (session_id, historial). Sin id, o con uno desconocido, la sesión empieza vacía."""
    if session_id and session_id in _sessions:
        _sessions.move_to_end(session_id)
        return session_id, list(_sessions[session_id])
    return session_id or uuid.uuid4().hex, []


def save_turn(session_id: str, user_message: str, answer: str) -> None:
    history = _sessions.setdefault(session_id, [])
    history += [ConversationMessage(role="user", content=user_message),
                ConversationMessage(role="assistant", content=answer)]
    del history[:-MAX_MESSAGES_PER_SESSION]
    _sessions.move_to_end(session_id)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


def count_tokens(text: str) -> int:
    return len(encoding.encode(text or ""))


def truncate_by_tokens(text: str, budget: int) -> str:
    tokens = encoding.encode(text or "")
    if len(tokens) <= budget:
        return text or ""
    return encoding.decode(tokens[:budget]).rstrip() + "..."


def format_message(message: ConversationMessage) -> str:
    role = "USUARIO" if message.role == "user" else "ASISTENTE"
    return f"{role}: {truncate_by_tokens(message.content, MESSAGE_TOKEN_CAP)}"


def split_history(history: list[ConversationMessage]) -> tuple[list[str], list[ConversationMessage]]:
    """Separa (mensajes recientes que caben en el presupuesto, mensajes antiguos descartados)."""
    recent: list[str] = []
    used = 0
    for message in reversed(history):
        line = format_message(message)
        cost = count_tokens(line)
        if recent and used + cost > CONTEXT_TOKEN_BUDGETS["history"]:
            break
        recent.insert(0, line)
        used += cost
    return recent, history[: len(history) - len(recent)]


async def summarize(messages: list[ConversationMessage]) -> str:
    text = "\n".join(format_message(m) for m in messages)
    prompt = f"""Resume esta conversación previa conservando el cambio analizado, los sistemas
afectados, los riesgos, los stakeholders y las preguntas abiertas. Devuelve un resumen compacto.

{text}"""
    try:
        summary = await llm.complete(prompt, temperature=0.1, max_tokens=900)
    except Exception as error:
        logger.warning("No se pudo resumir el historial: %s", error)
        return ""
    return truncate_by_tokens(summary, CONTEXT_TOKEN_BUDGETS["summary"])


async def build_context(history: list[ConversationMessage]) -> str:
    """Texto de contexto: resumen de lo antiguo + mensajes recientes."""
    if not history:
        return ""
    recent, dropped = split_history(history)
    summary = await summarize(dropped) if dropped else ""
    return "\n".join(filter(None, (
        f"Resumen previo: {summary}" if summary else "",
        "Historial reciente:\n" + "\n".join(recent),
    )))


async def standalone_query(message: str, context: str) -> str:
    """Convierte un seguimiento ("¿y qué riesgo tiene eso?") en una consulta autocontenida."""
    if not context:
        return message
    prompt = f"""Convierte el último mensaje en una consulta independiente para búsqueda documental.
Usa el contexto sólo para resolver pronombres, sistemas y referencias implícitas.
Devuelve únicamente la consulta reformulada, sin comillas.

{context}

Último mensaje: {message}"""
    try:
        return await llm.complete(prompt, temperature=0, max_tokens=600) or message
    except Exception as error:
        logger.warning("No se pudo reformular el seguimiento: %s", error)
        return message
