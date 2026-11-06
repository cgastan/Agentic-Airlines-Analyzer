"""Endpoints conversacionales: mismo workflow de impacto con memoria de sesión."""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from schemas.models import ChatRequest, ChatResponse, ImpactReportResponse
from services.agent_flow import casual_answer, classify_message_intent, run_agent_workflow_events
from services.context_service import (
    build_session_context,
    get_or_create_session,
    prepare_standalone_query,
    record_report_turn,
    record_turn,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])
COOKIE = "chat_session_id"


def resolve_session_id(session_id: str | None) -> str:
    return session_id or f"SES-{uuid.uuid4().hex[:16].upper()}"


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_readable_report(report: ImpactReportResponse) -> str:
    """Versión en texto del reporte para mostrar en un chat."""
    lines = [report.summary_report.strip(), "", f"Nivel de riesgo: {report.risk_level.value}"]
    if report.risk_justification:
        lines.append(report.risk_justification)
    if report.technical_changes:
        lines += ["", "Cambios técnicos:"]
        lines += [f"{i}. {c.description}" + (f" ({c.affected_system})" if c.affected_system else "")
                  for i, c in enumerate(report.technical_changes, start=1)]
    if report.affected_stakeholders:
        lines += ["", "Stakeholders afectados:"]
        lines += [f"- {s.stakeholder}: {s.impact} (severidad {s.severity.value})"
                  for s in report.affected_stakeholders]
    if report.required_mitigations:
        lines += ["", "Mitigaciones requeridas:"] + [f"- {m}" for m in report.required_mitigations]
    if report.open_questions:
        lines += ["", "Preguntas abiertas:"] + [f"- {q}" for q in report.open_questions]
    if report.sources:
        documents = dict.fromkeys(s.document_id for s in report.sources)
        lines += ["", "Fuentes: " + ", ".join(documents)]
    return "\n".join(lines)


async def run_chat_turn(payload: ChatRequest, request: Request, session_id: str):
    """Generador común a /chat y /chat/stream: eventos de progreso y resultado final."""
    state = get_or_create_session(session_id)
    intent = await classify_message_intent(payload.message)

    if intent == "casual_chat":
        answer = await casual_answer(payload.message)
        await record_turn(state, payload.message, answer)
        yield {"event": "result", "response": ChatResponse(
            session_id=session_id, answer=answer, intent=intent)}
        return

    yield {"event": "progress", "step": "context", "message": "Resolviendo contexto de la conversación"}
    query = await prepare_standalone_query(payload.message, state)
    async for event in run_agent_workflow_events(
        query,
        request.app.state.index,
        target_system=payload.target_system,
        conversation_context=build_session_context(state),
    ):
        if event["event"] == "progress":
            yield event
            continue
        report = ImpactReportResponse.model_validate(event["report"])
        await record_report_turn(state, payload.message, report)
        yield {"event": "result", "response": ChatResponse(
            session_id=session_id,
            answer=format_readable_report(report),
            intent=intent,
            report=report,
            sources=report.sources,
            insufficient_information=report.insufficient_information,
        )}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    chat_session_id: str | None = Cookie(default=None),
):
    """Consulta conversacional; mantiene la sesión con una cookie."""
    session_id = resolve_session_id(chat_session_id)
    response.set_cookie(COOKIE, session_id, httponly=True, samesite="lax")
    try:
        async for event in run_chat_turn(payload, request, session_id):
            if event["event"] == "result":
                return event["response"]
    except Exception as error:
        logger.exception("Error en /chat")
        raise HTTPException(500, f"Error interno ejecutando el agente: {error}")
    raise HTTPException(500, "El agente no devolvió respuesta.")


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    chat_session_id: str | None = Cookie(default=None),
):
    """SSE: `progress` por paso, `text` línea a línea y `final` con el reporte estructurado."""
    session_id = resolve_session_id(chat_session_id)

    async def event_stream():
        try:
            async for event in run_chat_turn(payload, request, session_id):
                if event["event"] == "progress":
                    yield sse("progress", {"step": event["step"], "message": event["message"]})
                    continue
                result: ChatResponse = event["response"]
                for line in result.answer.splitlines():
                    yield sse("text", line)
                yield sse("final", result.model_dump(mode="json"))
        except Exception as error:
            logger.exception("Error en /chat/stream")
            yield sse("error", {"type": "error", "detail": str(error)})

    streaming = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    streaming.set_cookie(COOKIE, session_id, httponly=True, samesite="lax")
    return streaming
