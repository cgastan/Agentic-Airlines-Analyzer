# routers/analysis.py

import uuid  # genera identificadores únicos
import inspect

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from schemas.models import (
    ChatRequest,
    ChatResponse,
    ImpactReportResponse,
    SourceReference,
)
from services.agent_flow import classify_message_intent, run_agent_workflow
from services.context_service import (
    get_or_create_session,
    prepare_context_message,
    record_report_turn,
)

router = APIRouter(prefix="/api", tags=["Impact Analysis"])


def create_request_id() -> str:
    return f"CHG-{uuid.uuid4().hex[:12].upper()}"


def resolve_session_id(session_id: str | None) -> str:
    return session_id or f"SES-{uuid.uuid4().hex[:16].upper()}"


def get_unique_sources(report: ImpactReportResponse) -> list[SourceReference]:
    sources = []
    seen = set()
    for source in report.sources:
        key = (source.document_id, source.chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    response: Response,
    request: Request,
    chat_session_id: str | None = Cookie(default=None),
):
    """Consulta conversacional con respuesta en texto y fuentes."""
    try:
        session_id = resolve_session_id(chat_session_id)
        response.set_cookie("chat_session_id", session_id,
                            httponly=True, samesite="lax")
        state = get_or_create_session(session_id)
        intent = classify_message_intent(payload.message)
        standalone_query, _, session_context = prepare_context_message(
            state, payload.message)
        prompt = "\n\n".join(filter(None, (session_context, standalone_query)))
        if payload.target_system:
            prompt = f"Sistema objetivo: {payload.target_system}. {prompt}"

        workflow = run_agent_workflow(
            analyst_prompt=prompt,
            index=request.app.state.index,
        )
        report = await workflow if inspect.isawaitable(workflow) else workflow
        report["request_id"] = create_request_id()
        validated = ImpactReportResponse.model_validate(report)
        record_report_turn(state, payload.message, validated)
        return ChatResponse(
            session_id=session_id,
            answer=validated.summary_report,
            sources=get_unique_sources(validated),
            insufficient_information=validated.insufficient_information,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interno ejecutando el agente: {str(e)}")


def format_readable_report(report: ImpactReportResponse) -> tuple[str, list[dict]]:
    lines = [report.summary_report.strip()]

    if report.technical_changes:
        lines.extend(["", "Cambios técnicos"])
        for index, change in enumerate(report.technical_changes, start=1):
            system = f" ({change.affected_system})" if change.affected_system else ""
            lines.append(f"{index}. {change.description}{system}")

    if report.affected_stakeholders:
        lines.extend(["", "Impacto en stakeholders"])
        for stakeholder in report.affected_stakeholders:
            lines.append(
                f"- {stakeholder.stakeholder}: {stakeholder.impact} "
                f"(nivel {stakeholder.severity.value.lower()})"
            )

    if report.required_mitigations:
        lines.extend(["", "Mitigaciones requeridas"])
        lines.extend(
            f"- {mitigation}" for mitigation in report.required_mitigations
        )

    lines.extend(["", f"Nivel de riesgo: {report.risk_level.value}"])

    citations = []
    seen_documents = set()
    for source in report.sources:
        key = (source.document_id, source.chunk_id)
        if key in seen_documents:
            continue
        seen_documents.add(key)
        citations.append({
            "document": source.document_id,
            "title": source.title,
            "fragment": source.chunk_id,
        })

    if citations:
        lines.extend(["", "Documentos citados"])
        lines.extend(
            f"- {citation['document']}"
            + (f" ({citation['fragment']})" if citation['fragment'] else "")
            for citation in citations
        )

    return "\n".join(lines), citations


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    chat_session_id: str | None = Cookie(default=None),
):
    """Transmite texto legible y termina con el reporte estructurado."""
    session_id = resolve_session_id(chat_session_id)
    get_or_create_session(session_id)

    async def event_stream():
        import json

        stage_labels = {
            "initialize": "Inicializando solicitud",
            "retrieve_specifications_rag": "Consultando especificaciones técnicas",
            "retrieve_stakeholder_notes_rag": "Consultando impactos de stakeholders",
            "grade_context": "Evaluando contexto recuperado",
        }
        intent = classify_message_intent(payload.message)
        if intent == "impact_analysis":
            for stage, label in stage_labels.items():
                yield f"event: progress\ndata: {json.dumps(label, ensure_ascii=False)}\n\n"
        else:
            yield "event: progress\ndata: Preparando una respuesta\n\n"

        state = get_or_create_session(session_id)
        standalone_query, _, session_context = prepare_context_message(
            state, payload.message)
        prompt = "\n\n".join(filter(None, (session_context, standalone_query)))
        if payload.target_system:
            prompt = f"Sistema objetivo: {payload.target_system}. {prompt}"

        workflow = run_agent_workflow(
            analyst_prompt=prompt,
            index=getattr(request.app.state, "index", None),
        )
        report = await workflow if inspect.isawaitable(workflow) else workflow
        report["request_id"] = create_request_id()
        validated = ImpactReportResponse.model_validate(report)
        sources = get_unique_sources(validated)
        record_report_turn(state, payload.message, validated)

        text_chunks = validated.summary_report.split("\n")
        for chunk in text_chunks:
            encoded_chunk = chunk.encode(
                "utf-8", errors="replace").decode("utf-8")
            yield f"event: text\ndata: {encoded_chunk}\n\n"

        final_payload = {
            "type": "complete",
            "session_id": session_id,
            "answer": validated.summary_report,
            "sources": [source.model_dump(mode="json") for source in sources],
            "insufficient_information": validated.insufficient_information,
        }
        yield (
            "event: final\n"
            "data: " + json.dumps(final_payload, ensure_ascii=False) + "\n\n"
        )

    streaming_response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    streaming_response.set_cookie(
        "chat_session_id", session_id, httponly=True, samesite="lax"
    )
    return streaming_response
