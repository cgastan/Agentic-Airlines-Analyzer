import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from schemas.models import ImpactReportResponse
from services.agent_flow import run_agent_workflow
from services.chunking import build_chunks_from_documents
from services.document_extraction import extract_document_text
from services.document_stores import operational_chunks, technical_chunks
from services.embeddings import embed_chunks

router = APIRouter(prefix="/api", tags=["Impact Analysis"])


def create_request_id() -> str:
    return f"CHG-{uuid.uuid4().hex[:12].upper()}"


async def process_analysis(analyst_prompt: str, file: UploadFile | None) -> ImpactReportResponse:
    active_technical_store = list(technical_chunks)
    active_operational_store = list(operational_chunks)

    if file:
        file_bytes = await file.read()
        extension = Path(file.filename).suffix.lower()
        if extension not in {".pdf", ".doc", ".docx", ".txt", ".md"}:
            raise HTTPException(
                status_code=415,
                detail="Formato no admitido. Use PDF, DOC, DOCX, TXT o MD.",
            )

        try:
            uploaded_text = extract_document_text(file_bytes, extension)
        except Exception as extraction_error:
            raise HTTPException(
                status_code=422,
                detail=f"No se pudo extraer texto del documento: {str(extraction_error)}",
            )

        if not uploaded_text.strip():
            raise HTTPException(
                status_code=422,
                detail="El documento está vacío o no contiene texto.",
            )

        uploaded_document = [{
            "id": file.filename,
            "title": file.filename,
            "text": uploaded_text,
        }]
        uploaded_chunks = embed_chunks(
            build_chunks_from_documents(uploaded_document)
        )
        active_technical_store.extend(uploaded_chunks)

    report = run_agent_workflow(
        analyst_prompt=analyst_prompt,
        tech_store=active_technical_store,
        stakeholder_store=active_operational_store,
        allow_casual=False,
    )
    report["request_id"] = create_request_id()
    return ImpactReportResponse.model_validate(report)


@router.post("/analyze", response_model=ImpactReportResponse)
async def analyze_change_request(
    analyst_prompt: str = Form(...,
                               description="Descripción del cambio propuesto"),
    file: UploadFile = File(
        None,
        description="Documento PDF, DOC, DOCX, TXT o MD con detalles adicionales",
    ),
):
    try:
        return await process_analysis(analyst_prompt, file)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno ejecutando el agente: {str(error)}",
        )


@router.post("/analyze/stream")
async def analyze_change_request_stream(
    analyst_prompt: str = Form(...,
                               description="Descripción del cambio propuesto"),
    file: UploadFile = File(
        None,
        description="Documento PDF, DOC, DOCX, TXT o MD con detalles adicionales",
    ),
):
    async def event_stream():
        stages = {
            "initialize": "Inicializando solicitud de cambio",
            "retrieve": "Recuperando arquitectura y dependencias",
            "grade": "Evaluando suficiencia del contexto",
            "generate": "Generando reporte de impacto",
        }
        for _, label in stages.items():
            yield "event: progress\n" + f"data: {json.dumps(label, ensure_ascii=False)}\n\n"

        try:
            report = await process_analysis(analyst_prompt, file)
            final_payload = report.model_dump(mode="json")
            yield "event: final\ndata: " + json.dumps(final_payload, ensure_ascii=False) + "\n\n"
        except HTTPException as error:
            payload = {"type": "error", "detail": error.detail}
            yield "event: error\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        except Exception as error:
            payload = {"type": "error", "detail": str(error)}
            yield "event: error\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
