import json
import uuid
import inspect
from contextvars import ContextVar
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from schemas.models import ImpactReportResponse
from services.agent_flow import run_agent_workflow
from services.chunking import build_chunks_from_documents
from services.document_extraction import extract_document_text
from services.ingestion_service import add_documents

# Prefijo común y etiqueta usados por FastAPI para agrupar los endpoints.
router = APIRouter(prefix="/api", tags=["Impact Analysis"])
current_index: ContextVar = ContextVar("current_index", default=None)


def create_request_id() -> str:
    """Genera un identificador corto único para cada solicitud."""
    return f"CHG-{uuid.uuid4().hex[:12].upper()}"


async def process_analysis(
    analyst_prompt: str,
    file: UploadFile | None,
) -> ImpactReportResponse:
    """Procesa el cambio solicitado y devuelve un informe de impacto validado."""
    # Se trabaja sobre copias para no modificar los almacenes cargados globalmente.
    if file:
        file_bytes = await file.read()
        extension = Path(file.filename).suffix.lower()
        # Rechaza formatos que el extractor de documentos no admite.
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

        # Un documento sin texto no aporta contexto útil al flujo de agentes.
        if not uploaded_text.strip():
            raise HTTPException(
                status_code=422,
                detail="El documento está vacío o no contiene texto.",
            )

        uploaded = Path(file.filename or "upload.txt")
        uploaded_index = await add_documents([(uploaded, file_bytes)])
        # Mantiene una sesión de consulta independiente de las cargas persistentes.
        # Las cargas usan temporalmente el índice recién construido para esta solicitud.
        index = uploaded_index

    # Ejecuta el flujo de agentes con el contexto técnico y operativo disponible.
    index = current_index.get()
    result = run_agent_workflow(
        analyst_prompt=analyst_prompt,
        index=index,
        allow_casual=False,
    )
    report = await result if inspect.isawaitable(result) else result
    report["request_id"] = create_request_id()
    return ImpactReportResponse.model_validate(report)


# Endpoint que espera el resultado completo antes de devolver la respuesta JSON.
@router.post("/analyze", response_model=ImpactReportResponse)
async def analyze_change_request(
    request: Request,
    analyst_prompt: str = Form(...,
                               description="Descripción del cambio propuesto"),
    file: UploadFile = File(
        None,
        description="Documento PDF, DOC, DOCX, TXT o MD con detalles adicionales",
    ),
):
    try:
        current_index.set(getattr(request.app.state, "index", None))
        return await process_analysis(analyst_prompt, file)
    except HTTPException:
        # Conserva los códigos HTTP específicos definidos por el procesamiento.
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno ejecutando el agente: {str(error)}",
        )


# Endpoint de progreso que devuelve el análisis mediante Server-Sent Events.
@router.post("/analyze/stream")
async def analyze_change_request_stream(
    request: Request,
    analyst_prompt: str = Form(...,
                               description="Descripción del cambio propuesto"),
    file: UploadFile = File(
        None,
        description="Documento PDF, DOC, DOCX, TXT o MD con detalles adicionales",
    ),
):
    async def event_stream():
        """Emite las fases del análisis y finalmente el informe o el error."""
        stages = {
            "initialize": "Inicializando solicitud de cambio",
            "retrieve": "Recuperando arquitectura y dependencias",
            "grade": "Evaluando suficiencia del contexto",
            "generate": "Generando reporte de impacto",
        }
        # Los mensajes informan al cliente qué etapa del análisis está ejecutándose.
        for _, label in stages.items():
            yield "event: progress\n" + f"data: {json.dumps(label, ensure_ascii=False)}\n\n"

        try:
            current_index.set(getattr(request.app.state, "index", None))
            report = await process_analysis(analyst_prompt, file)
            # Convierte el modelo validado a datos JSON para enviarlo por SSE.
            final_payload = report.model_dump(mode="json")
            yield "event: final\ndata: " + json.dumps(final_payload, ensure_ascii=False) + "\n\n"
        except HTTPException as error:
            payload = {"type": "error", "detail": error.detail}
            yield "event: error\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        except Exception as error:
            payload = {"type": "error", "detail": str(error)}
            yield "event: error\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    # Mantiene la conexión abierta mientras se generan los eventos.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
