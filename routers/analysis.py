"""Endpoints de análisis de impacto y gestión de la base documental."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from config import settings
from schemas.models import (
    TARGET_SYSTEM_DESCRIPTION,
    TARGET_SYSTEM_EXAMPLES,
    ChangeRequestPayload,
    DocumentsResponse,
    ImpactReportResponse,
    IngestedDocument,
)
from services.agent_flow import run_agent_workflow_events
from services.ingestion_service import DOC_TYPES, extend_index
from services.readers import SUPPORTED_EXTENSIONS, read_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Impact Analysis"])

FILE_DESCRIPTION = f"Documento ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) con la solicitud de cambio u otra evidencia"
DOC_TYPE_DESCRIPTION = f"Opcional: {', '.join(sorted(DOC_TYPES))}. Si se omite, se infiere del contenido."


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def validate_payload(analyst_prompt: str, target_system: str | None) -> ChangeRequestPayload:
    try:
        return ChangeRequestPayload(analyst_prompt=analyst_prompt, target_system=target_system or None)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_context=False))


def validate_doc_type(doc_type: str | None) -> str | None:
    if doc_type and doc_type not in DOC_TYPES:
        raise HTTPException(422, f"document_type inválido. Valores: {', '.join(sorted(DOC_TYPES))}")
    return doc_type or None


async def read_upload(file: UploadFile | None) -> tuple[str, bytes] | None:
    """Lee el archivo (antes de abrir el stream) y valida formato, tamaño y contenido."""
    if file is None or not file.filename:
        return None
    filename = Path(file.filename).name  # Evita rutas tipo ../../archivo
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Formato no admitido. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"El archivo supera {settings.MAX_UPLOAD_MB} MB.")
    if not data.strip():
        raise HTTPException(422, "El documento está vacío.")
    return filename, data


async def extract_upload_text(upload: tuple[str, bytes]) -> str:
    filename, data = upload
    try:
        pages = await asyncio.to_thread(read_document, filename, data)
    except Exception as error:
        raise HTTPException(422, f"No se pudo extraer texto de {filename}: {error}")
    text = "\n".join(page.text for page in pages).strip()
    if not text:
        raise HTTPException(422, f"{filename} no contiene texto extraíble (¿PDF escaneado?).")
    return text


async def build_request_index(request: Request, upload, doc_type: str | None):
    """Índice para esta solicitud: base + archivo subido. El índice global no se modifica."""
    base_index = request.app.state.index
    if upload is None:
        return base_index
    index, _ = await extend_index(base_index, [upload], doc_type)
    return index


@router.post("/analyze", response_model=ImpactReportResponse)
async def analyze_change_request(
    request: Request,
    analyst_prompt: str = Form(..., description="Descripción del cambio propuesto"),
    target_system: str | None = Form(None, description=TARGET_SYSTEM_DESCRIPTION,
                                     examples=TARGET_SYSTEM_EXAMPLES),
    document_type: str | None = Form(None, description=DOC_TYPE_DESCRIPTION),
    file: UploadFile | None = File(None, description=FILE_DESCRIPTION),
):
    """Ejecuta el workflow completo y devuelve el reporte de impacto validado."""
    payload = validate_payload(analyst_prompt, target_system)
    doc_type = validate_doc_type(document_type)
    upload = await read_upload(file)
    uploaded_text = await extract_upload_text(upload) if upload else ""
    try:
        index = await build_request_index(request, upload, doc_type)
        report = None
        async for event in run_agent_workflow_events(
            payload.analyst_prompt, index,
            target_system=payload.target_system, uploaded_text=uploaded_text,
        ):
            if event["event"] == "report":
                report = event["report"]
        return ImpactReportResponse.model_validate(report)
    except Exception as error:
        logger.exception("Error en /analyze")
        raise HTTPException(500, f"Error interno ejecutando el agente: {error}")


@router.post("/analyze/stream")
async def analyze_change_request_stream(
    request: Request,
    analyst_prompt: str = Form(..., description="Descripción del cambio propuesto"),
    target_system: str | None = Form(None, description=TARGET_SYSTEM_DESCRIPTION,
                                     examples=TARGET_SYSTEM_EXAMPLES),
    document_type: str | None = Form(None, description=DOC_TYPE_DESCRIPTION),
    file: UploadFile | None = File(None, description=FILE_DESCRIPTION),
):
    """Server-Sent Events: `progress` por cada paso del agente y `final` con el reporte."""
    # Validación y lectura del archivo ANTES del stream: errores con código HTTP real
    # y sin depender de un UploadFile que ya podría estar cerrado.
    payload = validate_payload(analyst_prompt, target_system)
    doc_type = validate_doc_type(document_type)
    upload = await read_upload(file)
    uploaded_text = await extract_upload_text(upload) if upload else ""

    async def event_stream():
        try:
            if upload:
                yield sse("progress", {"step": "ingest", "message": f"Indexando {upload[0]}"})
            index = await build_request_index(request, upload, doc_type)
            async for event in run_agent_workflow_events(
                payload.analyst_prompt, index,
                target_system=payload.target_system, uploaded_text=uploaded_text,
            ):
                if event["event"] == "progress":
                    yield sse("progress", {"step": event["step"], "message": event["message"]})
                else:
                    report = ImpactReportResponse.model_validate(event["report"])
                    yield sse("final", report.model_dump(mode="json"))
        except Exception as error:
            logger.exception("Error en /analyze/stream")
            yield sse("error", {"type": "error", "detail": str(error)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
async def list_documents(request: Request):
    """Documentos indexados y el tipo documental asignado a cada uno."""
    index = request.app.state.index
    if index is None:
        return DocumentsResponse(documents=[], total_chunks=0)
    counts: dict[str, int] = {}
    for chunk in index.chunks:
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    return DocumentsResponse(
        documents=[
            IngestedDocument(filename=name, doc_type=doc_type, chunks=counts[name])
            for name, doc_type in index.sources.items()
        ],
        total_chunks=len(index),
    )


@router.post("/documents", response_model=DocumentsResponse, tags=["Documents"])
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(..., description=FILE_DESCRIPTION),
    document_type: str | None = Form(None, description=DOC_TYPE_DESCRIPTION),
):
    """Agrega documentos a la base de conocimiento (se guardan en data/ y se indexan)."""
    doc_type = validate_doc_type(document_type)
    uploads = [upload for upload in [await read_upload(f) for f in files] if upload]
    if not uploads:
        raise HTTPException(422, "No se recibió ningún archivo.")
    for upload in uploads:
        await extract_upload_text(upload)  # 422 temprano si alguno no tiene texto

    async with request.app.state.index_lock:
        index, chunks = await extend_index(request.app.state.index, uploads, doc_type)
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for filename, data in uploads:
            (settings.DATA_DIR / filename).write_bytes(data)
        request.app.state.index = index

    counts: dict[str, IngestedDocument] = {}
    for chunk in chunks:
        item = counts.setdefault(chunk.source, IngestedDocument(
            filename=chunk.source, doc_type=chunk.doc_type, chunks=0))
        item.chunks += 1
    return DocumentsResponse(documents=list(counts.values()), total_chunks=len(index))
