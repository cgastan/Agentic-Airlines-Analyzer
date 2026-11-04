# routers/analysis.py

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
from schemas.models import ChatRequest, ImpactReportResponse
from services.document_extraction import extract_document_text
from services.agent_flow import run_agent_workflow
from services.chunking import build_chunks_from_documents
from services.embeddings import embed_chunks

router = APIRouter(prefix="/api", tags=["Impact Analysis"])


def create_request_id() -> str:
    return f"CHG-{uuid.uuid4().hex[:12].upper()}"


# Directorio de documentos iniciales de Agentic Airlines SRL
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def load_initial_pdfs() -> tuple[list[dict], list[dict]]:
    """Lee y procesa los PDFs iniciales colocados en la carpeta data/."""
    tech_docs = []
    stakeholder_notes = []

    if not list(DATA_DIR.glob("*.pdf")):
        # Si no hay PDFs, usamos datos por defecto para evitar fallos iniciales
        tech_docs.append({
            "id": "default_arch",
            "title": "Default Architecture Spec",
            "text": "Agentic Airlines core reservation and inventory system v3."
        })
    else:
        for file_path in DATA_DIR.glob("*.pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            doc_item = {"id": file_path.name,
                        "title": file_path.stem, "text": text}
            # Clasificamos documentos operativos en español e inglés.
            filename = file_path.name.lower()
            operational_markers = (
                "stakeholder",
                "policy",
                "politica",
                "política",
                "procedimiento",
                "feedback",
                "riesgo",
            )
            if any(marker in filename for marker in operational_markers):
                stakeholder_notes.append(doc_item)
            else:
                tech_docs.append(doc_item)

    return tech_docs, stakeholder_notes


# Cargar stores iniciales al iniciar el servidor
initial_tech, initial_stakeholder = load_initial_pdfs()
tech_chunks = embed_chunks(build_chunks_from_documents(initial_tech))
stakeholder_chunks = embed_chunks(
    build_chunks_from_documents(initial_stakeholder))


@router.post("/analyze", response_model=ImpactReportResponse)
async def analyze_change_request(
    analyst_prompt: str = Form(...,
                               description="Descripción del cambio propuesto"),
    file: UploadFile = File(
        None, description="Documento PDF, DOC, DOCX, TXT o MD con detalles adicionales")
):
    """
    Endpoint para procesar solicitudes de cambio combinando la base de datos inicial 
    de PDFs y los archivos subidos dinámicamente por el analista.
    """
    try:
        # Copiar los almacenes base para esta ejecución
        active_tech_store = list(tech_chunks)
        active_stakeholder_store = list(stakeholder_chunks)

        # Si el usuario subió un archivo en tiempo de ejecución, lo procesamos al vuelo
        if file:
            file_bytes = await file.read()
            extension = Path(file.filename).suffix.lower()
            if extension not in {".pdf", ".doc", ".docx", ".txt", ".md"}:
                raise HTTPException(
                    status_code=415,
                    detail="Formato no admitido. Use PDF, DOC, DOCX, TXT o MD.")
            try:
                uploaded_text = extract_document_text(file_bytes, extension)
            except Exception as extraction_error:
                raise HTTPException(
                    status_code=422,
                    detail=f"No se pudo extraer texto del documento: {str(extraction_error)}")

            if not uploaded_text.strip():
                raise HTTPException(
                    status_code=422, detail="El documento está vacío o no contiene texto.")

            uploaded_doc = [{"id": file.filename,
                             "title": file.filename, "text": uploaded_text}]
            uploaded_chunks = embed_chunks(
                build_chunks_from_documents(uploaded_doc))

            # Añadir dinámicamente al store técnico para la búsqueda RAG de esta petición
            active_tech_store.extend(uploaded_chunks)

        # Ejecutar el flujo completo del agente (con los 7 pasos requeridos)
        report = run_agent_workflow(
            analyst_prompt=analyst_prompt,
            tech_store=active_tech_store,
            stakeholder_store=active_stakeholder_store
        )
        report["request_id"] = create_request_id()

        return ImpactReportResponse.model_validate(report)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interno ejecutando el agente: {str(e)}")


@router.post("/chat", response_model=ImpactReportResponse)
async def chat(payload: ChatRequest):
    """Consulta conversacional que reutiliza el mismo flujo de análisis de impacto."""
    try:
        prompt = payload.message
        if payload.target_system:
            prompt = f"Sistema objetivo: {payload.target_system}. {prompt}"

        report = run_agent_workflow(
            analyst_prompt=prompt,
            tech_store=tech_chunks,
            stakeholder_store=stakeholder_chunks
        )
        report["request_id"] = create_request_id()
        return ImpactReportResponse.model_validate(report)
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
async def chat_stream(payload: ChatRequest):
    """Transmite progreso y un reporte narrativo más legible como JSON Lines."""
    async def event_stream():
        import json

        for stage in (
            "initialize",
            "retrieve_specifications_rag",
            "retrieve_stakeholder_notes_rag",
            "grade_context",
        ):
            yield json.dumps(
                {"type": "progress", "step": stage}, ensure_ascii=False) + "\n"

        report = run_agent_workflow(
            analyst_prompt=payload.message,
            tech_store=tech_chunks,
            stakeholder_store=stakeholder_chunks
        )
        report["request_id"] = create_request_id()
        validated = ImpactReportResponse.model_validate(report)
        narrative, citations = format_readable_report(validated)
        yield json.dumps(
            {
                "type": "complete",
                "request_id": validated.request_id,
                "narrative": narrative,
                "citations": citations,
                "report": validated.model_dump(mode="json"),
            },
            ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
