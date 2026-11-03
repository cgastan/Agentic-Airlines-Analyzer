# routers/analysis.py

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pypdf import PdfReader
from schemas.models import ImpactReportResponse
from services.agent_flow import run_agent_workflow
from services.chunking import build_chunks_from_documents
from services.embeddings import embed_chunks

router = APIRouter(prefix="/api", tags=["Impact Analysis"])

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
            # Clasificamos según el nombre del archivo (ej: si contiene 'stakeholder' o 'arch')
            if "stakeholder" in file_path.name.lower() or "policy" in file_path.name.lower():
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
    request_id: str = Form(..., description="ID único de la solicitud"),
    analyst_prompt: str = Form(...,
                               description="Descripción del cambio propuesto"),
    file: UploadFile = File(
        None, description="PDF opcional con detalles adicionales de la solicitud de cambio")
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
        if file and file.filename.endswith(".pdf"):
            file_bytes = await file.read()
            # Guardar temporalmente o leer desde memoria con pypdf
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            uploaded_text = ""
            for page in reader.pages:
                uploaded_text += page.extract_text() or ""

            uploaded_doc = [{"id": file.filename,
                             "title": file.filename, "text": uploaded_text}]
            uploaded_chunks = embed_chunks(
                build_chunks_from_documents(uploaded_doc))

            # Añadir dinámicamente al store técnico para la búsqueda RAG de esta petición
            active_tech_store.extend(uploaded_chunks)

        # Ejecutar el flujo completo del agente (con los 7 pasos requeridos)
        report_text = run_agent_workflow(
            analyst_prompt=analyst_prompt,
            tech_store=active_tech_store,
            stakeholder_store=active_stakeholder_store
        )

        return ImpactReportResponse(
            request_id=request_id,
            technical_changes=[
                "Actualización evaluada mediante RAG y archivos adjuntos"],
            risk_level="Medio-Alto",
            affected_stakeholders=["Operaciones de Vuelo",
                                   "Agentes de Tierra", "Pasajeros"],
            required_mitigations=[
                "Validar dependencias técnicas y actualizar documentación de capacitación"],
            summary_report=report_text
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interno ejecutando el agente: {str(e)}")
