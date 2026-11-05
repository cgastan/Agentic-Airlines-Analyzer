from pathlib import Path

from pypdf import PdfReader

from services.chunking import build_chunks_from_documents
from services.embeddings import embed_chunks

DATA_DIR = Path("data")


def load_initial_documents() -> tuple[list[dict], list[dict]]:
    DATA_DIR.mkdir(exist_ok=True)
    technical_documents = []
    operational_documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        technical_documents.append({
            "id": "default_arch",
            "title": "Default Architecture Spec",
            "text": "Agentic Airlines core reservation and inventory system v3.",
        })
        return technical_documents, operational_documents

    operational_markers = (
        "stakeholder",
        "policy",
        "politica",
        "política",
        "procedimiento",
        "feedback",
        "riesgo",
    )

    for file_path in pdf_files:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        document = {
            "id": file_path.name,
            "title": file_path.stem,
            "text": text,
        }
        filename = file_path.name.lower()

        if any(marker in filename for marker in operational_markers):
            operational_documents.append(document)
        else:
            technical_documents.append(document)

    return technical_documents, operational_documents


technical_documents, operational_documents = load_initial_documents()
technical_chunks = embed_chunks(
    build_chunks_from_documents(technical_documents)
)
operational_chunks = embed_chunks(
    build_chunks_from_documents(operational_documents)
)
