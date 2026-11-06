"""Funciones RAG del agente sobre el índice híbrido.

Cada recuperación consulta un subconjunto distinto del corpus según el tipo
documental inferido en la ingesta: especificaciones/solicitudes de cambio por un
lado y notas de stakeholders/políticas por el otro.
"""
from __future__ import annotations

from config import settings
from services.chunking import Chunk
from services.embeddings import get_embedding_async
from services.ingestion_service import SPECIFICATION_TYPES, STAKEHOLDER_TYPES
from services.vector_store import HybridIndex


def chunk_to_result(chunk: Chunk) -> dict:
    """Expone el fragmento y su procedencia al agente y al contrato de la API."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "doc_type": chunk.doc_type,
        "page": chunk.page,
        "section": chunk.section,
    }


async def search(
    index: HybridIndex | None,
    query: str,
    doc_types: frozenset[str],
    top_k: int,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    if index is None or not query.strip():
        return []
    embedding = await get_embedding_async(query)
    chunks = index.search(
        query, embedding, top_k=top_k, doc_types=set(doc_types), exclude_ids=exclude_ids)
    return [chunk_to_result(chunk) for chunk in chunks]


async def search_specifications(index: HybridIndex | None, query: str) -> list[dict]:
    """Arquitectura, flujos de proceso y solicitudes de cambio."""
    return await search(index, query, SPECIFICATION_TYPES, settings.SPEC_TOP_K)


async def search_stakeholder_notes(
    index: HybridIndex | None, query: str, exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Feedback de stakeholders y requisitos de políticas, sin repetir chunks técnicos."""
    return await search(index, query, STAKEHOLDER_TYPES, settings.STAKEHOLDER_TOP_K, exclude_ids)


def format_context(chunks: list[dict]) -> str:
    """Texto numerado con la cita de cada fragmento, para que el LLM pueda referenciarlo."""
    blocks = []
    for chunk in chunks:
        location = ", ".join(filter(None, (
            f"pág. {chunk['page']}" if chunk.get("page") else None,
            chunk.get("section"),
        )))
        citation = f"{chunk['source']}" + (f" ({location})" if location else "")
        blocks.append(f"[{chunk['id']}] Fuente: {citation}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks) or "(sin resultados)"
