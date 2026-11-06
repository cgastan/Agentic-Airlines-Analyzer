"""Consultas de alto nivel sobre el índice híbrido único."""
from __future__ import annotations

from services.embeddings import get_embedding_async
from services.vector_store import HybridIndex


async def retrieve_specifications_rag(index: HybridIndex, query: str) -> list[dict]:
    """Recupera sistemas, integraciones y dependencias."""
    embedding = await get_embedding_async(query)
    chunks = index.search(query, embedding)
    return [chunk_to_result(chunk) for chunk in chunks]


async def retrieve_stakeholder_notes_rag(
    index: HybridIndex,
    query: str,
    exclude_ids: set[str],
) -> list[dict]:
    """Recupera impacto operativo sin repetir chunks técnicos ya usados."""
    embedding = await get_embedding_async(query)
    chunks = index.search(query, embedding, exclude_ids=exclude_ids)
    return [chunk_to_result(chunk) for chunk in chunks]


def chunk_to_result(chunk) -> dict:
    """Expone el fragmento y su procedencia al agente y al contrato API."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "page": chunk.page,
        "section": chunk.section,
        "identifiers": chunk.identifiers,
    }
