"""Funciones RAG del agente sobre el índice híbrido.

Ambas recuperaciones buscan en el mismo índice; se diferencian por la consulta
(técnica vs. de stakeholders, generadas en `initialize`) y porque la segunda
excluye los fragmentos que ya trajo la primera.
"""
from __future__ import annotations

from config import settings
from services.chunking import Chunk
from services.embeddings import get_embedding_async
from services.vector_store import HybridIndex


def chunk_to_result(chunk: Chunk) -> dict:
    """Expone el fragmento y su procedencia al agente y al contrato de la API."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "page": chunk.page,
        "section": chunk.section,
    }


async def search(
    index: HybridIndex | None,
    query: str,
    top_k: int,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    if index is None or not query.strip():
        return []
    embedding = await get_embedding_async(query)
    chunks = index.search(query, embedding, top_k=top_k, exclude_ids=exclude_ids)
    return [chunk_to_result(chunk) for chunk in chunks]


async def search_specifications(index: HybridIndex | None, query: str) -> list[dict]:
    """Componentes, integraciones, dependencias y detalles de la solicitud de cambio."""
    return await search(index, query, settings.SPEC_TOP_K)


async def search_stakeholder_notes(
    index: HybridIndex | None, query: str, exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Impactos operativos, feedback de stakeholders y políticas, sin repetir chunks técnicos."""
    return await search(index, query, settings.STAKEHOLDER_TOP_K, exclude_ids)


def format_context(chunks: list[dict]) -> str:
    """Texto con la cita de cada fragmento, para que el LLM pueda referenciarlo."""
    blocks = []
    for chunk in chunks:
        location = ", ".join(filter(None, (
            f"pág. {chunk['page']}" if chunk.get("page") else None,
            chunk.get("section"),
        )))
        citation = f"{chunk['source']}" + (f" ({location})" if location else "")
        blocks.append(f"[{chunk['id']}] Fuente: {citation}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks) or "(sin resultados)"
