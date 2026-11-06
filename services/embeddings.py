"""Embeddings densos: Sentence Transformers local u OpenAI, según EMBEDDING_PROVIDER."""
from __future__ import annotations

import asyncio
from functools import lru_cache

from config import settings

BATCH_SIZE = 64


@lru_cache(maxsize=1)
def _local_encoder():
    # Import diferido: cargar el modelo tarda, así que sólo ocurre al primer uso.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def check_embedding_config() -> None:
    """Falla de inmediato (y con un mensaje claro) si falta la clave de OpenAI."""
    if settings.EMBEDDING_PROVIDER.lower() == "openai" and not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=openai requiere OPENAI_API_KEY en el archivo .env "
            "(o use EMBEDDING_PROVIDER=local para embeddings locales)."
        )


@lru_cache(maxsize=1)
def _openai_client():
    from openai import AsyncOpenAI
    check_embedding_config()
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings por lotes sin bloquear el event loop."""
    if not texts:
        return []
    if settings.EMBEDDING_PROVIDER.lower() == "openai":
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            response = await _openai_client().embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=texts[start:start + BATCH_SIZE],
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def encode() -> list[list[float]]:
        return _local_encoder().encode(texts, batch_size=BATCH_SIZE).tolist()

    return await asyncio.to_thread(encode)


async def get_embedding_async(text: str) -> list[float]:
    """Embedding de una sola consulta."""
    return (await embed_texts([text]))[0]
