"""Ingesta de documentos: extracción -> chunking -> embeddings -> índice.

Todos los documentos (arquitectura, solicitudes de cambio, notas de stakeholders,
políticas) pasan por el mismo pipeline y terminan en un único índice.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import numpy as np

from config import settings
from services.chunking import Chunk, chunk_pages
from services.embeddings import embed_texts
from services.readers import SUPPORTED_EXTENSIONS, read_document
from services.vector_store import HybridIndex

logger = logging.getLogger(__name__)


async def _embed_with_cache(chunks: list[Chunk]) -> list[list[float]]:
    """Embeddings de un documento, cacheados en disco por contenido + modelo."""
    cache_dir = settings.CACHE_DIR / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = [chunk.context_text for chunk in chunks]
    key = hashlib.sha256(
        (settings.embedding_model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{key}.npy"
    if cache_path.exists():
        return np.load(cache_path).tolist()
    vectors = await embed_texts(texts)
    np.save(cache_path, np.asarray(vectors, dtype=np.float32))
    return vectors


async def ingest_document(filename: str, data: bytes) -> tuple[list[Chunk], list[list[float]]]:
    """Lee, fragmenta y vectoriza un archivo (de data/ o subido por la API)."""
    pages = await asyncio.to_thread(read_document, filename, data)
    chunks = chunk_pages(pages)
    if not chunks:
        return [], []
    logger.info("Ingestado %s (%d chunks).", filename, len(chunks))
    return chunks, await _embed_with_cache(chunks)


async def _ingest_many(files: list[tuple[str, bytes]]) -> tuple[list[Chunk], list[list[float]]]:
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for filename, data in files:
        new_chunks, new_vectors = await ingest_document(filename, data)
        chunks += new_chunks
        vectors += new_vectors
    return chunks, vectors


async def build_index(data_dir: Path | None = None) -> HybridIndex | None:
    """Indexa todos los documentos soportados de data/. Devuelve None si no hay ninguno."""
    data_dir = data_dir or settings.DATA_DIR
    if not data_dir.exists():
        logger.warning("No existe %s; el índice base queda vacío.", data_dir)
        return None

    files = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append((path.name, path.read_bytes()))

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for file in files:
        try:
            new_chunks, new_vectors = await _ingest_many([file])
        except Exception as error:  # un archivo dañado no impide indexar el resto
            logger.error("No se pudo ingerir %s: %s", file[0], error)
            continue
        chunks += new_chunks
        vectors += new_vectors

    if not chunks:
        logger.warning("No hay documentos legibles en %s.", data_dir)
        return None
    return HybridIndex(chunks, vectors)


async def extend_index(
    index: HybridIndex | None, files: list[tuple[str, bytes]],
) -> tuple[HybridIndex, list[Chunk]]:
    """Devuelve un índice nuevo = índice actual + archivos (el actual no se modifica)."""
    chunks, vectors = await _ingest_many(files)
    if not chunks:
        raise ValueError("Los documentos no contienen texto extraíble.")
    if index is None:
        return HybridIndex(chunks, vectors), chunks
    return index.merged_with(chunks, vectors), chunks
