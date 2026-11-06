"""Ingesta asincrónica de documentos y construcción del índice híbrido."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

from config import settings
from services.chunking import Chunk, chunk_pages
from services.embeddings import get_embedding_async
from services.readers import READERS
from services.vector_store import HybridIndex

logger = logging.getLogger(__name__)


def load_manifest() -> dict[str, list[str]]:
    """Carga etiquetas opcionales sin hacerlas necesarias para la ingesta."""
    manifest_path = settings.DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        filename: list(metadata.get("tags", []))
        for filename, metadata in data.items()
    }


async def _read_and_chunk(path: Path, tags: list[str] | None = None) -> list[Chunk]:
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        logger.warning("Formato no admitido, se omitirá %s", path.name)
        return []
    pages = await asyncio.to_thread(reader, path)
    return chunk_pages(pages, tags)


async def _embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors: list[list[float]] = []

    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        for chunk in batch:
            cache_key = hashlib.sha256(
                f"{chunk.context_text}|{settings.EMBEDDING_MODEL}".encode("utf-8")
            ).hexdigest()
            cache_path = settings.CACHE_DIR / f"{cache_key}.json"
            if cache_path.exists():
                vectors.append(json.loads(cache_path.read_text(encoding="utf-8")))
                continue
            vector = await get_embedding_async(chunk.context_text)
            cache_path.write_text(json.dumps(vector), encoding="utf-8")
            vectors.append(vector)
    return vectors


async def build_index() -> HybridIndex:
    """Construye el índice base desde todos los archivos legibles."""
    manifest = load_manifest()
    paths = sorted(settings.DATA_DIR.iterdir()) if settings.DATA_DIR.exists() else []
    chunks: list[Chunk] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in READERS:
            chunks.extend(await _read_and_chunk(path, manifest.get(path.name)))

    if not chunks:
        raise RuntimeError(
            f"No hay documentos PDF, DOCX o TXT legibles en {settings.DATA_DIR}. "
            "Agregue documentos o use una carga explícita antes de analizar."
        )
    embeddings = await _embed_chunks(chunks)
    return HybridIndex(chunks, embeddings)


async def add_documents(
    files: Iterable[tuple[Path, bytes]],
    tags: list[str] | None = None,
) -> HybridIndex:
    """Ingesta cargas temporales con el mismo pipeline."""
    chunks: list[Chunk] = []
    for path, content in files:
        if path.suffix.lower() not in READERS:
            logger.warning("Formato de carga no admitido: %s", path.name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        chunks.extend(await _read_and_chunk(path, tags))
    if not chunks:
        raise ValueError("No se pudo leer contenido válido de las cargas.")
    embeddings = await _embed_chunks(chunks)
    return HybridIndex(chunks, embeddings)
