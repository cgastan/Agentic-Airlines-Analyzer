"""Ingesta de documentos: extracción -> clasificación -> chunking -> embeddings -> índice.

La clasificación documental (especificación técnica, solicitud de cambio, nota de
stakeholder o política) se infiere del CONTENIDO con el LLM, no del nombre del
archivo. Opcionalmente puede fijarse en data/manifest.json:

    {"mi_doc.pdf": {"doc_type": "policy"}}
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from config import settings
from services import llm
from services.chunking import Chunk, chunk_pages
from services.embeddings import embed_texts
from services.readers import SUPPORTED_EXTENSIONS, read_document
from services.vector_store import HybridIndex

logger = logging.getLogger(__name__)

DOC_TYPE_DESCRIPTIONS = {
    "specification": "arquitectura de sistemas, especificaciones técnicas, integraciones, APIs o flujos de proceso",
    "change_request": "solicitud de cambio: describe una modificación propuesta, su alcance o justificación",
    "stakeholder": "notas, minutas, entrevistas o feedback de stakeholders y áreas de negocio",
    "policy": "políticas, normativas, regulaciones, procedimientos o requisitos de cumplimiento",
}
DOC_TYPES = frozenset(DOC_TYPE_DESCRIPTIONS) | {"general"}

# Qué tipos consulta cada paso RAG del agente ("general" = sin clasificar, va a ambos).
SPECIFICATION_TYPES = frozenset({"specification", "change_request", "general"})
STAKEHOLDER_TYPES = frozenset({"stakeholder", "policy", "general"})


def _digest(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("No se pudo leer %s, se ignora.", path)
        return {}


def load_manifest() -> dict[str, str]:
    """Tipos documentales declarados manualmente (opcional)."""
    data = _load_json(settings.DATA_DIR / "manifest.json")
    return {
        name: meta["doc_type"]
        for name, meta in data.items()
        if isinstance(meta, dict) and meta.get("doc_type") in DOC_TYPES
    }


async def classify_document(filename: str, text: str, content_hash: str) -> str:
    """Clasifica el documento por su contenido usando el LLM, con caché por hash."""
    cache_path = settings.CACHE_DIR / "doc_types.json"
    cache = _load_json(cache_path)
    if content_hash in cache:
        return cache[content_hash]

    options = "\n".join(f"- {key}: {value}" for key, value in DOC_TYPE_DESCRIPTIONS.items())
    prompt = f"""Clasifica el documento en exactamente una categoría:
{options}

Nombre de archivo: {filename}
Extracto:
\"\"\"{text[:3000]}\"\"\"

Responde en JSON: {{"doc_type": "<categoría>"}}"""
    try:
        result = await llm.complete_json(prompt, temperature=0, max_tokens=400)
        doc_type = str(result.get("doc_type", "")).strip().lower()
    except Exception as error:  # Sin LLM disponible el documento sigue siendo indexable.
        logger.warning("No se pudo clasificar %s (%s); se usa 'general'.", filename, error)
        return "general"

    if doc_type not in DOC_TYPES:
        doc_type = "general"
    cache[content_hash] = doc_type
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return doc_type


async def _embed_with_cache(chunks: list[Chunk]) -> list[list[float]]:
    """Embeddings por documento, cacheados por contenido + modelo."""
    cache_dir = settings.CACHE_DIR / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = [chunk.context_text for chunk in chunks]
    key = _digest(settings.embedding_model_name + "\x00" + "\x00".join(texts))
    cache_path = cache_dir / f"{key}.npy"
    if cache_path.exists():
        return np.load(cache_path).tolist()
    vectors = await embed_texts(texts)
    np.save(cache_path, np.asarray(vectors, dtype=np.float32))
    return vectors


async def ingest_document(filename: str, data: bytes, doc_type: str | None = None) -> list[Chunk]:
    """Procesa un archivo (base o subido) con el mismo pipeline. Devuelve sus chunks."""
    pages = await asyncio.to_thread(read_document, filename, data)
    full_text = "\n".join(page.text for page in pages)
    if not full_text.strip():
        return []
    if doc_type not in DOC_TYPES:
        doc_type = await classify_document(filename, full_text, _digest(data))
    chunks = chunk_pages(pages, doc_type)
    logger.info("Ingestado %s como '%s' (%d chunks).", filename, doc_type, len(chunks))
    return chunks


async def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Agrupa por documento para aprovechar la caché de embeddings."""
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)
    vectors: dict[str, list[float]] = {}
    for group in by_source.values():
        for chunk, vector in zip(group, await _embed_with_cache(group)):
            vectors[chunk.id] = vector
    return [vectors[chunk.id] for chunk in chunks]


async def build_index(data_dir: Path | None = None) -> HybridIndex | None:
    """Indexa todos los documentos soportados de data/. Devuelve None si no hay ninguno."""
    data_dir = data_dir or settings.DATA_DIR
    if not data_dir.exists():
        logger.warning("No existe %s; el índice base queda vacío.", data_dir)
        return None

    manifest = load_manifest()
    chunks: list[Chunk] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            chunks.extend(await ingest_document(path.name, path.read_bytes(), manifest.get(path.name)))
        except Exception as error:
            logger.error("No se pudo ingerir %s: %s", path.name, error)

    if not chunks:
        logger.warning("No hay documentos legibles en %s.", data_dir)
        return None
    return HybridIndex(chunks, await embed_chunks(chunks))


async def extend_index(
    index: HybridIndex | None,
    files: list[tuple[str, bytes]],
    doc_type: str | None = None,
) -> tuple[HybridIndex, list[Chunk]]:
    """Devuelve un índice nuevo = índice base + archivos subidos (el base no se modifica)."""
    new_chunks: list[Chunk] = []
    for filename, data in files:
        new_chunks.extend(await ingest_document(filename, data, doc_type))
    if not new_chunks:
        raise ValueError("Los documentos no contienen texto extraíble.")
    embeddings = await embed_chunks(new_chunks)
    if index is None:
        return HybridIndex(new_chunks, embeddings), new_chunks
    return index.merged_with(new_chunks, embeddings), new_chunks
