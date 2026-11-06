"""Fragmentación de texto con solapamiento. Todos los documentos se procesan igual."""
from __future__ import annotations

from pydantic import BaseModel

from config import settings
from services.readers import Page


class Chunk(BaseModel):
    """Fragmento indexable con su texto original y procedencia."""

    id: str
    text: str
    context_text: str
    source: str
    page: int | None = None
    section: str | None = None


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Divide el texto por tamaño, prefiriendo cortar en párrafos y manteniendo solapamiento."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    normalized = "\n\n".join(
        line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            # Corta en el salto de párrafo o el fin de frase más cercano.
            for separator in ("\n\n", ". ", " "):
                cut = normalized.rfind(separator, start + chunk_size // 2, end)
                if cut > start:
                    end = cut + len(separator.rstrip())
                    break

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """Fragmenta las páginas de un documento conservando archivo, página y sección."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text):
            header = " > ".join(filter(None, (page.source, page.section)))
            location = f"p{page.page}" if page.page else "s"
            chunks.append(Chunk(
                # El contador garantiza IDs únicos aunque se repitan secciones.
                id=f"{page.source}#{location}-{len(chunks) + 1}",
                text=text,
                # El encabezado mejora el embedding y el ranking BM25.
                context_text=f"{header}: {text}",
                source=page.source,
                page=page.page,
                section=page.section,
            ))
    return chunks
