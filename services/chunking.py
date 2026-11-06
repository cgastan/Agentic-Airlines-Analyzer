# services/chunking.py

"""Fragmentación de texto con identificación de referencias."""
from __future__ import annotations

import re
from pathlib import Path
from pydantic import BaseModel, Field

from config import settings
from services.readers import Page


IDENTIFIER_PATTERNS = (
    re.compile(
        r"\b(?:ley|decreto|resoluci[oó]n|art[ií]culo)\s+\d+(?:[./-]\d+)*\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Z0-9]*-\d+(?:-\d+)*\b"),
    re.compile(r"\b(?:PCI\s+DSS)\b", re.IGNORECASE),
)


def extract_identifiers(text: str) -> list[str]:
    """Extrae referencias legales, códigos y estándares por su forma."""
    found = {
        match.group(0).strip()
        for pattern in IDENTIFIER_PATTERNS
        for match in pattern.finditer(text)
    }
    return sorted(found, key=str.casefold)


class Chunk(BaseModel):
    """Fragmento indexable con texto original y procedencia."""

    id: str
    text: str
    context_text: str
    source: str
    page: int | None = None
    section: str | None = None
    identifiers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Divide texto en chunks por tamaño, conservando solapamiento entre bloques."""
    normalized = "\n\n".join(
        paragraph.strip()
        for paragraph in text.splitlines()
        if paragraph.strip()
    )
    if not normalized:
        return []

    overlap = max(0, min(overlap, chunk_size - 1))
    chunks = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        # Preferir el salto de párrafo más cercano para no cortar una frase.
        if end < len(normalized):
            paragraph_break = normalized.rfind("\n\n", start, end)
            if paragraph_break > start:
                end = paragraph_break

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        # Mantiene el solapamiento configurado sin retroceder hasta el inicio.
        start = max(start + 1, end - overlap)

    return chunks


def chunk_pages(pages: list[Page], tags: list[str] | None = None) -> list[Chunk]:
    """Fragmenta páginas conservando documento, sección y página."""
    chunks: list[Chunk] = []
    for page in pages:
        title = Path(page.source).stem
        for index, text in enumerate(chunk_text(
            page.text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        ), start=1):
            section = page.section or ""
            # El contexto mejora embeddings y permite mostrar la jerarquía de la fuente.
            context = f"{title} > {section}: {text}".strip()
            chunks.append(Chunk(
                id=f"{page.source}:{page.page or 0}:{index}",
                text=text,
                context_text=context,
                source=page.source,
                page=page.page,
                section=page.section,
                identifiers=extract_identifiers(text),
                tags=list(tags or []),
            ))
    return chunks


def build_chunks_from_documents(documents: list[dict], max_paragraphs: int = 2) -> list[dict]:
    """Create searchable chunks and keep useful source metadata."""
    all_chunks = []

    for doc in documents:
        for index, text in enumerate(chunk_text(doc['text']), start=1):
            all_chunks.append({
                'chunk_id': f"{doc['id']}_chunk_{index}",
                'doc_id': doc['id'],
                'title': doc['title'],
                'text': text,
            })

    return all_chunks
