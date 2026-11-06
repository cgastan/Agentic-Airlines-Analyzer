"""Fragmentación de texto con solapamiento y extracción genérica de identificadores."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from config import settings
from services.readers import Page

# Patrones por FORMA, no por valor: no hay nombres de sistemas ni normas fijos.
IDENTIFIER_PATTERNS = (
    # Referencias normativas: "Ley 18.331", "Decreto 414/009", "Artículo 12".
    re.compile(
        r"\b(?:ley|decreto|resoluci[oó]n|art[ií]culo|norma)\s+\d+(?:[./-]\d+)*\b", re.IGNORECASE),
    # Códigos de documento o ticket: "ARQ-SIS-001", "CHG-2024-15", "POL-GC-002".
    re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d+\b"),
    # Estándares en mayúsculas con versión: "ISO 27001", "PCI DSS 4.0".
    re.compile(r"\b[A-Z]{2,}(?:\s[A-Z]{2,})*\s\d+(?:[.:-]\d+)*\b"),
)


def extract_identifiers(text: str) -> list[str]:
    """Extrae referencias legales, códigos y estándares presentes en el texto."""
    found = {
        match.group(0).strip()
        for pattern in IDENTIFIER_PATTERNS
        for match in pattern.finditer(text)
    }
    return sorted(found, key=str.casefold)


class Chunk(BaseModel):
    """Fragmento indexable con su texto original y procedencia."""

    id: str
    text: str
    context_text: str
    source: str
    doc_type: str = "general"
    page: int | None = None
    section: str | None = None
    identifiers: list[str] = Field(default_factory=list)


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


def chunk_pages(pages: list[Page], doc_type: str = "general") -> list[Chunk]:
    """Fragmenta páginas conservando documento, página, sección y tipo documental."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text):
            header = " > ".join(filter(None, (page.source, page.section)))
            location = f"p{page.page}" if page.page else "s"
            chunks.append(Chunk(
                # El contador global garantiza IDs únicos aunque se repitan secciones.
                id=f"{page.source}#{location}-{len(chunks) + 1}",
                text=text,
                # El encabezado mejora el embedding y el ranking BM25.
                context_text=f"{header}: {text}",
                source=page.source,
                doc_type=doc_type,
                page=page.page,
                section=page.section,
                identifiers=extract_identifiers(text),
            ))
    return chunks
