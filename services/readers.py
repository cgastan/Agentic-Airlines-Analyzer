"""Lectores de documentos con metadatos de procedencia."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from docx import Document
from pydantic import BaseModel
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class Page(BaseModel):
    """Representa una página o sección extraída de un documento."""

    text: str
    source: str
    page: int | None = None
    section: str | None = None


def read_pdf(path: Path) -> list[Page]:
    """Extrae una página independiente por cada página del PDF."""
    return [
        Page(text=page.extract_text() or "", source=str(path), page=index + 1)
        for index, page in enumerate(PdfReader(path).pages)
    ]


def read_docx(path: Path) -> list[Page]:
    """Agrupa párrafos por encabezados DOCX y conserva tablas con sus cabeceras."""
    document = Document(path)
    section = "Documento"
    parts: list[str] = []
    pages: list[Page] = []

    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith("Heading") and paragraph.text.strip():
            if parts:
                pages.append(Page(text="\n".join(parts), source=str(path), section=section))
            section = paragraph.text.strip()
            parts = []
        elif paragraph.text.strip():
            parts.append(paragraph.text.strip())

    for table in document.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        header = rows[0]
        parts.append(" | ".join(header))
        for row in rows[1:]:
            parts.append(" | ".join(f"{header[i]}: {row[i]}" for i in range(len(header))))

    if parts:
        pages.append(Page(text="\n".join(parts), source=str(path), section=section))
    return pages


def read_txt(path: Path) -> list[Page]:
    """Lee un archivo de texto como una sola sección."""
    return [Page(text=path.read_text(encoding="utf-8"), source=str(path))]


READERS: dict[str, Callable[[Path], list[Page]]] = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".txt": read_txt,
}
