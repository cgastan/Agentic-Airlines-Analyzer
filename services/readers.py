"""Extracción de texto con metadatos de procedencia para PDF, DOCX, DOC, TXT y MD.

Todos los lectores trabajan sobre bytes, de modo que el mismo pipeline sirve
para los documentos base de data/ y para los archivos subidos por la API.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from pydantic import BaseModel


class Page(BaseModel):
    """Una página (PDF) o sección (DOCX/MD) extraída de un documento."""

    text: str
    source: str
    page: int | None = None
    section: str | None = None


def read_pdf(data: bytes, source: str) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [
        Page(text=page.extract_text() or "", source=source, page=number)
        for number, page in enumerate(reader.pages, start=1)
    ]


def read_docx(data: bytes, source: str) -> list[Page]:
    """Agrupa párrafos por encabezado y conserva las tablas con sus cabeceras."""
    from docx import Document

    document = Document(io.BytesIO(data))
    pages: list[Page] = []
    section: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if parts:
            pages.append(Page(text="\n".join(parts), source=source, section=section))

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        if style.lower().startswith(("heading", "título", "titulo", "title")):
            flush()
            section, parts = text, []
        else:
            parts.append(text)
    flush()

    for number, table in enumerate(document.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        header = rows[0]
        lines = [" | ".join(header)]
        for row in rows[1:]:
            lines.append(" | ".join(
                f"{header[i] if i < len(header) else ''}: {value}" for i, value in enumerate(row)))
        pages.append(Page(text="\n".join(lines), source=source, section=f"Tabla {number}"))
    return pages


def read_markdown(data: bytes, source: str) -> list[Page]:
    """Divide Markdown por encabezados (#, ##, ...) para conservar la sección."""
    pages: list[Page] = []
    section: str | None = None
    parts: list[str] = []
    for line in data.decode("utf-8-sig", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            if parts:
                pages.append(Page(text="\n".join(parts), source=source, section=section))
            section, parts = line.lstrip("# ").strip() or None, []
        else:
            parts.append(line)
    if parts:
        pages.append(Page(text="\n".join(parts), source=source, section=section))
    return pages


def read_txt(data: bytes, source: str) -> list[Page]:
    return [Page(text=data.decode("utf-8-sig", errors="replace"), source=source)]


def _doc_with_libreoffice(data: bytes) -> str:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        return ""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "document.doc"
        source.write_bytes(data)
        subprocess.run(
            [executable, "--headless", "--convert-to", "txt:Text",
             "--outdir", directory, str(source)],
            check=True, capture_output=True, timeout=60,
        )
        output = Path(directory) / "document.txt"
        return output.read_text(encoding="utf-8-sig", errors="replace") if output.exists() else ""


class DocFormatError(ValueError):
    """El .doc no se puede leer (cifrado, Word 95 o anterior, o dañado)."""


# Caracteres de control de Word 97 -> texto plano.
_WORD_CONTROL_CHARS = str.maketrans({
    "\r": "\n",      # fin de párrafo
    "\x0b": "\n",    # salto de línea manual
    "\x0c": "\n",    # salto de página / sección
    "\x07": "\t",    # fin de celda o de fila de tabla
    "\x1e": "-",     # guion de no separación
    "\x1f": None,    # guion opcional
    "\xa0": " ",     # espacio de no separación
    "\x01": None,    # objeto incrustado / imagen
    "\x08": None,    # objeto dibujado
    "\x05": None,    # marca de comentario
    "\x02": None,    # marca de nota al pie
})


def _strip_word_fields(text: str) -> str:
    """Quita el código de los campos (0x13 código 0x14 resultado 0x15) y conserva el resultado.

    Ej.: "{PAGE}" -> "3", "{HYPERLINK "http://..."}texto" -> "texto". Soporta campos anidados.
    """
    output: list[str] = []
    stack: list[bool] = []  # True = dentro del código del campo (no se muestra)
    for char in text:
        if char == "\x13":
            stack.append(True)
        elif char == "\x14":
            if stack:
                stack[-1] = False
        elif char == "\x15":
            if stack:
                stack.pop()
        elif not any(stack):
            output.append(char)
    return "".join(output)


def _clean_word_text(text: str) -> str:
    # Cada celda termina en 0x07 y la fila agrega otra marca 0x07: dos seguidas = fin de fila.
    text = _strip_word_fields(text).replace("\x07\x07", "\x07\n").translate(_WORD_CONTROL_CHARS)
    lines = []
    for line in text.split("\n"):
        # Las tablas quedan como "celda | celda".
        cells = [cell.strip() for cell in line.split("\t")]
        lines.append(" | ".join(cell for cell in cells if cell))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _doc_word97_text(data: bytes) -> str:
    """Lee el texto del documento principal de un .doc Word 97-2003 (formato [MS-DOC]).

    1. El stream WordDocument empieza con la FIB, que indica cuántos caracteres tiene
       el cuerpo (ccpText) y dónde está la piece table (fcClx/lcbClx) en el stream 0Table/1Table.
    2. La piece table divide el texto en "piezas"; cada una está en cp1252 (comprimida)
       o en UTF-16LE dentro del stream WordDocument.
    """
    import struct

    import olefile

    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        if not ole.exists("WordDocument"):
            raise DocFormatError("El archivo OLE no contiene un documento de Word.")
        word = ole.openstream("WordDocument").read()
        if len(word) < 0x22 or struct.unpack_from("<H", word, 0)[0] != 0xA5EC:
            raise DocFormatError("Encabezado de Word inválido.")
        n_fib, = struct.unpack_from("<H", word, 2)
        flags, = struct.unpack_from("<H", word, 0x0A)
        if flags & 0x0100:
            raise DocFormatError("El documento está protegido con contraseña.")
        if n_fib < 101:
            raise DocFormatError("Formato Word 95 o anterior no soportado; guárdelo como .docx.")
        table_name = "1Table" if flags & 0x0200 else "0Table"
        if not ole.exists(table_name):
            raise DocFormatError(f"Falta el stream {table_name}.")
        table = ole.openstream(table_name).read()

    # FibBase (32 bytes) -> fibRgW -> fibRgLw (ccpText) -> fibRgFcLcb (fcClx, par nº 33).
    position = 32
    csw, = struct.unpack_from("<H", word, position)
    position += 2 + csw * 2
    cslw, = struct.unpack_from("<H", word, position)
    ccp_text, = struct.unpack_from("<I", word, position + 2 + 3 * 4)
    position += 2 + cslw * 4
    cb_rg_fc_lcb, = struct.unpack_from("<H", word, position)
    if cb_rg_fc_lcb <= 33:
        raise DocFormatError("FIB incompleta.")
    fc_clx, lcb_clx = struct.unpack_from("<II", word, position + 2 + 33 * 8)
    clx = table[fc_clx:fc_clx + lcb_clx]

    # Clx = Prc* (0x01, formato) seguido de Pcdt (0x02, piece table).
    offset = 0
    while offset < len(clx) and clx[offset] == 0x01:
        cb_grpprl, = struct.unpack_from("<h", clx, offset + 1)
        offset += 3 + cb_grpprl
    if offset >= len(clx) or clx[offset] != 0x02:
        raise DocFormatError("No se encontró la tabla de piezas del documento.")
    lcb_plc, = struct.unpack_from("<I", clx, offset + 1)
    plc = clx[offset + 5:offset + 5 + lcb_plc]
    pieces = (lcb_plc - 4) // 12  # (n+1) CPs de 4 bytes + n descriptores de 8 bytes
    cps = struct.unpack_from(f"<{pieces + 1}I", plc, 0)

    parts: list[str] = []
    remaining = ccp_text
    for number in range(pieces):
        if remaining <= 0:
            break
        fc_raw, = struct.unpack_from("<I", plc, 4 * (pieces + 1) + 8 * number + 2)
        length = min(cps[number + 1] - cps[number], remaining)
        fc = fc_raw & 0x3FFFFFFF
        if fc_raw & 0x40000000:  # comprimida: 1 byte por carácter
            parts.append(word[fc // 2:fc // 2 + length].decode("cp1252", errors="replace"))
        else:
            parts.append(word[fc:fc + 2 * length].decode("utf-16le", errors="replace"))
        remaining -= length
    return _clean_word_text("".join(parts))


def _rtf_to_text(rtf: str) -> str:
    """Conversión RTF -> texto suficiente para indexar (Word suele guardar RTF como .doc)."""
    skip_destinations = {
        "fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer",
        "listtable", "listoverridetable", "rsidtbl", "generator", "themedata",
        "colorschememapping", "latentstyles", "datastore", "xmlnstbl", "object",
    }
    output: list[str] = []
    stack: list[bool] = []
    skipping = False
    unicode_skip = 0
    token = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)",
                       re.IGNORECASE | re.DOTALL)
    for match in token.finditer(rtf):
        word, arg, hex_char, symbol, brace, char = match.groups()
        if brace == "{":
            stack.append(skipping)
        elif brace == "}":
            skipping = stack.pop() if stack else False
        elif symbol == "*":
            skipping = True
        elif skipping:
            continue
        elif word:
            if word in skip_destinations:
                skipping = True
            elif word in {"par", "line", "row", "sect", "page"}:
                output.append("\n")
            elif word in {"tab", "cell"}:
                output.append("\t")
            elif word == "u" and arg:
                output.append(chr(int(arg) % 65536))
                unicode_skip = 1
        elif hex_char:
            if unicode_skip:
                unicode_skip -= 1
            else:
                output.append(bytes([int(hex_char, 16)]).decode("cp1252", errors="replace"))
        elif symbol:
            output.append({"~": " ", "_": "-", "-": ""}.get(symbol, symbol))
        elif char:
            if unicode_skip:
                unicode_skip -= 1
            else:
                output.append(char)
    return _clean_word_text("".join(output).replace("\n", "\r"))


def _doc_with_libreoffice(data: bytes) -> str:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        return ""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "document.doc"
        source.write_bytes(data)
        subprocess.run(
            [executable, "--headless", "--convert-to", "txt:Text",
             "--outdir", directory, str(source)],
            check=True, capture_output=True, timeout=60,
        )
        output = Path(directory) / "document.txt"
        return output.read_text(encoding="utf-8-sig", errors="replace") if output.exists() else ""


def read_doc(data: bytes, source: str) -> list[Page]:
    """Lee .doc detectando el formato real por su firma, no por la extensión."""
    head = data[:512].lstrip()
    if data.startswith(b"PK"):  # en realidad es un .docx renombrado
        return read_docx(data, source)
    if head.startswith(b"{\\rtf"):  # RTF guardado con extensión .doc
        return [Page(text=_rtf_to_text(data.decode("latin-1")), source=source)]
    if head[:200].lower().startswith((b"<html", b"<!doctype", b"mime-version")):
        html = data.decode("utf-8", errors="replace")
        html = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", html)
        text = re.sub(r"(?i)<br\s*/?>|</p>|</tr>|</h\d>", "\n", html)
        return [Page(text=re.sub(r"<[^>]+>", " ", text), source=source)]

    import olefile
    if not olefile.isOleFile(io.BytesIO(data)):
        raise DocFormatError("El archivo no es un documento de Word válido.")
    try:
        text = _doc_word97_text(data)
    except DocFormatError:
        raise
    except Exception as error:  # estructura dañada: último intento con LibreOffice
        try:
            text = _doc_with_libreoffice(data)
        except (subprocess.SubprocessError, OSError):
            text = ""
        if not text.strip():
            raise DocFormatError(f"No se pudo interpretar el .doc ({error}).") from error
    return [Page(text=text, source=source)]


READERS: dict[str, Callable[[bytes, str], list[Page]]] = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".doc": read_doc,
    ".txt": read_txt,
    ".md": read_markdown,
}

SUPPORTED_EXTENSIONS = frozenset(READERS)


class UnsupportedFormatError(ValueError):
    pass


def read_document(filename: str, data: bytes) -> list[Page]:
    """Lee cualquier formato soportado y descarta páginas sin texto."""
    extension = Path(filename).suffix.lower()
    reader = READERS.get(extension)
    if reader is None:
        raise UnsupportedFormatError(
            f"Formato '{extension or 'sin extensión'}' no admitido. "
            f"Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    source = Path(filename).name
    return [page for page in reader(data, source) if page.text.strip()]
