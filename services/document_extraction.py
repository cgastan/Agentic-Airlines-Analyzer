import shutil
import subprocess
import tempfile
from pathlib import Path

import olefile
from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    import io

    document = Document(io.BytesIO(file_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            paragraphs.append(" | ".join(cell.text for cell in row.cells))

    return "\n".join(paragraphs)


def extract_text_with_libreoffice(file_bytes: bytes, extension: str) -> str:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        return ""

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / f"document{extension}"
        output_dir = Path(directory) / "output"
        output_dir.mkdir()
        source.write_bytes(file_bytes)

        subprocess.run(
            [
                executable,
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                str(output_dir),
                str(source),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

        output_files = list(output_dir.glob("*.txt"))
        if output_files:
            return output_files[0].read_text(encoding="utf-8-sig", errors="replace")

    return ""


def extract_legacy_doc_strings(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as temporary_file:
        temporary_file.write(file_bytes)
        temporary_path = Path(temporary_file.name)

    try:
        if not olefile.isOleFile(str(temporary_path)):
            return ""

        with olefile.OleFileIO(str(temporary_path)) as document_file:
            if not document_file.exists("WordDocument"):
                return ""
            word_document = document_file.openstream("WordDocument")
            raw = word_document.read()

        strings = []
        utf16_text = raw.decode("utf-16le", errors="ignore")
        for token in utf16_text.split("\x00"):
            if len(token.strip()) >= 4 and token.strip().isascii():
                strings.append(token.strip())

        ascii_text = " ".join(
            part.decode("ascii", errors="ignore")
            for part in raw.split(b"\x00")
            if len(part.decode("ascii", errors="ignore")) >= 4
        )
        strings.append(ascii_text)
        return "\n".join(dict.fromkeys(strings))
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_document_text(file_bytes: bytes, extension: str) -> str:
    extension = extension.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == ".docx":
        return extract_text_from_docx(file_bytes)
    if extension in {".txt", ".md"}:
        return file_bytes.decode("utf-8-sig", errors="replace")
    if extension == ".doc":
        converted = extract_text_with_libreoffice(file_bytes, extension)
        if converted.strip():
            return converted
        return extract_legacy_doc_strings(file_bytes)

    raise ValueError("Formato de documento no admitido")
