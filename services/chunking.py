# services/chunking.py

from typing import List


def chunk_text(text: str, chunk_size: int = 500, max_paragraphs: int = 2, overlap: int = 50) -> list[str]:
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
        if end < len(normalized):
            paragraph_break = normalized.rfind("\n\n", start, end)
            if paragraph_break > start:
                end = paragraph_break

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)

    return chunks


def build_chunks_from_documents(documents: list[dict], max_paragraphs: int = 2) -> list[dict]:
    """Create searchable chunks and keep useful source metadata."""
    all_chunks = []

    for doc in documents:
        for index, text in enumerate(chunk_text(doc['text'], max_paragraphs=max_paragraphs), start=1):
            all_chunks.append({
                'chunk_id': f"{doc['id']}_chunk_{index}",
                'doc_id': doc['id'],
                'title': doc['title'],
                'text': text,
            })

    return all_chunks
