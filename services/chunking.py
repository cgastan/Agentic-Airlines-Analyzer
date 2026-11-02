# services/chunking.py

from typing import List


def chunk_text(text: str, chunk_size: int = 500, max_paragraphs: int = 2, overlap: int = 50) -> list[str]:
    """Split text into small paragraph-based chunks."""
    paragraphs = [paragraph.strip()
                  for paragraph in text.split('\n\n') if paragraph.strip()]
    chunks = []

    for start in range(0, len(paragraphs), max_paragraphs):
        group = paragraphs[start:start + max_paragraphs]
        chunks.append('\n\n'.join(group))

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
