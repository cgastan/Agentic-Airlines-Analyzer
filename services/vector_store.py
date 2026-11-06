"""Índice híbrido denso y BM25."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from services.chunking import Chunk, extract_identifiers


def normalize_number(number: str) -> str:
    """Convierte puntuación legal en una forma buscable consistente."""
    return number.replace(".", "").replace(",", "").replace("-", "")


def tokenize(text: str) -> list[str]:
    """Tokeniza conservando códigos y normalizando números y acentos."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    tokens = re.findall(r"[a-z]+(?:-[a-z0-9]+)*|\d+(?:[.-]\d+)+|\d+", text)
    return [
        normalize_number(token) if re.fullmatch(r"\d+(?:[.-]\d+)+|\d+", token) else token
        for token in tokens
    ]


class HybridIndex:
    """Combina similitud densa, BM25, identificadores exactos y etiquetas."""

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("El índice híbrido requiere al menos un documento legible.")
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Una fila cero no debe producir divisiones por cero durante la búsqueda.
        self.embeddings = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk.context_text) for chunk in chunks])

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        pool: int | None = None,
        exclude_ids: set[str] | None = None,
        preferred_tags: list[str] | None = None,
    ) -> list[Chunk]:
        top_k = top_k or settings.TOP_K
        pool = pool or settings.POOL_SIZE
        exclude_ids = exclude_ids or set()
        candidates = [index for index, chunk in enumerate(self.chunks) if chunk.id not in exclude_ids]
        if not candidates:
            return []

        vector = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm else vector
        dense_scores = self.embeddings[candidates] @ vector

        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        dense_ranking = sorted(candidates, key=lambda index: (-dense_scores[candidates.index(index)], index))
        lexical_ranking = sorted(candidates, key=lambda index: (-bm25_scores[index], index))

        scores: dict[int, float] = {}
        for rank, index in enumerate(dense_ranking[:pool], start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (settings.RRF_K + rank)
        for rank, index in enumerate(lexical_ranking[:pool], start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (settings.RRF_K + rank)

        query_ids = {identifier.casefold() for identifier in extract_identifiers(query)}
        preferred = set(preferred_tags or [])
        for index in candidates:
            chunk = self.chunks[index]
            if query_ids.intersection(identifier.casefold() for identifier in chunk.identifiers):
                scores[index] = scores.get(index, 0.0) + 0.08
            if preferred.intersection(chunk.tags):
                scores[index] = scores.get(index, 0.0) + 0.03

        # El match exacto puede superar una distancia semántica desfavorable.
        ranked = sorted(candidates, key=lambda index: (-scores.get(index, 0.0), index))
        return [self.chunks[index] for index in ranked[:top_k]]
