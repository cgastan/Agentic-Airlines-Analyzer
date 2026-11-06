"""Índice vectorial denso en memoria combinado con BM25 mediante Reciprocal Rank Fusion."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from schemas.models import system_key
from services.chunking import Chunk, extract_identifiers


def tokenize(text: str) -> list[str]:
    """Tokeniza sin acentos ni mayúsculas, conservando códigos y números normativos."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens = re.findall(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*", text)
    # "18.331" y "18331" deben coincidir.
    return [re.sub(r"(?<=\d)[.\-/](?=\d)", "", token) for token in tokens]


def reciprocal_rank_fusion(rankings: Iterable[list[int]], k: int | None = None) -> dict[int, float]:
    """Combina rankings: score(d) = sum(1 / (k + rank_i(d)))."""
    k = k or settings.RRF_K
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return scores


class HybridIndex:
    """Matriz de embeddings normalizados + BM25 + bonus por identificadores exactos."""

    IDENTIFIER_BONUS = 0.05

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("El índice requiere al menos un fragmento.")
        if len(chunks) != len(embeddings):
            raise ValueError("Cantidad de chunks y embeddings no coincide.")
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(chunk.context_text) for chunk in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> dict[str, str]:
        """Documento -> tipo documental, para inspección vía API."""
        return {chunk.source: chunk.doc_type for chunk in self.chunks}

    def canonical_spelling(self, term: str) -> str | None:
        """Cómo escriben los documentos un término (p.ej. 'payhub' -> 'PayHub').

        Compara sin distinguir mayúsculas ni acentos y devuelve la grafía más
        frecuente encontrada en el corpus, o None si el término no aparece.
        """
        key = system_key(term)
        if not key:
            return None
        pattern = re.compile(r"\b" + r"\s+".join(map(re.escape, key.split())) + r"\b")
        spellings: Counter[str] = Counter()
        for chunk in self.chunks:
            text = unicodedata.normalize("NFC", chunk.text)
            # Texto "plano" (sin acentos, en minúsculas) y, para cada carácter plano,
            # su posición en el texto original: así se recupera la grafía real con acentos.
            plain_chars: list[str] = []
            positions: list[int] = []
            for position, char in enumerate(text):
                for base in unicodedata.normalize("NFD", char):
                    if not unicodedata.combining(base):
                        plain_chars.append(base.casefold())
                        positions.append(position)
            for match in pattern.finditer("".join(plain_chars)):
                start, end = positions[match.start()], positions[match.end() - 1] + 1
                spellings[" ".join(text[start:end].split())] += 1
        return spellings.most_common(1)[0][0] if spellings else None

    def merged_with(self, chunks: list[Chunk], embeddings: list[list[float]]) -> "HybridIndex":
        """Devuelve un índice nuevo con fragmentos adicionales, sin mutar el actual.

        Si un documento ya existía (mismo nombre), se reemplaza por la nueva versión.
        """
        new_sources = {chunk.source for chunk in chunks}
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.source not in new_sources]
        return HybridIndex(
            [self.chunks[i] for i in keep] + list(chunks),
            [self.embeddings[i].tolist() for i in keep] + list(embeddings),
        )

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        doc_types: set[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[Chunk]:
        """Búsqueda híbrida. doc_types restringe el corpus (p.ej. sólo especificaciones).

        Si el filtro deja el corpus vacío, se busca en todos los documentos para no
        devolver contexto vacío por una clasificación documental incompleta.
        """
        exclude_ids = exclude_ids or set()
        available = [i for i, c in enumerate(self.chunks) if c.id not in exclude_ids]
        candidates = [i for i in available if not doc_types or self.chunks[i].doc_type in doc_types]
        if not candidates:
            candidates = available
        if not candidates:
            return []

        vector = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm else vector
        dense_scores = self.embeddings @ vector
        lexical_scores = np.asarray(self.bm25.get_scores(tokenize(query)))

        pool = settings.POOL_SIZE
        dense_ranking = sorted(candidates, key=lambda i: -dense_scores[i])[:pool]
        lexical_ranking = [
            i for i in sorted(candidates, key=lambda i: -lexical_scores[i]) if lexical_scores[i] > 0
        ][:pool]
        scores = reciprocal_rank_fusion([dense_ranking, lexical_ranking])

        # Un identificador exacto (ej. "ARQ-SIS-001") supera una distancia semántica desfavorable.
        query_ids = {identifier.casefold() for identifier in extract_identifiers(query)}
        if query_ids:
            for i in candidates:
                if query_ids & {x.casefold() for x in self.chunks[i].identifiers}:
                    scores[i] = scores.get(i, 0.0) + self.IDENTIFIER_BONUS

        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in ranked[:top_k]]
