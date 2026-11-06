# services/rag.py

import os
import re
import unicodedata

import numpy as np
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from services.embeddings import get_embedding, cosine_similarity

load_dotenv(override=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL", "llama-3.3-70b-versatile")


def normalize_text(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(
        character for character in text if not unicodedata.combining(character))
    return re.findall(r"[a-z0-9]+", text)


def extract_exact_identifiers(text: str) -> set[str]:
    """Extrae identificadores con formato numérico o alfanumérico estable."""
    normalized = text.casefold().strip()
    identifiers = set(re.findall(r"\b\d+(?:[.-]\d+)+\b", normalized))
    identifiers.update(re.findall(r"\b[a-z][a-z0-9]*-[a-z0-9][a-z0-9.-]*\b", normalized))
    return identifiers


def reciprocal_rank_fusion(
    lexical_ranking: list[int],
    vector_ranking: list[int],
    rrf_constant: int = 60,
    lexical_weight: float = 1.0,
    vector_weight: float = 1.0,
    exact_ranking: list[int] | None = None,
    exact_weight: float = 1.0,
) -> list[int]:
    scores: dict[int, float] = {}

    for ranking, weight in (
        (lexical_ranking, lexical_weight),
        (vector_ranking, vector_weight),
        (exact_ranking or [], exact_weight),
    ):
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + weight / (
                rrf_constant + rank
            )

    return sorted(scores, key=lambda chunk_index: (-scores[chunk_index], chunk_index))


def build_exact_match_ranking(query: str, chunks: list[dict]) -> list[int]:
    """Ordena los chunks por coincidencias exactas y normalizadas."""
    query_tokens = normalize_text(query)
    query_text = " ".join(query_tokens)
    query_identifiers = extract_exact_identifiers(query)

    scored_chunks = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = normalize_text(chunk["text"])
        chunk_token_set = set(chunk_tokens)
        matched_tokens = sum(
            token in chunk_token_set for token in set(query_tokens))
        normalized_chunk = " ".join(chunk_tokens)
        chunk_identifiers = extract_exact_identifiers(chunk["text"])
        matched_identifiers = len(query_identifiers & chunk_identifiers)

        # Las frases completas tienen prioridad sobre coincidencias parciales.
        score = (
            1 if query_text and query_text in normalized_chunk else 0,
            matched_identifiers,
            matched_tokens,
        )
        scored_chunks.append((score, index))

    return [
        index
        for _, index in sorted(
            scored_chunks,
            key=lambda item: (
                -item[0][0],
                -item[0][1],
                -item[0][2],
                item[1],
            ),
        )
    ]


def search_similar_chunks(query: str, chunks_with_embeddings: list[dict], top_k: int = 3) -> list[dict]:
    """Combina coincidencia exacta, BM25 léxico y similitud semántica mediante RRF."""
    if not chunks_with_embeddings:
        return []

    tokenized_corpus = [normalize_text(chunk["text"])
                        for chunk in chunks_with_embeddings]
    bm25 = BM25Okapi(tokenized_corpus)
    lexical_scores = bm25.get_scores(normalize_text(query))
    lexical_ranking = sorted(
        range(len(chunks_with_embeddings)),
        key=lambda index: (-lexical_scores[index], index),
    )

    query_vector = np.array(get_embedding(query), dtype=np.float32)
    vector_scores = []
    for index, chunk in enumerate(chunks_with_embeddings):
        if "embedding" in chunk:
            vector_scores.append((
                cosine_similarity(query_vector, chunk["embedding"]), index
            ))
        else:
            vector_scores.append((float("-inf"), index))

    vector_ranking = [
        index for _, index in sorted(vector_scores, key=lambda item: (-item[0], item[1]))
    ]
    exact_ranking = build_exact_match_ranking(query, chunks_with_embeddings)
    fused_ranking = reciprocal_rank_fusion(
        lexical_ranking,
        vector_ranking,
        exact_ranking=exact_ranking,
        exact_weight=1.25,
    )

    exact_matches = [
        index
        for index in exact_ranking
        if extract_exact_identifiers(query)
        and extract_exact_identifiers(query)
        & extract_exact_identifiers(chunks_with_embeddings[index]["text"])
    ]
    if exact_matches:
        result_indexes = exact_matches[:top_k]
        result_indexes.extend(
            index for index in fused_ranking if index not in result_indexes
        )
        return [chunks_with_embeddings[index] for index in result_indexes[:top_k]]

    return [chunks_with_embeddings[index] for index in fused_ranking[:top_k]]
