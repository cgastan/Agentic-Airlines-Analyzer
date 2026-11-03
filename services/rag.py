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


def reciprocal_rank_fusion(
    lexical_ranking: list[int],
    vector_ranking: list[int],
    rrf_constant: int = 60,
    lexical_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[int]:
    scores: dict[int, float] = {}

    for ranking, weight in (
        (lexical_ranking, lexical_weight),
        (vector_ranking, vector_weight),
    ):
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + weight / (
                rrf_constant + rank
            )

    return sorted(scores, key=lambda chunk_index: (-scores[chunk_index], chunk_index))


def search_similar_chunks(query: str, chunks_with_embeddings: list[dict], top_k: int = 3) -> list[dict]:
    """Combina BM25 léxico y similitud semántica mediante RRF."""
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
    fused_ranking = reciprocal_rank_fusion(lexical_ranking, vector_ranking)

    return [chunks_with_embeddings[index] for index in fused_ranking[:top_k]]
