# services/rag.py

import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from services.embeddings import get_embedding, cosine_similarity

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL", "llama-3.3-70b-versatile")


def search_similar_chunks(query: str, chunks_with_embeddings: list[dict], top_k: int = 3) -> list[dict]:
    """
    Busca los fragmentos más relevantes utilizando la similitud de coseno
    entre el vector de la consulta y los vectores de los chunks almacenados.
    """
    if not chunks_with_embeddings:
        return []

    query_vector = np.array(get_embedding(query), dtype=np.float32)
    scored_chunks = []

    for chunk in chunks_with_embeddings:
        if 'embedding' not in chunk:
            continue
        score = cosine_similarity(query_vector, chunk['embedding'])
        scored_chunks.append((score, chunk))

    # Ordenar de mayor a menor similitud
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in scored_chunks[:top_k]]
