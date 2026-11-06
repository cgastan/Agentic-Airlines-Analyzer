# services/embeddings.py

import os
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Usamos un modelo liviano y gratuito de HuggingFace que corre localmente en Python
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
GENERATION_MODEL = os.environ.get(
    'GENERATION_MODEL', 'llama-3.3-70b-versatile')

EMBEDDING_PROVIDER = os.environ.get(
    "EMBEDDING_PROVIDER", "local").strip().lower()
OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

encoder = None
openai_embeddings_client = None
if EMBEDDING_PROVIDER == "local":
    encoder = SentenceTransformer(EMBEDDING_MODEL)
else:
    openai_embeddings_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    """Genera un embedding mediante el proveedor configurado."""
    if EMBEDDING_PROVIDER == "openai":
        response = openai_embeddings_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    embedding = encoder.encode(text)
    return embedding.tolist()


async def get_embedding_async(text: str) -> list[float]:
    """Ejecuta embeddings locales sin bloquear el event loop."""
    if EMBEDDING_PROVIDER == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = await client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    return await asyncio.to_thread(get_embedding, text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud de coseno entre dos vectores (exactamente como lo tenías)."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Recorre los chunks de tus documentos y les añade el vector embedding."""
    for chunk in chunks:
        # Genera el vector local y lo convierte en un array de numpy de 32 bits
        chunk['embedding'] = np.array(
            get_embedding(chunk['text']), dtype=np.float32)

    print(f'Generated embeddings for {len(chunks)} chunks.')
    if chunks:
        print(f"Embedding size: {len(chunks[0]['embedding'])}")

    return chunks


print('Setup complete: Embeddings service ready.')
