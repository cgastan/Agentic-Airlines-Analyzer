# services/embeddings.py

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Usamos un modelo liviano y gratuito de HuggingFace que corre localmente en Python
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
GENERATION_MODEL = os.environ.get(
    'GENERATION_MODEL', 'llama-3.3-70b-versatile')

# Inicializamos el codificador local para reemplazar la llamada de OpenAI en embeddings
encoder = SentenceTransformer(EMBEDDING_MODEL)


def get_embedding(text: str) -> list[float]:
    """Genera el vector embedding localmente de manera gratuita."""
    embedding = encoder.encode(text)
    return embedding.tolist()


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
