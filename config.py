"""Configuración central validada del proyecto."""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Carga variables de entorno sin depender del directorio de trabajo."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "embeddings"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_PROVIDER: str = "local"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    POOL_SIZE: int = 20
    RRF_K: int = 60

    LLM_PROVIDER: str = "groq"
    GROQ_GENERATION_MODEL: str = "openai/gpt-oss-20b"
    OPENAI_GENERATION_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GENERATION_MODEL: str = "llama-3.3-70b-versatile"


settings = Settings()

# Compatibilidad con módulos de contexto existentes.
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))
CONTEXT_WINDOW_USAGE_LIMIT = float(
    os.getenv("CONTEXT_WINDOW_USAGE_LIMIT", "0.75"))

CONTEXT_TOKEN_BUDGETS = {
    "system": int(os.getenv("CONTEXT_SYSTEM_TOKENS", "1000")),
    "structured_state": int(os.getenv("CONTEXT_STATE_TOKENS", "500")),
    "summary": int(os.getenv("CONTEXT_SUMMARY_TOKENS", "600")),
    "history": int(os.getenv("CONTEXT_HISTORY_TOKENS", "2000")),
    "rag": int(os.getenv("CONTEXT_RAG_TOKENS", "5000")),
    "output": int(os.getenv("CONTEXT_OUTPUT_TOKENS", "2000")),
}

MAX_CONTEXT_TOKENS = int(MODEL_CONTEXT_WINDOW * CONTEXT_WINDOW_USAGE_LIMIT)
