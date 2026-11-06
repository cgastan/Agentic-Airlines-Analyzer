"""Configuración central validada del proyecto."""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py vive en la raíz del proyecto, por eso se usa parent (no parents[1]).
PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Carga variables de entorno desde .env sin depender del directorio de trabajo."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Rutas
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CACHE_DIR: Path = PROJECT_ROOT / ".cache"

    # LLM (groq | openai)
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GENERATION_MODEL: Optional[str] = None
    # "auto" aplica "low" sólo a modelos de razonamiento (gpt-oss, o-series).
    LLM_REASONING_EFFORT: str = "auto"

    # Embeddings (local | openai)
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Chunking y recuperación
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    SPEC_TOP_K: int = 5
    STAKEHOLDER_TOP_K: int = 4
    POOL_SIZE: int = 20
    RRF_K: int = 60

    # Workflow del agente
    MAX_QUERY_REWRITES: int = 2
    MAX_UPLOAD_MB: int = 15

    @property
    def generation_model(self) -> str:
        """Modelo de generación según proveedor, salvo que se configure explícitamente."""
        if self.GENERATION_MODEL:
            return self.GENERATION_MODEL
        return "gpt-4o-mini" if self.LLM_PROVIDER.lower() == "openai" else "openai/gpt-oss-20b"

    @property
    def embedding_model_name(self) -> str:
        if self.EMBEDDING_PROVIDER.lower() == "openai":
            return self.OPENAI_EMBEDDING_MODEL
        return self.EMBEDDING_MODEL


settings = Settings()

# Presupuestos de tokens para la memoria conversacional del chat.
CONTEXT_TOKEN_BUDGETS = {
    "summary": 600,
    "history": 2000,
    "rag": 5000,
    "uploaded_document": 3000,
}
