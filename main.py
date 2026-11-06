"""Aplicación FastAPI: Agentic Airlines SRL - Automated Impact Analysis Summarizer."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from routers import analysis, chat
from services.ingestion_service import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El índice se construye una vez al arrancar (con caché de embeddings y clasificación).
    app.state.index_lock = asyncio.Lock()
    app.state.index = await build_index()
    if app.state.index is None:
        logger.warning("Índice vacío: agregue documentos en %s o vía POST /api/documents.",
                       settings.DATA_DIR)
    else:
        logger.info("Índice listo: %d chunks de %d documentos.",
                    len(app.state.index), len(app.state.index.sources))
    yield


app = FastAPI(
    title="Agentic Airlines - Impact Analyzer API",
    description="Análisis automatizado de impacto de cambios para Business Analysts (RAG híbrido + LLM).",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(analysis.router)
app.include_router(chat.router)


BINARY = {"type": "string", "format": "binary"}


def _is_upload(schema: dict) -> bool:
    options = schema.get("anyOf", [schema])
    return any(o.get("contentMediaType") == "application/octet-stream" for o in options)


def custom_openapi() -> dict:
    """Swagger UI sólo muestra el selector de archivos con `format: binary`.

    FastAPI genera `contentMediaType` (estilo OpenAPI 3.1) y Swagger lo dibuja como una
    caja de texto donde habría que pegar el contenido binario del archivo.
    """
    if app.openapi_schema is None:
        schema = FastAPI.openapi(app)
        for name, definition in schema.get("components", {}).get("schemas", {}).items():
            if not name.startswith("Body_"):
                continue
            for prop in definition.get("properties", {}).values():
                if _is_upload(prop):                                   # file: UploadFile | None
                    for stale in ("anyOf", "contentMediaType"):
                        prop.pop(stale, None)
                    prop.update(BINARY)
                elif prop.get("type") == "array" and _is_upload(prop.get("items", {})):
                    prop["items"] = dict(BINARY)                       # files: list[UploadFile]
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def root():
    return {
        "message": "Bienvenido al Agente Analizador de Impacto de Agentic Airlines SRL ✈️",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health():
    index = app.state.index
    return {"status": "ok", "documents": len(index.sources) if index else 0,
            "chunks": len(index) if index else 0}
