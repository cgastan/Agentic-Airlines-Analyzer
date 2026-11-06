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
