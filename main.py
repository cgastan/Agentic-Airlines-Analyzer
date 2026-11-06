# main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from services.ingestion_service import build_index
from routers import analysis, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El índice se construye una vez al arrancar, nunca durante la importación.
    app.state.index = await build_index()
    yield


app = FastAPI(
    title="Agentic Airlines - Impact Analyzer API",
    description="API para análisis automatizado de impacto de cambios para Business Analysts usando Groq y RAG local.",
    version="1.0.0",
    lifespan=lifespan,
)

# Registrar los routers separados de análisis y conversación
app.include_router(analysis.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "message": "Bienvenido al Agente Analizador de Impacto de Agentic Airlines SRL ✈️",
        "docs_url": "/docs"
    }
