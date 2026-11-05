# main.py

from fastapi import FastAPI
from routers import analysis, chat

app = FastAPI(
    title="Agentic Airlines - Impact Analyzer API",
    description="API para análisis automatizado de impacto de cambios para Business Analysts usando Groq y RAG local.",
    version="1.0.0"
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
