# Agentic Airlines - Impact Analyzer

Agente de IA para generar análisis automatizados de impacto de cambios a partir de solicitudes, especificaciones técnicas y notas de stakeholders.

## Arquitectura

- `routers/analysis.py`: endpoints FastAPI y carga de documentos.
- `services/agent_flow.py`: workflow de análisis, RAG, evaluación y generación.
- `services/rag.py`: búsqueda vectorial por similitud de coseno.
- `services/embeddings.py`: embeddings locales con Sentence Transformers.
- `services/chunking.py`: fragmentación configurable con solapamiento.
- `schemas/models.py`: contratos Pydantic de entrada y salida.

## Requisitos

- Python 3.10 o superior.
- Clave de Groq o proveedor LLM compatible configurado en el proyecto.

## Instalación en Windows

```powershell
python -m venv airl_env
.\airl_env\Scripts\activate
pip install -r requirements.txt
```

Crear un archivo `.env` en la raíz:

```env
GROQ_API_KEY=tu_clave
GENERATION_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Ejecución

```powershell
uvicorn main:app --reload
```

Documentación interactiva: `http://127.0.0.1:8000/docs`

## Endpoints

### `POST /api/analyze`

Recibe `request_id`, `analyst_prompt` y opcionalmente un documento PDF, TXT o MD mediante multipart.

```text
request_id: CHG-001
analyst_prompt: Evaluar la integración del nuevo control de pasaporte biométrico.
file: solicitud.pdf
```

### `POST /api/chat`

Consulta JSON reutilizando el workflow RAG.

```json
{
  "request_id": "CHG-001",
  "message": "¿Qué sistemas y stakeholders resultan afectados por el cambio de reserva?",
  "target_system": "Sistema de reservas"
}
```

### `POST /api/chat/stream`

Transmite progreso y reporte final como JSON Lines (`application/x-ndjson`).

## Workflow del agente

1. Inicializa la solicitud.
2. Recupera especificaciones técnicas con RAG.
3. Recupera notas de stakeholders y políticas.
4. Evalúa si el contexto es suficiente.
5. Reescribe la consulta hasta dos veces si falta información.
6. Genera un reporte estructurado.
7. Devuelve un fallback explícito cuando no puede confirmar el impacto.

Los documentos base se leen desde `data/`. Los archivos cuya clasificación contiene `stakeholder` o `policy` se consideran operativos; el resto se considera técnico.

## Validación

```powershell
python -m compileall -q .
```