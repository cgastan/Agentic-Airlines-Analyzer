# Agentic Airlines SRL - Automated Impact Analysis Summarizer

Agente de IA que genera reportes estructurados de análisis de impacto a partir de
solicitudes de cambio, especificaciones de arquitectura, notas de stakeholders y políticas
(PDF, DOCX, DOC, TXT o MD).

## Arquitectura

```
routers/
  analysis.py   POST /api/analyze, /api/analyze/stream (SSE), GET|POST /api/documents
  chat.py       POST /api/chat, /api/chat/stream (SSE) con memoria de sesión
services/
  readers.py           extracción de texto PDF/DOCX/DOC/TXT/MD con página y sección
  chunking.py          fragmentación con solapamiento, conservando archivo/página/sección
  embeddings.py        embeddings densos: OpenAI (por defecto) o Sentence Transformers local
  ingestion_service.py extracción -> chunks -> embeddings (con caché); igual para todo documento
  vector_store.py      índice híbrido: similitud coseno + BM25 combinados con RRF
  rag.py               funciones de recuperación del agente
  agent_flow.py        workflow del agente (6 pasos) con llamadas directas al LLM
  llm.py               cliente único Groq/OpenAI, modo JSON
  context_service.py   memoria conversacional (historial acotado + resumen)
schemas/models.py      contratos Pydantic de entrada, estado del agente y reporte
```

## Workflow del agente

1. **initialize**: el LLM extrae resumen del cambio, sistemas, perspectivas de stakeholders,
   identificadores y límites de alcance, y arma **dos consultas distintas**.
2. **retrieve_specifications_rag**: búsqueda híbrida con la consulta técnica (sistemas, integraciones, dependencias).
3. **retrieve_stakeholder_notes_rag**: búsqueda con la consulta de stakeholders/políticas, excluyendo los chunks ya recuperados.
4. **grade_context**: el LLM evalúa si el contexto alcanza e indica qué información falta.
5. **rewrite_query**: si falta información, reformula ambas consultas usando lo faltante y
   vuelve al paso 2 (máximo `MAX_QUERY_REWRITES=2`).
6. **generate_impact_report**: reporte JSON validado con cambios técnicos, nivel de riesgo,
   stakeholders afectados, mitigaciones, dependencias, preguntas abiertas y fuentes citadas.
   Si tras los reintentos el contexto sigue incompleto, el reporte se marca
   `insufficient_information=true` con confianza `Bajo` y lista lo faltante.

### Un único pipeline para todos los documentos

Arquitectura, solicitudes de cambio, notas de stakeholders y políticas se procesan
igual (extracción → chunks → embeddings) y van a un único índice. Las dos
recuperaciones se diferencian por la consulta que arma `initialize`. Los códigos
exactos (`POL-GC-002`, `Ley 17.250`) los encuentra BM25.

## Instalación (Windows)

```powershell
python -m venv airl_env
.\airl_env\Scripts\activate
pip install -r requirements.txt
```

`.env` en la raíz:

```env
LLM_PROVIDER=groq                 # o openai
GROQ_API_KEY=tu_clave
GENERATION_MODEL=openai/gpt-oss-20b
OPENAI_API_KEY=tu_clave           # requerida: embeddings de OpenAI (text-embedding-3-small)
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_PROVIDER=local        # alternativa sin OpenAI (Sentence Transformers)
# EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Ejecución

```powershell
uvicorn main:app --reload
```

Documentación interactiva: http://127.0.0.1:8000/docs. Los documentos de `data/` se indexan al arrancar.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/analyze` | multipart: `analyst_prompt`, `target_system?`, `file?` → `ImpactReportResponse` |
| POST | `/api/analyze/stream` | igual, en SSE: eventos `progress` (uno por paso real del agente), `final`, `error` |
| POST | `/api/chat` | JSON `{"message", "target_system?"}`; sesión por cookie |
| POST | `/api/chat/stream` | SSE: `progress`, `text`, `final` |
| GET | `/api/documents` | documentos indexados y cantidad de chunks |
| POST | `/api/documents` | agrega archivos a la base de conocimiento (`data/`) |
| GET | `/health` | estado del índice |

El archivo enviado a `/api/analyze` se usa sólo para esa solicitud (índice base + archivo);
para que quede disponible para siempre, usar `POST /api/documents`.

## Tests

```powershell
python -m pytest -q
```

Los tests usan un LLM y embeddings falsos: no requieren clave ni descargan modelos.
