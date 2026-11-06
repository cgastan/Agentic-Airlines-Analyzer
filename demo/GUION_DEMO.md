# Guion de la demo (≈5 minutos dentro de los 10)

Probado contra el agente real (Groq `openai/gpt-oss-120b` + embeddings de OpenAI) el 06/11/2026.
Las respuestas del LLM cambian un poco en cada ejecución: usa los **"debería aparecer"** como
lista de control, no como texto exacto.

## Antes de empezar (15 min antes)
1. `uvicorn main:app` (sin `--reload`) y esperar `Índice listo: 81 chunks de 3 documentos`.
2. Abrir `http://127.0.0.1:8000/docs` y recargar con **Ctrl+F5**.
3. Hacer **una** consulta de prueba para confirmar que Groq responde. Después, no repetir el guion
   entero: Groq gratis tiene límite por minuto y si lo agotas, cada respuesta tarda más.
4. Tener este archivo abierto para copiar y pegar.

> El archivo `demo/CR-2026-016_Reembolso_Automatico.docx` NO está en `data/` a propósito:
> así el agente lo ve por primera vez en vivo.

---

## Paso 1: Charla casual y límites (`POST /api/chat`) ~40 s

```json
{ "message": "Hola, ¿qué puedes hacer?", "session_id": null, "target_system": null }
```
**Debería aparecer:** `intent: "casual_chat"`, respuesta corta (≤ 60 palabras) ofreciendo analizar
cambios, riesgos, dependencias y stakeholders. **`report: null`**: no corrió el workflow.

**Copia el `session_id` de la respuesta.** Lo usas en todos los pasos siguientes.

Opcional (muestra que no se sale del tema):
```json
{ "message": "Escríbeme un poema sobre aviones", "session_id": "PEGAR_AQUI" }
```
**Debería aparecer:** `intent: "off_topic"` y *"No puedo ayudarte con eso porque está fuera de mi alcance..."*

**Qué decir:** "El agente primero clasifica la intención. Si no es un análisis de impacto, responde
corto o rechaza, y no gasta el workflow ni la búsqueda documental."

---

## Paso 2: Pregunta de stakeholders (`POST /api/chat`) ~1 min

```json
{ "message": "¿Qué opinan los stakeholders sobre el cambio de proveedor de SMS de NotifyX (CR-2026-015)?", "session_id": "PEGAR_AQUI" }
```
**Debería aparecer:**
- Nivel de riesgo **Alto**, justificado con **POL-GC-002** (datos personales a un tercero y Ley 18.331).
- Adaptador en **SkyBus**, sin cambiar la interfaz de **AirBook**; operación en paralelo 2 semanas,
  rollback ≤ 60 min.
- Límite de **50.000 mensajes por hora** de NotifyX.
- Stakeholders: Atención al Cliente (guiones, capacitación), Operaciones, Seguridad/Legal.
- `Fuentes:` al final, con las **Notas de Stakeholders** y la **Arquitectura**.

### 2b. Seguimiento con memoria (mismo `session_id`)
```json
{ "message": "¿Y qué aprobaciones necesita y por qué?", "session_id": "PEGAR_AQUI" }
```
**Debería aparecer:** CAB por ser riesgo Alto, más Finanzas y Legal/Cumplimiento. Entendió que
"eso" era CR-2026-015 **sin repetirlo**.

**Qué decir:** "El servidor guarda la conversación bajo el `session_id`; con eso reformula la
pregunta de seguimiento como una consulta completa antes de buscar."

---

## Paso 3: Pregunta técnica / de especificación (`POST /api/chat`) ~1 min

```json
{ "message": "¿Qué impacto tendría conectar CareDesk directamente con PayHub para procesar reembolsos, sin pasar por SkyBus?", "session_id": "PEGAR_AQUI" }
```
**Debería aparecer:**
- Riesgo **Alto**: reembolsos y datos personales (criterio de POL-GC-002).
- **Rompe la regla de integración**: todo pasa por SkyBus; la **única excepción** documentada es
  AirBook-PayHub (latencia de 800 ms).
- PCI DSS y aprobación del CAB.

### Cómo mostrar las dos recuperaciones RAG
Baja en la respuesta JSON a `report` → `trace` → `queries`:
- `spec_query`: consulta **técnica** (componentes, integraciones), y
- `stakeholder_query`: consulta de **stakeholders y políticas**.

Y en `report.sources` se ven fragmentos de la **Arquitectura** (técnica) y de las
**Notas de Stakeholders / Política** (organizacional).

**Qué decir (preciso y honesto):** "Hay una base vectorial híbrida, con embeddings de OpenAI y BM25,
y el agente hace **dos recuperaciones distintas**: una técnica y otra de stakeholders, cada una con su
propia consulta generada por el LLM. La segunda excluye lo que ya trajo la primera, para no repetir
contexto. Además siempre suma los criterios de riesgo de la política, para clasificar con las
reglas de la empresa."

---

## Paso 4: Subir una solicitud nueva (`POST /api/analyze`) ~1 min

En Swagger, `/api/analyze` → *Try it out*:
- `analyst_prompt`: `Analiza el impacto de la solicitud de cambio adjunta`
- `target_system`: `payhub` (en minúscula a propósito: se normaliza a `PayHub`)
- `file`: **Choose File** → `demo/CR-2026-016_Reembolso_Automatico.docx`

**El momento clave:** el documento dice que el solicitante considera el riesgo **"Bajo"** y que basta
con el Dueño del Proceso.

**Debería aparecer:**
- `risk_level: "Alto"`, contradiciendo al solicitante con la política (afecta reembolsos y
  experiencia del pasajero), y aprobación del **CAB**.
- Sistemas: PayHub, AirBook, NotifyX (y CareDesk).
- Mitigaciones que **no están en el archivo subido**, sino en las notas de stakeholders: revisión
  **PCI DSS** (≈10 días hábiles), regresión de AirBook, límite de 50.000 SMS/hora, y a veces el tope
  de **USD 1.500** que pide Finanzas o la simulación de cancelaciones masivas.
- `sources`: el `.docx` subido + Arquitectura + Notas + Política.

**Qué decir:** "El documento nuevo se indexa solo para esta consulta y se cruza con la base. El
agente no se queda con la evaluación del solicitante: aplica la política y agrega lo que dijeron los
stakeholders en reuniones previas."

---

## Plan B
| Problema | Qué hacer |
|---|---|
| Tarda mucho (más de 30 s) | Es el límite por minuto de Groq: espera; el cliente reintenta solo. Sigue hablando del workflow mientras tanto. |
| Error 500 / se cae | Mostrar `GET /api/documents` y `/health`, y explicar la respuesta esperada con este guion. |
| `session_id` dice `"string"` | Cámbialo por `null` en el primer mensaje. |
| Swagger queda cargando | Hay un uvicorn viejo en el puerto 8000: ciérralo o usa `--port 8001`. |
| Sale un riesgo distinto al esperado | Mostrar `risk_justification` y decir que el LLM puede variar; el criterio está en POL-GC-002. |

## Preguntas que te pueden hacer
- **¿Por qué una sola base y no dos?** Los dos RAG buscan sobre el mismo índice con consultas
  distintas; así no hay que clasificar los documentos a mano y un documento nuevo entra sin configurar nada.
- **¿Qué pasa si el contexto no alcanza?** `grade_context` lo evalúa y `rewrite_query` reformula hasta
  2 veces; si sigue faltando, el reporte marca `insufficient_information` y lista lo faltante.
- **¿Inventa?** El prompt exige basarse solo en los fragmentos y citarlos; lo no respaldado va a `open_questions`.
