# services/agent_flow.py

import os
from groq import Groq
from dotenv import load_dotenv
from services.rag import search_similar_chunks

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL", "llama-3.3-70b-versatile")


def initialize(analyst_prompt: str) -> dict:
    """
    Step 1: initialize
    Captures the analyst's request, extracts proposed change details, 
    identifies stakeholder perspectives, and sets up execution state.
    """
    print("[Agent Step] initialize: Capturing request and setting up state.")
    return {
        "original_query": analyst_prompt,
        "current_query": analyst_prompt,
        "state": "initialized"
    }


def retrieve_specifications_rag(query: str, chunks_store: list[dict]) -> list[dict]:
    """
    Step 2: retrieve_specifications_rag
    Runs custom vector search across system architecture documents and process flows.
    """
    print("[Agent Step] retrieve_specifications_rag: Searching architecture and technical specs.")
    return search_similar_chunks(query, chunks_store, top_k=3)


def retrieve_stakeholder_notes_rag(query: str, stakeholder_store: list[dict]) -> list[dict]:
    """
    Step 3: retrieve_stakeholder_notes_rag
    Queries dense vector indexes storing stakeholder feedback notes and policy requirements.
    """
    print("[Agent Step] retrieve_stakeholder_notes_rag: Querying stakeholder feedback and policies.")
    return search_similar_chunks(query, stakeholder_store, top_k=2)


def grade_context(query: str, context_text: str) -> str:
    """
    Step 4: grade_context
    Runs a targeted LLM evaluation to check if retrieved info provides sufficient detail.
    """
    print("[Agent Step] grade_context: Evaluating retrieved context sufficiency.")
    prompt = f"""
    Eres un evaluador de calidad para Business Analysts en Agentic Airlines SRL.
    Evalúa si el siguiente contexto provee suficiente detalle técnico y operativo para responder a la consulta.
    
    Consulta: {query}
    Contexto: {context_text}
    
    Responde estrictamente con una sola palabra: 'yes' si es suficiente o 'no' si falta información crítica.
    """
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()


def rewrite_query(query: str) -> str:
    """
    Step 5: rewrite_query
    Reformulates vague or incomplete analyst prompts and re-runs RAG retrievals (up to 2 times).
    """
    print("[Agent Step] rewrite_query: Reformulating vague or incomplete prompt.")
    prompt = f"""
    La siguiente solicitud de cambio de un Business Analyst de Agentic Airlines SRL es ambigua o incompleta. 
    Por favor, reescribe la consulta añadiendo contexto técnico y términos clave orientados a la arquitectura de una aerolínea.
    
    Consulta original: {query}
    Devuelve únicamente la consulta reescrita, sin texto adicional.
    """
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()


def fallback_answer() -> str:
    """
    Step 7: fallback_answer
    Safely replies when technical dependencies or impact data cannot be confirmed.
    """
    print("[Agent Step] fallback_answer: Information could not be confirmed.")
    return "I don't know based on the provided change requests and architecture documents."


def generate_impact_report(query: str, tech_chunks: list[dict], stakeholder_chunks: list[dict]) -> str:
    """
    Step 6: generate_impact_report
    Prompts the LLM to synthesize a structured impact analysis report.
    """
    print("[Agent Step] generate_impact_report: Synthesizing final structured report with Groq.")

    tech_context = "\n\n---\n\n".join([c['text'] for c in tech_chunks])
    stakeholder_context = "\n\n---\n\n".join([c['text']
                                             for c in stakeholder_chunks])

    system_prompt = (
        "Eres un asistente experto en análisis de impacto para Business Analysts en Agentic Airlines SRL. "
        "Tu tarea es analizar los cambios propuestos basándote estrictamente en la documentación técnica y notas de stakeholders."
    )

    user_prompt = f"""
    Solicitud de Cambio: {query}

    --- Especificaciones Técnicas y Arquitectura ---
    {tech_context}

    --- Notas de Stakeholders y Políticas ---
    {stakeholder_context}

    Genera un reporte de impacto estructurado que detalle:
    1. Cambios técnicos necesarios.
    2. Niveles de riesgo.
    3. Stakeholders o áreas afectadas.
    4. Mitigaciones requeridas.
    """

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


def run_agent_workflow(analyst_prompt: str, tech_store: list[dict], stakeholder_store: list[dict]) -> str:
    """Orquesta el pipeline completo cumpliendo con los pasos exigidos."""
    state = initialize(analyst_prompt)
    current_query = state["current_query"]

    max_retries = 2
    retry_count = 0
    tech_retrieved = []
    stakeholder_retrieved = []

    while retry_count <= max_retries:
        tech_retrieved = retrieve_specifications_rag(current_query, tech_store)
        stakeholder_retrieved = retrieve_stakeholder_notes_rag(
            current_query, stakeholder_store)

        combined_text = "\n".join(
            [c['text'] for c in tech_retrieved + stakeholder_retrieved])

        if combined_text:
            grade = grade_context(current_query, combined_text)
            if 'yes' in grade:
                break

        retry_count += 1
        if retry_count <= max_retries:
            current_query = rewrite_query(current_query)
        else:
            return fallback_answer()

    return generate_impact_report(analyst_prompt, tech_retrieved, stakeholder_retrieved)


print('Setup complete: Full agent workflow with professor requirements ready.')
