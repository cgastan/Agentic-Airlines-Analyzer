# services/agent_flow.py

import json
import os
from groq import Groq
from dotenv import load_dotenv
from services.rag import search_similar_chunks

load_dotenv(override=True)

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
    Evalúa si el contexto permite identificar al menos sistemas afectados, riesgos o controles operativos para preparar un análisis preliminar.
    No exijas que el cambio específico aparezca literalmente. Responde 'no' solo si el contexto no tiene información técnica ni operativa útil para la consulta.

    Consulta: {query}
    Contexto: {context_text}

    Responde estrictamente con una sola palabra: 'yes' si es suficiente o 'no' si falta información crítica.
    """
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        reasoning_effort="low"
    )
    content = response.choices[0].message.content or ""
    return content.strip().lower()


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


def generate_impact_report(query: str, tech_chunks: list[dict], stakeholder_chunks: list[dict]) -> dict:
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

    Devuelve exclusivamente un objeto JSON con esta estructura:
    {{
      "technical_changes": [{{"description": "...", "affected_system": "..."}}],
      "risk_level": "Bajo|Medio|Alto|Critico",
      "affected_stakeholders": [{{"stakeholder": "...", "impact": "...", "severity": "Bajo|Medio|Alto|Critico", "mitigation": "..."}}],
      "required_mitigations": ["..."],
      "dependencies": ["..."],
      "summary_report": "...",
      "confidence": "Bajo|Medio|Alto",
      "insufficient_information": false
    }}

    No inventes información ausente. Si falta evidencia, usa listas vacías, marca
    insufficient_information como true y explica esa limitación en summary_report.
    """

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1500,
        reasoning_effort="low"
    )
    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise ValueError("El modelo no devolvió un reporte JSON válido")


def run_agent_workflow(analyst_prompt: str, tech_store: list[dict], stakeholder_store: list[dict]) -> dict:
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
            return {
                "technical_changes": [],
                "risk_level": "Medio",
                "affected_stakeholders": [],
                "required_mitigations": [],
                "dependencies": [],
                "summary_report": fallback_answer(),
                "confidence": "Bajo",
                "insufficient_information": True,
            }

    report = generate_impact_report(
        analyst_prompt, tech_retrieved, stakeholder_retrieved)
    report.setdefault(
        "summary_report", "El contexto recuperado permite identificar impactos preliminares, pero la síntesis detallada requiere mayor información verificable.")
    report.setdefault("confidence", "Medio")
    report.setdefault("insufficient_information", False)
    report.setdefault("risk_level", "Medio")
    report.setdefault("technical_changes", [])
    report.setdefault("affected_stakeholders", [])
    report.setdefault("required_mitigations", [])
    report.setdefault("dependencies", [])
    report["sources"] = [
        {
            "document_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "chunk_id": chunk.get("chunk_id"),
        }
        for chunk in tech_retrieved + stakeholder_retrieved
    ]
    return report


print('Setup complete: Full agent workflow with professor requirements ready.')
