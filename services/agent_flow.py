# services/agent_flow.py

import json
import os
from typing import TypedDict

from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from schemas.models import LLMImpactReport
from services.rag import search_similar_chunks

load_dotenv(override=True)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
if LLM_PROVIDER == "openai":
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    GENERATION_MODEL = os.environ.get("OPENAI_GENERATION_MODEL", "gpt-4o-mini")
else:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "openai/gpt-oss-20b")


def provider_options(include_reasoning: bool = True) -> dict:
    if include_reasoning and LLM_PROVIDER == "groq":
        return {"reasoning_effort": "low"}
    return {}


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
        **provider_options()
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


def classify_message_intent(message: str) -> str:
    prompt = f"""
Clasifica el mensaje del usuario para una API de análisis de impacto de Agentic Airlines.

Responde únicamente con una de estas categorías:
- impact_analysis: solicita analizar, comparar, identificar, evaluar, resumir o explicar cambios, sistemas, dependencias, riesgos, stakeholders o documentos relacionados con impacto de cambios.
- casual_chat: saludos, charla general, agradecimientos, preguntas personales al asistente, temas ajenos al análisis o solicitudes que el agente no puede realizar.

Mensaje: {message}
"""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
        **provider_options()
    )
    intent = (response.choices[0].message.content or "").strip().lower()
    return "impact_analysis" if "impact_analysis" in intent else "casual_chat"


def casual_answer(message: str) -> str:
    prompt = f"""
Actúa como asistente de análisis de impacto de cambios de Agentic Airlines.
Mantén una conversación breve, natural y profesional en español, limitada a un máximo de 60 palabras.

El usuario puede hablar de temas cotidianos o preguntar qué puedes hacer.
No escribas poemas, código, traducciones, cálculos ajenos, contenidos creativos ni realices ninguna tarea fuera del análisis de impacto.
Si pide una tarea ajena, responde exactamente con este formato lógico: "No puedo ejecutar esa tarea porque está fuera de mi alcance. Puedo ayudarte a analizar cambios, riesgos, dependencias, stakeholders o documentos de Agentic Airlines."
Si está relacionada con impacto de cambios, sistemas, riesgos, stakeholders o documentos, conversa con naturalidad y invita a describir el cambio concreto.
No afirmes haber ejecutado herramientas, leído archivos o realizado acciones que el sistema no haya confirmado.

Mensaje del usuario: {message}
Responde únicamente el texto destinado al usuario.
"""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=80,
        **provider_options()
    )
    answer = (response.choices[0].message.content or "").strip()
    return truncate_casual_answer(answer)


def truncate_casual_answer(answer: str, max_words: int = 60) -> str:
    words = answer.split()
    if len(words) <= max_words:
        return answer

    truncated = " ".join(words[:max_words]).rstrip(" ,.;:")
    last_sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_sentence_end >= 0:
        return truncated[:last_sentence_end + 1]
    return f"{truncated}."


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

    No inventes información ausente. Limita cada afirmación al contenido recuperado
    y usa listas vacías cuando no exista evidencia verificable.
    """

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1500,
        **provider_options()
    )
    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(content[start:end + 1])
        else:
            raise ValueError("El modelo no devolvió un reporte JSON válido")

    try:
        validated = LLMImpactReport.model_validate(data)
        return validated.model_dump(mode="json")
    except ValidationError as validation_error:
        raise ValueError(
            "El reporte generado no cumple el contrato de salida"
        ) from validation_error


class AgentState(TypedDict, total=False):
    original_query: str
    current_query: str
    tech_store: list[dict]
    stakeholder_store: list[dict]
    tech_retrieved: list[dict]
    stakeholder_retrieved: list[dict]
    retry_count: int
    context_grade: str
    intent: str
    allow_casual: bool
    report: dict


def initialize_node(state: AgentState) -> AgentState:
    initialized = initialize(state["original_query"])
    intent = "impact_analysis"
    if state.get("allow_casual", True):
        intent = classify_message_intent(state["original_query"])
    return {
        **state,
        "current_query": initialized["current_query"],
        "retry_count": 0,
        "intent": intent,
    }


def casual_node(state: AgentState) -> AgentState:
    report = {
        "technical_changes": [],
        "risk_level": "Medio",
        "affected_stakeholders": [],
        "required_mitigations": [],
        "dependencies": [],
        "summary_report": casual_answer(state["original_query"]),
        "confidence": "Alto",
        "insufficient_information": False,
        "sources": [],
    }
    return {**state, "report": report}


def retrieve_node(state: AgentState) -> AgentState:
    tech_retrieved = retrieve_specifications_rag(
        state["current_query"], state["tech_store"]
    )
    stakeholder_retrieved = retrieve_stakeholder_notes_rag(
        state["current_query"], state["stakeholder_store"]
    )
    return {
        **state,
        "tech_retrieved": tech_retrieved,
        "stakeholder_retrieved": stakeholder_retrieved,
    }


def grade_node(state: AgentState) -> AgentState:
    combined_text = "\n".join(
        chunk["text"]
        for chunk in state["tech_retrieved"] + state["stakeholder_retrieved"]
    )
    context_grade = grade_context(state["current_query"], combined_text)
    has_both_sources = bool(
        state["tech_retrieved"] and state["stakeholder_retrieved"])
    has_enough_context = len(combined_text.strip()) >= 500
    is_sufficient = "yes" in context_grade or (
        has_both_sources and has_enough_context)
    return {**state, "context_grade": "yes" if is_sufficient else "no"}


def rewrite_node(state: AgentState) -> AgentState:
    retry_count = state["retry_count"] + 1
    return {
        **state,
        "current_query": rewrite_query(state["current_query"]),
        "retry_count": retry_count,
    }


def normalize_generated_report(report: dict, retrieved_chunks: list[dict]) -> dict:
    has_retrieved_evidence = bool(retrieved_chunks)
    has_impact_details = bool(
        report["technical_changes"]
        or report["affected_stakeholders"]
        or report["dependencies"]
    )
    has_sufficient_information = has_retrieved_evidence and has_impact_details

    report["insufficient_information"] = not has_sufficient_information
    if not has_sufficient_information:
        report["summary_report"] = (
            "El contexto recuperado permite confirmar fuentes disponibles, pero no evidencia suficiente "
            "para determinar cambios, dependencias o stakeholders con precisión."
        )
        report["confidence"] = "Bajo"
    elif report["confidence"] not in {"Bajo", "Medio", "Alto"}:
        report["confidence"] = "Medio"

    return report


def generate_node(state: AgentState) -> AgentState:
    retrieved_chunks = state["tech_retrieved"] + state["stakeholder_retrieved"]
    report = generate_impact_report(
        state["original_query"],
        state["tech_retrieved"],
        state["stakeholder_retrieved"],
    )
    report = normalize_generated_report(report, retrieved_chunks)
    report.setdefault("risk_level", "Medio")
    report["sources"] = [
        {
            "document_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "chunk_id": chunk.get("chunk_id"),
        }
        for chunk in retrieved_chunks
    ]
    return {**state, "report": report}


def fallback_node(state: AgentState) -> AgentState:
    report = {
        "technical_changes": [],
        "risk_level": "Medio",
        "affected_stakeholders": [],
        "required_mitigations": [],
        "dependencies": [],
        "summary_report": fallback_answer(),
        "confidence": "Bajo",
        "insufficient_information": True,
        "sources": [],
    }
    return {**state, "report": report}


def route_after_grade(state: AgentState) -> str:
    if state.get("context_grade") == "yes":
        return "generate"
    if state["retry_count"] >= 2:
        return "fallback"
    return "rewrite"


def build_impact_graph():
    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize_node)
    graph.add_node("casual", casual_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "initialize")
    graph.add_conditional_edges(
        "initialize",
        lambda state: "casual" if state["intent"] == "casual_chat" else "retrieve",
        {"casual": "casual", "retrieve": "retrieve"},
    )
    graph.add_edge("casual", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


impact_graph = build_impact_graph()


def run_agent_workflow(
    analyst_prompt: str,
    tech_store: list[dict],
    stakeholder_store: list[dict],
    allow_casual: bool = True,
) -> dict:
    """Ejecuta el agente LangGraph y devuelve el reporte generado."""
    final_state = impact_graph.invoke({
        "original_query": analyst_prompt,
        "tech_store": tech_store,
        "stakeholder_store": stakeholder_store,
        "allow_casual": allow_casual,
    })
    return final_state["report"]


print('Setup complete: Full agent workflow with professor requirements ready.')
