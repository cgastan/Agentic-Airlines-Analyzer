"""Cliente LLM único (Groq u OpenAI) con llamadas directas y asíncronas."""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client():
    """Crea el cliente asíncrono una sola vez, según LLM_PROVIDER."""
    if settings.LLM_PROVIDER.lower() == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    from groq import AsyncGroq
    return AsyncGroq(api_key=settings.GROQ_API_KEY)


def _reasoning_options() -> dict:
    effort = settings.LLM_REASONING_EFFORT.strip().lower()
    model = settings.generation_model.lower()
    if effort == "auto":
        is_reasoning_model = "gpt-oss" in model or model.startswith(("o1", "o3", "o4"))
        return {"reasoning_effort": "low"} if is_reasoning_model else {}
    if effort in {"", "none", "off"}:
        return {}
    return {"reasoning_effort": effort}


async def complete(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Ejecuta una llamada de chat completion y devuelve el texto de respuesta.

    max_tokens incluye los tokens de razonamiento en modelos como gpt-oss,
    por eso los valores son holgados incluso para respuestas cortas.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = _reasoning_options()
    if json_mode:
        options["response_format"] = {"type": "json_object"}

    response = await get_client().chat.completions.create(
        model=settings.generation_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **options,
    )
    return (response.choices[0].message.content or "").strip()


def parse_json_object(content: str) -> dict:
    """Extrae un objeto JSON aunque el modelo lo envuelva en ``` o texto extra."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("El modelo no devolvió un objeto JSON válido")
        data = json.loads(content[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Se esperaba un objeto JSON")
    return data


async def complete_json(prompt: str, system: str | None = None, **kwargs) -> dict:
    """Llama al LLM en modo JSON y devuelve el diccionario parseado."""
    content = await complete(prompt, system=system, json_mode=True, **kwargs)
    return parse_json_object(content)
