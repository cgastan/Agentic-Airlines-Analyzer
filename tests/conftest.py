"""Fixtures: LLM y embeddings falsos para testear el pipeline sin red ni modelos."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from services import embeddings, llm  # noqa: E402


class FakeEncoder:
    """Bag-of-words con hashing: textos con palabras en común quedan cerca."""

    def encode(self, texts, batch_size=64):
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                vectors[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % 256] += 1
        return vectors


class FakeLLM:
    """Responde según el paso del workflow que reconoce en el prompt."""

    def __init__(self):
        self.calls: list[str] = []
        self.grade_sufficient = True

    async def __call__(self, prompt, system=None, **kwargs):
        if "Clasifica el mensaje" in prompt:
            self.calls.append("classify_intent")
            intent = "casual_chat" if "hola" in prompt.lower() else "impact_analysis"
            return json.dumps({"intent": intent})
        if "planificar una búsqueda documental" in prompt:
            self.calls.append("initialize")
            return json.dumps({
                "change_summary": "Migrar el motor de reservas",
                "affected_systems": ["Reservas"],
                "stakeholder_perspectives": ["Atención al cliente"],
                "spec_query": "motor de reservas integración pagos",
                "stakeholder_query": "impacto atención al cliente política de cambios",
            })
        if "evaluador de calidad de contexto" in prompt:
            self.calls.append("grade_context")
            return json.dumps({"sufficient": self.grade_sufficient,
                               "missing_information": [] if self.grade_sufficient else ["nombre del sistema"]})
        if "Consultas ya probadas" in prompt:
            self.calls.append("rewrite_query")
            n = self.calls.count("rewrite_query")
            return json.dumps({"spec_query": f"reservas v{n}", "stakeholder_query": f"clientes v{n}"})
        if "Criterios de riesgo" in prompt:
            self.calls.append("generate")
            return json.dumps({
                "summary_report": "El cambio afecta Reservas y Pagos.",
                "technical_changes": [{"description": "Actualizar API", "affected_system": "Reservas"}],
                "risk_level": "Alto",
                "affected_stakeholders": [{"stakeholder": "Atención al cliente", "impact": "Nuevos flujos"}],
                "required_mitigations": ["Plan de rollback"],
                "confidence": "Alto",
            })
        self.calls.append("other")
        return "consulta reformulada"


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "complete", fake)
    return fake


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Embeddings falsos y caché/data en carpetas temporales para cada test."""
    monkeypatch.setattr(embeddings, "_local_encoder", lambda: FakeEncoder())
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(settings, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    return tmp_path
