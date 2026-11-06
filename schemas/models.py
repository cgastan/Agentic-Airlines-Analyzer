"""Contratos Pydantic de entrada, estado intermedio del agente y reporte final."""
from __future__ import annotations

import unicodedata
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "Bajo"
    MEDIUM = "Medio"
    HIGH = "Alto"
    CRITICAL = "Critico"


class ConfidenceLevel(str, Enum):
    LOW = "Bajo"
    MEDIUM = "Medio"
    HIGH = "Alto"


# ---------- Entrada ----------

TARGET_SYSTEM_DESCRIPTION = (
    "Sistema o componente principal que se quiere modificar (opcional). "
    "Ayuda al agente a buscar en la documentación del sistema correcto cuando el mensaje es vago. "
    "Use el nombre tal como aparece en sus documentos; no distingue mayúsculas, minúsculas ni espacios extra. "
    "Ejemplos: 'AirBook', 'PayHub', 'motor de reservas', 'SkyBus'."
)
TARGET_SYSTEM_EXAMPLES = ["AirBook", "PayHub", "motor de reservas"]


def system_key(value: str) -> str:
    """Forma comparable de un nombre de sistema: minúsculas, sin acentos y sin espacios repetidos."""
    decomposed = unicodedata.normalize("NFKD", value)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(plain.casefold().split())


def clean_system_name(value: Optional[str]) -> Optional[str]:
    """Recorta y colapsa espacios; un valor vacío equivale a no indicar sistema."""
    cleaned = " ".join((value or "").split())
    return cleaned or None


class ChangeRequestPayload(BaseModel):
    """Valida la solicitud de cambio enviada por el Business Analyst."""
    analyst_prompt: str = Field(..., min_length=5, max_length=8000,
                                description="Descripción del cambio propuesto.")
    target_system: Optional[str] = Field(
        None, max_length=200, description=TARGET_SYSTEM_DESCRIPTION, examples=TARGET_SYSTEM_EXAMPLES)

    @field_validator("analyst_prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("La descripción del cambio es demasiado corta.")
        return value

    @field_validator("target_system")
    @classmethod
    def normalize_target_system(cls, value: Optional[str]) -> Optional[str]:
        return clean_system_name(value)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    target_system: Optional[str] = Field(
        None, max_length=200, description=TARGET_SYSTEM_DESCRIPTION, examples=TARGET_SYSTEM_EXAMPLES)

    @field_validator("target_system")
    @classmethod
    def normalize_target_system(cls, value: Optional[str]) -> Optional[str]:
        return clean_system_name(value)


# ---------- Estado intermedio del agente ----------

class ChangeDetails(BaseModel):
    """Salida del paso initialize: detalles del cambio y consultas RAG separadas."""
    change_summary: str
    affected_systems: List[str] = Field(default_factory=list)
    stakeholder_perspectives: List[str] = Field(default_factory=list)
    identifiers: List[str] = Field(default_factory=list)
    scope_limits: List[str] = Field(default_factory=list)
    spec_query: str
    stakeholder_query: str


class ContextGrade(BaseModel):
    """Salida del paso grade_context."""
    sufficient: bool
    missing_information: List[str] = Field(default_factory=list)
    reasoning: str = ""


class RewrittenQueries(BaseModel):
    """Salida del paso rewrite_query."""
    spec_query: str
    stakeholder_query: str


# ---------- Reporte ----------

class SourceReference(BaseModel):
    document_id: str
    title: str
    chunk_id: Optional[str] = None
    page: Optional[int] = None


class TechnicalChange(BaseModel):
    description: str
    affected_system: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    evidence: List[str] = Field(
        default_factory=list, description="IDs de los fragmentos que respaldan el cambio.")


class StakeholderImpact(BaseModel):
    stakeholder: str
    impact: str
    severity: RiskLevel = RiskLevel.MEDIUM
    mitigation: Optional[str] = None


class LLMImpactReport(BaseModel):
    """Campos producidos por el LLM antes de agregar ID y fuentes."""
    summary_report: str
    technical_changes: List[TechnicalChange] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_justification: str = ""
    affected_stakeholders: List[StakeholderImpact] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class AgentTrace(BaseModel):
    """Trazabilidad del workflow, útil para depurar y explicar el resultado."""
    steps: List[str] = Field(default_factory=list)
    queries: List[RewrittenQueries] = Field(default_factory=list)
    rewrites: int = 0
    grade: Optional[ContextGrade] = None


class ImpactReportResponse(LLMImpactReport):
    """Estructura validada para el reporte de impacto final."""
    request_id: str
    change_summary: str = ""
    insufficient_information: bool = False
    missing_information: List[str] = Field(default_factory=list)
    sources: List[SourceReference] = Field(default_factory=list)
    trace: Optional[AgentTrace] = None


# ---------- Chat ----------

class ConversationMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class SessionState(BaseModel):
    session_id: str
    change_request_id: Optional[str] = None
    systems: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    risk_level: Optional[RiskLevel] = None
    summary: str = ""
    messages: List[ConversationMessage] = Field(default_factory=list)
    last_report: Optional[ImpactReportResponse] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str = "impact_analysis"
    report: Optional[ImpactReportResponse] = None
    sources: List[SourceReference] = Field(default_factory=list)
    insufficient_information: bool = False


class IngestedDocument(BaseModel):
    filename: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: List[IngestedDocument]
    total_chunks: int
