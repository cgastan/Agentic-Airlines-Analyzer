"""Contratos Pydantic de entrada, estado intermedio del agente y reporte final."""
from __future__ import annotations

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

class ChangeRequestPayload(BaseModel):
    """Valida la solicitud de cambio enviada por el Business Analyst."""
    analyst_prompt: str = Field(..., min_length=5, max_length=8000,
                                description="Descripción del cambio propuesto.")
    target_system: Optional[str] = Field(
        None, max_length=200, description="Sistema o componente principal afectado.")

    @field_validator("analyst_prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("La descripción del cambio es demasiado corta.")
        return value


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    target_system: Optional[str] = Field(None, max_length=200)


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
    doc_type: Optional[str] = None


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
    doc_type: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: List[IngestedDocument]
    total_chunks: int
