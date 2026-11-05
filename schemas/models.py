# schemas/models.py

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "Bajo"
    MEDIUM = "Medio"
    HIGH = "Alto"
    CRITICAL = "Critico"


class SourceReference(BaseModel):
    document_id: str
    title: str
    chunk_id: Optional[str] = None


class TechnicalChange(BaseModel):
    description: str
    affected_system: Optional[str] = None
    evidence_sources: List[SourceReference] = Field(default_factory=list)


class StakeholderImpact(BaseModel):
    stakeholder: str
    impact: str
    severity: RiskLevel = RiskLevel.MEDIUM
    mitigation: Optional[str] = None


class ChangeRequestPayload(BaseModel):
    """Valida la solicitud de cambio enviada por el Business Analyst."""
    analyst_prompt: str = Field(
        ..., description="Descripción del cambio propuesto por el analista.")
    target_system: Optional[str] = Field(
        None, description="Sistema o componente principal afectado.")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    target_system: Optional[str] = None


class ConversationMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class CompactReport(BaseModel):
    executive_summary: str
    risk_level: Optional[RiskLevel] = None
    key_mitigations: List[str] = Field(default_factory=list)


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
    sources: List[SourceReference] = Field(default_factory=list)
    insufficient_information: bool = False


class LLMImpactReport(BaseModel):
    """Campos producidos por el LLM antes de agregar ID y fuentes."""
    technical_changes: List[TechnicalChange] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    affected_stakeholders: List[StakeholderImpact] = Field(
        default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    summary_report: str
    confidence: str = Field(
        description="Nivel de confianza: Bajo, Medio o Alto")
    insufficient_information: bool = False


class ImpactReportResponse(BaseModel):
    """Estructura validada para el reporte de impacto final."""
    request_id: str
    technical_changes: List[TechnicalChange] = Field(default_factory=list)
    risk_level: RiskLevel
    affected_stakeholders: List[StakeholderImpact] = Field(
        default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    summary_report: str
    confidence: str = Field(
        description="Nivel de confianza: Bajo, Medio o Alto")
    insufficient_information: bool = False
    sources: List[SourceReference] = Field(default_factory=list)
