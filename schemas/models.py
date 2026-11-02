# schemas/models.py

from pydantic import BaseModel, Field
from typing import List, Optional


class ChangeRequestPayload(BaseModel):
    """Valida la solicitud de cambio enviada por el Business Analyst[cite: 5]."""
    request_id: str = Field(...,
                            description="Identificador único de la solicitud de cambio.")
    analyst_prompt: str = Field(
        ..., description="Descripción del cambio propuesto por el analista.")
    target_system: Optional[str] = Field(
        None, description="Sistema o componente principal afectado.")


class ImpactReportResponse(BaseModel):
    """Estructura de salida para el reporte de impacto final[cite: 5]."""
    request_id: str
    technical_changes: List[str] = Field(
        ..., description="Cambios técnicos necesarios identificados[cite: 5].")
    risk_level: str = Field(
        ..., description="Nivel de riesgo evaluado (ej. Bajo, Medio, Alto)[cite: 5].")
    affected_stakeholders: List[str] = Field(
        ..., description="Stakeholders o áreas operativas afectadas[cite: 5].")
    required_mitigations: List[str] = Field(...,
                                            description="Mitigaciones requeridas[cite: 5].")
    summary_report: str = Field(...,
                                description="Texto completo del reporte generado por el LLM.")
