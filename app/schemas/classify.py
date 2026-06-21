# app/schemas/classify.py
"""
Pydantic models for public-facing classification endpoints.

Used by routers/public.py: /classify, /classify/batch, /health.
"""

from pydantic import BaseModel, Field, field_validator


# /classify
class ClassifyRequest(BaseModel):
    report_text: str = Field(
        ...,
        min_length=20,
        description="Full text of the pathology report to classify."
    )

    @field_validator("report_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("report_text cannot be blank or whitespace only")
        return v


class ClassifyResponse(BaseModel):
    prediction: str = Field(..., description="Predicted cancer type code, e.g. 'BRCA'")
    prediction_label: str = Field(..., description="Full cancer type name")
    routing_score: float = Field(..., description="Stage 2 NLI routing score (0-1)")
    routed_to: str = Field(..., description="'hybrid' or the fallback backend name used")
    reliability: str = Field(..., description="'high' or 'low' based on routing score")
    top_k_candidates: list[str] = Field(..., description="Stage 1 candidate codes")
    prediction_id: str = Field(..., description="UUID for this prediction, used for /explain")
    latency_ms: int = Field(..., description="Total processing time in milliseconds")


# /classify/batch
class BatchClassifyRequest(BaseModel):
    reports: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of report texts to classify, max 100 per batch."
    )


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total: int
    succeeded: int
    failed: int


# /health
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    version: str


# Shared error response 
class ErrorResponse(BaseModel):
    detail: str