# app/schemas.py
"""
Pydantic models for request/response validation.

These define the public contract of the API — what clients send and
what they get back. FastAPI uses these for automatic validation,
serialization, and OpenAPI docs generation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# Public: /classify
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


# Public: /classify/batch
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


# Public: /health
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    version: str


# Admin: /admin/health
class AdminHealthResponse(BaseModel):
    status: str
    version: str
    stage1_loaded: bool
    stage2_loaded: bool
    llm_backend: str
    llm_backend_reachable: bool
    database_connected: bool


#  Admin: /admin/model-info
class ModelInfoResponse(BaseModel):
    confidence_threshold: float
    top_k: int
    chunk_tokens: int
    chunk_stride: int
    max_chunks: int
    nli_model: str
    llm_backend: str
    llm_model: str
    num_labels: int


# Admin: /admin/metrics
class MetricsResponse(BaseModel):
    total_requests: int
    routed_to_hybrid: int
    routed_to_fallback: int
    fallback_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_routing_score: float


# Admin: /admin/threshold
class ThresholdUpdateRequest(BaseModel):
    new_threshold: float = Field(..., ge=0.0, le=1.0)


class ThresholdUpdateResponse(BaseModel):
    previous_threshold: float
    new_threshold: float
    updated_at: datetime


# Admin: /admin/evaluate
class EvaluateResponse(BaseModel):
    total_reports: int
    overall_accuracy: float
    high_confidence_accuracy: float
    low_confidence_accuracy: float
    high_confidence_count: int
    low_confidence_count: int


# Admin: /classify/explain/{prediction_id}

class ChunkEvidence(BaseModel):
    chunk_index: int
    chunk_text: str
    top_label: str
    score: float


class ExplainResponse(BaseModel):
    prediction_id: str
    prediction: str
    routing_score: float
    top_k_candidates: list[str]
    chunk_evidence: list[ChunkEvidence]


# Error response (used across all endpoints) 
class ErrorResponse(BaseModel):
    detail: str