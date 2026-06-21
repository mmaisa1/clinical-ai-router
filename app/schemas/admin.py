# app/schemas/admin.py
"""
Pydantic models for admin-facing endpoints.

Used by routers/admin.py: /admin/health, /admin/model-info, /admin/metrics,
/admin/threshold, /admin/evaluate, /classify/explain/{id}.

Note: metrics, threshold, evaluate, and explain schemas are defined here
ahead of their implementation (Phase 4-5) so the contract is locked even
though the routes currently return 501.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ── /admin/health ──────────────────────────────────────────────────────────────

class AdminHealthResponse(BaseModel):
    status: str
    version: str
    stage1_loaded: bool
    stage2_loaded: bool
    llm_backend: str
    llm_backend_reachable: bool
    database_connected: bool


# ── /admin/model-info ─────────────────────────────────────────────────────────

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


# ── /admin/metrics (Phase 4 — not yet implemented) ────────────────────────────

class MetricsResponse(BaseModel):
    total_requests: int
    routed_to_hybrid: int
    routed_to_fallback: int
    fallback_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_routing_score: float


# ── /admin/threshold (Phase 5 — not yet implemented) ──────────────────────────

class ThresholdUpdateRequest(BaseModel):
    new_threshold: float = Field(..., ge=0.0, le=1.0)


class ThresholdUpdateResponse(BaseModel):
    previous_threshold: float
    new_threshold: float
    updated_at: datetime


# ── /admin/evaluate (Phase 5 — not yet implemented) ───────────────────────────

class EvaluateResponse(BaseModel):
    total_reports: int
    overall_accuracy: float
    high_confidence_accuracy: float
    low_confidence_accuracy: float
    high_confidence_count: int
    low_confidence_count: int


# ── /classify/explain/{prediction_id} (Phase 4 — not yet implemented) ─────────

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