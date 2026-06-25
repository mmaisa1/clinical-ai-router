# app/routers/admin.py
"""
Admin-facing routes: require X-Admin-Key header on every route.

Currently applicable for health and model-info only. 
Metrics, evaluate, threshold, and explain endpoints require PostgreSQL logging (not yet built)
and are stubbed with a clear NotImplementedError-style response.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Depends
from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.core.auth import require_admin_key
from app.core.config import (
    CONFIDENCE_THRESHOLD,
    TOP_K,
    CHUNK_TOKENS,
    CHUNK_STRIDE,
    MAX_CHUNKS,
    NLI_MODEL_NAME,
    LLM_BACKEND,
    QWEN_MODEL_NAME,
    GROQ_MODEL,
    ANTHROPIC_MODEL,
    APP_VERSION,
)
from app.core.labels import LABEL_MAP
from app.core.database import get_db_session
from app.core.models import Prediction
from app.schemas.admin import AdminHealthResponse, ModelInfoResponse, MetricsResponse

router = APIRouter(dependencies=[Depends(require_admin_key)])

def _active_llm_model_name() -> str:
    return {
        "local_qwen": QWEN_MODEL_NAME,
        "groq": GROQ_MODEL,
        "anthropic": ANTHROPIC_MODEL,
    }.get(LLM_BACKEND, "unknown")


@router.get("/admin/health", response_model=AdminHealthResponse)
async def admin_health(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)

    return AdminHealthResponse(
        status="ok" if pipeline else "degraded",
        version=APP_VERSION,
        stage1_loaded=pipeline is not None,
        stage2_loaded=pipeline is not None,
        llm_backend=LLM_BACKEND,
        llm_backend_reachable=pipeline is not None,
        database_connected=False,  # not built yet
    )


@router.get("/admin/model-info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        top_k=TOP_K,
        chunk_tokens=CHUNK_TOKENS,
        chunk_stride=CHUNK_STRIDE,
        max_chunks=MAX_CHUNKS,
        nli_model=NLI_MODEL_NAME,
        llm_backend=LLM_BACKEND,
        llm_model=_active_llm_model_name(),
        num_labels=len(LABEL_MAP),
    )


@router.get("/admin/metrics", response_model=MetricsResponse)
async def metrics(db: Session = Depends(get_db_session)):
    total_requests = db.query(func.count(Prediction.id)).scalar()

    if total_requests == 0:
        return MetricsResponse(
            total_requests=0,
            routed_to_hybrid=0,
            routed_to_fallback=0,
            fallback_rate=0.0,
            low_reliability_count=0,
            low_reliability_rate=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            avg_routing_score=0.0,
        )

    routed_to_hybrid = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.routed_to == "hybrid")
        .scalar()
    )
    routed_to_fallback = total_requests - routed_to_hybrid

    low_reliability_count = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.reliability == "low")
        .scalar()
    )

    avg_latency = db.query(func.avg(Prediction.latency_ms)).scalar()
    avg_routing_score = db.query(func.avg(Prediction.routing_score)).scalar()

    p50, p95 = db.query(
        func.percentile_cont(0.5).within_group(Prediction.latency_ms),
        func.percentile_cont(0.95).within_group(Prediction.latency_ms),
    ).one()

    return MetricsResponse(
        total_requests=total_requests,
        routed_to_hybrid=routed_to_hybrid,
        routed_to_fallback=routed_to_fallback,
        fallback_rate=round(routed_to_fallback / total_requests, 4),
        low_reliability_count=low_reliability_count,
        low_reliability_rate=round(low_reliability_count / total_requests, 4),
        avg_latency_ms=round(float(avg_latency), 2),
        p50_latency_ms=round(float(p50), 2),
        p95_latency_ms=round(float(p95), 2),
        avg_routing_score=round(float(avg_routing_score), 4),
    )


@router.post("/admin/threshold")
async def update_threshold():
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — planned for Phase 5.",
    )


@router.post("/admin/evaluate")
async def evaluate():
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — planned for Phase 5.",
    )


@router.get("/classify/explain/{prediction_id}")
async def explain(prediction_id: str):
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — requires prediction logging (Phase 4).",
    )