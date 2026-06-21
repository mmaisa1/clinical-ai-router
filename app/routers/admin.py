# app/routers/admin.py
"""
Admin-facing routes: require X-Admin-Key header on every route.

Currently applicable for health and model-info only. 
Metrics, evaluate, threshold, and explain endpoints require PostgreSQL logging (not yet built)
and are stubbed with a clear NotImplementedError-style response.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

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
from app.schemas.admin import AdminHealthResponse, ModelInfoResponse

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


@router.get("/admin/metrics")
async def metrics():
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet — requires PostgreSQL prediction logging (Phase 4).",
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