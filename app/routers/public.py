# app/routers/public.py
"""
Public-facing routes: no authentication required.
"""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.classify import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    HealthResponse,
)
from app.core.config import APP_VERSION

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest, request: Request):
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.classify(payload.report_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    return ClassifyResponse(**result)


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(payload: BatchClassifyRequest, request: Request):
    pipeline = request.app.state.pipeline
    results = []
    succeeded = 0
    failed = 0

    for report_text in payload.reports:
        try:
            result = pipeline.classify(report_text)
            results.append(ClassifyResponse(**result))
            succeeded += 1
        except Exception:
            failed += 1

    return BatchClassifyResponse(
        results=results,
        total=len(payload.reports),
        succeeded=succeeded,
        failed=failed,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline_loaded = hasattr(request.app.state, "pipeline")
    return HealthResponse(
        status="ok" if pipeline_loaded else "degraded",
        version=APP_VERSION,
    )