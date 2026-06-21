# app/routers/public.py — updated /classify route
"""
Public-facing routes: no authentication required.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session

from app.schemas.classify import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    HealthResponse,
)
from app.core.config import APP_VERSION
from app.core.database import get_db_session
from app.core.models import Prediction

router = APIRouter()


def _log_prediction(db: Session, report_text: str, result: dict) -> None:
    """Persist a classification result. Logging failures should not
    break the API response — log and continue, don't raise."""
    try:
        row = Prediction(
            id=result["prediction_id"],
            report_text=report_text,
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            routing_score=result["routing_score"],
            routed_to=result["routed_to"],
            reliability=result["reliability"],
            top_k_candidates=",".join(result["top_k_candidates"]),
            latency_ms=result["latency_ms"],
        )
        db.add(row)
        db.commit()
    except Exception as e:
        print(f"WARNING: failed to log prediction {result.get('prediction_id')}: {e}")
        db.rollback()


@router.post("/classify", response_model=ClassifyResponse)
async def classify(
    payload: ClassifyRequest,
    request: Request,
    db: Session = Depends(get_db_session),
):
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.classify(payload.report_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    _log_prediction(db, payload.report_text, result)

    return ClassifyResponse(**result)


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(
    payload: BatchClassifyRequest,
    request: Request,
    db: Session = Depends(get_db_session),
):
    pipeline = request.app.state.pipeline
    results = []
    succeeded = 0
    failed = 0

    for report_text in payload.reports:
        try:
            result = pipeline.classify(report_text)
            _log_prediction(db, report_text, result)
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