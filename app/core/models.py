# app/core/models.py
"""
SQLAlchemy ORM models — the actual database table definitions.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean
from sqlalchemy.sql import func

from app.core.database import Base


class Prediction(Base):
    """
    One row per /classify request. This is what powers /admin/metrics,
    /admin/evaluate, and /classify/explain/{id} once those are built.
    """
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)  # the prediction_id UUID

    report_text = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    prediction_label = Column(String, nullable=False)
    routing_score = Column(Float, nullable=False)
    routed_to = Column(String, nullable=False)       # "hybrid" | "groq" | "local_qwen" | "anthropic"
    reliability = Column(String, nullable=False)      # "high" | "low"
    top_k_candidates = Column(String, nullable=False)  # stored as comma-separated string
    latency_ms = Column(Integer, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())