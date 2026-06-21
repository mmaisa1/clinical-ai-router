# app/pipeline/pipeline.py
"""
Orchestrates the full hybrid classification pipeline.

Stage 1 (TF-IDF) → Stage 2 (NLI) → routing decision → optional LLM fallback.
This is the single entry point routers/public.py calls for /classify.
"""

import time
import uuid

from app.core.config import CONFIDENCE_THRESHOLD, LLM_BACKEND
from app.core.labels import LABEL_MAP
from app.pipeline.classifier import Stage1Classifier
from app.pipeline.nli import Stage2Reranker
from app.pipeline.llm import get_llm_fallback


class ClassificationPipeline:
    """
    Loads all models once at construction. Instantiate a single instance
    at app startup (see main.py) and reuse it across all requests.
    """

    def __init__(self):
        self.stage1 = Stage1Classifier()
        self.stage2 = Stage2Reranker()
        self.llm_fallback = get_llm_fallback()

    def classify(self, report_text: str) -> dict:
        """
        Run the full pipeline on a single report.

        Returns a dict matching the ClassifyResponse schema fields.
        """
        start = time.perf_counter()

        # Stage 1 — narrow to Top-K candidates
        codes, verbose = self.stage1.get_top_k_with_verbose(report_text)

        # Stage 2 — chunked NLI reranking over full document
        hybrid_code, routing_score = self.stage2.rerank(report_text, codes, verbose)

        # Routing decision
        if routing_score >= CONFIDENCE_THRESHOLD:
            prediction = hybrid_code
            routed_to = "hybrid"
            reliability = "high"
        else:
            fallback_code, was_valid = self.llm_fallback.classify(report_text, codes)
            prediction = fallback_code
            routed_to = LLM_BACKEND
            reliability = "high" if was_valid else "low"

        latency_ms = int((time.perf_counter() - start) * 1000)

        return {
            "prediction": prediction,
            "prediction_label": LABEL_MAP[prediction],
            "routing_score": round(routing_score, 4),
            "routed_to": routed_to,
            "reliability": reliability,
            "top_k_candidates": codes,
            "prediction_id": str(uuid.uuid4()),
            "latency_ms": latency_ms,
        }