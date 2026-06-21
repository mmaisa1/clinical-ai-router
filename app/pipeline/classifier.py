# app/classifier.py
"""
Stage 1 of the hybrid pipeline: TF-IDF + LinearSVC.

Narrows the full 32-label space down to the Top-K most likely candidates
based on term frequency patterns. Loads the serialized models produced
by train.py.
"""

import numpy as np
import joblib

from app.core.config import VECTORIZER_PATH, SVM_MODEL_PATH, TOP_K
from app.core.labels import LABEL_MAP


class Stage1Classifier:
    """Wraps the trained TF-IDF + LinearSVC model for candidate narrowing."""

    def __init__(self):
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.svm_model = joblib.load(SVM_MODEL_PATH)

    def focus_text(self, report: str, max_chars: int = 2000) -> str:
        """
        Extract the diagnostically relevant portion of a report.

        Must stay identical to the version in train.py — the model was
        trained on text processed this same way.
        """
        if report is None:
            return ""
        txt = str(report)
        lower = txt.lower()
        anchors = [
            "final diagnosis", "pathologic diagnosis", "diagnosis",
            "impression", "comment", "microscopic description",
        ]
        start = 0
        for anchor in anchors:
            pos = lower.find(anchor)
            if pos != -1:
                start = pos
                break
        return txt[start: start + max_chars]

    def get_top_k_candidates(self, report_text: str, k: int = TOP_K) -> list[str]:
        """
        Return the Top-K most likely cancer type codes for a report,
        ranked by decision function score (highest first).
        """
        tfidf_vec = self.vectorizer.transform([report_text])

        # CalibratedClassifierCV wraps LinearSVC — access decision_function
        # via the first calibrated classifier's base estimator.
        base = self.svm_model.calibrated_classifiers_[0].estimator
        scores = base.decision_function(tfidf_vec)
        scores = np.asarray(scores).reshape(-1)

        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.svm_model.classes_[i] for i in top_idx]

    def get_top_k_with_verbose(self, report_text: str, k: int = TOP_K) -> tuple[list[str], list[str]]:
        """Return (codes, verbose_names) for the Top-K candidates."""
        codes = self.get_top_k_candidates(report_text, k)
        verbose = [LABEL_MAP[c] for c in codes]
        return codes, verbose