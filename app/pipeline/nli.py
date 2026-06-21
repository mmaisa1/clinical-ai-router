# app/nli.py
"""
Stage 2 of the hybrid pipeline: chunked NLI reranking.

Splits the full report into overlapping token windows and scores each
window against the Stage 1 Top-K candidates using BART-large-MNLI
zero-shot classification. The highest-scoring chunk prediction wins.

This stage exists because 67.3% of TCGA reports exceed the 512-token
input limit of most NLI models — chunking gives full document coverage
instead of silently truncating the report.
"""

import torch
from transformers import pipeline as hf_pipeline, AutoTokenizer

from app.core.config import NLI_MODEL_NAME, CHUNK_TOKENS, CHUNK_STRIDE, MAX_CHUNKS
from app.core.labels import VERBOSE_TO_CODE


class Stage2Reranker:
    """Wraps BART-large-MNLI for chunked zero-shot reranking."""

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL_NAME,
            device=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=True)
        self.device_label = "GPU" if device == 0 else "CPU"

    def chunk_by_tokens(self, text: str) -> list[str]:
        """
        Split text into overlapping token windows.

        Uses CHUNK_TOKENS-sized windows with CHUNK_STRIDE overlap,
        capped at MAX_CHUNKS windows total to bound latency on very
        long reports.
        """
        ids = self.tokenizer(
            text, add_special_tokens=False, truncation=False
        )["input_ids"]

        chunks = []
        start = 0
        while start < len(ids) and len(chunks) < MAX_CHUNKS:
            end = min(start + CHUNK_TOKENS, len(ids))
            chunk_txt = self.tokenizer.decode(ids[start:end], skip_special_tokens=True)
            chunks.append(chunk_txt)
            if end == len(ids):
                break
            start = max(0, end - CHUNK_STRIDE)

        return chunks if chunks else [str(text)]

    def rerank(
        self,
        report_text: str,
        candidate_codes: list[str],
        candidate_verbose: list[str],
    ) -> tuple[str, float]:
        """
        Rerank Stage 1 candidates using chunked NLI scoring.

        Returns (best_code, best_score) — the highest-confidence chunk
        prediction across the entire document.
        """
        chunks = self.chunk_by_tokens(report_text)

        best_code = candidate_codes[0]  # fallback if nothing scores higher
        best_score = -1.0

        for chunk in chunks:
            result = self.pipe(
                chunk,
                candidate_labels=candidate_verbose,
                hypothesis_template="This pathology report describes {}.",
                truncation=True,
                max_length=1024,
            )
            top_verbose = result["labels"][0]
            top_score = float(result["scores"][0])

            if top_score > best_score:
                best_score = top_score
                best_code = VERBOSE_TO_CODE[top_verbose]

        return best_code, best_score