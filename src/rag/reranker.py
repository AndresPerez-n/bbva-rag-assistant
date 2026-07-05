"""Result re-ranking (bonus).

Design pattern: **Strategy**. Re-ranking is an interchangeable algorithm behind
the :class:`Reranker` interface. Two strategies are provided:

- :class:`NoOpReranker` — keep the vector-similarity order (used when disabled).
- :class:`CrossEncoderReranker` — a local cross-encoder re-scores each
  (query, chunk) pair jointly, which is far more accurate than the bi-encoder
  similarity used for the first-stage recall.

The strategy is selected from configuration by :func:`build_reranker`, so the
retriever code never branches on which one is active.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from src.config import Settings, get_settings
from src.rag.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        ...


class NoOpReranker(Reranker):
    """Pass-through: trust the vector store's ordering, just truncate to top_k."""

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        return chunks[:top_k]


class CrossEncoderReranker(Reranker):
    """Re-score candidates with a cross-encoder and return the best top_k."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"] = float(score)
        ranked = sorted(chunks, key=lambda c: c.metadata.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]


def build_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if not settings.reranker_enabled:
        logger.info("Reranker disabled; using NoOpReranker")
        return NoOpReranker()
    try:
        logger.info("Loading cross-encoder reranker: %s", settings.reranker_model)
        return CrossEncoderReranker(settings.reranker_model)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully if the model can't load
        logger.warning("Failed to load reranker (%s); falling back to NoOpReranker", exc)
        return NoOpReranker()
