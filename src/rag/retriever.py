"""Two-stage retrieval: vector recall then re-ranking.

1. Embed the query and pull ``top_k_retrieval`` candidates from the vector store
   (wide recall, cheap bi-encoder similarity).
2. Re-rank with the configured strategy and keep ``top_k_final`` (precise).

Returns the final chunks plus a confidence signal derived from the best score,
used both to label answers and to detect out-of-scope questions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.config import Settings, get_settings
from src.ingestion.embedder import Embedder, EmbedderFactory
from src.rag.reranker import Reranker, build_reranker
from src.rag.vector_store import RetrievedChunk, VectorStore, create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    top_score: float

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def confidence(self) -> str:
        if self.is_empty:
            return "out_of_scope"
        if self.top_score >= 0.55:
            return "high"
        if self.top_score >= 0.40:
            return "medium"
        return "low"


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        top_k_retrieval: int,
        top_k_final: int,
        similarity_threshold: float,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> RetrievalResult:
        query_vector = self.embedder.embed_query(query)
        candidates = self.store.search(
            query_vector, top_k=self.top_k_retrieval, score_threshold=self.similarity_threshold
        )
        if not candidates:
            return RetrievalResult(chunks=[], top_score=0.0)

        # Confidence uses the first-stage similarity (comparable 0..1 across queries).
        top_score = max(c.score for c in candidates)
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_final)
        return RetrievalResult(chunks=reranked, top_score=top_score)


def build_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    return Retriever(
        embedder=EmbedderFactory.create(settings),
        store=create_vector_store(settings),
        reranker=build_reranker(settings),
        top_k_retrieval=settings.top_k_retrieval,
        top_k_final=settings.top_k_final,
        similarity_threshold=settings.similarity_threshold,
    )
