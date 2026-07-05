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

from typing import Optional

from src.config import Settings, get_settings
from src.ingestion.embedder import Embedder, EmbedderFactory
from src.rag.faiss_index import FaissCrossCheck, build_faiss_crosscheck
from src.rag.reranker import Reranker, build_reranker
from src.rag.vector_store import RetrievedChunk, VectorStore, create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    top_score: float
    crosscheck: Optional[dict] = None

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
        faiss_crosscheck: Optional[FaissCrossCheck] = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.similarity_threshold = similarity_threshold
        self.faiss_crosscheck = faiss_crosscheck

    def retrieve(self, query: str) -> RetrievalResult:
        query_vector = self.embedder.embed_query(query)
        candidates = self.store.search(
            query_vector, top_k=self.top_k_retrieval, score_threshold=self.similarity_threshold
        )
        crosscheck = self._crosscheck(query_vector, candidates)
        if not candidates:
            return RetrievalResult(chunks=[], top_score=0.0, crosscheck=crosscheck)

        # Confidence uses the first-stage similarity (comparable 0..1 across queries).
        top_score = max(c.score for c in candidates)
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_final)
        return RetrievalResult(chunks=reranked, top_score=top_score, crosscheck=crosscheck)

    def _crosscheck(self, query_vector, candidates) -> Optional[dict]:
        """Compare Qdrant's top vector hit with an independent FAISS search."""
        if not self.faiss_crosscheck or self.faiss_crosscheck.size == 0:
            return None
        qdrant_top = max(candidates, key=lambda c: c.score) if candidates else None
        faiss_hits = self.faiss_crosscheck.search(query_vector, k=1)
        faiss_top = faiss_hits[0] if faiss_hits else None
        agree = bool(
            qdrant_top and faiss_top and qdrant_top.source_url
            and qdrant_top.source_url == faiss_top.url
        )
        return {
            "faiss_indexed": self.faiss_crosscheck.size,
            "qdrant_top": (
                {"url": qdrant_top.source_url, "score": round(qdrant_top.score, 4)}
                if qdrant_top else None
            ),
            "faiss_top": (
                {"url": faiss_top.url, "score": round(faiss_top.score, 4)} if faiss_top else None
            ),
            "agree": agree,
        }


def build_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    store = create_vector_store(settings)
    crosscheck = build_faiss_crosscheck(store) if settings.faiss_crosscheck_enabled else None
    return Retriever(
        embedder=EmbedderFactory.create(settings),
        store=store,
        reranker=build_reranker(settings),
        top_k_retrieval=settings.top_k_retrieval,
        top_k_final=settings.top_k_final,
        similarity_threshold=settings.similarity_threshold,
        faiss_crosscheck=crosscheck,
    )
