"""In-process FAISS index used purely as an independent retrieval cross-check.

The system's vector database is **Qdrant** (a real, self-hosted service). This
module builds a small FAISS ``IndexFlatIP`` in memory from the *same* vectors
stored in Qdrant, so every query can be answered a second time, independently,
and compared against Qdrant's top hit. Agreement between the two engines is a
confirmation that retrieval is consistent and not an artifact of one backend.

Vectors are L2-normalized at embed time, so inner product == cosine similarity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FaissHit:
    text: str
    url: str
    title: str
    score: float


class FaissCrossCheck:
    def __init__(self) -> None:
        self._index = None
        self._meta: List[dict] = []
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def build(self, records) -> "FaissCrossCheck":
        import faiss
        import numpy as np

        if not records:
            logger.warning("FAISS cross-check: no records to index")
            return self
        vectors = np.asarray([r.vector for r in records], dtype="float32")
        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._meta = [
            {"text": r.text, "url": r.metadata.get("url", ""), "title": r.metadata.get("title", "")}
            for r in records
        ]
        self._size = len(records)
        logger.info("FAISS cross-check index built: %d vectors, dim=%d", self._size, dim)
        return self

    def search(self, query_vector: List[float], k: int = 1) -> List[FaissHit]:
        if self._index is None or self._size == 0:
            return []
        import numpy as np

        q = np.asarray([query_vector], dtype="float32")
        scores, idxs = self._index.search(q, min(k, self._size))
        hits: List[FaissHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            m = self._meta[int(idx)]
            hits.append(FaissHit(text=m["text"], url=m["url"], title=m["title"], score=float(score)))
        return hits


def build_faiss_crosscheck(store: VectorStore) -> FaissCrossCheck:
    """Build the cross-check index from whatever is currently in Qdrant."""
    cc = FaissCrossCheck()
    try:
        cc.build(store.fetch_all())
    except Exception as exc:  # noqa: BLE001 — the cross-check is optional, never fatal
        logger.warning("FAISS cross-check unavailable: %s", exc)
    return cc
