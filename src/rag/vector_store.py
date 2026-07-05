"""Vector database access.

Design pattern: **Repository**. :class:`VectorStore` defines a storage-agnostic
interface (ensure_collection / upsert / search / count); :class:`QdrantVectorStore`
implements it against Qdrant. The ingestion and retrieval layers depend only on
the interface, so the backing database could be swapped (Weaviate, pgvector, …)
without changing any caller — the collaboration also reads as an **Adapter** over
the Qdrant client.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import Settings, get_settings


@dataclass
class VectorRecord:
    """A chunk to be stored: its vector plus payload metadata."""

    text: str
    vector: List[float]
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A search hit returned to the retriever."""

    text: str
    score: float
    metadata: Dict = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        ...

    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> int:
        ...

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int, score_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class QdrantVectorStore(VectorStore):
    """Repository implementation backed by a Qdrant service."""

    def __init__(self, url: str, collection: str) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        self._client = QdrantClient(url=url, timeout=30.0)

    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        exists = self._client.collection_exists(self.collection)
        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, records: List[VectorRecord]) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=r.vector,
                payload={"text": r.text, **r.metadata},
            )
            for r in records
        ]
        if points:
            self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self, query_vector: List[float], top_k: int, score_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results: List[RetrievedChunk] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(RetrievedChunk(text=text, score=float(hit.score), metadata=payload))
        return results

    def count(self) -> int:
        try:
            return self._client.count(self.collection, exact=True).count
        except Exception:  # noqa: BLE001 — collection may not exist yet
            return 0


def create_vector_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return QdrantVectorStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
