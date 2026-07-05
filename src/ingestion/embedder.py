"""Embedding models behind a factory.

Design pattern: **Factory Method**. :class:`EmbedderFactory` decides which
concrete :class:`Embedder` to instantiate from configuration, so the rest of the
system depends only on the abstract interface. Swapping the embedding model (or
adding an API-based embedder) is a change here, not in every call site.

The default is a local sentence-transformers model — free, runs on CPU, no API
cost — with L2-normalized vectors so cosine similarity equals the dot product.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from src.config import Settings, get_settings


class Embedder(ABC):
    """Abstract embedding interface used by ingestion and retrieval."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


class SentenceTransformerEmbedder(Embedder):
    """Local, CPU-friendly embedder using sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


class EmbedderFactory:
    """Create the configured embedder. Extend the mapping to add providers."""

    @staticmethod
    def create(settings: Settings | None = None) -> Embedder:
        settings = settings or get_settings()
        model = settings.embedding_model
        # Only the local provider today; the factory is the single place to add
        # e.g. an OpenAI/Vertex embedder without touching callers.
        return SentenceTransformerEmbedder(model)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide cached embedder (loading the model is expensive)."""
    return EmbedderFactory.create()
