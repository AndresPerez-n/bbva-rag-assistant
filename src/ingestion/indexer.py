"""Index clean documents into the vector store.

Reads the clean JSON documents produced by the scraper, splits them into
overlapping chunks, embeds each chunk locally, and upserts the vectors (with
source metadata for citations) into the vector store.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.config import Settings, get_settings
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.embedder import Embedder, EmbedderFactory
from src.rag.vector_store import VectorRecord, VectorStore, create_vector_store

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunker: RecursiveChunker,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker

    def ingest_directory(self, clean_dir: str, recreate: bool = True) -> int:
        clean_path = Path(clean_dir)
        files = sorted(clean_path.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"No clean documents found in {clean_dir}. Run the scraper first."
            )

        self.store.ensure_collection(self.embedder.dimension, recreate=recreate)

        total_chunks = 0
        for file in files:
            doc = json.loads(file.read_text(encoding="utf-8"))
            metadata = {
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "source_file": file.name,
            }
            chunks = self.chunker.split(doc.get("text", ""), metadata=metadata)
            if not chunks:
                continue

            vectors = self.embedder.embed_documents([c.text for c in chunks])
            records: List[VectorRecord] = [
                VectorRecord(text=c.text, vector=v, metadata=c.metadata)
                for c, v in zip(chunks, vectors)
            ]
            written = self.store.upsert(records)
            total_chunks += written
            logger.info("indexed %s -> %d chunks", file.name, written)

        logger.info("ingestion complete: %d chunks across %d documents", total_chunks, len(files))
        return total_chunks


def build_indexer(settings: Settings | None = None) -> Indexer:
    settings = settings or get_settings()
    embedder = EmbedderFactory.create(settings)
    store = create_vector_store(settings)
    chunker = RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
    return Indexer(embedder, store, chunker)
