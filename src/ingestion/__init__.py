"""Ingestion: turn clean documents into embedded, indexed vector chunks."""

from src.ingestion.chunker import RecursiveChunker, TextChunk
from src.ingestion.embedder import Embedder, EmbedderFactory

__all__ = ["RecursiveChunker", "TextChunk", "Embedder", "EmbedderFactory"]
