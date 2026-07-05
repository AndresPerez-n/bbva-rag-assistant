"""Recursive character text splitter.

Splits long documents into overlapping chunks that respect natural boundaries
(paragraph -> line -> sentence -> word) before falling back to a hard cut. This
keeps a chunk semantically coherent — one topic per chunk — which improves
retrieval precision. Implemented directly (no framework dependency) so the
behaviour is explicit and easy to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Ordered from coarsest to finest boundary.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class TextChunk:
    text: str
    metadata: Dict = field(default_factory=dict)


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Dict | None = None) -> List[TextChunk]:
        metadata = metadata or {}
        pieces = self._split_recursive(text.strip(), _SEPARATORS)
        merged = self._merge_with_overlap(pieces)
        return [
            TextChunk(text=chunk, metadata={**metadata, "chunk_index": i})
            for i, chunk in enumerate(merged)
            if chunk.strip()
        ]

    # --- internals ---------------------------------------------------------
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator = separators[0]
        rest = separators[1:]

        if separator == "":
            # Hard fallback: slice by size.
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        out: List[str] = []
        for part in parts:
            piece = part + separator
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split_recursive(piece, rest))
        return out

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Greedily pack pieces up to chunk_size, carrying an overlap tail."""
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current.strip())
                if len(piece) > self.chunk_size:
                    # A single oversized piece: emit as-is (already size-sliced upstream).
                    chunks.append(piece.strip())
                    current = ""
                else:
                    overlap = current[-self.chunk_overlap:] if current else ""
                    current = overlap + piece
        if current.strip():
            chunks.append(current.strip())
        return chunks
