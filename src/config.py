"""Centralized, externalized configuration.

Design pattern: **Singleton**. A single ``Settings`` instance is created once,
reads every tunable parameter from the environment (with sensible defaults), and
is shared across the whole application via :func:`get_settings`. This guarantees
one consistent source of truth for configuration and avoids re-parsing the
environment on every call.

Every parameter the test asks to be configurable — N (conversation window),
model, chunk size, reranker, retrieval depth — is defined here and driven by
environment variables / the ``.env`` file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env once, at import time, before Settings reads the environment.
load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable, environment-driven configuration object."""

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 1024))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Embeddings. Default is the compact, reliable all-MiniLM-L6-v2. For better
    # Spanish ranking, a multilingual model is recommended (see .env.example) —
    # set EMBEDDING_MODEL to override; the pipeline is model-agnostic.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Vector database
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "bbva_web"))

    # Chunking
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 120))

    # Retrieval
    top_k_retrieval: int = field(default_factory=lambda: _get_int("TOP_K_RETRIEVAL", 20))
    top_k_final: int = field(default_factory=lambda: _get_int("TOP_K_FINAL", 6))
    similarity_threshold: float = field(default_factory=lambda: _get_float("SIMILARITY_THRESHOLD", 0.35))

    # FAISS retrieval cross-check: build an in-process FAISS index from Qdrant's
    # vectors and confirm, per query, that it agrees with Qdrant's top hit.
    faiss_crosscheck_enabled: bool = field(default_factory=lambda: _get_bool("FAISS_CROSSCHECK", True))

    # Reranker
    reranker_enabled: bool = field(default_factory=lambda: _get_bool("RERANKER_ENABLED", True))
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # Conversation history — N previous messages (configurable)
    conversation_window: int = field(default_factory=lambda: _get_int("CONVERSATION_WINDOW", 8))
    history_db_path: str = field(default_factory=lambda: os.getenv("HISTORY_DB_PATH", "./data/history.db"))

    # Scraper
    scrape_start_url: str = field(default_factory=lambda: os.getenv("SCRAPE_START_URL", "https://www.bbva.com.co/"))
    scrape_max_pages: int = field(default_factory=lambda: _get_int("SCRAPE_MAX_PAGES", 40))
    scrape_max_depth: int = field(default_factory=lambda: _get_int("SCRAPE_MAX_DEPTH", 2))
    raw_data_dir: str = field(default_factory=lambda: os.getenv("RAW_DATA_DIR", "./data/raw"))
    clean_data_dir: str = field(default_factory=lambda: os.getenv("CLEAN_DATA_DIR", "./data/clean"))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _get_int("API_PORT", 8000))


# --- Singleton access -------------------------------------------------------
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton, creating it once."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
