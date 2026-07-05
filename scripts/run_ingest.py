"""Entrypoint: chunk, embed, and index the clean documents into Qdrant.

Usage:
    python -m scripts.run_ingest
"""
from __future__ import annotations

import logging
import sys

from src.config import get_settings
from src.ingestion.indexer import build_indexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def main() -> int:
    settings = get_settings()
    logger.info("Ingesting clean docs from %s into Qdrant collection '%s'",
                settings.clean_data_dir, settings.qdrant_collection)
    indexer = build_indexer(settings)
    try:
        total = indexer.ingest_directory(settings.clean_data_dir, recreate=True)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    if total == 0:
        logger.error("No chunks indexed — clean documents may be empty.")
        return 1
    logger.info("Done: %d chunks indexed.", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
