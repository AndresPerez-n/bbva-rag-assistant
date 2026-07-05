"""One-shot data bootstrap: scrape the site, then index it into Qdrant.

Usage (inside the app container or a local venv):
    python -m scripts.bootstrap

Equivalent to running scripts.run_scrape followed by scripts.run_ingest.
"""
from __future__ import annotations

import logging
import sys

from scripts import run_ingest, run_scrape

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bootstrap")


def main() -> int:
    logger.info("Step 1/2 — scraping the site")
    rc = run_scrape.main()
    if rc != 0:
        logger.error("Scraping failed; aborting before ingestion.")
        return rc

    logger.info("Step 2/2 — indexing into Qdrant")
    return run_ingest.main()


if __name__ == "__main__":
    sys.exit(main())
