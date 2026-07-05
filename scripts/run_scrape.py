"""Entrypoint: scrape the configured bank website into data/raw + data/clean.

Usage:
    python -m scripts.run_scrape

Tries the JavaScript-aware Playwright scraper first (required for BBVA's
client-rendered site); if Chromium is unavailable it falls back to a plain HTTP
scraper and warns that JS-only content will be missed.
"""
from __future__ import annotations

import logging
import sys

from src.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scrape")


def build_scraper():
    settings = get_settings()
    common = dict(
        start_url=settings.scrape_start_url,
        raw_dir=settings.raw_data_dir,
        clean_dir=settings.clean_data_dir,
        max_pages=settings.scrape_max_pages,
        max_depth=settings.scrape_max_depth,
    )
    try:
        import playwright  # noqa: F401
        from src.scraper.playwright_scraper import PlaywrightScraper

        logger.info("Using Playwright (JS-aware) scraper")
        return PlaywrightScraper(**common)
    except Exception as exc:  # noqa: BLE001
        from src.scraper.http_scraper import HttpScraper

        logger.warning("Playwright unavailable (%s); falling back to plain HTTP scraper", exc)
        return HttpScraper(**common)


def main() -> int:
    settings = get_settings()
    logger.info("Scraping %s (max_pages=%d, max_depth=%d)",
                settings.scrape_start_url, settings.scrape_max_pages, settings.scrape_max_depth)
    scraper = build_scraper()
    docs = scraper.run()
    if not docs:
        logger.error("No documents scraped. The site may be blocking automated access.")
        return 1
    logger.info("Done: %d clean documents written to %s", len(docs), settings.clean_data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
