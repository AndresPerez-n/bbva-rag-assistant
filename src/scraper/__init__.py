"""Web scraping layer: crawl a bank website and persist raw + clean documents."""

from src.scraper.base import BaseScraper, ScrapedDocument
from src.scraper.cleaner import HtmlCleaner

__all__ = ["BaseScraper", "ScrapedDocument", "HtmlCleaner"]
