"""Crawl skeleton.

Design pattern: **Template Method**. :class:`BaseScraper` defines the fixed
crawling algorithm in :meth:`run` — seed a queue, breadth-first visit pages up
to a page/depth budget, persist the raw HTML (*crudos*), clean it, persist the
clean document (*limpios*), and enqueue in-domain links. The single step that
varies between implementations — *how a page's HTML is fetched* — is deferred to
the abstract :meth:`fetch` primitive. Subclasses (Playwright, plain HTTP) supply
only that step and inherit the entire crawl/persist workflow unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

from src.scraper.cleaner import HtmlCleaner

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """One cleaned page ready for ingestion."""

    url: str
    title: str
    text: str
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_path: Optional[str] = None
    clean_path: Optional[str] = None


class BaseScraper(ABC):
    """Breadth-first, same-domain crawler with raw + clean persistence."""

    def __init__(
        self,
        start_url: str,
        raw_dir: str,
        clean_dir: str,
        max_pages: int = 40,
        max_depth: int = 2,
    ) -> None:
        self.start_url = start_url
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.domain = urlparse(start_url).netloc
        self.cleaner = HtmlCleaner()
        self.visited: Set[str] = set()

    # --- Template method (fixed algorithm) ---------------------------------
    def run(self) -> List[ScrapedDocument]:
        """Execute the crawl. Subclasses do not override this."""
        self.setup()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        queue: deque = deque([(self._canonical(self.start_url), 0)])
        documents: List[ScrapedDocument] = []

        while queue and len(self.visited) < self.max_pages:
            url, depth = queue.popleft()
            if url in self.visited or depth > self.max_depth:
                continue
            self.visited.add(url)

            try:
                html = self.fetch(url)  # <-- the varying primitive
            except Exception as exc:  # noqa: BLE001 — one bad page must not kill the crawl
                logger.warning("fetch failed for %s: %s", url, exc)
                continue
            if not html:
                continue

            raw_path = self._save_raw(url, html)
            title = self.cleaner.title(html)
            text = self.cleaner.clean_text(html)

            if len(text) >= 200:  # skip near-empty shells
                doc = ScrapedDocument(url=url, title=title, text=text, raw_path=str(raw_path))
                doc.clean_path = str(self._save_clean(url, doc))
                documents.append(doc)
                logger.info("saved [%d/%d] %s (%d chars)", len(documents), self.max_pages, url, len(text))

            if depth < self.max_depth:
                for link in self._extract_links(html, url):
                    if link not in self.visited:
                        queue.append((link, depth + 1))

        self.teardown()
        logger.info("crawl complete: %d pages saved", len(documents))
        return documents

    # --- Primitive operation (must be implemented) -------------------------
    @abstractmethod
    def fetch(self, url: str) -> Optional[str]:
        """Return the page HTML for ``url`` (or None to skip it)."""

    # --- Hooks (optional overrides) ----------------------------------------
    def setup(self) -> None:  # noqa: D401 — optional lifecycle hook
        """Called once before the crawl."""

    def teardown(self) -> None:
        """Called once after the crawl (e.g. close a browser)."""

    # --- Shared helpers ----------------------------------------------------
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            absolute = self._canonical(urljoin(base_url, href))
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != self.domain:
                continue  # stay on the same domain
            if re.search(r"\.(pdf|jpg|jpeg|png|gif|zip|docx?|xlsx?|mp4|svg)$", parsed.path, re.I):
                continue
            links.append(absolute)
        return links

    @staticmethod
    def _canonical(url: str) -> str:
        url, _ = urldefrag(url)  # drop #fragments
        return url.rstrip("/") or url

    def _slug(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_") or "index"
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "-", path)[:60]
        return f"{safe}-{digest}"

    def _save_raw(self, url: str, html: str) -> Path:
        path = self.raw_dir / f"{self._slug(url)}.html"
        path.write_text(html, encoding="utf-8")
        return path

    def _save_clean(self, url: str, doc: ScrapedDocument) -> Path:
        path = self.clean_dir / f"{self._slug(url)}.json"
        path.write_text(json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
