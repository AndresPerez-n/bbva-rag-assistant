"""Lightweight fallback scraper using only the standard library.

Used when Playwright/Chromium is unavailable (e.g. a slim environment without
the browser installed). It cannot execute JavaScript, so it only recovers
server-rendered HTML — documented as a limitation in the README. Same crawl
algorithm as the Playwright variant; only :meth:`fetch` differs.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.request import Request, urlopen

from src.scraper.base import BaseScraper

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class HttpScraper(BaseScraper):
    """Fetch pages with a plain HTTP GET (no JS execution)."""

    def __init__(self, *args, timeout_s: int = 20, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timeout_s = timeout_s

    def fetch(self, url: str) -> Optional[str]:
        req = Request(url, headers={"User-Agent": _USER_AGENT, "Accept-Language": "es-CO,es;q=0.9"})
        with urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 — trusted, configured URL
            ctype = resp.headers.get_content_type()
            if ctype and "html" not in ctype:
                return None
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
