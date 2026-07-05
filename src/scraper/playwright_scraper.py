"""JavaScript-aware scraper.

BBVA Colombia's site is heavily client-rendered and sits behind bot protection,
so a plain HTTP GET returns an near-empty shell. This implementation drives a
headless Chromium via Playwright: it renders the page, waits for the network to
settle, and returns the fully-hydrated DOM. Only the :meth:`fetch` primitive is
implemented here — the crawl/persist algorithm is inherited from
:class:`~src.scraper.base.BaseScraper`.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.scraper.base import BaseScraper

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class PlaywrightScraper(BaseScraper):
    """Fetch pages with a headless Chromium so JS-rendered content is captured."""

    def __init__(self, *args, wait_ms: int = 2500, timeout_ms: int = 30000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wait_ms = wait_ms
        self.timeout_ms = timeout_ms
        self._pw = None
        self._browser = None
        self._context = None

    def setup(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        self._context = self._browser.new_context(
            user_agent=_USER_AGENT,
            locale="es-CO",
            viewport={"width": 1366, "height": 900},
        )
        logger.info("Playwright Chromium started")

    def teardown(self) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer:
                    closer.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pw:
            self._pw.stop()
        logger.info("Playwright Chromium stopped")

    def fetch(self, url: str) -> Optional[str]:
        page = self._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
            except Exception:  # noqa: BLE001 — networkidle can time out on chatty pages
                pass
            page.wait_for_timeout(self.wait_ms)
            return page.content()
        finally:
            page.close()
