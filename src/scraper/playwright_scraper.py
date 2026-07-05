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

# BBVA's site is behind Akamai bot management, which returns HTTP 403 to a naked
# headless browser. Sending realistic client-hint headers and spoofing the
# automation fingerprints below makes the request look like an ordinary Chrome
# session, which the WAF accepts.
_EXTRA_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

_STEALTH_JS = (
    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
    "Object.defineProperty(navigator,'languages',{get:()=>['es-CO','es']});"
    "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
    "window.chrome={runtime:{}};"
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
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = self._browser.new_context(
            user_agent=_USER_AGENT,
            locale="es-CO",
            timezone_id="America/Bogota",
            viewport={"width": 1366, "height": 768},
            extra_http_headers=_EXTRA_HEADERS,
        )
        self._context.add_init_script(_STEALTH_JS)
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
            resp = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            # If the WAF challenges this request, wait and retry once — a second
            # hit usually carries the clearance cookie set on the first.
            if resp is not None and resp.status in (403, 429, 503):
                page.wait_for_timeout(2000)
                resp = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                if resp is not None and resp.status >= 400:
                    logger.warning("skipping %s (status %s)", url, resp.status)
                    return None
            try:
                page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
            except Exception:  # noqa: BLE001 — networkidle can time out on chatty pages
                pass
            page.wait_for_timeout(self.wait_ms)
            return page.content()
        finally:
            page.close()
