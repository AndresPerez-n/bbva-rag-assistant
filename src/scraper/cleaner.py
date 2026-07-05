"""HTML -> clean text extraction.

Kept separate from the crawl logic so the cleaning strategy can evolve
independently of *how* pages are fetched. Uses BeautifulSoup to drop
non-content elements (scripts, styles, nav, footer, cookie banners) and
collapse whitespace into readable paragraphs.
"""
from __future__ import annotations

import re
from typing import List

from bs4 import BeautifulSoup

# Tags that never carry answerable content.
_NOISE_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "form",
    "nav", "footer", "header", "aside",
]

# Substrings in class/id that usually mark chrome (menus, cookies, banners).
_NOISE_HINTS = ["cookie", "menu", "navbar", "breadcrumb", "footer", "header", "banner", "modal"]


class HtmlCleaner:
    """Turn raw HTML into a title + clean body text."""

    def title(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else ""

    def clean_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        for el in soup.find_all(True):
            ident = " ".join(filter(None, [
                " ".join(el.get("class", [])),
                el.get("id", "") or "",
            ])).lower()
            if any(hint in ident for hint in _NOISE_HINTS):
                el.decompose()

        text = soup.get_text(separator="\n")
        return self._normalize(text)

    @staticmethod
    def _normalize(text: str) -> str:
        lines: List[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            # Drop empty and one-word menu leftovers.
            if len(line) >= 2:
                lines.append(line)
        # Collapse 3+ blank runs; join with single newlines.
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
