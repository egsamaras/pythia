"""RSS/Atom ingestion: fetch configured feeds and normalize entries."""

from __future__ import annotations

import calendar
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx

logger = logging.getLogger(__name__)

_MAX_SUMMARY_CHARS = 500
# Some feeds publish 100-300 entries (The Economist, Google News); cap what one
# feed can contribute per fetch so ingestion stays proportionate.
_MAX_ENTRIES_PER_FEED = 25
_FETCH_WORKERS = 8

# Friendly source labels by feed hostname — many feeds self-describe with
# unhelpful titles ("News", "Section Feed", "International homepage"). These
# labels appear in the UI and in the oracle prompt.
_SOURCE_LABELS = {
    "feeds.bbci.co.uk": "BBC News",
    "www.theguardian.com": "The Guardian",
    "rss.nytimes.com": "New York Times",
    "feeds.a.dj.com": "Wall Street Journal",
    "feeds.skynews.com": "Sky News",
    "www.france24.com": "France 24",
    "rss.dw.com": "Deutsche Welle",
    "www.cnbc.com": "CNBC",
    "www.economist.com": "The Economist",
    "www.ft.com": "Financial Times",
    "feeds.content.dowjones.io": "MarketWatch",
    "www.arabnews.com": "Arab News",
    "www.thenationalnews.com": "The National",
    "www.middleeasteye.net": "Middle East Eye",
    "www.federalreserve.gov": "US Federal Reserve",
    "www.ecb.europa.eu": "European Central Bank",
    "www.bankofengland.co.uk": "Bank of England",
}


class _TagStripper(HTMLParser):
    """Collect text content, dropping tags and script/style bodies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join("".join(self._chunks).split())


def strip_html(raw: str) -> str:
    """Reduce an HTML fragment to plain text (tags removed, entities decoded)."""
    if not raw:
        return ""
    stripper = _TagStripper()
    stripper.feed(html.unescape(raw) if "<" not in raw else raw)
    text = stripper.text()
    return re.sub(r"\s+", " ", text).strip()


def _entry_published(entry: Any) -> str:
    """Best-effort published timestamp for an entry, ISO-8601 UTC."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            ts = calendar.timegm(parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_feed(source_url: str, content: bytes) -> list[dict[str, Any]]:
    """Parse raw feed bytes into normalized article dicts.

    Entries without a title or an http(s) link are dropped — they cannot be
    displayed or safely linked. Google News query feeds get their per-entry
    publisher extracted from the "Headline - Publisher" title convention.
    """
    parsed = feedparser.parse(content)
    hostname = urlparse(source_url).hostname or ""
    is_google_news = hostname == "news.google.com"
    feed_source = _SOURCE_LABELS.get(
        hostname, strip_html(parsed.feed.get("title", "")) or source_url
    )
    articles: list[dict[str, Any]] = []
    for entry in parsed.entries[:_MAX_ENTRIES_PER_FEED]:
        title = strip_html(getattr(entry, "title", ""))
        link = (getattr(entry, "link", "") or "").strip()
        if not title or not link.startswith(("http://", "https://")):
            continue
        source = feed_source
        if is_google_news:
            head, sep, publisher = title.rpartition(" - ")
            if sep:
                title, source = head.strip(), publisher.strip() or "Google News"
            else:
                source = "Google News"
        summary = strip_html(getattr(entry, "summary", ""))[:_MAX_SUMMARY_CHARS]
        articles.append(
            {
                "source": source,
                "title": title,
                "url": link,
                "summary": summary,
                "published_at": _entry_published(entry),
            }
        )
    return articles


def fetch_all(
    feed_urls: tuple[str, ...] | list[str],
    timeout_s: float = 15.0,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch every feed concurrently; returns (articles, {feed_url: error}).

    A failing feed never aborts the scan — it is logged and reported.
    ``transport`` is injectable for tests (httpx.MockTransport).
    """
    articles: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    headers = {"User-Agent": "Pythia/0.1 (+local research tool)"}
    with httpx.Client(
        timeout=timeout_s, follow_redirects=True, headers=headers, transport=transport
    ) as client:

        def fetch_one(url: str) -> tuple[str, list[dict[str, Any]], str | None]:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                return url, parse_feed(url, resp.content), None
            except Exception as exc:  # noqa: BLE001 - report per-feed, keep going
                return url, [], f"{type(exc).__name__}: {exc}"

        workers = min(_FETCH_WORKERS, max(1, len(feed_urls)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for url, parsed, error in pool.map(fetch_one, feed_urls):
                if error is None:
                    articles.extend(parsed)
                    logger.info("feed ok: %s (%d entries)", url, len(parsed))
                else:
                    errors[url] = error
                    logger.warning("feed failed: %s (%s)", url, error)
    return articles, errors
