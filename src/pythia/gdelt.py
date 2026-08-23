"""GDELT structured signals: worldwide coverage volume as an instrument.

Pythia's first non-editorial source. For each configured watch query, the
GDELT DOC 2.0 API (free, keyless, 15-minute cadence, 100+ languages) supplies:

- recent worldwide articles matching the watch (fed into the archive like any
  feed, labeled "(via GDELT)" with origin "gdelt:<watch>"), and
- a coverage-volume timeline, from which an escalation ratio is computed —
  the last 48 hours against the prior baseline.

The ratios become a plain-language snapshot injected into the forecast prompt
and stored on the scan row, so the signal the oracle read stays auditable.
Coverage volume measures the world's *attention*, not verified events — the
snapshot says so explicitly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_ARTICLES_PER_WATCH = 8
_TIMESPAN_ARTICLES = "2d"
_TIMESPAN_TIMELINE = "14d"
_RECENT_HOURS = 48
_RISING = 1.5
_FALLING = 0.67
# Below this many matched articles in the recent window, a ratio is sampling
# noise: the watch stays silent rather than dressing noise up as signal.
_MIN_RECENT_ARTICLES = 10
# GDELT's contract is one request per 5 seconds per IP, and its block
# ESCALATES with every request made while blocked — so requests are
# sequential with 6s spacing, and the first 429 aborts the whole pull
# (retrying against a block only extends it). The scan proceeds without
# signals and the next scan tries again fresh.
_REQUEST_SPACING_S = 6.0


@dataclass(frozen=True)
class GdeltResult:
    """Everything one signals pull produced."""

    articles: list[dict[str, Any]]
    snapshot: str | None
    errors: dict[str, str]


def _parse_seendate(raw: str) -> str:
    """GDELT's compact stamp ('20260822T101500Z') to ISO-8601 UTC.

    An unparseable stamp falls back to the start of the pull window rather
    than now(), so a broken entry can never win the balancer's recency race.
    """
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="seconds")
    except ValueError:
        fallback = datetime.now(timezone.utc) - timedelta(hours=_RECENT_HOURS)
        return fallback.isoformat(timespec="seconds")


def parse_artlist(watch: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a DOC ArtList payload into Pythia article dicts."""
    articles: list[dict[str, Any]] = []
    for entry in (payload.get("articles") or [])[:_ARTICLES_PER_WATCH]:
        title = (entry.get("title") or "").strip()
        url = (entry.get("url") or "").strip()
        if not title or not url.startswith(("http://", "https://")):
            continue
        language = (entry.get("language") or "").strip()
        country = (entry.get("sourcecountry") or "").strip()
        summary = ""
        if language and language.lower() != "english":
            # Flag non-English items so the oracle knows what it is reading.
            summary = f"({language}-language coverage from {country or 'unknown country'})"
        articles.append(
            {
                "source": f"{entry.get('domain', 'unknown')} (via GDELT)",
                "title": title,
                "url": url,
                "summary": summary,
                "published_at": _parse_seendate(entry.get("seendate", "")),
                "origin": f"gdelt:{watch}",
            }
        )
    return articles


def escalation_ratio(payload: dict[str, Any]) -> tuple[float, float, int] | None:
    """(recent rate, baseline rate, recent matched count) from a TimelineVolRaw.

    Each window's rate is the POOLED ratio sum(value)/sum(norm) across its
    intervals — norm-weighted, so a partial trailing bucket cannot dominate
    and the result does not depend on the wall-clock hour of the scan. Rates
    are matched-article share of all GDELT-monitored coverage, so global
    news-volume swings cancel out. Returns None when the timeline is empty
    or the window split leaves either side empty.
    """
    series = (payload.get("timeline") or [{}])[0].get("data") or []
    points = []
    for p in series:
        norm = p.get("norm") or 0
        if norm > 0:
            points.append((p.get("date", ""), p.get("value") or 0, norm))
    if len(points) < 4:
        return None
    cutoff_dt = datetime.now(timezone.utc)
    recent_v = recent_n = base_v = base_n = 0.0
    for raw_date, value, norm in points:
        try:
            dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (cutoff_dt - dt).total_seconds() <= _RECENT_HOURS * 3600:
            recent_v += value
            recent_n += norm
        else:
            base_v += value
            base_n += norm
    if recent_n <= 0 or base_n <= 0:
        return None
    return (recent_v / recent_n, base_v / base_n, int(recent_v))


def _snapshot_line(
    watch: str, recent: float, baseline: float, recent_count: int
) -> str | None:
    """One snapshot line, or None when the evidence is too thin to label."""
    if recent_count < _MIN_RECENT_ARTICLES:
        # A dormant or barely-matched watch is silence, never a fabricated
        # "new signal" — a ratio over a handful of articles is noise.
        return None
    if baseline <= 0:
        return (
            f"- {watch}: coverage appeared from a near-zero baseline "
            f"({recent_count} matched articles in the recent window) (new signal)"
        )
    ratio = recent / baseline
    trend = "rising" if ratio >= _RISING else "falling" if ratio <= _FALLING else "steady"
    return (
        f"- {watch}: coverage at {ratio:.1f}x its prior 12-day baseline, "
        f"{recent_count} matched articles in the recent ~2-day window ({trend})"
    )


class RateLimited(RuntimeError):
    """GDELT returned 429 — the IP is blocked; stop the pull entirely."""


def _spaced_get(
    client: httpx.Client, params: dict[str, str], first: bool
) -> dict[str, Any]:
    """One rate-respectful DOC API call; a 429 is pull-fatal by design."""
    if not first:
        time.sleep(_REQUEST_SPACING_S)
    resp = client.get(DOC_API, params=params)
    if resp.status_code == 429:
        raise RateLimited("GDELT rate limit hit; aborting this pull")
    resp.raise_for_status()
    return resp.json()


def fetch_signals(
    watches: tuple[tuple[str, str], ...], timeout_s: float = 20.0
) -> GdeltResult:
    """Pull articles and escalation ratios for every watch, fail-soft.

    A failing watch is reported in ``errors`` and skipped; a fully failed
    pull returns empty articles and no snapshot, never an exception.
    """
    articles: list[dict[str, Any]] = []
    lines: list[str] = []
    errors: dict[str, str] = {}
    first = True
    rate_limited = False
    with httpx.Client(
        timeout=timeout_s,
        follow_redirects=True,
        headers={"User-Agent": "Pythia/0.1 (+local research tool)"},
    ) as client:
        for name, query in watches:
            if rate_limited:
                errors[name] = "skipped: GDELT rate limit in effect"
                continue
            try:
                art_payload = _spaced_get(
                    client,
                    {
                        "query": query,
                        "mode": "ArtList",
                        "format": "json",
                        "maxrecords": str(_ARTICLES_PER_WATCH),
                        "sort": "DateDesc",
                        "timespan": _TIMESPAN_ARTICLES,
                    },
                    first,
                )
            except RateLimited as exc:
                first = False
                rate_limited = True
                errors[name] = str(exc)
                logger.warning("gdelt pull aborted at watch %s: %s", name, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - report per-watch, keep going
                first = False
                errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("gdelt watch failed: %s (%s)", name, errors[name])
                continue
            first = False
            # Bank the articles now: a timeline failure must not discard them.
            arts = parse_artlist(name, art_payload)
            articles.extend(arts)
            try:
                vol_payload = _spaced_get(
                    client,
                    {
                        "query": query,
                        "mode": "TimelineVolRaw",
                        "format": "json",
                        "timespan": _TIMESPAN_TIMELINE,
                    },
                    False,
                )
            except RateLimited as exc:
                rate_limited = True
                errors[name] = str(exc)
                logger.warning("gdelt pull aborted at watch %s: %s", name, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"timeline failed: {type(exc).__name__}: {exc}"
                logger.warning("gdelt watch %s: %s", name, errors[name])
                continue
            rates = escalation_ratio(vol_payload)
            if rates:
                line = _snapshot_line(name, *rates)
                if line:
                    lines.append(line)
            logger.info("gdelt watch ok: %s (%d articles)", name, len(arts))

    snapshot = None
    if lines:
        snapshot = (
            "Worldwide coverage-volume ratios per watch topic (GDELT, machine-"
            "coded across 100+ languages). These measure global news ATTENTION, "
            "not verified events: attention can spike with no underlying change. "
            "Use them to decide where to look — never as direct evidence that an "
            "event occurred, and never raise an event's probability on attention "
            "alone:\n" + "\n".join(lines)
        )
    return GdeltResult(articles=articles, snapshot=snapshot, errors=errors)
