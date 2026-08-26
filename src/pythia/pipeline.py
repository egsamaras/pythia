"""The scan pipeline: fetch news -> consult the oracle -> persist forecasts.

Runs in a background thread started by the API; all progress is written to the
scan row so the UI can poll it.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

from pythia import dedup, feeds, gdelt, scoring, store
from pythia.config import Settings
from pythia.oracle import Oracle, OracleError, get_oracle
from pythia.oracle.base import MODES

logger = logging.getLogger(__name__)


def run_scan(
    settings: Settings,
    scan_id: int,
    oracle: Oracle | None = None,
    today: date | None = None,
) -> None:
    """Execute one full scan, recording progress and errors on the scan row.

    ``oracle`` and ``today`` are injectable for tests; production callers pass
    neither. Never raises — a failure marks the scan as errored.
    """
    db = settings.db_path
    today = today or date.today()
    try:
        store.update_scan(db, scan_id, detail="Fetching news feeds")
        articles, feed_errors = feeds.fetch_all(settings.feeds, settings.feed_timeout_s)
        if not articles:
            raise OracleError(
                "no articles fetched; feed errors: "
                + "; ".join(f"{u}: {e}" for u, e in feed_errors.items())
            )
        inserted = store.upsert_articles(db, articles)
        logger.info("scan %d: %d articles fetched, %d new", scan_id, len(articles), inserted)

        # Structured signals: GDELT coverage-volume ratios + worldwide articles.
        signals_text: str | None = None
        gdelt_note = ""
        if settings.gdelt_enabled:
            store.update_scan(db, scan_id, detail="Reading GDELT signals")
            result = gdelt.fetch_signals(settings.gdelt_watches, settings.gdelt_timeout_s)
            if result.articles:
                store.upsert_articles(db, result.articles)
            signals_text = result.snapshot
            if result.errors and not result.snapshot and not result.articles:
                gdelt_note = " (GDELT unavailable)"
            elif result.errors:
                gdelt_note = f" ({len(result.errors)} GDELT watches failed)"
            if signals_text:
                store.update_scan(db, scan_id, signals=signals_text)

        oracle = oracle or get_oracle(settings)
        cap = (
            settings.max_articles_anthropic
            if oracle.provider == "anthropic"
            else settings.max_articles_ollama
        )
        # Balanced across sources so one prolific feed cannot dominate the batch.
        batch = store.recent_articles_balanced(db, limit=cap)
        store.record_scan_articles(db, scan_id, [a["id"] for a in batch])
        store.update_scan(
            db,
            scan_id,
            article_count=len(batch),
            detail=f"Consulting the oracle ({oracle.provider}:{oracle.model})",
        )

        # Self-learning loop: the measured track record feeds the prompt.
        # Lifetime (time-weighted) probabilities, so revisions can't skew it.
        track_record = scoring.track_record_summary(
            scoring.lifetime_pairs(store.resolved_records(db)),
            {m: scoring.lifetime_pairs(store.resolved_records(db, m)) for m in MODES},
        )
        drafts = oracle.generate(
            batch,
            today,
            store.open_statements(db),
            track_record=track_record,
            signals=signals_text,
        )
        # Enforcement, not instruction: the prompt forbids duplicates and
        # negation pairs, but models oblige unevenly. A ledger holding both X
        # and NOT-X scores itself into meaninglessness, so collisions are
        # refused here against every open claim and within the batch itself.
        seen = store.open_statements(db, limit=1000)
        refused = 0
        for draft in drafts:
            collision = dedup.duplicate_of(draft.statement, seen)
            if collision:
                kind, other = collision
                logger.info(
                    "refusing %s forecast: %r collides with %r", kind,
                    draft.statement[:60], other[:60],
                )
                refused += 1
                continue
            seen.append(draft.statement)
            evidence_ids = [batch[i]["id"] for i in draft.evidence_indices if i < len(batch)]
            store.insert_prediction(
                db,
                scan_id=scan_id,
                statement=draft.statement,
                probability=draft.probability,
                horizon_date=draft.horizon_date,
                category=draft.category,
                rationale=draft.rationale,
                evidence_ids=evidence_ids,
                mode=draft.mode,
                basis=draft.basis,
                region=draft.region,
            )
        kept = len(drafts) - refused
        detail = f"{kept} new forecasts from {len(batch)} articles"
        if refused:
            detail += f" ({refused} duplicates refused)"
        if feed_errors:
            detail += f" ({len(feed_errors)} feeds failed)"
        detail += gdelt_note
        store.update_scan(db, scan_id, status="complete", detail=detail, finished=True)
        logger.info("scan %d complete: %s", scan_id, detail)
    except Exception as exc:  # noqa: BLE001 - boundary: report on the scan row
        logger.exception("scan %d failed", scan_id)
        try:
            store.update_scan(
                db,
                scan_id,
                status="error",
                detail="Scan failed",
                error=f"{type(exc).__name__}: {exc}",
                finished=True,
            )
        except Exception:  # noqa: BLE001 - the startup sweep will reap the row
            logger.exception("scan %d: could not record the failure", scan_id)


def run_resolution_check(
    settings: Settings,
    scan_id: int,
    oracle: Oracle | None = None,
    today: date | None = None,
    target_ids: list[int] | None = None,
) -> None:
    """Check open forecasts against fresh news: propose resolutions, and apply
    evidence-backed probability updates (with full revision history).

    Resolutions stay human-confirmed; probability updates auto-apply unless
    PYTHIA_AUTO_UPDATE=0. ``target_ids`` restricts the check to specific
    forecasts (the per-forecast "Check now" flow). Never raises.
    """
    db = settings.db_path
    today = today or date.today()
    try:
        open_forecasts = store.open_predictions_for_check(
            db,
            today.isoformat(),
            limit=len(target_ids) if target_ids else settings.max_resolution_forecasts,
            ids=target_ids,
        )
        if not open_forecasts:
            store.update_scan(
                db, scan_id, status="complete",
                detail="No open forecasts to check", finished=True,
            )
            return

        store.update_scan(db, scan_id, detail="Fetching news feeds")
        articles, feed_errors = feeds.fetch_all(settings.feeds, settings.feed_timeout_s)
        if articles:
            store.upsert_articles(db, articles)
        elif feed_errors:
            logger.warning("resolution check %d: all feeds failed, using archive", scan_id)

        oracle = oracle or get_oracle(settings)
        cap = (
            settings.max_articles_anthropic
            if oracle.provider == "anthropic"
            else settings.max_articles_ollama
        )
        batch = store.recent_articles_balanced(db, limit=cap)
        # Targeted checks always get the archive treatment for their forecasts.
        batch = _supplement_with_archive(
            db, open_forecasts, batch, today, include_all=bool(target_ids)
        )
        store.record_scan_articles(db, scan_id, [a["id"] for a in batch])
        store.update_scan(
            db,
            scan_id,
            article_count=len(batch),
            detail=f"Judging {len(open_forecasts)} forecasts ({oracle.provider}:{oracle.model})",
        )

        drafts = oracle.resolve(open_forecasts, batch, today)
        # An absence-based NO needs the horizon well past (reporting lag) —
        # and before the horizon, NO requires cited impossibility.
        grace_iso = (today - timedelta(days=2)).isoformat()
        today_iso = today.isoformat()
        prior_revisions = store.revisions_by_prediction(
            db, [fc["id"] for fc in open_forecasts]
        )
        proposed = 0
        updated = 0
        for draft in drafts:
            if draft.verdict == "unresolved":
                continue
            fc = open_forecasts[draft.forecast_index]
            evidence_ids = [batch[i]["id"] for i in draft.evidence_indices if i < len(batch)]
            if draft.verdict == "update":
                if not settings.auto_update_probabilities:
                    logger.info("auto-update disabled; ignoring update for %d", fc["id"])
                    continue
                if fc["horizon_date"] < today_iso:
                    # After the horizon the event already happened or didn't;
                    # revising the number now is hindsight, not forecasting.
                    logger.info("skipping post-horizon update for prediction %d", fc["id"])
                    continue
                if draft.new_probability is None or (
                    round(abs(draft.new_probability - fc["probability"]), 3) < 0.05
                ):
                    logger.info(
                        "skipping negligible update for prediction %d (%.2f -> %s)",
                        fc["id"],
                        fc["probability"],
                        draft.new_probability,
                    )
                    continue
                latest = (prior_revisions.get(fc["id"]) or [None])[0]
                if latest is not None:
                    # Drift guard: one update per day, and never on the same
                    # evidence the last revision already priced in.
                    if latest["created_at"][:10] == today_iso:
                        logger.info("skipping same-day re-update for %d", fc["id"])
                        continue
                    if set(evidence_ids) <= set(latest["evidence_ids"]):
                        logger.info("skipping update for %d: no new evidence", fc["id"])
                        continue
                if store.apply_probability_revision(
                    db,
                    fc["id"],
                    draft.new_probability,
                    note=draft.justification,
                    evidence_ids=evidence_ids,
                ):
                    updated += 1
                continue
            if draft.verdict == "did_not_happen" and not draft.evidence_indices:
                if fc["horizon_date"] >= grace_iso:
                    logger.info(
                        "skipping evidence-free NO for prediction %d (horizon %s too recent)",
                        fc["id"],
                        fc["horizon_date"],
                    )
                    continue
            if store.set_proposal(
                db,
                fc["id"],
                outcome=draft.verdict == "happened",
                note=draft.justification,
                evidence_ids=evidence_ids,
            ):
                proposed += 1
        detail = (
            f"{proposed} resolution proposals, {updated} probability updates "
            f"from {len(open_forecasts)} checked"
        )
        if feed_errors:
            detail += f" ({len(feed_errors)} feeds failed)"
        store.update_scan(db, scan_id, status="complete", detail=detail, finished=True)
        logger.info("resolution check %d complete: %s", scan_id, detail)
    except Exception as exc:  # noqa: BLE001 - boundary: report on the scan row
        logger.exception("resolution check %d failed", scan_id)
        try:
            store.update_scan(
                db,
                scan_id,
                status="error",
                detail="Resolution check failed",
                error=f"{type(exc).__name__}: {exc}",
                finished=True,
            )
        except Exception:  # noqa: BLE001 - the startup sweep will reap the row
            logger.exception("resolution check %d: could not record the failure", scan_id)


def _search_terms(statement: str, max_terms: int = 4) -> list[str]:
    """Distinctive word prefixes from a forecast statement for archive lookup.

    Truncating to a 6-char prefix is cheap stemming: "quota%" matches quota
    and quotas, "increa%" matches increase and increased.
    """
    words = re.findall(r"[A-Za-z][A-Za-z-]{4,}", statement)
    seen: list[str] = []
    for w in words:
        prefix = w.lower()[:6]
        if prefix not in ("happen", "announ", "before") and prefix not in seen:
            seen.append(prefix)
        if len(seen) >= max_terms:
            break
    return seen


def _supplement_with_archive(
    db: Any,
    open_forecasts: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    today: date,
    max_extras: int = 10,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    """Add archived articles relevant to overdue forecasts to the batch.

    The recent batch spans a day or two; the archive holds everything ever
    fetched. Without this, an event reported weeks ago would look like it
    never happened. ``include_all`` supplements every given forecast, not
    only overdue ones (used by targeted checks).
    """
    today_iso = today.isoformat()
    known_ids = {a["id"] for a in batch}
    extras: list[dict[str, Any]] = []
    for fc in open_forecasts:
        if (not include_all and fc["horizon_date"] >= today_iso) or len(extras) >= max_extras:
            continue
        # Evidence window starts a week before the forecast was made — context
        # published just before creation is still legitimate update evidence.
        since_iso = None
        created = (fc.get("created_at") or "")[:10]
        if created:
            try:
                since_iso = (date.fromisoformat(created) - timedelta(days=7)).isoformat()
            except ValueError:
                pass
        # Targeted checks study one forecast: afford it a deeper evidence pull.
        matches = store.search_articles(
            db,
            _search_terms(fc["statement"]),
            limit=6 if include_all else 3,
            since_iso=since_iso,
        )
        for a in matches:
            if a["id"] not in known_ids and len(extras) < max_extras:
                known_ids.add(a["id"])
                extras.append(a)
    if extras:
        logger.info("resolution check: %d archive articles added for overdue forecasts", len(extras))
    return batch + extras


def scan_metadata(settings: Settings) -> dict[str, Any]:
    """Provider/model pair a new scan would use right now."""
    from pythia.oracle import resolve_provider

    provider = resolve_provider(settings)
    model = settings.anthropic_model if provider == "anthropic" else settings.ollama_model
    return {"provider": provider, "model": model}
