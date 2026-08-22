"""FastAPI application: JSON API plus the static dashboard.

Local, single-user tool — bind to 127.0.0.1; there is no authentication.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pythia import pipeline, scoring, store
from pythia.config import (
    CATEGORIES,
    LEGACY_DB_PATH,
    REGIONS,
    db_path_is_default,
    get_settings,
)
from pythia.oracle import OracleError
from pythia.oracle.base import MODES
from pythia.oracle.ollama_oracle import ollama_available

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if db_path_is_default() and store.migrate_legacy_db(LEGACY_DB_PATH, settings.db_path):
        logger.warning(
            "database migrated from %s to %s (legacy file left in place; "
            "Documents is often iCloud-synced, which corrupts SQLite WAL files)",
            LEGACY_DB_PATH,
            settings.db_path,
        )
    store.init_db(settings.db_path)
    # Age-based sweep: a launchd CLI run may be in flight while the server
    # starts, so only clearly-stale rows are reaped here (and again lazily on
    # each run request, so a stale row self-heals without a restart).
    orphaned = store.fail_orphaned_scans(
        settings.db_path, older_than_s=settings.oracle_timeout_s + 900
    )
    if orphaned:
        logger.warning("marked %d stale running scan(s) as interrupted", orphaned)
    logger.info("db ready at %s", settings.db_path)
    yield


app = FastAPI(title="Pythia", version="0.1.0", lifespan=_lifespan)

_scan_lock = threading.Lock()


class ResolveRequest(BaseModel):
    outcome: bool
    note: str = Field(default="", max_length=2000)


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    try:
        meta = pipeline.scan_metadata(settings)
        provider: dict[str, Any] = {"available": True, **meta}
    except OracleError as exc:
        provider = {"available": False, "error": str(exc)}
    return {
        "status": "ok",
        "provider": provider,
        "ollama_reachable": ollama_available(settings.ollama_url),
        "categories": list(CATEGORIES),
        "modes": list(MODES),
        "regions": list(REGIONS),
    }


def _start_background_run(kind: str, target_ids: list[int] | None = None) -> dict[str, Any]:
    """Shared launch path for forecast scans and resolution checks.

    One run at a time regardless of kind — they share the model backend.
    """
    settings = get_settings()
    with _scan_lock:
        store.fail_orphaned_scans(
            settings.db_path, older_than_s=settings.oracle_timeout_s + 900
        )
        try:
            meta = pipeline.scan_metadata(settings)
        except OracleError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        # Atomic gate: creation succeeds only when no run is in progress,
        # closing the race against a concurrently-launched CLI run.
        scan_id = store.create_scan_exclusive(
            settings.db_path, meta["provider"], meta["model"], kind
        )
        if scan_id is None:
            running = store.running_scan(settings.db_path)
            raise HTTPException(
                status_code=409,
                detail={"scan_id": running["id"] if running else None},
            )

    if kind == "forecast":
        thread_target, args = pipeline.run_scan, (settings, scan_id)
    else:
        thread_target = pipeline.run_resolution_check
        args = (settings, scan_id, None, None, target_ids)
    thread = threading.Thread(
        target=thread_target, args=args, daemon=True, name=f"{kind}-{scan_id}"
    )
    thread.start()
    return {"scan_id": scan_id, "kind": kind, **meta}


@app.post("/api/scan", status_code=202)
def start_scan() -> dict[str, Any]:
    return _start_background_run("forecast")


@app.post("/api/resolution-check", status_code=202)
def start_resolution_check() -> dict[str, Any]:
    return _start_background_run("resolution")


@app.post("/api/predictions/{pred_id}/check", status_code=202)
def check_one_prediction(pred_id: int) -> dict[str, Any]:
    """Targeted resolution check for a single open forecast."""
    pred = store.get_prediction(get_settings().db_path, pred_id)
    if pred is None or pred["status"] != "open":
        raise HTTPException(status_code=404, detail="open prediction not found")
    return _start_background_run("resolution", target_ids=[pred_id])


@app.get("/api/scans")
def scans() -> list[dict[str, Any]]:
    return store.list_scans(get_settings().db_path)


@app.get("/api/scans/{scan_id}")
def scan_detail(scan_id: int) -> dict[str, Any]:
    scan = store.get_scan(get_settings().db_path, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return scan


@app.get("/api/scans/{scan_id}/articles")
def scan_digest(scan_id: int) -> list[dict[str, Any]]:
    settings = get_settings()
    if store.get_scan(settings.db_path, scan_id) is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return store.articles_for_scan(settings.db_path, scan_id)


@app.get("/api/predictions")
def predictions(
    status: str | None = Query(default=None, pattern="^(open|resolved)$"),
) -> list[dict[str, Any]]:
    settings = get_settings()
    preds = store.list_predictions(settings.db_path, status)
    # Attach evidence articles so the UI can link headlines without N+1 calls.
    all_ids = sorted(
        {
            i
            for p in preds
            for i in p["evidence_ids"] + p["proposal_evidence_ids"] + p["resolution_evidence_ids"]
        }
    )
    by_id = {a["id"]: a for a in store.articles_by_ids(settings.db_path, all_ids)}
    revisions = store.revisions_by_prediction(settings.db_path, [p["id"] for p in preds])
    today = date.today().isoformat()
    for p in preds:
        p["revisions"] = revisions.get(p["id"], [])
        p["evidence"] = [by_id[i] for i in p["evidence_ids"] if i in by_id]
        p["proposal_evidence"] = [
            by_id[i] for i in p["proposal_evidence_ids"] if i in by_id
        ]
        p["resolution_evidence"] = [
            by_id[i] for i in p["resolution_evidence_ids"] if i in by_id
        ]
        p["due"] = p["status"] == "open" and p["horizon_date"] < today
    return preds


@app.post("/api/predictions/{pred_id}/resolve")
def resolve(pred_id: int, body: ResolveRequest) -> dict[str, Any]:
    updated = store.resolve_prediction(
        get_settings().db_path, pred_id, body.outcome, body.note
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    return updated


@app.post("/api/predictions/{pred_id}/proposal/accept")
def accept_proposal(pred_id: int) -> dict[str, Any]:
    updated = store.resolve_from_proposal(get_settings().db_path, pred_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="no pending proposal")
    return updated


@app.post("/api/predictions/{pred_id}/proposal/reject")
def reject_proposal(pred_id: int) -> dict[str, Any]:
    if not store.clear_proposal(get_settings().db_path, pred_id):
        raise HTTPException(status_code=404, detail="no pending proposal")
    return {"ok": True}


@app.post("/api/predictions/{pred_id}/reopen")
def reopen(pred_id: int) -> dict[str, Any]:
    updated = store.reopen_prediction(get_settings().db_path, pred_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    return updated


@app.get("/api/articles")
def articles(limit: int = Query(default=30, ge=1, le=200)) -> list[dict[str, Any]]:
    return store.recent_articles(get_settings().db_path, limit=limit)


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    settings = get_settings()
    # Lifetime (time-weighted) probabilities: a late revision only counts for
    # the slice of the forecast's open life it was actually held.
    pairs = scoring.lifetime_pairs(store.resolved_records(settings.db_path))
    preds = store.list_predictions(settings.db_path)
    today = date.today().isoformat()
    open_preds = [p for p in preds if p["status"] == "open"]
    by_mode = {}
    for mode in MODES:
        mode_pairs = scoring.lifetime_pairs(
            store.resolved_records(settings.db_path, mode=mode)
        )
        by_mode[mode] = {
            "open": sum(1 for p in open_preds if p["mode"] == mode),
            "resolved": len(mode_pairs),
            "brier": scoring.brier_score(mode_pairs),
        }
    return {
        "open": len(open_preds),
        "due": sum(1 for p in open_preds if p["horizon_date"] < today),
        "resolved": len(pairs),
        "brier": scoring.brier_score(pairs),
        "hit_rate": scoring.hit_rate(pairs),
        "calibration": scoring.calibration_bins(pairs),
        "by_mode": by_mode,
    }


# Mounted last so /api/* wins; html=True serves index.html at /.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
