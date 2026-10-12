"""SQLite persistence for articles, scans, and predictions.

Every function opens its own short-lived connection, so callers may run from
any thread (the scan pipeline runs in a background thread).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id           INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL UNIQUE,
    summary      TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    origin       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles (published_at DESC);

CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'forecast',
    status        TEXT NOT NULL DEFAULT 'running',
    detail        TEXT NOT NULL DEFAULT '',
    article_count INTEGER NOT NULL DEFAULT 0,
    signals       TEXT NOT NULL DEFAULT '',
    error         TEXT
);

CREATE TABLE IF NOT EXISTS probability_revisions (
    id              INTEGER PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES predictions (id),
    old_probability REAL NOT NULL,
    new_probability REAL NOT NULL,
    note            TEXT NOT NULL DEFAULT '',
    evidence_ids    TEXT NOT NULL DEFAULT '[]',
    source          TEXT NOT NULL DEFAULT 'oracle',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revisions_prediction
    ON probability_revisions (prediction_id);

CREATE TABLE IF NOT EXISTS scan_articles (
    scan_id    INTEGER NOT NULL REFERENCES scans (id),
    article_id INTEGER NOT NULL REFERENCES articles (id),
    position   INTEGER NOT NULL,
    PRIMARY KEY (scan_id, article_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY,
    scan_id         INTEGER NOT NULL REFERENCES scans (id),
    statement       TEXT NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'event',
    region          TEXT NOT NULL DEFAULT 'global',
    probability     REAL NOT NULL,
    horizon_date    TEXT NOT NULL,
    category        TEXT NOT NULL,
    rationale       TEXT NOT NULL DEFAULT '',
    basis           TEXT NOT NULL DEFAULT '',
    evidence_ids    TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    outcome         INTEGER,
    resolved_at     TEXT,
    resolution_note TEXT,
    resolution_evidence_ids TEXT NOT NULL DEFAULT '[]',
    proposal_outcome      INTEGER,
    proposal_note         TEXT,
    proposal_evidence_ids TEXT NOT NULL DEFAULT '[]',
    proposal_at           TEXT,
    proposal_dismissed_outcome INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions (status);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode and dict-like rows."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        _ensure_prediction_columns(conn)


def _ensure_prediction_columns(conn: sqlite3.Connection) -> None:
    """Additive migration for databases created before newer fields.

    Pre-existing predictions were all first-order forecasts, so 'event' is the
    semantically correct default for them.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    additions = {
        "mode": "TEXT NOT NULL DEFAULT 'event'",
        "basis": "TEXT NOT NULL DEFAULT ''",
        "region": "TEXT NOT NULL DEFAULT 'global'",
        "resolution_evidence_ids": "TEXT NOT NULL DEFAULT '[]'",
        "proposal_outcome": "INTEGER",
        "proposal_note": "TEXT",
        "proposal_evidence_ids": "TEXT NOT NULL DEFAULT '[]'",
        "proposal_at": "TEXT",
        "proposal_dismissed_outcome": "INTEGER",
    }
    for name, decl in additions.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {name} {decl}")
    scan_cols = {row["name"] for row in conn.execute("PRAGMA table_info(scans)")}
    if "kind" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN kind TEXT NOT NULL DEFAULT 'forecast'")
    if "signals" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN signals TEXT NOT NULL DEFAULT ''")
    article_cols = {row["name"] for row in conn.execute("PRAGMA table_info(articles)")}
    if "origin" not in article_cols:
        conn.execute("ALTER TABLE articles ADD COLUMN origin TEXT NOT NULL DEFAULT ''")


def migrate_legacy_db(legacy_path: Path, target_path: Path) -> bool:
    """Copy a legacy database to the new location, once, WAL-safely.

    Uses SQLite's backup API so an in-flight WAL on the legacy file is
    captured consistently, staged through a temp file so a failed copy never
    leaves a stub target that would block future migration. The legacy file
    is left untouched. Returns True when a copy was performed.
    """
    if target_path.exists() or not legacy_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    staging = target_path.with_suffix(".migrating")
    try:
        src = sqlite3.connect(legacy_path)
        try:
            dst = sqlite3.connect(staging)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        staging.replace(target_path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return True


# --- articles ---------------------------------------------------------------


def upsert_articles(db_path: Path, articles: list[dict[str, Any]]) -> int:
    """Insert articles; refresh a stored URL when it was republished newer.

    Government advisory feeds update entries in place under a stable URL —
    an escalation must replace the stale text and re-enter the recency pool.
    Returns rows inserted or refreshed.
    """
    now = _now_iso()
    changed = 0
    with connect(db_path) as conn:
        for a in articles:
            cur = conn.execute(
                """
                INSERT INTO articles
                    (source, title, url, summary, published_at, fetched_at, origin)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url) DO UPDATE SET
                    source = excluded.source,
                    title = excluded.title,
                    summary = excluded.summary,
                    published_at = excluded.published_at,
                    fetched_at = excluded.fetched_at,
                    origin = excluded.origin
                WHERE excluded.published_at > articles.published_at
                  -- A GDELT re-sighting never downgrades a curated feed's row
                  -- (its richer summary and provenance would be wiped).
                  AND NOT (excluded.origin LIKE 'gdelt:%'
                           AND articles.origin NOT LIKE 'gdelt:%')
                """,
                (
                    a["source"],
                    a["title"],
                    a["url"],
                    a.get("summary", ""),
                    a["published_at"],
                    now,
                    a.get("origin", ""),
                ),
            )
            changed += cur.rowcount
    return changed


def recent_articles(db_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, source, title, url, summary, published_at, fetched_at
            FROM articles
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_articles_balanced(
    db_path: Path, limit: int, per_origin: int = 4, max_age_days: int = 14
) -> list[dict[str, Any]]:
    """Recent articles, round-robin balanced across originating feeds.

    The pool is the newest ``per_origin`` articles per FEED (window function),
    not a global newest-N slice — so a weekly-publishing analysis source is
    always present alongside high-volume wires, and a query feed's many
    publishers still count as one feed. Rotation takes one article per feed
    (feeds ordered by their newest article) until ``limit``. Rows older than
    ``max_age_days`` are excluded so stale items cannot resurface forever;
    legacy rows without an origin fall back to their source label.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=max_age_days)
    ).isoformat(timespec="seconds")
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT id, source, title, url, summary, published_at, origin,
                       ROW_NUMBER() OVER (
                           PARTITION BY CASE WHEN origin != '' THEN origin ELSE source END
                           ORDER BY published_at DESC
                       ) AS rn
                FROM articles
                WHERE published_at >= ?
            )
            SELECT id, source, title, url, summary, published_at, origin
            FROM ranked
            WHERE rn <= ?
            ORDER BY published_at DESC
            """,
            (cutoff, per_origin),
        ).fetchall()
    pool = [dict(r) for r in rows]
    by_feed: dict[str, list[dict[str, Any]]] = {}
    for article in pool:  # pool is newest-first; insertion order keeps recency
        key = article["origin"] or article["source"]
        by_feed.setdefault(key, []).append(article)
    queues = list(by_feed.values())
    batch: list[dict[str, Any]] = []
    while len(batch) < limit and queues:
        queues = [q for q in queues if q]
        for q in queues:
            if len(batch) >= limit:
                break
            batch.append(q.pop(0))
    return batch


def record_scan_articles(db_path: Path, scan_id: int, article_ids: list[int]) -> None:
    """Record the exact article batch a scan fed to the oracle (the digest)."""
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO scan_articles (scan_id, article_id, position)
            VALUES (?, ?, ?)
            ON CONFLICT (scan_id, article_id) DO NOTHING
            """,
            [(scan_id, aid, pos) for pos, aid in enumerate(article_ids)],
        )


def articles_for_scan(db_path: Path, scan_id: int) -> list[dict[str, Any]]:
    """The digest: articles a scan actually sent to the oracle, in batch order."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.source, a.title, a.url, a.summary, a.published_at
            FROM scan_articles sa
            JOIN articles a ON a.id = sa.article_id
            WHERE sa.scan_id = ?
            ORDER BY sa.position
            """,
            (scan_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def articles_by_ids(db_path: Path, ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, source, title, url, summary, published_at
            FROM articles
            WHERE id IN ({placeholders})
            """,
            ids,
        ).fetchall()
    return [dict(r) for r in rows]


# --- scans ------------------------------------------------------------------


def create_scan_exclusive(
    db_path: Path, provider: str, model: str, kind: str = "forecast"
) -> int | None:
    """Create a scan only if no run is in progress — atomic across processes.

    A single INSERT...WHERE NOT EXISTS statement, serialized by SQLite's
    writer lock, so the CLI and the server cannot both slip past the gate.
    Returns the new scan id, or None when another run holds the gate.
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scans (started_at, provider, model, kind, status, detail)
            SELECT ?, ?, ?, ?, 'running', 'Starting scan'
            WHERE NOT EXISTS (SELECT 1 FROM scans WHERE status = 'running')
            """,
            (_now_iso(), provider, model, kind),
        )
        if cur.rowcount == 0:
            return None
        scan_id = cur.lastrowid
    assert scan_id is not None
    return scan_id


def create_scan(db_path: Path, provider: str, model: str, kind: str = "forecast") -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scans (started_at, provider, model, kind, status, detail)
            VALUES (?, ?, ?, ?, 'running', 'Starting scan')
            """,
            (_now_iso(), provider, model, kind),
        )
    scan_id = cur.lastrowid
    assert scan_id is not None
    return scan_id


def update_scan(
    db_path: Path,
    scan_id: int,
    *,
    status: str | None = None,
    detail: str | None = None,
    article_count: int | None = None,
    signals: str | None = None,
    error: str | None = None,
    finished: bool = False,
) -> None:
    sets: list[str] = []
    args: list[Any] = []
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if detail is not None:
        sets.append("detail = ?")
        args.append(detail)
    if article_count is not None:
        sets.append("article_count = ?")
        args.append(article_count)
    if signals is not None:
        sets.append("signals = ?")
        args.append(signals)
    if error is not None:
        sets.append("error = ?")
        args.append(error)
    if finished:
        sets.append("finished_at = ?")
        args.append(_now_iso())
    if not sets:
        return
    args.append(scan_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE scans SET {', '.join(sets)} WHERE id = ?", args)


def get_scan(db_path: Path, scan_id: int) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, started_at, finished_at, provider, model, kind, status,
                   detail, article_count, signals, error
            FROM scans
            WHERE id = ?
            """,
            (scan_id,),
        ).fetchone()
    return dict(row) if row else None


def list_scans(db_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, finished_at, provider, model, kind, status,
                   detail, article_count, signals, error
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def fail_orphaned_scans(db_path: Path, older_than_s: float | None = None) -> int:
    """Mark 'running' scans as errored so they stop blocking new runs.

    With ``older_than_s`` set, only scans older than that are reaped — the
    safe form when another process (server vs CLI) may legitimately have a
    run in flight. With None, every running scan is reaped.
    """
    query = """
        UPDATE scans
        SET status = 'error', detail = 'Scan failed',
            error = 'interrupted or stale; reaped by startup sweep', finished_at = ?
        WHERE status = 'running'
    """
    args: list[Any] = [_now_iso()]
    if older_than_s is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=older_than_s)
        query += " AND started_at < ?"
        args.append(cutoff.isoformat(timespec="seconds"))
    with connect(db_path) as conn:
        cur = conn.execute(query, args)
    return cur.rowcount


def running_scan(db_path: Path) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, started_at, provider, model, kind, status, detail
            FROM scans
            WHERE status = 'running'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    return dict(row) if row else None


# --- predictions ------------------------------------------------------------


def insert_prediction(
    db_path: Path,
    *,
    scan_id: int,
    statement: str,
    probability: float,
    horizon_date: str,
    category: str,
    rationale: str,
    evidence_ids: list[int],
    mode: str = "event",
    basis: str = "",
    region: str = "global",
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (scan_id, statement, mode, region, probability, horizon_date,
                 category, rationale, basis, evidence_ids, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                statement,
                mode,
                region,
                probability,
                horizon_date,
                category,
                rationale,
                basis,
                json.dumps(evidence_ids),
                _now_iso(),
            ),
        )
    pred_id = cur.lastrowid
    assert pred_id is not None
    return pred_id


def list_predictions(db_path: Path, status: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT id, scan_id, statement, mode, region, probability, horizon_date,
               category, rationale, basis, evidence_ids, created_at, status, outcome,
               resolved_at, resolution_note, resolution_evidence_ids,
               proposal_outcome, proposal_note, proposal_evidence_ids, proposal_at
        FROM predictions
    """
    args: list[Any] = []
    if status in ("open", "resolved", "void"):
        query += " WHERE status = ?"
        args.append(status)
    query += " ORDER BY id DESC"
    with connect(db_path) as conn:
        rows = conn.execute(query, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["evidence_ids"] = json.loads(d["evidence_ids"])
        d["proposal_evidence_ids"] = json.loads(d["proposal_evidence_ids"] or "[]")
        d["resolution_evidence_ids"] = json.loads(d["resolution_evidence_ids"] or "[]")
        out.append(d)
    return out


def open_statements(
    db_path: Path, limit: int = 40, include_void: bool = False
) -> list[str]:
    """Statements of open forecasts, newest first.

    The default cap suits the prompt (avoid duplicate suggestions); pass a
    larger limit for duplicate ENFORCEMENT, which must see every open claim.
    ``include_void`` also returns voided statements: voiding one says "this
    cannot be scored", so re-proposing it tomorrow would simply hand the user
    the same unscoreable claim again. Reopening a void clears that block.
    """
    statuses = ("open", "void") if include_void else ("open",)
    placeholders = ",".join("?" for _ in statuses)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT statement FROM predictions WHERE status IN ({placeholders}) "
            "ORDER BY id DESC LIMIT ?",
            (*statuses, limit),
        ).fetchall()
    return [r["statement"] for r in rows]


def delete_open_predictions(db_path: Path, pred_ids: list[int]) -> int:
    """Delete OPEN predictions by id, with their revision history.

    Resolved forecasts are immune: the resolution ledger is append-only, so a
    scored forecast can never be quietly removed from the record.
    """
    if not pred_ids:
        return 0
    placeholders = ",".join("?" for _ in pred_ids)
    with connect(db_path) as conn:
        conn.execute(
            f"""
            DELETE FROM probability_revisions
            WHERE prediction_id IN (
                SELECT id FROM predictions
                WHERE id IN ({placeholders}) AND status = 'open'
            )
            """,
            pred_ids,
        )
        cur = conn.execute(
            f"DELETE FROM predictions WHERE id IN ({placeholders}) AND status = 'open'",
            pred_ids,
        )
    return cur.rowcount


def void_prediction(db_path: Path, pred_id: int, reason: str) -> dict[str, Any] | None:
    """Retire an open forecast without scoring it.

    For statements too vague to settle honestly ("a major European city",
    "increased diplomatic pressure"). Guessing an outcome would put noise
    into the calibration record; deleting it would hide that the system
    produced something unscoreable. A void keeps the forecast and its
    reason in the ledger, and every scoring query ignores it because they
    all select status = 'resolved'.
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE predictions
            SET status = 'void', outcome = NULL, resolved_at = ?,
                resolution_note = ?, resolution_evidence_ids = '[]',
                proposal_outcome = NULL, proposal_note = NULL,
                proposal_evidence_ids = '[]', proposal_at = NULL
            WHERE id = ? AND status = 'open'
            """,
            (_now_iso(), reason, pred_id),
        )
        if cur.rowcount == 0:
            return None
    return get_prediction(db_path, pred_id)


def resolve_prediction(
    db_path: Path, pred_id: int, outcome: bool, note: str = ""
) -> dict[str, Any] | None:
    """Score an outcome against a forecast.

    Re-resolving a resolved forecast is allowed (correcting a mis-click), but
    a VOID one is refused: it was retired as unscoreable, and scoring it has
    to go through an explicit reopen so the intent is on the record.
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE predictions
            SET status = 'resolved', outcome = ?, resolved_at = ?, resolution_note = ?,
                proposal_outcome = NULL, proposal_note = NULL,
                proposal_evidence_ids = '[]', proposal_at = NULL
            WHERE id = ? AND status != 'void'
            """,
            (1 if outcome else 0, _now_iso(), note, pred_id),
        )
        if cur.rowcount == 0:
            return None
    preds = [p for p in list_predictions(db_path) if p["id"] == pred_id]
    return preds[0] if preds else None


def reopen_prediction(db_path: Path, pred_id: int) -> dict[str, Any] | None:
    """Return a resolved OR voided forecast to the open ledger."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE predictions
            SET status = 'open', outcome = NULL, resolved_at = NULL, resolution_note = NULL,
                resolution_evidence_ids = '[]',
                proposal_outcome = NULL, proposal_note = NULL,
                proposal_evidence_ids = '[]', proposal_at = NULL,
                proposal_dismissed_outcome = NULL
            WHERE id = ?
            """,
            (pred_id,),
        )
        if cur.rowcount == 0:
            return None
    preds = [p for p in list_predictions(db_path) if p["id"] == pred_id]
    return preds[0] if preds else None


def open_predictions_for_check(
    db_path: Path, today_iso: str, limit: int = 20, ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """Open predictions to offer the resolution oracle: due first, then soonest.

    ``ids`` restricts the queue to specific predictions (targeted check).
    """
    query = """
        SELECT id, statement, mode, probability, horizon_date, category,
               created_at, rationale, basis
        FROM predictions
        WHERE status = 'open'
    """
    args: list[Any] = []
    if ids:
        query += f" AND id IN ({','.join('?' for _ in ids)})"
        args.extend(ids)
    query += " ORDER BY CASE WHEN horizon_date < ? THEN 0 ELSE 1 END, horizon_date ASC LIMIT ?"
    args.extend([today_iso, limit])
    with connect(db_path) as conn:
        rows = conn.execute(query, args).fetchall()
    return [dict(r) for r in rows]


def apply_probability_revision(
    db_path: Path,
    pred_id: int,
    new_probability: float,
    note: str,
    evidence_ids: list[int],
    source: str = "oracle",
) -> bool:
    """Revise an open forecast's probability, keeping the full audit history.

    One transaction, with the UPDATE itself as the guard: if the prediction
    was resolved (or its probability changed) between read and write, nothing
    is written and False is returned.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT probability FROM predictions WHERE id = ? AND status = 'open'",
            (pred_id,),
        ).fetchone()
        if row is None:
            return False
        cur = conn.execute(
            """
            UPDATE predictions SET probability = ?
            WHERE id = ? AND status = 'open' AND probability = ?
            """,
            (new_probability, pred_id, row["probability"]),
        )
        if cur.rowcount == 0:
            return False
        conn.execute(
            """
            INSERT INTO probability_revisions
                (prediction_id, old_probability, new_probability, note,
                 evidence_ids, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred_id,
                row["probability"],
                new_probability,
                note,
                json.dumps(evidence_ids),
                source,
                _now_iso(),
            ),
        )
    return True


def revisions_by_prediction(
    db_path: Path, pred_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    """Revision histories keyed by prediction id, newest first."""
    if not pred_ids:
        return {}
    placeholders = ",".join("?" for _ in pred_ids)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT prediction_id, old_probability, new_probability, note,
                   evidence_ids, source, created_at
            FROM probability_revisions
            WHERE prediction_id IN ({placeholders})
            ORDER BY id DESC
            """,
            pred_ids,
        ).fetchall()
    out: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        d = dict(r)
        d["evidence_ids"] = json.loads(d["evidence_ids"])
        out.setdefault(d.pop("prediction_id"), []).append(d)
    return out


def search_articles(
    db_path: Path, terms: list[str], limit: int = 3, since_iso: str | None = None
) -> list[dict[str, Any]]:
    """Newest archive articles whose title or summary matches any term.

    Lets the resolution judge see coverage that has aged out of the recent
    batch — the archive keeps everything ever fetched.
    """
    cleaned = [t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for t in terms if t]
    if not cleaned:
        return []
    clause = " OR ".join(
        "(title LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\')" for _ in cleaned
    )
    args: list[Any] = []
    for t in cleaned:
        args.extend([f"%{t}%", f"%{t}%"])
    query = f"""
        SELECT id, source, title, url, summary, published_at
        FROM articles
        WHERE ({clause})
    """
    if since_iso:
        query += " AND published_at >= ?"
        args.append(since_iso)
    query += " ORDER BY published_at DESC LIMIT ?"
    args.append(limit)
    with connect(db_path) as conn:
        rows = conn.execute(query, args).fetchall()
    return [dict(r) for r in rows]


def set_proposal(
    db_path: Path, pred_id: int, outcome: bool, note: str, evidence_ids: list[int]
) -> bool:
    """Attach an oracle-proposed resolution to an open prediction.

    A proposal the user already dismissed is not re-raised with the same
    outcome; the oracle must change its verdict (or the user reopen) first.
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE predictions
            SET proposal_outcome = ?, proposal_note = ?,
                proposal_evidence_ids = ?, proposal_at = ?
            WHERE id = ? AND status = 'open'
              AND (proposal_dismissed_outcome IS NULL OR proposal_dismissed_outcome != ?)
            """,
            (
                1 if outcome else 0,
                note,
                json.dumps(evidence_ids),
                _now_iso(),
                pred_id,
                1 if outcome else 0,
            ),
        )
    return cur.rowcount > 0


def clear_proposal(db_path: Path, pred_id: int) -> bool:
    """Dismiss a pending proposal, remembering the dismissed outcome."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE predictions
            SET proposal_dismissed_outcome = proposal_outcome,
                proposal_outcome = NULL, proposal_note = NULL,
                proposal_evidence_ids = '[]', proposal_at = NULL
            WHERE id = ? AND proposal_outcome IS NOT NULL
            """,
            (pred_id,),
        )
    return cur.rowcount > 0


def resolve_from_proposal(db_path: Path, pred_id: int) -> dict[str, Any] | None:
    """Accept a pending proposal: one guarded UPDATE, so a concurrent manual
    resolve or reopen can never be overwritten. Evidence citations are kept
    on the resolved row for later audit."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE predictions
            SET status = 'resolved', outcome = proposal_outcome, resolved_at = ?,
                resolution_note = RTRIM('[oracle] ' || COALESCE(proposal_note, '')),
                resolution_evidence_ids = proposal_evidence_ids,
                proposal_outcome = NULL, proposal_note = NULL,
                proposal_evidence_ids = '[]', proposal_at = NULL
            WHERE id = ? AND status = 'open' AND proposal_outcome IS NOT NULL
            """,
            (_now_iso(), pred_id),
        )
        if cur.rowcount == 0:
            return None
    preds = [p for p in list_predictions(db_path) if p["id"] == pred_id]
    return preds[0] if preds else None


def resolved_records(db_path: Path, mode: str | None = None) -> list[dict[str, Any]]:
    """Resolved predictions with their full probability history, for scoring.

    Each record carries the final probability, outcome, open lifetime bounds,
    and the revision list (newest first) so scoring can reconstruct the
    credence trajectory rather than trusting only the last number.
    """
    query = """
        SELECT id, probability, outcome, created_at, resolved_at
        FROM predictions
        WHERE status = 'resolved'
    """
    args: list[Any] = []
    if mode is not None:
        query += " AND mode = ?"
        args.append(mode)
    with connect(db_path) as conn:
        rows = conn.execute(query, args).fetchall()
    records = [dict(r) for r in rows]
    revisions = revisions_by_prediction(db_path, [r["id"] for r in records])
    for r in records:
        r["revisions"] = revisions.get(r["id"], [])
    return records


def get_prediction(db_path: Path, pred_id: int) -> dict[str, Any] | None:
    """One prediction row by id, JSON fields decoded."""
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, scan_id, statement, mode, probability, horizon_date, category,
                   rationale, basis, evidence_ids, created_at, status, outcome,
                   resolved_at, resolution_note, resolution_evidence_ids,
                   proposal_outcome, proposal_note, proposal_evidence_ids, proposal_at
            FROM predictions
            WHERE id = ?
            """,
            (pred_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["evidence_ids"] = json.loads(d["evidence_ids"])
    d["proposal_evidence_ids"] = json.loads(d["proposal_evidence_ids"] or "[]")
    d["resolution_evidence_ids"] = json.loads(d["resolution_evidence_ids"] or "[]")
    return d


def resolved_pairs(db_path: Path, mode: str | None = None) -> list[tuple[float, int]]:
    """(probability, outcome) pairs for resolved predictions, optionally per mode."""
    query = "SELECT probability, outcome FROM predictions WHERE status = 'resolved'"
    args: list[Any] = []
    if mode is not None:
        query += " AND mode = ?"
        args.append(mode)
    with connect(db_path) as conn:
        rows = conn.execute(query, args).fetchall()
    return [(r["probability"], r["outcome"]) for r in rows]
