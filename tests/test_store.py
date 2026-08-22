"""Persistence-layer tests on a temporary database."""

from pathlib import Path

import pytest

from pythia import store


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    store.init_db(path)
    return path


ARTICLE = {
    "source": "Test Wire",
    "title": "Synthetic headline",
    "url": "https://example.com/a1",
    "summary": "A synthetic summary.",
    "published_at": "2026-08-19T09:30:00+00:00",
}


def test_upsert_articles_deduplicates_on_url(db: Path) -> None:
    assert store.upsert_articles(db, [ARTICLE]) == 1
    assert store.upsert_articles(db, [ARTICLE]) == 0
    assert len(store.recent_articles(db)) == 1


def test_recent_articles_balanced_across_sources(db: Path) -> None:
    articles = []
    # Wire A floods with 10 newer articles; Wire B has 3 older ones.
    for i in range(10):
        articles.append(
            {
                "source": "Wire A",
                "title": f"A{i}",
                "url": f"https://a.example/{i}",
                "published_at": f"2026-08-20T12:{i:02d}:00+00:00",
            }
        )
    for i in range(3):
        articles.append(
            {
                "source": "Wire B",
                "title": f"B{i}",
                "url": f"https://b.example/{i}",
                "published_at": f"2026-08-19T12:{i:02d}:00+00:00",
            }
        )
    store.upsert_articles(db, articles)

    batch = store.recent_articles_balanced(db, limit=6)
    counts: dict[str, int] = {}
    for a in batch:
        counts[a["source"]] = counts.get(a["source"], 0) + 1
    assert counts == {"Wire A": 3, "Wire B": 3}
    # Within a source, newest first.
    a_titles = [a["title"] for a in batch if a["source"] == "Wire A"]
    assert a_titles == ["A9", "A8", "A7"]


def test_recent_articles_balanced_fills_from_remaining_sources(db: Path) -> None:
    articles = [
        {
            "source": "Wire A",
            "title": f"A{i}",
            "url": f"https://a.example/{i}",
            "published_at": f"2026-08-20T12:{i:02d}:00+00:00",
        }
        for i in range(8)
    ] + [
        {
            "source": "Wire B",
            "title": "B0",
            "url": "https://b.example/0",
            "published_at": "2026-08-20T13:00:00+00:00",
        }
    ]
    store.upsert_articles(db, articles)
    batch = store.recent_articles_balanced(db, limit=5)
    assert len(batch) == 5  # B exhausts after 1; A fills the rest
    assert sum(1 for a in batch if a["source"] == "Wire B") == 1


def test_scan_lifecycle(db: Path) -> None:
    scan_id = store.create_scan(db, "ollama", "llama3.1:8b")
    assert store.running_scan(db)["id"] == scan_id
    store.update_scan(db, scan_id, status="complete", detail="done", finished=True)
    assert store.running_scan(db) is None
    scan = store.get_scan(db, scan_id)
    assert scan["status"] == "complete"
    assert scan["finished_at"] is not None


def _backdate_scan(db: Path, scan_id: int, iso: str) -> None:
    with store.connect(db) as conn:
        conn.execute("UPDATE scans SET started_at = ? WHERE id = ?", (iso, scan_id))


def test_fail_orphaned_scans_unblocks_after_crash(db: Path) -> None:
    scan_id = store.create_scan(db, "ollama", "m")
    done_id = store.create_scan(db, "ollama", "m")
    store.update_scan(db, done_id, status="complete", finished=True)

    assert store.fail_orphaned_scans(db) == 1
    assert store.running_scan(db) is None
    orphaned = store.get_scan(db, scan_id)
    assert orphaned["status"] == "error"
    assert "reaped" in orphaned["error"]
    assert store.get_scan(db, done_id)["status"] == "complete"


def test_age_based_sweep_spares_fresh_runs(db: Path) -> None:
    fresh = store.create_scan(db, "ollama", "m")
    stale = store.create_scan(db, "ollama", "m")
    _backdate_scan(db, stale, "2026-08-01T00:00:00+00:00")

    assert store.fail_orphaned_scans(db, older_than_s=3600) == 1
    assert store.get_scan(db, stale)["status"] == "error"
    assert store.get_scan(db, fresh)["status"] == "running"  # a live run survives


def test_mode_and_basis_roundtrip_and_per_mode_pairs(db: Path) -> None:
    scan_id = store.create_scan(db, "ollama", "m")
    event_id = store.insert_prediction(
        db, scan_id=scan_id, statement="E", probability=0.6,
        horizon_date="2026-09-30", category="markets", rationale="",
        evidence_ids=[],
    )
    prec_id = store.insert_prediction(
        db, scan_id=scan_id, statement="P", probability=0.8,
        horizon_date="2026-09-30", category="economy", rationale="",
        evidence_ids=[], mode="precedent", basis="7 of 9 past episodes",
    )
    preds = {p["id"]: p for p in store.list_predictions(db)}
    assert preds[event_id]["mode"] == "event"
    assert preds[prec_id]["mode"] == "precedent"
    assert preds[prec_id]["basis"] == "7 of 9 past episodes"

    store.resolve_prediction(db, event_id, True)
    store.resolve_prediction(db, prec_id, False)
    assert store.resolved_pairs(db) == [(0.6, 1), (0.8, 0)]
    assert store.resolved_pairs(db, mode="precedent") == [(0.8, 0)]
    assert store.resolved_pairs(db, mode="consequence") == []


def test_init_db_migrates_pre_mode_schema(tmp_path: Path) -> None:
    old_db = tmp_path / "old.db"
    with store.connect(old_db) as conn:
        conn.execute(
            """
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY, scan_id INTEGER NOT NULL,
                statement TEXT NOT NULL, probability REAL NOT NULL,
                horizon_date TEXT NOT NULL, category TEXT NOT NULL,
                rationale TEXT NOT NULL DEFAULT '',
                evidence_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
                outcome INTEGER, resolved_at TEXT, resolution_note TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO predictions
                (scan_id, statement, probability, horizon_date, category, created_at)
            VALUES (1, 'Legacy forecast', 0.5, '2026-09-01', 'markets', '2026-08-01')
            """
        )
    store.init_db(old_db)
    preds = store.list_predictions(old_db)
    assert preds[0]["mode"] == "event"  # legacy rows default to first-order
    assert preds[0]["basis"] == ""
    assert preds[0]["proposal_outcome"] is None
    assert preds[0]["proposal_evidence_ids"] == []


def _open_prediction(db: Path, statement: str = "P", horizon: str = "2026-09-30") -> int:
    scan_id = store.create_scan(db, "ollama", "m")
    return store.insert_prediction(
        db, scan_id=scan_id, statement=statement, probability=0.5,
        horizon_date=horizon, category="markets", rationale="", evidence_ids=[],
    )


def test_proposal_accept_resolves_with_oracle_note(db: Path) -> None:
    pred_id = _open_prediction(db)
    store.upsert_articles(db, [ARTICLE])
    article_id = store.recent_articles(db)[0]["id"]

    assert store.set_proposal(db, pred_id, True, "confirmed by wire", [article_id])
    pending = store.list_predictions(db)[0]
    assert pending["proposal_outcome"] == 1
    assert pending["proposal_evidence_ids"] == [article_id]

    resolved = store.resolve_from_proposal(db, pred_id)
    assert resolved["status"] == "resolved"
    assert resolved["outcome"] == 1
    assert resolved["resolution_note"] == "[oracle] confirmed by wire"
    assert resolved["resolution_evidence_ids"] == [article_id]  # audit trail kept
    assert resolved["proposal_outcome"] is None
    # A second accept has nothing to act on.
    assert store.resolve_from_proposal(db, pred_id) is None


def test_proposal_reject_and_manual_paths_clear_proposals(db: Path) -> None:
    pred_id = _open_prediction(db)
    store.set_proposal(db, pred_id, False, "no evidence", [])
    assert store.clear_proposal(db, pred_id) is True
    assert store.clear_proposal(db, pred_id) is False  # already cleared

    # A dismissed verdict is not re-raised with the same outcome...
    assert store.set_proposal(db, pred_id, False, "again", []) is False
    # ...but the opposite verdict is a genuinely new claim.
    assert store.set_proposal(db, pred_id, True, "new evidence", []) is True

    store.resolve_prediction(db, pred_id, True)  # manual resolve wins
    assert store.list_predictions(db)[0]["proposal_outcome"] is None
    store.reopen_prediction(db, pred_id)
    reopened = store.list_predictions(db)[0]
    assert reopened["proposal_outcome"] is None
    # Reopening resets the dismissal memory: fresh eyes.
    assert store.set_proposal(db, reopened["id"], False, "fresh look", []) is True


def test_probability_revision_history(db: Path) -> None:
    pred_id = _open_prediction(db)  # p=0.5
    assert store.apply_probability_revision(db, pred_id, 0.7, "shift up", [1]) is True
    assert store.apply_probability_revision(db, pred_id, 0.4, "shift down", []) is True

    pred = store.list_predictions(db)[0]
    assert pred["probability"] == 0.4
    history = store.revisions_by_prediction(db, [pred_id])[pred_id]
    assert [(h["old_probability"], h["new_probability"]) for h in history] == [
        (0.7, 0.4),  # newest first
        (0.5, 0.7),
    ]
    assert history[1]["evidence_ids"] == [1]

    store.resolve_prediction(db, pred_id, True)
    assert store.apply_probability_revision(db, pred_id, 0.9, "too late", []) is False
    assert store.revisions_by_prediction(db, []) == {}

    # Ledger contract: raw pairs carry the final (revised) probability, and
    # resolved_records exposes the history for time-weighted scoring.
    assert store.resolved_pairs(db) == [(0.4, 1)]
    records = store.resolved_records(db)
    assert len(records) == 1
    assert records[0]["probability"] == 0.4
    assert len(records[0]["revisions"]) == 2
    assert records[0]["created_at"] and records[0]["resolved_at"]


def test_get_prediction_single_row(db: Path) -> None:
    pred_id = _open_prediction(db, "Solo lookup")
    pred = store.get_prediction(db, pred_id)
    assert pred is not None
    assert pred["statement"] == "Solo lookup"
    assert pred["evidence_ids"] == []
    assert store.get_prediction(db, 9999) is None


def test_create_scan_exclusive_gate(db: Path) -> None:
    first = store.create_scan_exclusive(db, "ollama", "m")
    assert first is not None
    assert store.create_scan_exclusive(db, "ollama", "m") is None  # gate held
    store.update_scan(db, first, status="complete", finished=True)
    second = store.create_scan_exclusive(db, "ollama", "m", kind="resolution")
    assert second is not None
    assert store.get_scan(db, second)["kind"] == "resolution"


def test_scan_articles_digest_roundtrip(db: Path) -> None:
    scan_id = store.create_scan(db, "ollama", "m")
    articles = [
        {
            "source": "Wire",
            "title": f"Headline {i}",
            "url": f"https://example.com/{i}",
            "published_at": f"2026-08-2{i}T00:00:00+00:00",
        }
        for i in range(3)
    ]
    store.upsert_articles(db, articles)
    ids = [a["id"] for a in store.recent_articles(db)]

    batch_order = [ids[2], ids[0]]  # deliberate non-chronological batch order
    store.record_scan_articles(db, scan_id, batch_order)
    store.record_scan_articles(db, scan_id, batch_order)  # idempotent

    digest = store.articles_for_scan(db, scan_id)
    assert [a["id"] for a in digest] == batch_order  # batch order preserved
    assert store.articles_for_scan(db, 999) == []


def test_search_articles_matches_titles_since_date(db: Path) -> None:
    store.upsert_articles(
        db,
        [
            {
                "source": "Wire", "title": "OPEC agrees new quota cut",
                "url": "https://x.example/opec", "published_at": "2026-08-10T00:00:00+00:00",
            },
            {
                "source": "Wire", "title": "Unrelated sports story",
                "url": "https://x.example/sports", "published_at": "2026-08-11T00:00:00+00:00",
            },
            {
                "source": "Wire", "title": "OPEC ministers meet, too old",
                "url": "https://x.example/old", "published_at": "2026-07-01T00:00:00+00:00",
            },
        ],
    )
    hits = store.search_articles(db, ["OPEC", "quota"], since_iso="2026-08-01")
    assert [h["url"] for h in hits] == ["https://x.example/opec"]
    assert store.search_articles(db, []) == []
    # LIKE wildcards in a statement must not act as wildcards.
    assert store.search_articles(db, ["100%"]) == []


def test_search_articles_matches_summaries_too(db: Path) -> None:
    store.upsert_articles(
        db,
        [
            {
                "source": "Wire",
                "title": "Ministers meet in Vienna",
                "url": "https://x.example/summary-hit",
                "summary": "Discussion focused on raising production quotas next month.",
                "published_at": "2026-08-15T00:00:00+00:00",
            }
        ],
    )
    hits = store.search_articles(db, ["quota"])
    assert [h["url"] for h in hits] == ["https://x.example/summary-hit"]


def test_set_proposal_refuses_resolved_predictions(db: Path) -> None:
    pred_id = _open_prediction(db)
    store.resolve_prediction(db, pred_id, True)
    assert store.set_proposal(db, pred_id, False, "late", []) is False


def test_open_predictions_for_check_orders_due_first(db: Path) -> None:
    later = _open_prediction(db, "future far", "2026-12-01")
    due = _open_prediction(db, "overdue", "2026-08-01")
    soon = _open_prediction(db, "future near", "2026-09-01")
    resolved = _open_prediction(db, "already done", "2026-07-01")
    store.resolve_prediction(db, resolved, True)

    queue = store.open_predictions_for_check(db, today_iso="2026-08-22")
    assert [q["id"] for q in queue] == [due, soon, later]


def test_migrate_legacy_db_copies_once(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.db"
    target = tmp_path / "new" / "pythia.db"
    store.init_db(legacy)
    scan_id = store.create_scan(legacy, "ollama", "m")

    assert store.migrate_legacy_db(legacy, target) is True
    assert store.get_scan(target, scan_id)["provider"] == "ollama"
    assert legacy.exists()  # original untouched
    # Second call is a no-op: target already exists.
    assert store.migrate_legacy_db(legacy, target) is False
    # Missing legacy is also a no-op.
    assert store.migrate_legacy_db(tmp_path / "nope.db", tmp_path / "x.db") is False


def test_prediction_resolve_and_reopen_cycle(db: Path) -> None:
    scan_id = store.create_scan(db, "ollama", "m")
    store.upsert_articles(db, [ARTICLE])
    article_id = store.recent_articles(db)[0]["id"]
    pred_id = store.insert_prediction(
        db,
        scan_id=scan_id,
        statement="Synthetic event happens",
        probability=0.7,
        horizon_date="2026-09-30",
        category="markets",
        rationale="test",
        evidence_ids=[article_id],
    )

    open_preds = store.list_predictions(db, "open")
    assert [p["id"] for p in open_preds] == [pred_id]
    assert open_preds[0]["evidence_ids"] == [article_id]
    assert store.open_statements(db) == ["Synthetic event happens"]

    resolved = store.resolve_prediction(db, pred_id, True, "confirmed by wire")
    assert resolved["status"] == "resolved"
    assert resolved["outcome"] == 1
    assert store.resolved_pairs(db) == [(0.7, 1)]
    assert store.list_predictions(db, "open") == []

    reopened = store.reopen_prediction(db, pred_id)
    assert reopened["status"] == "open"
    assert reopened["outcome"] is None
    assert store.resolved_pairs(db) == []


def test_resolve_missing_prediction_returns_none(db: Path) -> None:
    assert store.resolve_prediction(db, 999, True) is None
    assert store.reopen_prediction(db, 999) is None
