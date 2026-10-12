"""API tests with a fake oracle and no network."""

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pythia import pipeline, store
from pythia.config import Settings
from pythia.oracle.base import ForecastDraft, OracleError

def _recent(hours_ago: float = 2) -> str:
    """An article timestamp inside the batcher's freshness window.

    Hardcoded dates rot: recent_articles_balanced ignores anything older than
    two weeks, so fixtures pinned to a literal date silently stop being
    batched once that date ages out.
    """
    return (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).isoformat(timespec="seconds")



class FakeOracle:
    provider = "fake"
    model = "fake-1"

    def generate(
        self,
        articles: list[dict[str, Any]],
        today: date,
        open_statements: list[str],
        track_record: str | None = None,
        signals: str | None = None,
    ) -> list[ForecastDraft]:
        # Relative horizon so assertions never rot as the wall clock advances.
        horizon = (date.today() + timedelta(days=30)).isoformat()
        return [
            ForecastDraft(
                statement="Synthetic index closes higher this month",
                probability=0.6,
                horizon_date=horizon,
                category="markets",
                rationale="synthetic",
                evidence_indices=[0],
            ),
            ForecastDraft(
                statement="Funding costs rise after the reported stress",
                probability=0.4,
                horizon_date=horizon,
                category="economy",
                rationale="synthetic",
                evidence_indices=[0],
                mode="consequence",
                basis="stress -> spreads widen -> funding costs rise",
            ),
            ForecastDraft(
                statement="The historical pattern repeats",
                probability=0.7,
                horizon_date=horizon,
                category="geopolitics",
                rationale="synthetic",
                evidence_indices=[],
                mode="precedent",
                basis="7 of 9 comparable episodes since 2000",
                region="mena",
            ),
        ]


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("PYTHIA_DB", str(tmp_path / "api.db"))
    monkeypatch.setenv("PYTHIA_GDELT", "0")  # tests never touch the network
    from pythia.config import get_settings

    s = get_settings()
    store.init_db(s.db_path)
    return s


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    from pythia.server import app

    return TestClient(app)


def _seed_scan(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real pipeline with fake feeds + fake oracle."""
    articles = [
        {
            "source": "Wire",
            "title": "Synthetic headline",
            "url": "https://example.com/x",
            "summary": "s",
            "published_at": _recent(),
        }
    ]
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (articles, {}))
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=FakeOracle(), today=date(2026, 8, 21))
    scan = store.get_scan(settings.db_path, scan_id)
    assert scan["status"] == "complete", scan


def test_health(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from pythia import server

    monkeypatch.setattr(server, "ollama_available", lambda url: False)
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == {"available": True, "provider": "fake", "model": "fake-1"}
    assert body["ollama_reachable"] is False
    assert "geopolitics" in body["categories"]
    assert body["modes"] == ["event", "consequence", "precedent"]
    assert body["regions"][0] == "global" and "mena" in body["regions"]


def test_health_reports_unavailable_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pythia import server

    monkeypatch.setattr(server, "ollama_available", lambda url: False)

    def _raise(settings: Settings) -> dict[str, Any]:
        raise OracleError("no backend")

    monkeypatch.setattr(pipeline, "scan_metadata", _raise)
    body = client.get("/api/health").json()
    assert body["provider"]["available"] is False
    assert "no backend" in body["provider"]["error"]


def test_scan_pipeline_to_predictions(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = client.get("/api/predictions").json()
    assert len(preds) == 3
    by_mode = {p["mode"]: p for p in preds}
    assert set(by_mode) == {"event", "consequence", "precedent"}
    assert by_mode["event"]["evidence"][0]["title"] == "Synthetic headline"
    assert by_mode["consequence"]["basis"].startswith("stress ->")
    assert by_mode["precedent"]["basis"] == "7 of 9 comparable episodes since 2000"
    assert by_mode["precedent"]["region"] == "mena"
    assert by_mode["event"]["region"] == "global"
    assert all(p["status"] == "open" and p["due"] is False for p in preds)


def test_resolve_flow_updates_stats(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = client.get("/api/predictions").json()
    event_id = next(p["id"] for p in preds if p["mode"] == "event")

    resp = client.post(f"/api/predictions/{event_id}/resolve", json={"outcome": True})
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"

    stats = client.get("/api/stats").json()
    assert stats["resolved"] == 1
    assert stats["open"] == 2
    assert stats["brier"] == pytest.approx((0.6 - 1) ** 2)
    assert stats["hit_rate"] == 1.0
    assert len(stats["calibration"]) == 1
    assert stats["by_mode"]["event"]["resolved"] == 1
    assert stats["by_mode"]["event"]["brier"] == pytest.approx((0.6 - 1) ** 2)
    assert stats["by_mode"]["consequence"] == {"open": 1, "resolved": 0, "brier": None}
    assert stats["by_mode"]["precedent"] == {"open": 1, "resolved": 0, "brier": None}

    resp = client.post(f"/api/predictions/{event_id}/reopen")
    assert resp.status_code == 200
    assert client.get("/api/stats").json()["open"] == 3


def test_void_retires_without_scoring(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = client.get("/api/predictions").json()
    target = next(p for p in preds if p["mode"] == "event")

    resp = client.post(
        f"/api/predictions/{target['id']}/void",
        json={"reason": '"a major European city" cannot be resolved'},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "void"
    assert body["outcome"] is None
    assert body["resolution_note"] == '"a major European city" cannot be resolved'

    stats = client.get("/api/stats").json()
    assert stats["void"] == 1
    assert stats["open"] == 2
    # The whole point: a void never reaches the score.
    assert stats["resolved"] == 0
    assert stats["brier"] is None
    assert stats["by_mode"]["event"]["resolved"] == 0

    assert [p["id"] for p in client.get("/api/predictions?status=void").json()] == [
        target["id"]
    ]
    # Voiding is not resolving: it must not be scoreable afterwards either.
    assert client.post(
        f"/api/predictions/{target['id']}/resolve", json={"outcome": True}
    ).status_code == 404
    # But it can be reopened if the statement is salvageable.
    assert client.post(f"/api/predictions/{target['id']}/reopen").status_code == 200
    assert client.get("/api/stats").json()["open"] == 3


def test_void_requires_a_reason_and_an_open_forecast(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    pid = client.get("/api/predictions").json()[0]["id"]
    assert client.post(f"/api/predictions/{pid}/void", json={"reason": ""}).status_code == 422
    assert client.post("/api/predictions/99999/void", json={"reason": "x"}).status_code == 404
    client.post(f"/api/predictions/{pid}/resolve", json={"outcome": True})
    # A scored forecast is history; it cannot be quietly voided out of the record.
    assert client.post(
        f"/api/predictions/{pid}/void", json={"reason": "second thoughts"}
    ).status_code == 404


def test_voided_statement_is_not_proposed_again(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = store.list_predictions(settings.db_path, "open")
    target = next(p for p in preds if p["mode"] == "event")
    store.void_prediction(settings.db_path, target["id"], "too vague")

    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (
        [{"source": "Wire", "title": "Synthetic headline",
          "url": "https://example.com/x", "summary": "s",
          "published_at": _recent()}], {},
    ))
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=FakeOracle())
    # FakeOracle re-offers all three statements: two collide with still-open
    # forecasts, and the third with the voided one.
    assert "(3 duplicates refused)" in store.get_scan(settings.db_path, scan_id)["detail"]
    statements = [p["statement"] for p in store.list_predictions(settings.db_path, "open")]
    assert target["statement"] not in statements


def test_resolve_missing_prediction_404(client: TestClient) -> None:
    resp = client.post("/api/predictions/12345/resolve", json={"outcome": False})
    assert resp.status_code == 404


def test_due_flag_for_past_horizon(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    store.insert_prediction(
        settings.db_path,
        scan_id=scan_id,
        statement="Already past its horizon",
        probability=0.5,
        horizon_date=(date.today() - timedelta(days=2)).isoformat(),
        category="economy",
        rationale="",
        evidence_ids=[],
    )
    preds = {p["statement"]: p for p in client.get("/api/predictions").json()}
    assert preds["Already past its horizon"]["due"] is True
    assert preds["Synthetic index closes higher this month"]["due"] is False
    assert client.get("/api/stats").json()["due"] == 1


def test_start_scan_returns_202_and_runs(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def fake_run(s: Settings, scan_id: int) -> None:
        store.update_scan(s.db_path, scan_id, status="complete", detail="done", finished=True)

    monkeypatch.setattr(pipeline, "run_scan", fake_run)
    resp = client.post("/api/scan")
    assert resp.status_code == 202
    scan_id = resp.json()["scan_id"]
    for _ in range(50):  # the stub runs in a thread; allow it a moment
        scan = client.get(f"/api/scans/{scan_id}").json()
        if scan["status"] != "running":
            break
        time.sleep(0.05)
    assert scan["status"] == "complete"


def test_start_scan_conflict_while_running(client: TestClient, settings: Settings) -> None:
    running_id = store.create_scan(settings.db_path, "fake", "fake-1")
    resp = client.post("/api/scan")
    assert resp.status_code == 409
    assert resp.json()["detail"]["scan_id"] == running_id


def test_start_scan_503_when_no_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(settings: Settings) -> dict[str, Any]:
        raise OracleError("no backend available")

    monkeypatch.setattr(pipeline, "scan_metadata", _raise)
    resp = client.post("/api/scan")
    assert resp.status_code == 503
    assert "no backend" in resp.json()["detail"]


class FakeResolutionOracle:
    provider = "fake"
    model = "fake-1"

    def resolve(
        self,
        forecasts: list[dict[str, Any]],
        articles: list[dict[str, Any]],
        today: date,
    ) -> list["ForecastDraft"]:
        from pythia.oracle.base import ResolutionDraft

        return [
            ResolutionDraft(0, "happened", "reported by the wire", [0]),
            ResolutionDraft(1, "unresolved", "no evidence either way", []),
        ]


def test_resolution_check_pipeline_sets_proposals(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)  # creates 3 open forecasts + 1 article
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=FakeResolutionOracle(), today=date.today()
    )
    scan = store.get_scan(settings.db_path, check_id)
    assert scan["status"] == "complete", scan
    assert scan["kind"] == "resolution"
    assert "1 resolution proposals, 0 probability updates from 3 checked" in scan["detail"]

    preds = store.list_predictions(settings.db_path)
    proposals = [p for p in preds if p["proposal_outcome"] is not None]
    assert len(proposals) == 1
    assert proposals[0]["proposal_note"] == "reported by the wire"
    assert len(proposals[0]["proposal_evidence_ids"]) == 1


class EvidenceFreeNoOracle:
    provider = "fake"
    model = "fake-1"

    def resolve(
        self,
        forecasts: list[dict[str, Any]],
        articles: list[dict[str, Any]],
        today: date,
    ) -> list[Any]:
        from pythia.oracle.base import ResolutionDraft

        # An absence-based NO for every forecast — none of which is old enough.
        return [
            ResolutionDraft(i, "did_not_happen", "no reporting found", [])
            for i in range(len(forecasts))
        ]


def test_evidence_free_no_needs_horizon_well_past(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)  # 3 open forecasts, horizons +30d
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    store.insert_prediction(
        settings.db_path, scan_id=scan_id, statement="Passed yesterday",
        probability=0.5, horizon_date=(date.today() - timedelta(days=1)).isoformat(),
        category="markets", rationale="", evidence_ids=[],
    )
    store.insert_prediction(
        settings.db_path, scan_id=scan_id, statement="Passed long ago",
        probability=0.5, horizon_date=(date.today() - timedelta(days=10)).isoformat(),
        category="markets", rationale="", evidence_ids=[],
    )
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=EvidenceFreeNoOracle(), today=date.today()
    )
    proposals = {
        p["statement"]: p["proposal_outcome"]
        for p in store.list_predictions(settings.db_path)
        if p["proposal_outcome"] is not None
    }
    # Only the long-past horizon may take an evidence-free NO; the future
    # horizons (early NO without impossibility) and yesterday (reporting lag)
    # are all guarded out.
    assert proposals == {"Passed long ago": 0}


def test_track_record_reaches_the_forecast_prompt(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seed 30 resolved forecasts so the feedback threshold is met.
    seed = store.create_scan(settings.db_path, "fake", "fake-1")
    for i in range(30):
        pid = store.insert_prediction(
            settings.db_path, scan_id=seed, statement=f"Seed {i}",
            probability=0.7, horizon_date="2026-09-01", category="markets",
            rationale="", evidence_ids=[],
        )
        store.resolve_prediction(settings.db_path, pid, i % 5 == 0)  # 20% happen

    captured: dict[str, Any] = {}

    class RecordingForecastOracle:
        provider = "fake"
        model = "fake-1"

        def generate(self, articles, today, open_statements, track_record=None, signals=None):  # noqa: ANN001
            captured["track_record"] = track_record
            return FakeOracle().generate(articles, today, open_statements)

    articles = [{
        "source": "Wire", "title": "Synthetic headline",
        "url": "https://example.com/tr", "summary": "s",
        "published_at": _recent(),
    }]
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (articles, {}))
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=RecordingForecastOracle())

    assert store.get_scan(settings.db_path, scan_id)["status"] == "complete"
    assert captured["track_record"] is not None
    assert "Brier" in captured["track_record"]
    assert "70-80% range" in captured["track_record"]  # 70% forecast, 20% happened


def test_gdelt_signals_flow_into_prompt_scan_row_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHIA_DB", str(tmp_path / "gdelt.db"))
    monkeypatch.setenv("PYTHIA_GDELT", "1")
    from pythia.config import get_settings
    from pythia.gdelt import GdeltResult

    settings = get_settings()
    store.init_db(settings.db_path)

    gdelt_articles = [{
        "source": "example.ir (via GDELT)", "title": "Coverage spike item",
        "url": "https://example.ir/spike", "summary": "",
        "published_at": _recent(), "origin": "gdelt:iran",
    }]
    canned = GdeltResult(
        articles=gdelt_articles,
        snapshot="- iran: coverage at 3.0x its prior 12-day baseline over the last 48h (rising)",
        errors={},
    )
    monkeypatch.setattr(pipeline.gdelt, "fetch_signals", lambda w, t: canned)
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (
        [{
            "source": "Wire", "title": "Synthetic headline",
            "url": "https://example.com/x", "summary": "s",
            "published_at": _recent(),
        }], {},
    ))

    captured: dict[str, Any] = {}

    class RecordingOracle:
        provider = "fake"
        model = "fake-1"

        def generate(self, articles, today, open_statements, track_record=None, signals=None):  # noqa: ANN001
            captured["signals"] = signals
            captured["titles"] = [a["title"] for a in articles]
            return FakeOracle().generate(articles, today, open_statements)

    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=RecordingOracle())

    scan = store.get_scan(settings.db_path, scan_id)
    assert scan["status"] == "complete", scan
    assert "3.0x its prior 12-day baseline" in scan["signals"]  # audit trail
    assert captured["signals"] == canned.snapshot  # the oracle read it
    assert "Coverage spike item" in captured["titles"]  # GDELT article batched


def test_gdelt_partial_failure_notes_but_still_batches_articles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHIA_DB", str(tmp_path / "gdelt2.db"))
    monkeypatch.setenv("PYTHIA_GDELT", "1")
    from pythia.config import get_settings
    from pythia.gdelt import GdeltResult

    settings = get_settings()
    store.init_db(settings.db_path)
    canned = GdeltResult(
        articles=[{
            "source": "x.ir (via GDELT)", "title": "Signalless article",
            "url": "https://x.ir/a", "summary": "",
            "published_at": _recent(), "origin": "gdelt:iran",
        }],
        snapshot=None,  # timelines failed; articles survived
        errors={"gulf": "timeline failed: ReadTimeout: slow"},
    )
    monkeypatch.setattr(pipeline.gdelt, "fetch_signals", lambda w, t: canned)
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (
        [{
            "source": "Wire", "title": "Synthetic headline",
            "url": "https://example.com/x", "summary": "s",
            "published_at": _recent(),
        }], {},
    ))

    captured: dict[str, Any] = {}

    class RecordingOracle:
        provider = "fake"
        model = "fake-1"

        def generate(self, articles, today, open_statements, track_record=None, signals=None):  # noqa: ANN001
            captured["signals"] = signals
            captured["titles"] = [a["title"] for a in articles]
            return FakeOracle().generate(articles, today, open_statements)

    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=RecordingOracle())
    scan = store.get_scan(settings.db_path, scan_id)
    assert scan["status"] == "complete"
    assert scan["signals"] == ""  # no stale/false snapshot stored
    assert "(1 GDELT watches failed)" in scan["detail"]
    assert captured["signals"] is None
    assert "Signalless article" in captured["titles"]


class DuplicatingOracle:
    """A model that ignores the prompt's no-duplicates rule, as they do."""

    provider = "fake"
    model = "fake-1"

    def generate(self, articles, today, open_statements, track_record=None, signals=None):  # noqa: ANN001
        from datetime import timedelta as _td

        horizon = (date.today() + _td(days=30)).isoformat()

        def draft(stmt: str) -> ForecastDraft:
            return ForecastDraft(
                statement=stmt, probability=0.5, horizon_date=horizon,
                category="geopolitics", rationale="r", evidence_indices=[],
            )

        return [
            draft("The EU will impose new sanctions on Iran by December 1, 2026."),
            # same claim restated
            draft("The EU imposes new sanctions on Iran by December 1, 2026."),
            # the same claim asserted the other way
            draft("The EU will not impose new sanctions on Iran by December 1, 2026."),
            # genuinely different
            draft("Japan will raise its policy rate by December 1, 2026."),
        ]


def test_duplicate_and_negation_forecasts_are_refused(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (
        [{
            "source": "Wire", "title": "Synthetic headline",
            "url": "https://example.com/x", "summary": "s",
            "published_at": _recent(),
        }], {},
    ))
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=DuplicatingOracle())

    scan = store.get_scan(settings.db_path, scan_id)
    assert scan["status"] == "complete", scan
    assert "2 new forecasts" in scan["detail"]
    assert "(2 duplicates refused)" in scan["detail"]
    statements = [p["statement"] for p in store.list_predictions(settings.db_path)]
    assert len(statements) == 2
    assert any("Japan" in s for s in statements)
    # Never both a claim and its negation.
    assert not (
        any("will impose new sanctions on Iran" in s for s in statements)
        and any("will not impose new sanctions on Iran" in s for s in statements)
    )


def test_duplicates_refused_against_previously_open_forecasts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = store.create_scan(settings.db_path, "fake", "fake-1")
    store.insert_prediction(
        settings.db_path, scan_id=seed,
        statement="Japan will raise its policy rate by December 1, 2026.",
        probability=0.4, horizon_date=(date.today() + timedelta(days=30)).isoformat(),
        category="economy", rationale="", evidence_ids=[],
    )
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: (
        [{
            "source": "Wire", "title": "Synthetic headline",
            "url": "https://example.com/x", "summary": "s",
            "published_at": _recent(),
        }], {},
    ))
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=DuplicatingOracle())
    statements = [p["statement"] for p in store.list_predictions(settings.db_path)]
    # The Japan forecast was already open, so this scan adds only the EU one.
    assert sum("Japan" in s for s in statements) == 1
    assert "(3 duplicates refused)" in store.get_scan(settings.db_path, scan_id)["detail"]


def test_gdelt_disabled_skips_the_pull(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(watches, timeout):  # noqa: ANN001
        raise AssertionError("gdelt must not be called when disabled")

    monkeypatch.setattr(pipeline.gdelt, "fetch_signals", _boom)
    _seed_scan(settings, monkeypatch)  # settings fixture sets PYTHIA_GDELT=0
    assert store.list_scans(settings.db_path)[0]["signals"] == ""


def test_resolution_check_digest_includes_archive_supplement(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)  # 1 recent article in the archive
    # An archived article that matches the overdue statement but is too old to
    # be in the recent batch window would normally be invisible to the judge.
    store.upsert_articles(
        settings.db_path,
        [
            {
                "source": "Wire",
                "title": "Quarterly synthetic dividend announced by megacorp",
                "url": "https://example.com/archive-hit",
                "summary": "",
                "published_at": "2026-01-05T00:00:00+00:00",
            }
        ],
    )
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    store.insert_prediction(
        settings.db_path, scan_id=scan_id,
        statement="Megacorp announces a synthetic dividend",
        probability=0.5, horizon_date=(date.today() - timedelta(days=10)).isoformat(),
        category="markets", rationale="", evidence_ids=[],
    )
    with store.connect(settings.db_path) as conn:  # created before the article aged out
        conn.execute(
            "UPDATE predictions SET created_at = '2026-01-01T00:00:00+00:00' "
            "WHERE statement LIKE 'Megacorp%'"
        )

    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=FakeResolutionOracle(), today=date.today()
    )
    digest = store.articles_for_scan(settings.db_path, check_id)
    titles = [a["title"] for a in digest]
    # The digest holds everything the judge read: recent batch + archive extras.
    assert "Synthetic headline" in titles
    assert "Quarterly synthetic dividend announced by megacorp" in titles
    assert store.get_scan(settings.db_path, check_id)["article_count"] == len(digest)


class UpdatingOracle:
    provider = "fake"
    model = "fake-1"

    def __init__(self, new_probability: float) -> None:
        self.new_probability = new_probability

    def resolve(
        self,
        forecasts: list[dict[str, Any]],
        articles: list[dict[str, Any]],
        today: date,
    ) -> list[Any]:
        from pythia.oracle.base import ResolutionDraft

        return [
            ResolutionDraft(0, "update", "signals shifted", [0], self.new_probability)
        ]


def test_probability_update_applies_with_history(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)  # event forecast at p=0.6
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=UpdatingOracle(0.8), today=date.today()
    )
    scan = store.get_scan(settings.db_path, check_id)
    assert "1 probability updates" in scan["detail"]

    preds = store.list_predictions(settings.db_path)
    updated = next(p for p in preds if p["probability"] == 0.8)
    assert updated["status"] == "open"  # updated, not resolved
    history = store.revisions_by_prediction(settings.db_path, [updated["id"]])
    assert history[updated["id"]][0]["old_probability"] == 0.6
    assert history[updated["id"]][0]["note"] == "signals shifted"
    assert history[updated["id"]][0]["source"] == "oracle"

    # The API contract the UI's revision line consumes.
    api_pred = next(
        p for p in client.get("/api/predictions").json() if p["id"] == updated["id"]
    )
    rev = api_pred["revisions"][0]
    assert rev["old_probability"] == 0.6
    assert rev["new_probability"] == 0.8
    assert rev["note"] == "signals shifted"

    # Drift guard: a same-day re-update on the same evidence is refused.
    again = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, again, oracle=UpdatingOracle(0.9), today=date.today()
    )
    assert "0 probability updates" in store.get_scan(settings.db_path, again)["detail"]


def test_update_boundary_exactly_five_points_applies(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)  # p=0.6
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=UpdatingOracle(0.55), today=date.today()
    )
    assert "1 probability updates" in store.get_scan(settings.db_path, check_id)["detail"]


def test_update_skipped_after_horizon(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    store.upsert_articles(
        settings.db_path,
        [{
            "source": "Wire", "title": "Synthetic headline",
            "url": "https://example.com/x", "summary": "s",
            "published_at": _recent(),
        }],
    )
    store.insert_prediction(
        settings.db_path, scan_id=scan_id, statement="Already overdue",
        probability=0.5, horizon_date=(date.today() - timedelta(days=5)).isoformat(),
        category="markets", rationale="", evidence_ids=[],
    )
    monkeypatch.setattr(pipeline.feeds, "fetch_all", lambda urls, t: ([], {}))
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=UpdatingOracle(0.9), today=date.today()
    )
    # Post-horizon updates are hindsight, not forecasting: refused.
    assert "0 probability updates" in store.get_scan(settings.db_path, check_id)["detail"]
    assert store.list_predictions(settings.db_path)[0]["probability"] == 0.5


def test_negligible_update_is_skipped(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=UpdatingOracle(0.62), today=date.today()  # |Δ| < 0.05
    )
    assert "0 probability updates" in store.get_scan(settings.db_path, check_id)["detail"]
    assert all(p["probability"] != 0.62 for p in store.list_predictions(settings.db_path))


def test_auto_update_flag_disables_revisions(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    monkeypatch.setenv("PYTHIA_AUTO_UPDATE", "0")
    from pythia.config import get_settings

    frozen = get_settings()
    check_id = store.create_scan(frozen.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(frozen, check_id, oracle=UpdatingOracle(0.9), today=date.today())
    assert "0 probability updates" in store.get_scan(frozen.db_path, check_id)["detail"]
    # And nothing changed in the database — not merely an uncounted apply.
    assert all(p["probability"] != 0.9 for p in store.list_predictions(frozen.db_path))
    all_ids = [p["id"] for p in store.list_predictions(frozen.db_path)]
    assert store.revisions_by_prediction(frozen.db_path, all_ids) == {}


def test_targeted_check_endpoint_and_pipeline(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = client.get("/api/predictions").json()
    target = preds[0]["id"]
    other_ids = [p["id"] for p in preds if p["id"] != target]

    captured: dict[str, Any] = {}

    def fake_check(s: Settings, scan_id: int, oracle=None, today=None, target_ids=None) -> None:
        captured["target_ids"] = target_ids
        store.update_scan(s.db_path, scan_id, status="complete", finished=True)

    monkeypatch.setattr(pipeline, "run_resolution_check", fake_check)
    resp = client.post(f"/api/predictions/{target}/check")
    assert resp.status_code == 202
    assert resp.json()["kind"] == "resolution"
    for _ in range(50):
        if "target_ids" in captured:
            break
        time.sleep(0.05)
    assert captured["target_ids"] == [target]
    assert target not in other_ids

    assert client.post("/api/predictions/9999/check").status_code == 404


def test_targeted_pipeline_checks_only_target(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = store.list_predictions(settings.db_path)
    target = preds[0]["id"]

    seen: dict[str, Any] = {}

    class RecordingOracle:
        provider = "fake"
        model = "fake-1"

        def resolve(self, forecasts, articles, today):  # noqa: ANN001
            seen["ids"] = [f["id"] for f in forecasts]
            return []

    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(
        settings, check_id, oracle=RecordingOracle(), today=date.today(),
        target_ids=[target],
    )
    assert seen["ids"] == [target]


def test_resolution_check_with_no_open_forecasts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_id = store.create_scan(settings.db_path, "fake", "fake-1", kind="resolution")
    pipeline.run_resolution_check(settings, check_id, oracle=FakeResolutionOracle())
    scan = store.get_scan(settings.db_path, check_id)
    assert scan["status"] == "complete"
    assert scan["detail"] == "No open forecasts to check"


def test_proposal_accept_and_reject_endpoints(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    preds = client.get("/api/predictions").json()
    a, b = preds[0]["id"], preds[1]["id"]
    article_id = store.recent_articles(settings.db_path)[0]["id"]
    store.set_proposal(settings.db_path, a, True, "seen on the wire", [article_id])
    store.set_proposal(settings.db_path, b, False, "deadline passed", [])

    listed = {p["id"]: p for p in client.get("/api/predictions").json()}
    assert listed[a]["proposal_outcome"] == 1
    assert listed[a]["proposal_evidence"][0]["title"] == "Synthetic headline"

    accepted = client.post(f"/api/predictions/{a}/proposal/accept")
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "resolved"
    assert accepted.json()["resolution_note"] == "[oracle] seen on the wire"
    assert accepted.json()["resolution_evidence_ids"] == [article_id]

    resolved_view = next(
        p for p in client.get("/api/predictions").json() if p["id"] == a
    )
    assert resolved_view["resolution_evidence"][0]["title"] == "Synthetic headline"

    assert client.post(f"/api/predictions/{b}/proposal/reject").status_code == 200
    assert client.post(f"/api/predictions/{b}/proposal/reject").status_code == 404
    assert client.post(f"/api/predictions/{a}/proposal/accept").status_code == 404


def test_start_resolution_check_endpoint(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def fake_check(
        s: Settings, scan_id: int, oracle=None, today=None, target_ids=None
    ) -> None:
        store.update_scan(s.db_path, scan_id, status="complete", detail="done", finished=True)

    monkeypatch.setattr(pipeline, "run_resolution_check", fake_check)
    resp = client.post("/api/resolution-check")
    assert resp.status_code == 202
    assert resp.json()["kind"] == "resolution"
    scan_id = resp.json()["scan_id"]
    for _ in range(50):
        scan = client.get(f"/api/scans/{scan_id}").json()
        if scan["status"] != "running":
            break
        time.sleep(0.05)
    assert scan["status"] == "complete"
    assert scan["kind"] == "resolution"


def test_scan_digest_endpoint(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_scan(settings, monkeypatch)
    scan_id = client.get("/api/scans").json()[0]["id"]
    digest = client.get(f"/api/scans/{scan_id}/articles").json()
    assert [a["title"] for a in digest] == ["Synthetic headline"]
    assert client.get("/api/scans/9999/articles").status_code == 404


def test_articles_limit_validation(client: TestClient) -> None:
    assert client.get("/api/articles?limit=-5").status_code == 422
    assert client.get("/api/articles?limit=9999").status_code == 422
    assert client.get("/api/predictions?status=bogus").status_code == 422


def test_pipeline_records_error_when_feeds_empty(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline.feeds, "fetch_all", lambda urls, t: ([], {"https://x": "boom"})
    )
    scan_id = store.create_scan(settings.db_path, "fake", "fake-1")
    pipeline.run_scan(settings, scan_id, oracle=FakeOracle())
    scan = store.get_scan(settings.db_path, scan_id)
    assert scan["status"] == "error"
    assert "boom" in scan["error"]


def test_stats_empty_db(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    empty_mode = {"open": 0, "resolved": 0, "brier": None}
    assert stats == {
        "open": 0,
        "due": 0,
        "resolved": 0,
        "void": 0,
        "brier": None,
        "hit_rate": None,
        "calibration": [],
        "by_mode": {"event": empty_mode, "consequence": empty_mode, "precedent": empty_mode},
    }
