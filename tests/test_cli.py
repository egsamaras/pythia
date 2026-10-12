"""CLI tests: headless runs and launchd schedule generation."""

import plistlib
import sys
from pathlib import Path

import pytest

from pythia import cli, pipeline, store
from pythia.config import Settings


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PYTHIA_DB", str(tmp_path / "cli.db"))
    return tmp_path


def test_cli_scan_runs_and_reports(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def fake_run(settings: Settings, scan_id: int) -> None:
        store.update_scan(
            settings.db_path, scan_id, status="complete", detail="5 forecasts", finished=True
        )

    monkeypatch.setattr(pipeline, "run_scan", fake_run)
    assert cli.main(["scan"]) == 0
    out = capsys.readouterr().out
    assert "forecast run 1: complete — 5 forecasts" in out


def test_cli_daily_runs_check_then_scan(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def fake_scan(settings: Settings, scan_id: int) -> None:
        order.append("scan")
        store.update_scan(settings.db_path, scan_id, status="complete", finished=True)

    def fake_check(settings: Settings, scan_id: int) -> None:
        order.append("check")
        store.update_scan(settings.db_path, scan_id, status="complete", finished=True)

    monkeypatch.setattr(pipeline, "run_scan", fake_scan)
    monkeypatch.setattr(pipeline, "run_resolution_check", fake_check)
    assert cli.main(["daily"]) == 0
    assert order == ["check", "scan"]


def test_cli_refuses_when_run_already_in_progress(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pythia.config import get_settings

    settings = get_settings()
    store.init_db(settings.db_path)
    store.create_scan(settings.db_path, "ollama", "m")  # fresh running scan

    assert cli.main(["scan"]) == 2
    assert "already in progress" in capsys.readouterr().err
    # The live run was NOT clobbered by the CLI's stale sweep.
    assert store.running_scan(settings.db_path) is not None


def test_cli_sweeps_stale_run_then_proceeds(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pythia.config import get_settings

    settings = get_settings()
    store.init_db(settings.db_path)
    stale = store.create_scan(settings.db_path, "ollama", "m")
    with store.connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE scans SET started_at = '2026-08-01T00:00:00+00:00' WHERE id = ?",
            (stale,),
        )
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def fake_run(settings: Settings, scan_id: int) -> None:
        store.update_scan(settings.db_path, scan_id, status="complete", finished=True)

    monkeypatch.setattr(pipeline, "run_scan", fake_run)
    assert cli.main(["scan"]) == 0
    assert store.get_scan(settings.db_path, stale)["status"] == "error"


def test_cli_run_reports_failure_exit_code(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def fake_run(settings: Settings, scan_id: int) -> None:
        store.update_scan(
            settings.db_path, scan_id, status="error", error="boom", finished=True
        )

    monkeypatch.setattr(pipeline, "run_scan", fake_run)
    assert cli.main(["scan"]) == 1


def _seed_messy_ledger(db_path) -> dict[str, int]:  # noqa: ANN001
    scan_id = store.create_scan(db_path, "fake", "fake-1")
    ids = {}
    for name, stmt in [
        ("good", "The EU will impose new sanctions on Iran by December 1, 2026."),
        ("dup", "The EU imposes new sanctions on Iran by December 1, 2026."),
        ("neg", "The EU will not impose new sanctions on Iran by December 1, 2026."),
        ("baserate", "6 of 8 emergency OPEC meetings since 2008 ended in a quota cut."),
        ("other", "Japan will raise its policy rate by December 1, 2026."),
    ]:
        ids[name] = store.insert_prediction(
            db_path, scan_id=scan_id, statement=stmt, probability=0.5,
            horizon_date="2026-12-01", category="geopolitics", rationale="",
            evidence_ids=[],
        )
    return ids


def test_cli_prune_dry_run_changes_nothing(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pythia.config import get_settings

    settings = get_settings()
    store.init_db(settings.db_path)
    _seed_messy_ledger(settings.db_path)

    assert cli.main(["prune", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 of 5 open forecasts would be retired" in out
    assert "Dry run" in out
    assert len(store.list_predictions(settings.db_path, "open")) == 5


def test_cli_prune_retires_only_the_bad_rows(cli_env: Path) -> None:
    from pythia.config import get_settings

    settings = get_settings()
    store.init_db(settings.db_path)
    ids = _seed_messy_ledger(settings.db_path)

    assert cli.main(["prune"]) == 0
    survivors = {p["id"] for p in store.list_predictions(settings.db_path, "open")}
    assert survivors == {ids["good"], ids["other"]}


def test_prune_never_touches_resolved_forecasts(cli_env: Path) -> None:
    from pythia.config import get_settings

    settings = get_settings()
    store.init_db(settings.db_path)
    ids = _seed_messy_ledger(settings.db_path)
    # A duplicate that has already been scored is history: immutable.
    store.resolve_prediction(settings.db_path, ids["dup"], True)

    assert cli.main(["prune"]) == 0
    resolved = store.list_predictions(settings.db_path, "resolved")
    assert [p["id"] for p in resolved] == [ids["dup"]]
    assert store.delete_open_predictions(settings.db_path, [ids["dup"]]) == 0


def test_backup_snapshot_is_readable_and_pruned(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pythia.config import get_settings

    backup_dir = cli_env / "backups"
    monkeypatch.setenv("PYTHIA_BACKUP_DIR", str(backup_dir))
    monkeypatch.setenv("PYTHIA_BACKUP_KEEP", "3")
    settings = get_settings()
    store.init_db(settings.db_path)
    scan = store.create_scan(settings.db_path, "fake", "fake-1")
    store.insert_prediction(
        settings.db_path, scan_id=scan, statement="Something happens by December",
        probability=0.5, horizon_date="2026-12-01", category="markets",
        rationale="", evidence_ids=[],
    )

    assert cli.main(["backup"]) == 0
    snapshots = sorted(backup_dir.glob("pythia-*.db"))
    assert len(snapshots) == 1
    # The snapshot must be a real, queryable ledger — not a truncated file.
    assert [p["statement"] for p in store.list_predictions(snapshots[0])] == [
        "Something happens by December"
    ]
    assert "Backed up to" in capsys.readouterr().out

    # Retention keeps the newest N by filename date.
    for day in ("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"):
        (backup_dir / f"pythia-{day}.db").write_bytes(b"old")
    assert cli.main(["backup"]) == 0
    kept = sorted(p.name for p in backup_dir.glob("pythia-*.db"))
    assert len(kept) == 3
    assert kept[-1].startswith("pythia-20")  # today's survives as the newest


def test_backup_failure_never_stops_the_daily_run(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(db_path, dest):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(store, "backup_database", boom)
    monkeypatch.setattr(
        pipeline, "scan_metadata", lambda s: {"provider": "fake", "model": "fake-1"}
    )

    def done(settings: Settings, scan_id: int) -> None:
        store.update_scan(settings.db_path, scan_id, status="complete", finished=True)

    monkeypatch.setattr(pipeline, "run_scan", done)
    monkeypatch.setattr(pipeline, "run_resolution_check", done)
    # The forecasting work still runs even when the snapshot cannot be written.
    assert cli.main(["daily"]) == 0


def test_cli_schedule_writes_plist_without_activating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHIA_OLLAMA_MODEL", "mistral-small:22b")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    assert cli.main(["schedule", "--hour", "6", "--minute", "45"]) == 0

    plist_path = tmp_path / "Library" / "LaunchAgents" / "com.pythia.daily.plist"
    assert plist_path.exists()
    with open(plist_path, "rb") as f:
        plist = plistlib.load(f)
    assert plist["Label"] == "com.pythia.daily"
    assert plist["ProgramArguments"] == [sys.executable, "-m", "pythia.cli", "daily"]
    assert plist["StartCalendarInterval"] == {"Hour": 6, "Minute": 45}
    assert plist["EnvironmentVariables"]["PYTHIA_OLLAMA_MODEL"] == "mistral-small:22b"
    assert "PYTHONPATH" in plist["EnvironmentVariables"]
    # Activation is the user's call: the CLI prints the command, never runs it.
    out = capsys.readouterr().out
    assert "launchctl load -w" in out
