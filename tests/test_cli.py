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
