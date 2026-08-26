"""Headless command-line entry points: run scans without the dashboard.

Used directly (``python -m pythia.cli scan``) and by the launchd schedule
that ``python -m pythia.cli schedule`` generates.
"""

from __future__ import annotations

import argparse
import logging
import os
import plistlib
import sys
from pathlib import Path

from pythia import dedup, pipeline, store
from pythia.config import (
    LEGACY_DB_PATH,
    PROJECT_ROOT,
    Settings,
    db_path_is_default,
    get_settings,
)

logger = logging.getLogger(__name__)

LAUNCH_AGENT_LABEL = "com.pythia.daily"


def _prepare(settings: Settings) -> None:
    if db_path_is_default() and store.migrate_legacy_db(LEGACY_DB_PATH, settings.db_path):
        logger.warning("database migrated from %s to %s", LEGACY_DB_PATH, settings.db_path)
    store.init_db(settings.db_path)
    # Age-based sweep only: the dashboard server may legitimately have a run
    # in flight right now, and it must not be clobbered from the CLI.
    stale = store.fail_orphaned_scans(
        settings.db_path, older_than_s=settings.oracle_timeout_s + 900
    )
    if stale:
        logger.warning("marked %d stale running scan(s) as interrupted", stale)


def _run(kind: str) -> int:
    """Run one scan or resolution check synchronously; exit 0 on success."""
    settings = get_settings()
    _prepare(settings)
    meta = pipeline.scan_metadata(settings)
    # Atomic gate shared with the server: creation fails when a run is live.
    scan_id = store.create_scan_exclusive(
        settings.db_path, meta["provider"], meta["model"], kind
    )
    if scan_id is None:
        running = store.running_scan(settings.db_path)
        detail = (
            f"#{running['id']} {running['kind']}, started {running['started_at']}"
            if running
            else "unknown"
        )
        print(
            f"another run is already in progress ({detail}); try again later",
            file=sys.stderr,
        )
        return 2
    logger.info("%s run %d starting on %s:%s", kind, scan_id, meta["provider"], meta["model"])
    if kind == "forecast":
        pipeline.run_scan(settings, scan_id)
    else:
        pipeline.run_resolution_check(settings, scan_id)
    scan = store.get_scan(settings.db_path, scan_id)
    assert scan is not None
    print(f"{kind} run {scan_id}: {scan['status']} — {scan['detail']}")
    if scan["error"]:
        print(f"error: {scan['error']}", file=sys.stderr)
    return 0 if scan["status"] == "complete" else 1


def cmd_scan(_: argparse.Namespace) -> int:
    return _run("forecast")


def cmd_check(_: argparse.Namespace) -> int:
    return _run("resolution")


def cmd_daily(args: argparse.Namespace) -> int:
    """Morning routine: propose resolutions for what's open, then scan."""
    check_rc = _run("resolution")
    scan_rc = _run("forecast")
    return max(check_rc, scan_rc)


def cmd_prune(args: argparse.Namespace) -> int:
    """Retire open forecasts that today's validation would refuse.

    Statement hygiene (no base rates, no duplicates, no negation pairs) became
    an enforced gate after some forecasts were already stored. Those rows
    would corrupt the first real calibration numbers, so this retires them.
    Resolved forecasts are never touched — the ledger's history is immutable.
    """
    settings = get_settings()
    _prepare(settings)
    open_preds = [
        p for p in store.list_predictions(settings.db_path, "open")
    ]
    open_preds.sort(key=lambda p: p["id"])  # keep the earliest of a colliding set

    kept: list[str] = []
    doomed: list[tuple[dict, str, str]] = []
    for p in open_preds:
        if not dedup.is_future_claim(p["statement"]):
            doomed.append((p, "not-a-forecast", ""))
            continue
        collision = dedup.duplicate_of(p["statement"], kept)
        if collision:
            doomed.append((p, collision[0], collision[1]))
            continue
        kept.append(p["statement"])

    if not doomed:
        print(f"Nothing to prune: all {len(open_preds)} open forecasts are clean.")
        return 0

    for p, kind, other in doomed:
        print(f"[{kind:14}] #{p['id']} {p['statement'][:66]}")
        if other:
            print(f"{'':17} collides with: {other[:62]}")
    print(f"\n{len(doomed)} of {len(open_preds)} open forecasts would be retired.")

    if args.dry_run:
        print("Dry run — nothing changed. Re-run without --dry-run to apply.")
        return 0
    removed = store.delete_open_predictions(settings.db_path, [p["id"] for p, _, _ in doomed])
    print(f"Retired {removed} forecasts; {len(kept)} remain open.")
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    """Write a launchd agent plist for a daily run. Activation stays manual."""
    log_path = Path.home() / "Library" / "Logs" / "pythia.log"
    env = {k: v for k, v in os.environ.items() if k.startswith("PYTHIA_")}
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    plist = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [sys.executable, "-m", "pythia.cli", "daily"],
        "EnvironmentVariables": env,
        "StartCalendarInterval": {"Hour": args.hour, "Minute": args.minute},
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
    }
    agent_path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agent_path, "wb") as f:
        plistlib.dump(plist, f)
    print(f"Wrote {agent_path}")
    print(f"Daily run at {args.hour:02d}:{args.minute:02d}, logging to {log_path}")
    if env.keys() - {"PYTHONPATH"}:
        captured = ", ".join(sorted(env.keys() - {"PYTHONPATH"}))
        print(f"Captured current environment: {captured}")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        # Never write secrets into a plist; launchd won't inherit the shell env.
        print(
            "\nNote: your shell has Anthropic credentials, but launchd jobs do not "
            "inherit shell environment variables and this tool never writes secrets "
            "into the plist. For the scheduled run to use the Claude backend, use an "
            "`ant auth login` profile (stored on disk, found automatically) — "
            "otherwise the job will fall back to Ollama."
        )
    print("\nTo activate:")
    print(f"  launchctl load -w {agent_path}")
    print("To deactivate:")
    print(f"  launchctl unload -w {agent_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="pythia", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="run one forecast scan").set_defaults(func=cmd_scan)
    sub.add_parser("check", help="run one resolution check").set_defaults(func=cmd_check)
    sub.add_parser("daily", help="resolution check, then forecast scan").set_defaults(
        func=cmd_daily
    )
    prune = sub.add_parser(
        "prune", help="retire open forecasts today's validation would refuse"
    )
    prune.add_argument("--dry-run", action="store_true", help="list them, change nothing")
    prune.set_defaults(func=cmd_prune)
    schedule = sub.add_parser("schedule", help="write a launchd agent for a daily run")
    schedule.add_argument("--hour", type=int, default=7, choices=range(24))
    schedule.add_argument("--minute", type=int, default=30, choices=range(60))
    schedule.set_defaults(func=cmd_schedule)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
