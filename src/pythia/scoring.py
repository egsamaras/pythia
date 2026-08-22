"""Calibration metrics over resolved predictions.

Pure functions: input is (probability, outcome) pairs where outcome is 0 or 1.
Resolved forecasts are scored on their LIFETIME probability — the
time-weighted average credence over the forecast's open life, reconstructed
from the revision history — so a last-minute revision cannot game the ledger.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def lifetime_probability(record: dict[str, Any]) -> float:
    """Time-weighted average probability over a resolved forecast's open life.

    ``record`` needs final ``probability``, ``created_at``, ``resolved_at``,
    and ``revisions`` (each with old/new_probability and created_at, any
    order). Falls back to the final probability when timestamps are unusable.
    """
    final_p = record["probability"]
    revisions = sorted(
        (r for r in record.get("revisions", []) if _parse(r.get("created_at"))),
        key=lambda r: r["created_at"],
    )
    start = _parse(record.get("created_at"))
    end = _parse(record.get("resolved_at"))
    if not revisions or start is None or end is None or end <= start:
        return final_p
    # Trajectory: initial probability until the first revision, each revised
    # value until the next, the last revised value until resolution.
    points = [(start, revisions[0]["old_probability"])]
    for r in revisions:
        ts = _parse(r["created_at"])
        assert ts is not None
        points.append((min(max(ts, start), end), r["new_probability"]))
    total = (end - start).total_seconds()
    weighted = 0.0
    for i, (ts, p) in enumerate(points):
        nxt = points[i + 1][0] if i + 1 < len(points) else end
        weighted += p * max((nxt - ts).total_seconds(), 0.0)
    return round(weighted / total, 4)


def lifetime_pairs(records: list[dict[str, Any]]) -> list[tuple[float, int]]:
    """(lifetime probability, outcome) pairs for scoring resolved records."""
    return [(lifetime_probability(r), r["outcome"]) for r in records]


def brier_score(pairs: list[tuple[float, int]]) -> float | None:
    """Mean squared error between forecast probability and outcome.

    0.0 is a perfect forecaster, 0.25 matches always saying 50%, 1.0 is
    perfectly wrong. Returns None when nothing is resolved yet.
    """
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def hit_rate(pairs: list[tuple[float, int]]) -> float | None:
    """Share of resolved predictions where the >=50% side matched the outcome."""
    if not pairs:
        return None
    hits = sum(1 for p, o in pairs if (p >= 0.5) == bool(o))
    return hits / len(pairs)


def track_record_summary(
    pairs: list[tuple[float, int]],
    by_mode: dict[str, list[tuple[float, int]]],
    min_resolved: int = 30,
) -> str | None:
    """Plain-language calibration feedback for the forecasting prompt.

    The self-learning loop: once enough forecasts have resolved, the system's
    measured miscalibration is described back to the oracle so future
    probabilities can correct for it. Thresholds are deliberately strict —
    a bin's gap must exceed twice its binomial standard error on at least 10
    samples before it becomes a directive, so statistical noise is never fed
    back as instruction. Returns None below ``min_resolved``.
    """
    if len(pairs) < min_resolved:
        return None
    overall = brier_score(pairs)
    assert overall is not None
    lines = [
        f"Overall Brier score {overall:.3f} over {len(pairs)} resolved forecasts "
        "(0 is perfect, 0.25 is a coin flip)."
    ]
    for mode, mode_pairs in by_mode.items():
        mode_brier = brier_score(mode_pairs)
        if mode_brier is not None and len(mode_pairs) >= 5:
            lines.append(f"Mode '{mode}': Brier {mode_brier:.3f} (n={len(mode_pairs)}).")
    for b in calibration_bins(pairs):
        n = b["count"]
        if n < 10:
            continue
        p = b["mean_predicted"]
        gap = p - b["observed_rate"]
        if abs(gap) <= 2 * math.sqrt(max(p * (1 - p), 1e-9) / n):
            continue
        lines.append(
            f"In the {round(b['bin_low'] * 100)}-{round(b['bin_high'] * 100)}% range "
            f"you forecast {round(p * 100)}% on average but outcomes occurred "
            f"{round(b['observed_rate'] * 100)}% of the time (n={n}) — move "
            f"forecasts in this range toward {round(b['observed_rate'] * 100)}%."
        )
    return "\n".join(lines)


def calibration_bins(
    pairs: list[tuple[float, int]], n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Bucket forecasts into probability bins; compare predicted vs observed.

    Returns only non-empty bins, each with the bin bounds, the mean forecast
    probability, the observed outcome rate, and the count.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        buckets[idx].append((p, o))
    bins: list[dict[str, float | int]] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        bins.append(
            {
                "bin_low": i / n_bins,
                "bin_high": (i + 1) / n_bins,
                "mean_predicted": sum(p for p, _ in bucket) / len(bucket),
                "observed_rate": sum(o for _, o in bucket) / len(bucket),
                "count": len(bucket),
            }
        )
    return bins
