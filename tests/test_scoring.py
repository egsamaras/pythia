"""Calibration metric tests: edge cases first."""

import pytest

from pythia.scoring import brier_score, calibration_bins, hit_rate


def test_brier_empty_returns_none() -> None:
    assert brier_score([]) is None


def test_brier_perfect_forecaster() -> None:
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0


def test_brier_perfectly_wrong() -> None:
    assert brier_score([(1.0, 0), (0.0, 1)]) == 1.0


def test_brier_coin_flip() -> None:
    assert brier_score([(0.5, 1), (0.5, 0)]) == pytest.approx(0.25)


def test_hit_rate_empty_returns_none() -> None:
    assert hit_rate([]) is None


def test_hit_rate_counts_majority_side() -> None:
    # 0.7->1 hit, 0.3->0 hit, 0.9->0 miss
    assert hit_rate([(0.7, 1), (0.3, 0), (0.9, 0)]) == pytest.approx(2 / 3)


def test_calibration_empty() -> None:
    assert calibration_bins([]) == []


def test_calibration_bins_group_and_average() -> None:
    pairs = [(0.72, 1), (0.78, 0), (0.15, 0)]
    bins = calibration_bins(pairs)
    assert len(bins) == 2
    low, high = bins
    assert low["bin_low"] == pytest.approx(0.1)
    assert low["count"] == 1
    assert high["mean_predicted"] == pytest.approx(0.75)
    assert high["observed_rate"] == pytest.approx(0.5)
    assert high["count"] == 2


def test_calibration_probability_one_lands_in_top_bin() -> None:
    bins = calibration_bins([(1.0, 1)])
    assert len(bins) == 1
    assert bins[0]["bin_high"] == pytest.approx(1.0)


def test_calibration_rejects_bad_bin_count() -> None:
    with pytest.raises(ValueError):
        calibration_bins([(0.5, 1)], n_bins=0)


def test_track_record_summary_needs_volume() -> None:
    from pythia.scoring import track_record_summary

    assert track_record_summary([(0.5, 1)] * 29, {}) is None


def test_track_record_summary_names_significant_bins_with_direction() -> None:
    from pythia.scoring import track_record_summary

    # Twenty forecasts at 75% of which only 4 happened (gap 0.55 >> 2*SE) plus
    # ten near-calibrated ones at 35% (gap 0.05, inside noise): only the first
    # bin may become a directive, phrased toward the observed frequency.
    bad = [(0.75, 1)] * 4 + [(0.75, 0)] * 16
    fine = [(0.35, 1)] * 4 + [(0.35, 0)] * 6
    pairs = bad + fine
    summary = track_record_summary(pairs, {"event": pairs})
    assert summary is not None
    assert "Brier" in summary
    assert "Mode 'event'" in summary
    assert "70-80% range" in summary
    assert "move forecasts in this range toward 20%" in summary
    assert "30-40% range" not in summary  # noise stays out of the directives


def test_lifetime_probability_without_revisions_is_final() -> None:
    from pythia.scoring import lifetime_probability

    record = {
        "probability": 0.6, "outcome": 1, "revisions": [],
        "created_at": "2026-08-01T00:00:00+00:00",
        "resolved_at": "2026-08-11T00:00:00+00:00",
    }
    assert lifetime_probability(record) == 0.6


def test_lifetime_probability_weights_by_time_held() -> None:
    from pythia.scoring import lifetime_probability

    # 0.20 held for 8 days, then revised to 0.90 for the final 2 days:
    # a late jump moves the score only a little (0.2*0.8 + 0.9*0.2 = 0.34).
    record = {
        "probability": 0.9, "outcome": 1,
        "created_at": "2026-08-01T00:00:00+00:00",
        "resolved_at": "2026-08-11T00:00:00+00:00",
        "revisions": [
            {
                "old_probability": 0.2, "new_probability": 0.9,
                "created_at": "2026-08-09T00:00:00+00:00",
            }
        ],
    }
    assert lifetime_probability(record) == pytest.approx(0.34)


def test_lifetime_probability_survives_bad_timestamps() -> None:
    from pythia.scoring import lifetime_probability

    record = {
        "probability": 0.7, "outcome": 0,
        "created_at": "garbage", "resolved_at": None,
        "revisions": [
            {"old_probability": 0.2, "new_probability": 0.7, "created_at": "also-bad"}
        ],
    }
    assert lifetime_probability(record) == 0.7
