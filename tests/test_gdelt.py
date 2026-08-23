"""GDELT adapter tests — canned payloads, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from pythia import gdelt


def _stamp(hours_ago: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return dt.strftime("%Y%m%dT%H%M%SZ")


ARTLIST = {
    "articles": [
        {
            "url": "https://example.ir/report",
            "title": "گزارش جدید درباره برنامه هسته‌ای",
            "seendate": _stamp(2),
            "domain": "example.ir",
            "language": "Farsi",
            "sourcecountry": "Iran",
        },
        {
            "url": "https://example.com/en",
            "title": "Inspectors denied access to enrichment site",
            "seendate": _stamp(3),
            "domain": "example.com",
            "language": "English",
            "sourcecountry": "United States",
        },
        {"url": "not-a-url", "title": "Broken", "seendate": _stamp(1)},
        {"url": "https://example.com/untitled", "title": "", "seendate": _stamp(1)},
    ]
}


def test_parse_artlist_normalizes_and_labels() -> None:
    articles = gdelt.parse_artlist("iran", ARTLIST)
    assert len(articles) == 2  # broken url and empty title dropped
    farsi, english = articles
    assert farsi["source"] == "example.ir (via GDELT)"
    assert farsi["origin"] == "gdelt:iran"
    assert farsi["summary"] == "(Farsi-language coverage from Iran)"
    assert english["summary"] == ""  # English needs no language flag
    assert farsi["published_at"].endswith("+00:00")


def _timeline(recent_rate: float, baseline_rate: float) -> dict[str, Any]:
    data = []
    for day in range(13, 2, -1):  # baseline: 3-13 days ago
        data.append({"date": _stamp(day * 24), "value": baseline_rate * 1000, "norm": 1000})
    for h in (36, 24, 12):  # recent 48h
        data.append({"date": _stamp(h), "value": recent_rate * 1000, "norm": 1000})
    return {"timeline": [{"data": data}]}


def test_escalation_ratio_pools_windows_and_counts() -> None:
    rates = gdelt.escalation_ratio(_timeline(recent_rate=0.06, baseline_rate=0.02))
    assert rates is not None
    recent, baseline, count = rates
    assert recent == pytest.approx(0.06)
    assert baseline == pytest.approx(0.02)
    assert count == 180  # 3 recent buckets x 60 matched articles


def test_escalation_ratio_is_norm_weighted_not_mean_of_ratios() -> None:
    # A tiny partial trailing bucket (norm=10) with an extreme share must not
    # dominate: pooled = (60+1)/(1000+10) ~= 0.0604, while a naive mean of
    # ratios would report (0.06 + 0.10) / 2 = 0.08.
    payload = _timeline(recent_rate=0.06, baseline_rate=0.02)
    data = payload["timeline"][0]["data"]
    data = [p for p in data if True][:-2]  # keep one recent bucket
    data.append({"date": _stamp(0.5), "value": 1, "norm": 10})  # partial bucket
    rates = gdelt.escalation_ratio({"timeline": [{"data": data}]})
    assert rates is not None
    assert rates[0] == pytest.approx(61 / 1010, rel=1e-3)


def test_snapshot_line_branches() -> None:
    thin = gdelt._snapshot_line("quiet", 0.0, 0.0, 0)
    assert thin is None  # dormant watch: silence, never a "new signal"
    assert gdelt._snapshot_line("thin", 0.5, 0.1, 9) is None  # below floor
    new = gdelt._snapshot_line("emergent", 0.05, 0.0, 25)
    assert new is not None and "(new signal)" in new and "25 matched" in new
    rising = gdelt._snapshot_line("hot", 0.06, 0.02, 180)
    assert rising is not None and "3.0x" in rising and "(rising)" in rising
    steady = gdelt._snapshot_line("flat", 0.021, 0.02, 50)
    assert steady is not None and "(steady)" in steady
    falling = gdelt._snapshot_line("cool", 0.01, 0.02, 40)
    assert falling is not None and "(falling)" in falling


def test_escalation_ratio_guards_thin_or_lopsided_data() -> None:
    assert gdelt.escalation_ratio({"timeline": []}) is None
    assert gdelt.escalation_ratio({"timeline": [{"data": []}]}) is None
    only_recent = {
        "timeline": [{"data": [{"date": _stamp(h), "value": 5, "norm": 100} for h in (1, 2, 3, 4)]}]
    }
    assert gdelt.escalation_ratio(only_recent) is None
    zero_norm = {
        "timeline": [{"data": [{"date": _stamp(h), "value": 5, "norm": 0} for h in (1, 30, 60, 90)]}]
    }
    assert gdelt.escalation_ratio(zero_norm) is None


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        if params and params.get("mode") == "ArtList":
            return _FakeResponse(ARTLIST)
        return _FakeResponse(_timeline(recent_rate=0.06, baseline_rate=0.02))


def test_fetch_signals_builds_snapshot_and_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdelt.httpx, "Client", _FakeClient)
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)  # no real spacing
    result = gdelt.fetch_signals((("iran", "iran nuclear"), ("gulf", "gulf security")))
    assert result.errors == {}
    assert len(result.articles) == 4  # two valid per watch
    assert result.snapshot is not None
    assert "ATTENTION, not verified events" in result.snapshot
    assert "never as direct evidence" in result.snapshot
    assert "- iran: coverage at 3.0x its prior 12-day baseline" in result.snapshot
    assert "180 matched articles" in result.snapshot
    assert "(rising)" in result.snapshot


def test_fetch_signals_fails_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenClient(_FakeClient):
        def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
            raise gdelt.httpx.ConnectError("offline")

    monkeypatch.setattr(gdelt.httpx, "Client", _BrokenClient)
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    result = gdelt.fetch_signals((("iran", "iran nuclear"),))
    assert result.articles == []
    assert result.snapshot is None
    assert "iran" in result.errors


def test_fetch_signals_aborts_pull_on_first_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Resp429:
        status_code = 429

        def raise_for_status(self) -> None:  # pragma: no cover - 429 short-circuits
            pass

        def json(self) -> dict[str, Any]:
            return {}

    class _ThrottledClient(_FakeClient):
        def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
            calls.append(params["query"] if params else "")
            return _Resp429()

    monkeypatch.setattr(gdelt.httpx, "Client", _ThrottledClient)
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    watches = tuple((f"w{i}", f"query {i}") for i in range(5))
    result = gdelt.fetch_signals(watches)
    assert result.snapshot is None
    assert len(result.errors) == 5
    # Exactly ONE request hits the blocked API; everything else is skipped —
    # requests made against an active block extend it.
    assert len(calls) == 1
    assert all("skipped" in result.errors[f"w{i}"] for i in (1, 2, 3, 4))


def test_fetch_signals_mixed_failure_keeps_good_watches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MixedClient(_FakeClient):
        def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
            if params and "broken" in params.get("query", ""):
                raise gdelt.httpx.ConnectError("boom")
            return super().get(url, params)

    monkeypatch.setattr(gdelt.httpx, "Client", _MixedClient)
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    result = gdelt.fetch_signals((("bad", "broken query"), ("iran", "iran nuclear")))
    assert list(result.errors) == ["bad"]  # non-429 failure never aborts the pull
    assert result.snapshot is not None and "- iran:" in result.snapshot
    assert len(result.articles) == 2


def test_timeline_failure_keeps_artlist_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TimelineBrokenClient(_FakeClient):
        def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
            if params and params.get("mode") == "TimelineVolRaw":
                raise gdelt.httpx.ReadTimeout("slow")
            return super().get(url, params)

    monkeypatch.setattr(gdelt.httpx, "Client", _TimelineBrokenClient)
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    result = gdelt.fetch_signals((("iran", "iran nuclear"),))
    assert len(result.articles) == 2  # banked before the timeline call failed
    assert result.snapshot is None
    assert "timeline failed" in result.errors["iran"]


def test_parse_seendate_fallback_cannot_win_recency() -> None:
    broken = {
        "articles": [
            {"url": "https://example.com/nodate", "title": "Undated", "seendate": "garbage"}
        ]
    }
    articles = gdelt.parse_artlist("iran", broken)
    published = datetime.fromisoformat(articles[0]["published_at"])
    age_hours = (datetime.now(timezone.utc) - published).total_seconds() / 3600
    assert 47 <= age_hours <= 49  # window start, never "now"


def test_watches_file_parsing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pythia.config import get_settings

    watches = tmp_path / "gdelt.txt"
    watches.write_text(
        "# comment\n"
        "iran | iran (nuclear OR sanctions)\n"
        "malformed line without pipe\n"
        "gulf | opec security\n"
    )
    monkeypatch.setenv("PYTHIA_GDELT_WATCHES_FILE", str(watches))
    assert get_settings().gdelt_watches == (
        ("iran", "iran (nuclear OR sanctions)"),
        ("gulf", "opec security"),
    )
