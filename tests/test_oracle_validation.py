"""Validation gate between raw model JSON and the database."""

from datetime import date

import pytest

from pythia.oracle.base import (
    FORECAST_JSON_SCHEMA,
    MODES,
    RESOLUTION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    OracleError,
    build_resolution_prompt,
    build_user_prompt,
    validate_drafts,
    validate_resolution_drafts,
)

TODAY = date(2026, 8, 21)


def _forecast(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "statement": "The ECB cuts rates at its September meeting",
        "mode": "event",
        "probability": 0.35,
        "horizon_date": "2026-09-15",
        "category": "economy",
        "rationale": "Base rate of cuts plus dovish signals.",
        "basis": "",
        "evidence_indices": [0, 2],
    }
    base.update(overrides)
    return base


def test_valid_forecast_passes() -> None:
    drafts = validate_drafts({"forecasts": [_forecast()]}, n_articles=3, today=TODAY)
    assert len(drafts) == 1
    d = drafts[0]
    assert d.probability == 0.35
    assert d.evidence_indices == [0, 2]


def test_out_of_range_evidence_indices_are_dropped_not_fatal() -> None:
    drafts = validate_drafts(
        {"forecasts": [_forecast(evidence_indices=[0, 99, -1])]}, n_articles=3, today=TODAY
    )
    assert drafts[0].evidence_indices == [0]


@pytest.mark.parametrize(
    "bad",
    [
        _forecast(probability=1.5),
        _forecast(probability=0.0),
        _forecast(horizon_date="2026-08-21"),  # not in the future
        _forecast(horizon_date="not-a-date"),
        _forecast(category="astrology"),
        _forecast(statement="   "),
        _forecast(mode="prophecy"),
        _forecast(mode="consequence", basis=""),  # non-event needs an audit trail
        _forecast(mode="precedent", basis="   "),
    ],
)
def test_invalid_forecast_is_skipped(bad: dict[str, object]) -> None:
    raw = {"forecasts": [bad, _forecast()]}
    drafts = validate_drafts(raw, n_articles=3, today=TODAY)
    assert len(drafts) == 1


def test_all_invalid_raises() -> None:
    with pytest.raises(OracleError):
        validate_drafts({"forecasts": [_forecast(probability=5)]}, n_articles=3, today=TODAY)


def test_non_object_raises() -> None:
    with pytest.raises(OracleError):
        validate_drafts(["not", "a", "dict"], n_articles=3, today=TODAY)


def test_base_rate_statement_is_dropped() -> None:
    # The model's favourite failure: putting the basis in the statement field.
    raw = {
        "forecasts": [
            _forecast(
                statement="6 of 8 emergency OPEC meetings since 2008 ended in a quota cut.",
                mode="precedent",
                basis="Reference class: 8 emergency OPEC meetings since 2008.",
            ),
            _forecast(),
        ]
    }
    drafts = validate_drafts(raw, n_articles=3, today=TODAY)
    assert len(drafts) == 1
    assert drafts[0].statement.startswith("The ECB")


def test_precedent_prompt_demands_a_future_statement() -> None:
    assert "The STATEMENT is still a future event" in SYSTEM_PROMPT
    assert "Never state a historical fact or a base rate as the statement" in SYSTEM_PROMPT


def test_region_captured_defaulted_and_validated() -> None:
    tagged = _forecast(region="mena")
    untagged = _forecast()
    bogus = _forecast(region="atlantis")
    drafts = validate_drafts(
        {"forecasts": [tagged, untagged, bogus]}, n_articles=3, today=TODAY
    )
    assert [d.region for d in drafts] == ["mena", "global"]  # bogus dropped


def test_schema_and_prompt_carry_regions() -> None:
    from pythia.config import REGIONS

    props = FORECAST_JSON_SCHEMA["properties"]["forecasts"]["items"]["properties"]
    assert props["region"]["enum"] == list(REGIONS)
    assert '"global" only when the consequences are clearly worldwide' in SYSTEM_PROMPT


def test_mode_and_basis_are_captured() -> None:
    raw = {
        "forecasts": [
            _forecast(
                mode="precedent",
                basis="6 of 8 emergency OPEC meetings since 2008 ended in a cut",
            )
        ]
    }
    drafts = validate_drafts(raw, n_articles=3, today=TODAY)
    assert drafts[0].mode == "precedent"
    assert drafts[0].basis.startswith("6 of 8")


def test_missing_mode_defaults_to_event() -> None:
    f = _forecast()
    del f["mode"], f["basis"]
    drafts = validate_drafts({"forecasts": [f]}, n_articles=3, today=TODAY)
    assert drafts[0].mode == "event"
    assert drafts[0].basis == ""


def test_system_prompt_defines_all_modes() -> None:
    for mode in MODES:
        assert f'mode "{mode}"' in SYSTEM_PROMPT


def test_schema_orders_reasoning_before_probability() -> None:
    """Constrained decoding emits fields in schema order; the probability must
    come after basis and rationale so the number conditions on the reasoning."""
    props = list(FORECAST_JSON_SCHEMA["properties"]["forecasts"]["items"]["properties"])
    assert props.index("basis") < props.index("probability")
    assert props.index("rationale") < props.index("probability")


def test_near_certain_probability_is_clamped_not_dropped() -> None:
    drafts = validate_drafts(
        {"forecasts": [_forecast(probability=0.999)]}, n_articles=3, today=TODAY
    )
    assert drafts[0].probability == 0.99


def test_evidence_indices_capped_at_six() -> None:
    drafts = validate_drafts(
        {"forecasts": [_forecast(evidence_indices=list(range(10)))]},
        n_articles=10,
        today=TODAY,
    )
    assert drafts[0].evidence_indices == [0, 1, 2, 3, 4, 5]


def test_horizon_slack_boundary() -> None:
    accepted = _forecast(horizon_date="2027-08-16")  # today + 360d: inside 2x slack
    rejected = _forecast(horizon_date="2027-08-17")  # today + 361d: beyond it
    drafts = validate_drafts(
        {"forecasts": [accepted, rejected]}, n_articles=3, today=TODAY
    )
    assert len(drafts) == 1
    assert drafts[0].horizon_date == "2027-08-16"


def _resolution(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "forecast_index": 0,
        "justification": "The wire reported the event on the 20th.",
        "evidence_indices": [1],
        "verdict": "happened",
    }
    base.update(overrides)
    return base


def test_resolution_drafts_valid_and_unresolved_kept() -> None:
    raw = {"resolutions": [_resolution(), _resolution(forecast_index=1, verdict="unresolved")]}
    drafts = validate_resolution_drafts(raw, n_forecasts=2, n_articles=3)
    assert [d.verdict for d in drafts] == ["happened", "unresolved"]


@pytest.mark.parametrize(
    "bad",
    [
        _resolution(verdict="maybe"),
        _resolution(forecast_index=5),
        _resolution(forecast_index=-1),
        _resolution(evidence_indices=[]),  # 'happened' needs cited evidence
        _resolution(evidence_indices=[99]),  # out-of-range citations stripped -> empty
    ],
)
def test_invalid_resolution_is_skipped(bad: dict[str, object]) -> None:
    raw = {"resolutions": [bad, _resolution(forecast_index=1)]}
    drafts = validate_resolution_drafts(raw, n_forecasts=2, n_articles=3)
    assert len(drafts) == 1
    assert drafts[0].forecast_index == 1


def test_update_verdict_carries_clamped_probability() -> None:
    raw = {"resolutions": [_resolution(verdict="update", new_probability=0.995)]}
    drafts = validate_resolution_drafts(raw, n_forecasts=1, n_articles=3)
    assert drafts[0].verdict == "update"
    assert drafts[0].new_probability == 0.99


def test_update_without_evidence_or_number_is_dropped() -> None:
    no_evidence = _resolution(verdict="update", new_probability=0.7, evidence_indices=[])
    no_number = _resolution(verdict="update")
    raw = {"resolutions": [no_evidence, no_number, _resolution(forecast_index=1)]}
    drafts = validate_resolution_drafts(raw, n_forecasts=2, n_articles=3)
    assert len(drafts) == 1
    assert drafts[0].verdict == "happened"


def test_non_update_verdicts_ignore_the_number() -> None:
    raw = {"resolutions": [_resolution(new_probability=0.5)]}
    drafts = validate_resolution_drafts(raw, n_forecasts=1, n_articles=3)
    assert drafts[0].new_probability is None


def test_duplicate_resolution_index_keeps_first() -> None:
    raw = {
        "resolutions": [
            _resolution(justification="first"),
            _resolution(justification="second", verdict="did_not_happen"),
        ]
    }
    drafts = validate_resolution_drafts(raw, n_forecasts=1, n_articles=3)
    assert len(drafts) == 1
    assert drafts[0].justification == "first"


def test_empty_resolutions_is_not_an_error() -> None:
    assert validate_resolution_drafts({"resolutions": []}, 3, 3) == []


def test_resolution_non_object_raises() -> None:
    with pytest.raises(OracleError):
        validate_resolution_drafts("nope", 1, 1)


def test_resolution_prompt_guards_against_absence_of_evidence() -> None:
    from pythia.oracle.base import RESOLUTION_SYSTEM_PROMPT

    assert "SMALL SAMPLE" in RESOLUTION_SYSTEM_PROMPT
    assert "NOT evidence it did not happen" in RESOLUTION_SYSTEM_PROMPT


def test_resolution_schema_orders_reasoning_before_verdict() -> None:
    props = list(RESOLUTION_JSON_SCHEMA["properties"]["resolutions"]["items"]["properties"])
    assert props.index("justification") < props.index("verdict")
    assert props.index("evidence_indices") < props.index("verdict")
    assert props.index("verdict") < props.index("new_probability")  # number last


def test_resolution_prompt_marks_overdue_and_current_probability() -> None:
    forecasts = [
        {"statement": "Overdue thing", "horizon_date": "2026-08-01", "probability": 0.35},
        {"statement": "Future thing", "horizon_date": "2026-12-01"},
    ]
    articles = [
        {"source": "Wire", "title": "H", "summary": "", "published_at": "2026-08-20T00:00:00"}
    ]
    prompt = build_resolution_prompt(forecasts, articles, TODAY)
    assert "[F0] (by 2026-08-01 (horizon passed), current p=0.35) Overdue thing" in prompt
    assert "[F1] (by 2026-12-01) Future thing" in prompt
    assert "[0]" in prompt and "2026-08-21" in prompt


def test_forecast_prompt_includes_track_record_when_given() -> None:
    articles = [
        {"source": "Wire", "title": "H", "summary": "", "published_at": "2026-08-20T00:00:00"}
    ]
    prompt = build_user_prompt(articles, TODAY, [], track_record="Brier 0.31; overconfident.")
    assert "Your measured calibration record" in prompt
    assert "Brier 0.31; overconfident." in prompt
    without = build_user_prompt(articles, TODAY, [])
    assert "calibration record" not in without


def test_forecast_prompt_includes_structured_signals_when_given() -> None:
    articles = [
        {"source": "Wire", "title": "H", "summary": "", "published_at": "2026-08-20T00:00:00"}
    ]
    prompt = build_user_prompt(articles, TODAY, [], signals="- iran: 3.0x (rising)")
    assert "Structured signals (machine-computed):" in prompt
    assert "- iran: 3.0x (rising)" in prompt
    assert "Structured signals" not in build_user_prompt(articles, TODAY, [])


def test_prompt_contains_indexed_articles_and_open_statements() -> None:
    articles = [
        {"source": "Wire", "title": "Headline A", "summary": "S", "published_at": "2026-08-20T00:00:00"},
        {"source": "Wire", "title": "Headline B", "summary": "", "published_at": "2026-08-20T00:00:00"},
    ]
    prompt = build_user_prompt(articles, TODAY, ["Existing forecast"])
    assert "[0]" in prompt and "Headline A" in prompt
    assert "[1]" in prompt and "Headline B" in prompt
    assert "Existing forecast" in prompt
    assert "2026-08-21" in prompt
    assert "at least two in each" in prompt  # demands the three-mode mix
    assert "precedent forecasts are mandatory" in prompt


def test_system_prompt_forbids_negation_pairs() -> None:
    assert "Never also emit its negation" in SYSTEM_PROMPT
