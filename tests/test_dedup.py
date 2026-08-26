"""Statement hygiene: base-rate rejection and duplicate/negation detection."""

from __future__ import annotations

import pytest

from pythia import dedup


@pytest.mark.parametrize(
    "statement",
    [
        "6 of 8 emergency OPEC meetings since 2008 ended in a quota cut.",
        "In 6 of the last 8 emergency meetings, OPEC cut quotas.",
        "75% of past summits produced a joint communique.",
        "Historically, OPEC cuts quotas after emergency meetings.",
        "Since 2015 the council was deadlocked on every vote.",
    ],
)
def test_base_rates_and_retrospectives_are_not_forecasts(statement: str) -> None:
    assert dedup.is_future_claim(statement) is False


@pytest.mark.parametrize(
    "statement",
    [
        # No "will", no deadline word — still a perfectly good forecast.
        "The ECB cuts its deposit rate at its September meeting.",
        "The EU imposes new sanctions on Iran by September 15, 2026.",
        "France will deliver the interceptor missiles to Ukraine by September 15, 2026.",
        "Funding costs rise after the reported stress.",
        "OPEC announces increased production quotas at its September meeting.",
    ],
)
def test_real_forecasts_survive(statement: str) -> None:
    assert dedup.is_future_claim(statement) is True


def test_negation_detection() -> None:
    assert dedup.is_negated("The US will not extend New START beyond February.") is True
    assert dedup.is_negated("The US won't extend New START beyond February.") is True
    assert dedup.is_negated("The US will extend New START beyond February.") is False


def test_stemmer_leaves_short_tokens_intact() -> None:
    # "US" must never be stemmed to "U" — it is a load-bearing token.
    assert "us" in dedup.content_key("The US will impose tariffs")
    assert dedup._stem("imposes") == dedup._stem("impose")
    assert dedup._stem("sanctions") == "sanction"


def test_duplicate_of_catches_restatements_and_contradictions() -> None:
    open_now = ["The US will impose new sanctions on China by December 31, 2026."]

    exact = dedup.duplicate_of(
        "The US will impose new sanctions on China by December 31, 2026.", open_now
    )
    assert exact is not None and exact[0] == "duplicate"

    # Same claim, different conjugation.
    conjugated = dedup.duplicate_of(
        "The US imposes new sanctions on China by December 31, 2026.", open_now
    )
    assert conjugated is not None and conjugated[0] == "duplicate"

    # The same claim asserted the other way: scoring both guarantees one hit
    # and one miss whatever the model's skill.
    negated = dedup.duplicate_of(
        "The US will not impose new sanctions on China by December 31, 2026.", open_now
    )
    assert negated is not None and negated[0] == "negation"


@pytest.mark.parametrize(
    "statement",
    [
        # Different horizon: a genuinely different question.
        "The US will impose new sanctions on China by March 31, 2027.",
        # Different actor.
        "The EU will impose new sanctions on China by December 31, 2026.",
        # Unrelated.
        "France will deliver interceptor missiles to Ukraine by September 15, 2026.",
    ],
)
def test_distinct_forecasts_are_not_refused(statement: str) -> None:
    open_now = ["The US will impose new sanctions on China by December 31, 2026."]
    assert dedup.duplicate_of(statement, open_now) is None


def test_duplicate_of_empty_history() -> None:
    assert dedup.duplicate_of("Anything at all happens by December.", []) is None
