"""Statement hygiene: near-duplicate and negation-pair detection.

The forecast prompt forbids duplicates, negation pairs, and base-rate
statements, but a mid-size local model ignores instructions it finds
inconvenient. These pure functions are the enforcement layer: they run at
persist time so a contradictory pair can never reach the ledger.

Why it matters beyond tidiness: scoring both X and NOT-X guarantees one hit
and one miss whatever the model's skill, manufacturing a ~0.25 Brier score
out of noise — precisely the self-measurement the system exists to provide.
"""

from __future__ import annotations

import re

# Words carrying no claim content; dropped before comparison so phrasing
# differences ("the EU" vs "EU") do not hide a duplicate.
_FILLER = frozenset(
    """a an the of to in on at by for from with its their this that these those
    is are be been being as into over under out up down and or but if then than
    new will would shall may might can could does do did has have had""".split()
)
_NEGATIONS = frozenset({"not", "never", "no", "cannot", "wont", "nor", "without"})

# A statement must be a claim, not a recitation of history. These patterns
# match base-rate and retrospective shapes only — the rule REJECTS known-bad
# forms rather than requiring known-good ones, because legitimate forecasts
# are phrased too freely to whitelist ("The ECB cuts its deposit rate at its
# September meeting" carries no "will" and no deadline word).
_BASE_RATE = re.compile(
    r"(^\s*(in\s+)?\d+(\.\d+)?\s*%"          # "75% of ..."
    r"|\b\d+\s+(of|out\s+of)\s+(the\s+)?(last\s+)?\d+\b"  # "6 of 8 ..."
    r"|^\s*(historically|in the past|traditionally)\b)",
    re.IGNORECASE,
)
# Retrospective framing: a past-year reference reported in the past tense.
_RETROSPECTIVE = re.compile(
    r"\b(since|in)\s+(19|20)\d{2}\b.{0,60}?"
    r"\b(ended|were|was|occurred|happened|resulted|led|proved|closed|fell|rose)\b",
    re.IGNORECASE,
)

_DUPLICATE_SIMILARITY = 0.80


def _stem(word: str) -> str:
    """Crude suffix stripping so "imposes"/"impose" compare equal.

    Length guards keep short tokens intact — "US" must never become "U".
    """
    if len(word) >= 5 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]  # imposes -> impose, sanctions -> sanction
    if len(word) >= 7 and word.endswith("ing"):
        return word[:-3]
    if len(word) >= 6 and word.endswith("ed"):
        return word[:-2]
    return word


def _tokens(statement: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", statement.lower())
    return [_stem(w) for w in words if w not in _FILLER]


def content_key(statement: str) -> frozenset[str]:
    """Comparable content of a statement, with negation words removed."""
    return frozenset(t for t in _tokens(statement) if t not in _NEGATIONS)


def is_negated(statement: str) -> bool:
    """True when the statement asserts that something will NOT happen."""
    lowered = statement.lower()
    if re.search(r"\b(won't|doesn't|isn't|aren't|hasn't|haven't)\b", lowered):
        return True
    return any(t in _NEGATIONS for t in _tokens(statement))


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of two statements' content, ignoring negation."""
    ka, kb = content_key(a), content_key(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def is_future_claim(statement: str) -> bool:
    """False when the statement recites history instead of claiming a future.

    Catches base rates ("6 of 8 past meetings ended in a cut") and
    retrospectives, which belong in a forecast's `basis`, never in its
    statement — they are unresolvable and would sit in the ledger forever.
    Deliberately permissive otherwise: rejecting a real forecast costs more
    than letting an oddly-phrased one through, since the horizon date and
    the resolution flow still have to make sense of it.
    """
    return not (_BASE_RATE.search(statement) or _RETROSPECTIVE.search(statement))


def duplicate_of(statement: str, existing: list[str]) -> tuple[str, str] | None:
    """First (kind, matched statement) this duplicates or contradicts.

    ``kind`` is "duplicate" for a restatement of the same claim, or
    "negation" for the same claim asserted the other way — both are refused,
    because a ledger holding X and NOT-X scores itself into meaninglessness.
    """
    negated = is_negated(statement)
    for other in existing:
        if similarity(statement, other) < _DUPLICATE_SIMILARITY:
            continue
        return ("negation" if is_negated(other) != negated else "duplicate", other)
    return None
