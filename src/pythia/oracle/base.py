"""Shared oracle contract: prompts, output schema, and draft validation.

Both backends (Anthropic API and local Ollama) receive the same prompts and
must return JSON matching FORECAST_JSON_SCHEMA; validation here is the single
gate between raw model output and the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

from pythia import dedup
from pythia.config import CATEGORIES, REGIONS

logger = logging.getLogger(__name__)

MIN_HORIZON_DAYS = 3
MAX_HORIZON_DAYS = 180

# The three forecasting methods. Order matters only for display.
MODES: tuple[str, ...] = ("event", "consequence", "precedent")

RESOLUTION_VERDICTS: tuple[str, ...] = ("happened", "did_not_happen", "unresolved", "update")

FORECAST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "forecasts": {
            "type": "array",
            "minItems": 3,
            "maxItems": 10,
            "items": {
                "type": "object",
                # Property order is load-bearing: constrained decoding emits
                # fields in schema order, and the probability must be committed
                # AFTER the model has written its basis and rationale —
                # reason-then-number, not number-then-justification.
                "properties": {
                    "statement": {"type": "string"},
                    "mode": {"type": "string", "enum": list(MODES)},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "region": {"type": "string", "enum": list(REGIONS)},
                    "horizon_date": {"type": "string"},
                    "basis": {"type": "string"},
                    "rationale": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                    "evidence_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 6,
                    },
                },
                "required": [
                    "statement",
                    "mode",
                    "category",
                    "region",
                    "horizon_date",
                    "basis",
                    "rationale",
                    "probability",
                    "evidence_indices",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["forecasts"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""You are Pythia, a calibrated geopolitical and economic \
forecasting system. From a batch of news headlines you produce probabilistic \
forecasts about concrete future events, using three distinct methods (the \
"mode" field):

- mode "event" — the news reports that something MAY happen; forecast whether \
it actually will (e.g. proposed sanctions actually imposed by the deadline). \
The forecast target is named by the articles themselves.
- mode "consequence" — forecast a DOWNSTREAM outcome the articles do not \
directly discuss: the market move, policy response, or corporate/state \
reaction that would follow from the reported developments. Put the causal \
chain in "basis" (e.g. "attacks on shipping -> rerouting -> container rates \
rise").
- mode "precedent" — reference-class forecasting: identify the closest \
historical analogues to the current situation and forecast what typically \
follows. The STATEMENT is still a future event ("OPEC cuts quotas at its \
September meeting"), never the base rate itself. Put the reference class, \
the historical episodes with their outcomes, and the resulting base rate in \
"basis" (e.g. "6 of 8 emergency OPEC meetings since 2008 ended in a quota \
cut"). Only cite historical episodes you are confident actually happened.

Mode tiebreak: if your primary evidence is a historical base rate from named \
episodes, the mode is "precedent" even when the statement is a downstream \
outcome; use "consequence" only when you are reasoning from the specifics of \
the current situation without a usable reference class.

Rules for every forecast:
- The statement must describe a FUTURE event — something that has not yet \
happened and could still fail to happen. Never state a historical fact or a \
base rate as the statement. It must be a single, concrete, externally \
verifiable event with a clear yes/no resolution by the horizon date (e.g. "The ECB cuts its deposit rate \
at its September meeting"), never a vague trend ("tensions will rise").
- Statements must be resolvable from public reporting alone.
- probability is your honest credence between 0.01 and 0.99. Always start \
from the base rate of the reference class, then adjust for current evidence. \
Most newsworthy speculation resolves NO — do not overshoot. Use the full \
range rather than clustering on 0.5, but reserve values above 0.90 or below \
0.10 for overwhelming evidence or strong historical regularity.
- horizon_date is an ISO date (YYYY-MM-DD) between {MIN_HORIZON_DAYS} and \
{MAX_HORIZON_DAYS} days from today, chosen so the event will clearly have or \
have not happened by then.
- category is one of: {", ".join(CATEGORIES)}.
- region is the region primarily affected, one of: {", ".join(REGIONS)}. \
Use "global" only when the consequences are clearly worldwide — major-power \
conflict, global markets or supply chains, pandemics — not merely newsworthy \
everywhere.
- rationale: 1-3 sentences naming the signals in the news.
- basis: may be empty for mode "event"; for "consequence" the causal chain \
(required); for "precedent" the reference class and its base rate (required).
- Write basis and rationale first and only then choose the probability: state \
the base rate, adjust for the current evidence, then commit the number.
- evidence_indices: the [n] indices of the articles that motivated the forecast.
- Do not duplicate any forecast in the "already open" list, and do not restate \
the same underlying claim in more than one mode.
- State each claim exactly once, in its positive form ("X will happen", with \
your probability). Never also emit its negation — "X will not happen" is the \
same forecast and wastes a slot.

Return only JSON matching the required schema."""


RESOLUTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                # Property order is load-bearing (constrained decoding):
                # justification and evidence come BEFORE the verdict, and the
                # revised probability comes LAST so the number conditions on
                # everything stated before it.
                "properties": {
                    "forecast_index": {"type": "integer"},
                    "justification": {"type": "string"},
                    "evidence_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 6,
                    },
                    "verdict": {"type": "string", "enum": list(RESOLUTION_VERDICTS)},
                    "new_probability": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                },
                "required": [
                    "forecast_index",
                    "justification",
                    "evidence_indices",
                    "verdict",
                    "new_probability",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["resolutions"],
    "additionalProperties": False,
}

RESOLUTION_SYSTEM_PROMPT = """You are Pythia's resolution judge. You receive \
open probabilistic forecasts and a batch of recent news articles, and you \
decide which forecasts can now be resolved. You only PROPOSE resolutions — a \
human confirms them — but propose only what the evidence supports.

The articles you receive are a SMALL SAMPLE of world coverage, not an \
archive. Absence of an event from this batch is NOT evidence it did not \
happen.

Verdict rules:
- "happened": only when the articles provide clear evidence that the stated \
event occurred. Cite the articles in evidence_indices.
- "did_not_happen": only for forecasts whose horizon date has passed, and \
only when at least one of these holds: (a) the event was tied to a specific \
scheduled occasion or deadline whose contrary outcome is reported in the \
articles or certain from your own knowledge; (b) the event has become \
logically impossible; or (c) the event is so consequential that any sample \
of world coverage from after the date would reflect it (a war, a state \
collapse, a major institution dissolving) and this one does not.
- "unresolved": everything else. Insufficient evidence, ambiguity about what \
the statement means, or an ordinary event that may simply have missed this \
batch. This is the correct default — a wrong proposal pollutes the \
calibration ledger; an unresolved one merely waits.
- "update": the forecast should STAY OPEN, but the cited articles materially \
change the odds relative to the current probability shown for it. Give your \
revised probability in new_probability. Use it only for genuine evidence \
shifts of at least 5 points, always with cited articles — never to restate \
the current number or to drift on general mood. Never update a forecast whose \
horizon has passed: resolve it or leave it unresolved. Base an update only on \
information published AFTER the forecast was made — its creation reasoning is \
shown to you and is already priced into the current probability.
- A forecast whose horizon has NOT passed can be resolved "happened" early, \
but "did_not_happen" only via impossibility (cite the deciding article).
- Each forecast shows the date it was made. Only events occurring after that \
date can resolve it "happened" — reports about earlier events do not count.
- justification: 1-2 sentences written BEFORE the verdict, naming the \
evidence (or its absence) that decides it.
- new_probability: your revised probability for verdict "update"; for every \
other verdict simply repeat the forecast's current probability.

Return only JSON matching the required schema, with one entry per forecast \
you were given (use "unresolved" for those you cannot decide)."""


def build_resolution_prompt(
    forecasts: list[dict[str, Any]], articles: list[dict[str, Any]], today: date
) -> str:
    """Assemble the resolution-check prompt: forecasts and evidence, indexed."""
    lines: list[str] = [f"Today's date: {today.isoformat()}", "", "Open forecasts:"]
    today_iso = today.isoformat()
    for i, fc in enumerate(forecasts):
        overdue = " (horizon passed)" if fc["horizon_date"] < today_iso else ""
        made = f"made {fc['created_at'][:10]}, " if fc.get("created_at") else ""
        current = (
            f", current p={fc['probability']:.2f}" if fc.get("probability") is not None else ""
        )
        lines.append(
            f"[F{i}] ({made}by {fc['horizon_date']}{overdue}{current}) {fc['statement']}"
        )
        reasoning = " ".join(
            part for part in (fc.get("rationale", ""), fc.get("basis", "")) if part
        ).strip()
        if reasoning:
            lines.append(f"      reasoning at creation (already priced in): {reasoning[:200]}")
    lines.append("")
    lines.append("Recent news articles:")
    for i, a in enumerate(articles):
        summary = f" — {a['summary']}" if a.get("summary") else ""
        lines.append(f"[{i}] ({a['source']}, {a['published_at'][:10]}) {a['title']}{summary}")
    lines.append("")
    lines.append(
        "For each forecast F0..F"
        + str(len(forecasts) - 1)
        + ", give your justification and verdict per the rules."
    )
    return "\n".join(lines)


def build_user_prompt(
    articles: list[dict[str, Any]],
    today: date,
    open_statements: list[str],
    track_record: str | None = None,
    signals: str | None = None,
) -> str:
    """Assemble the per-scan prompt: date, track record, signals, forecasts, articles."""
    lines: list[str] = [f"Today's date: {today.isoformat()}", ""]
    if track_record:
        lines.append("Your measured calibration record (learn from it):")
        lines.append(track_record)
        lines.append("")
    if signals:
        lines.append("Structured signals (machine-computed):")
        lines.append(signals)
        lines.append("")
    if open_statements:
        lines.append("Already open forecasts (do NOT duplicate):")
        lines.extend(f"- {s}" for s in open_statements)
        lines.append("")
    lines.append("News articles:")
    for i, a in enumerate(articles):
        summary = f" — {a['summary']}" if a.get("summary") else ""
        lines.append(f"[{i}] ({a['source']}, {a['published_at'][:10]}) {a['title']}{summary}")
    lines.append("")
    lines.append(
        "Produce 6-9 forecasts grounded in these articles: at least two in each "
        "mode (event, consequence, precedent), spread across categories where "
        "the news supports it. The precedent forecasts are mandatory — pick the "
        "two stories with the clearest historical analogues and name the "
        "episodes and base rate in basis."
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class ForecastDraft:
    """A validated forecast candidate, not yet persisted."""

    statement: str
    probability: float
    horizon_date: str
    category: str
    rationale: str
    evidence_indices: list[int]
    mode: str = "event"
    basis: str = ""
    region: str = "global"


@dataclass(frozen=True)
class ResolutionDraft:
    """A validated resolution proposal or probability update from the oracle."""

    forecast_index: int
    verdict: str
    justification: str
    evidence_indices: list[int]
    new_probability: float | None = None  # set only for verdict "update"


class OracleError(RuntimeError):
    """Raised when a backend cannot produce usable forecasts."""


class Oracle(Protocol):
    """A forecast generator and resolution-judge backend."""

    provider: str
    model: str

    def generate(
        self,
        articles: list[dict[str, Any]],
        today: date,
        open_statements: list[str],
        track_record: str | None = None,
        signals: str | None = None,
    ) -> list[ForecastDraft]:
        """Return validated forecast drafts for the given articles."""
        ...

    def resolve(
        self, forecasts: list[dict[str, Any]], articles: list[dict[str, Any]], today: date
    ) -> list[ResolutionDraft]:
        """Return validated resolution proposals for the given open forecasts."""
        ...


def validate_resolution_drafts(
    raw: Any, n_forecasts: int, n_articles: int
) -> list[ResolutionDraft]:
    """Validate raw resolution JSON, dropping malformed entries.

    Duplicate forecast_index entries keep the first occurrence. Unlike
    forecast validation, an empty result is fine — "nothing resolvable" is a
    legitimate outcome, so no error is raised.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("resolutions"), list):
        raise OracleError("model output is not a {'resolutions': [...]} object")
    drafts: list[ResolutionDraft] = []
    seen: set[int] = set()
    for i, r in enumerate(raw["resolutions"]):
        try:
            idx = int(r["forecast_index"])
            verdict = str(r["verdict"]).strip().lower()
            justification = str(r.get("justification", "")).strip()
            evidence = [
                int(e) for e in r.get("evidence_indices", []) if 0 <= int(e) < n_articles
            ][:6]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("dropping resolution %d: malformed (%s)", i, exc)
            continue
        if not 0 <= idx < n_forecasts:
            logger.warning("dropping resolution %d: forecast_index %d out of range", i, idx)
            continue
        if verdict not in RESOLUTION_VERDICTS:
            logger.warning("dropping resolution %d: unknown verdict %r", i, verdict)
            continue
        if idx in seen:
            logger.warning("dropping resolution %d: duplicate forecast_index %d", i, idx)
            continue
        if verdict == "happened" and not evidence:
            # A positive resolution with no cited evidence is unauditable.
            logger.warning("dropping resolution %d: 'happened' without evidence", i)
            continue
        new_probability: float | None = None
        if verdict == "update":
            if not evidence:
                logger.warning("dropping resolution %d: 'update' without evidence", i)
                continue
            try:
                new_probability = float(r["new_probability"])
            except (KeyError, TypeError, ValueError):
                logger.warning("dropping resolution %d: 'update' without a number", i)
                continue
            if not 0.0 < new_probability < 1.0:
                logger.warning(
                    "dropping resolution %d: new_probability %s out of range",
                    i,
                    new_probability,
                )
                continue
            new_probability = round(min(max(new_probability, 0.01), 0.99), 3)
        seen.add(idx)
        drafts.append(
            ResolutionDraft(
                forecast_index=idx,
                verdict=verdict,
                justification=justification,
                evidence_indices=evidence,
                new_probability=new_probability,
            )
        )
    return drafts


def validate_drafts(raw: Any, n_articles: int, today: date) -> list[ForecastDraft]:
    """Validate raw model JSON into ForecastDrafts, dropping bad entries.

    Fails loudly (OracleError) only when nothing valid remains; individual bad
    forecasts are logged and skipped so one malformed entry does not sink a scan.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("forecasts"), list):
        raise OracleError("model output is not a {'forecasts': [...]} object")

    min_horizon = today + timedelta(days=1)
    # Deliberate slack: accept up to 2x the prompted maximum — models
    # occasionally overshoot the horizon, and a long-dated forecast is still
    # scoreable; anything beyond that is treated as malformed.
    max_horizon = today + timedelta(days=MAX_HORIZON_DAYS * 2)
    drafts: list[ForecastDraft] = []
    for i, f in enumerate(raw["forecasts"]):
        try:
            statement = str(f["statement"]).strip()
            probability = float(f["probability"])
            horizon = date.fromisoformat(str(f["horizon_date"]).strip()[:10])
            category = str(f["category"]).strip().lower()
            rationale = str(f.get("rationale", "")).strip()
            mode = str(f.get("mode", "event")).strip().lower()
            basis = str(f.get("basis", "")).strip()
            region = str(f.get("region", "global")).strip().lower()
            evidence = [
                int(e) for e in f.get("evidence_indices", []) if 0 <= int(e) < n_articles
            ][:6]  # schema cap, enforced here for backends that ignore maxItems
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("dropping forecast %d: malformed (%s)", i, exc)
            continue
        if not statement:
            logger.warning("dropping forecast %d: empty statement", i)
            continue
        if not 0.0 < probability < 1.0:
            logger.warning("dropping forecast %d: probability %s out of range", i, probability)
            continue
        if not 0.01 <= probability <= 0.99:
            # A near-certain credence is kept but bounded to the schema range.
            logger.warning("clamping forecast %d: probability %s", i, probability)
            probability = min(max(probability, 0.01), 0.99)
        if category not in CATEGORIES:
            logger.warning("dropping forecast %d: unknown category %r", i, category)
            continue
        if mode not in MODES:
            logger.warning("dropping forecast %d: unknown mode %r", i, mode)
            continue
        if region not in REGIONS:
            logger.warning("dropping forecast %d: unknown region %r", i, region)
            continue
        if not dedup.is_future_claim(statement):
            # A base rate ("6 of 8 past meetings ended in a cut") is not a
            # forecast: it is unresolvable and belongs in `basis`.
            logger.warning(
                "dropping forecast %d: statement is not a future claim (%r)",
                i, statement[:70],
            )
            continue
        if mode != "event" and not basis:
            # basis is the audit trail for these modes; without it the mode
            # label is unverifiable and would pollute the per-mode Brier.
            logger.warning("dropping forecast %d: mode %r with empty basis", i, mode)
            continue
        if not min_horizon <= horizon <= max_horizon:
            logger.warning("dropping forecast %d: horizon %s out of range", i, horizon)
            continue
        drafts.append(
            ForecastDraft(
                statement=statement,
                probability=round(probability, 3),
                horizon_date=horizon.isoformat(),
                category=category,
                rationale=rationale,
                evidence_indices=evidence,
                mode=mode,
                basis=basis,
                region=region,
            )
        )
    if not drafts:
        raise OracleError("model returned no valid forecasts")
    return drafts
