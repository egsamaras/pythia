# Limitations

Honest boundaries of the Pythia prototype — read before trusting any number
it produces.

## Forecast validity

- **An LLM is not a crystal ball.** Forecasts are generated from a single batch
  of recent headlines. The model has no market data, no polling, no base-rate
  database — its probabilities are educated guesses conditioned on limited text.
- **Quality tracks the backend model.** `llama3.1:8b` (local default) produces
  rough, sometimes generic forecasts; Claude Opus 5 is substantially better but
  still bounded by the input news. Compare backends before drawing conclusions.
- **Statement ambiguity.** The system prompts for falsifiable statements, but the
  model can still produce vague ones. Resolution is manual and honest resolution
  discipline is on the user; ambiguous statements should be resolved
  conservatively or reopened and ignored.
- **Calibration needs volume.** Brier scores and calibration curves are
  meaningless below a few dozen resolved forecasts. Expect noise for weeks —
  and the per-mode breakdown needs volume *per mode*.
- **"Precedent" base rates come from model memory, not a database.** When a
  precedent-mode forecast cites "6 of 8 past episodes", that tally is the
  model's recollection of history, not a verified dataset; models can
  misremember or invent tidy-sounding base rates. Treat the cited episodes as
  a claim to spot-check, and judge the mode by its measured Brier score, not
  by how authoritative its basis sounds.
- **"Consequence" forecasts compound uncertainty.** They stack an inference on
  top of reported events, so expect them to be the least calibrated mode —
  that is visible in the per-mode Brier breakdown, which is the point.
- **Resolution proposals are hints, not verdicts.** The judge sees a small
  news window plus keyword-matched archive articles — not the historical
  record. Guards force it toward "unresolved" (evidence-free NOs need the
  horizon well past; early NOs need cited impossibility; "happened" needs
  cited articles), so expect it to miss resolvable forecasts rather than
  invent outcomes. You are the resolver; it is the shortlist.
- **The judge grades its own forecaster.** Proposals come from the same model
  family that made the forecasts, a mild self-grading bias. The human
  confirmation step is the control; resolve skeptically.
- **Probability revisions change what the ledger measures.** Resolution checks
  auto-apply evidence-backed probability updates (disable with
  `PYTHIA_AUTO_UPDATE=0`); every change is kept in `probability_revisions`.
  To stop late updates from gaming the score, resolved forecasts are scored on
  their **time-weighted lifetime probability** — a revision counts only for
  the slice of the forecast's open life it was actually held, so the Brier
  ledger measures sustained judgment, not the final glance at the news.
  Guards additionally block updates after the horizon (hindsight), same-day
  re-updates, and updates citing no new evidence — but a drifting local model
  is still a risk worth watching in the revision histories.
- **The feedback loop is deliberately slow to speak.** Calibration feedback
  reaches the forecast prompt only after 30 resolutions, and only names biases
  exceeding twice their binomial standard error on 10+ samples — below that,
  "feedback" would just be noise taught as instruction.
- **Selection bias.** Forecasts are generated about whatever the feeds cover —
  headline-driven, English-language, recency-biased. Absence of a forecast is
  not evidence of stability.

## Data and pipeline

- **Feeds are best-effort.** Publishers change or drop RSS endpoints without
  notice; a failing feed is logged and skipped, which silently narrows coverage.
  Check the scan detail line for failed-feed counts.
- **News text is untrusted input.** Headlines flow into the model prompt; a
  hostile or broken feed could steer forecasts (prompt injection). Mitigations:
  structured-output schema, app-side validation, no tool use, and the dashboard
  renders all text inertly — but steering of forecast *content* by feed text is
  inherent to the design.
- **Timezones and horizons.** Horizon dates are calendar dates without timezone
  semantics; "by date X" resolution is user judgment.
- **No retention policy.** Articles and predictions accumulate indefinitely in
  SQLite; prune manually if it grows.

## Security and deployment

- Localhost, single-user, **no authentication** — never expose the port.
- With the Anthropic backend, headlines and summaries are sent to the Claude
  API. For a fully on-premise/air-gapped posture, force
  `PYTHIA_PROVIDER=ollama` (feed fetching still needs outbound HTTP).

## Monitoring triggers (when to distrust the system)

- Brier score drifting above 0.25 (worse than a coin flip) on ≥ 30 resolutions.
- Calibration curve consistently above/below the diagonal (systematic
  over/under-confidence).
- A large share of forecasts expiring unresolved (statements too vague to score).
