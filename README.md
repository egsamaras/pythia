# Pythia

A news-driven probabilistic forecasting system. Pythia scans a configurable
set of RSS feeds, asks a language model to turn the day's news into **concrete,
falsifiable forecasts** (statement, probability, horizon date, category, evidence),
and keeps a **resolution ledger** so the system's calibration is measurable over
time (Brier score, calibration curve).

> Forecasts are LLM-generated probabilistic estimates for research and
> experimentation. They are not advice — financial, investment, or otherwise.

## Architecture

```
RSS feeds ──> feeds.py ──> SQLite (articles)
                              │
                              ▼
                        pipeline.py ──> oracle (Anthropic Claude API │ local Ollama)
                              │              structured JSON forecasts
                              ▼
                        SQLite (predictions ledger)
                              │
                              ▼
                        FastAPI ──> static dashboard (forecasts, resolution, calibration)
```

## Setup

```bash
cd pythia
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m uvicorn --app-dir src pythia.server:app --host 127.0.0.1 --port 8792
```

Open http://127.0.0.1:8792 and press **Run scan**.

The server binds to localhost and has **no authentication** — do not expose it.

## Oracle backends

| Backend | When used | Model (default) |
|---|---|---|
| Anthropic Claude API | `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` present (or forced) | `claude-opus-5` |
| Local Ollama | No Anthropic credentials and Ollama reachable (or forced) | `llama3.1:8b` |

Auto-detection prefers the Claude API when credentials exist; otherwise it falls
back to local Ollama — fully on-premise, no data leaves the machine. Forecast
quality tracks the model: a small local model produces noticeably rougher
forecasts than Claude Opus 5.

On the Anthropic backend, server-side **refusal fallbacks** are enabled by
default (`fallbacks="default"`): if the primary model declines a request for
safety reasons, the API retries it on a fallback model within the same call.
Set `PYTHIA_ANTHROPIC_FALLBACKS=0` to disable.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PYTHIA_PROVIDER` | `auto` | `auto` \| `anthropic` \| `ollama` |
| `PYTHIA_ANTHROPIC_MODEL` | `claude-opus-5` | Claude model id |
| `PYTHIA_ANTHROPIC_FALLBACKS` | `1` | server-side refusal fallbacks |
| `PYTHIA_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `PYTHIA_OLLAMA_MODEL` | `llama3.1:8b` | local model |
| `PYTHIA_DB` | `~/Library/Application Support/Pythia/pythia.db` | SQLite path |
| `PYTHIA_MAX_RESOLUTION_FORECASTS` | `20` | open forecasts per resolution check |
| `PYTHIA_AUTO_UPDATE` | `1` | oracle probability updates on open forecasts |
| `PYTHIA_FEEDS` | — | comma-separated RSS URLs (overrides the feeds file) |
| `PYTHIA_FEEDS_FILE` | `feeds.txt` | path to the feeds file |
| `PYTHIA_FEED_TIMEOUT_S` | `15` | per-feed HTTP timeout |
| `PYTHIA_ORACLE_TIMEOUT_S` | `1800` | model request timeout (large local models are slow) |
| `PYTHIA_MAX_ARTICLES_ANTHROPIC` | `80` | article batch cap (Claude) |
| `PYTHIA_MAX_ARTICLES_OLLAMA` | `30` | article batch cap (Ollama) |

## News sources

The active source list lives in [feeds.txt](feeds.txt) — one URL per line,
`#` for comments; edit it and restart the server. It ships with ~24 verified
feeds: global news (BBC, Guardian, NYT, WSJ, Sky, France 24, DW), business &
markets (MarketWatch, CNBC, FT, The Economist), Middle East / GCC (Arab News,
The National, Middle East Eye), central banks (Fed, ECB, Bank of England), and
**Google News query feeds** — topic-targeted searches (GCC banking, Gulf
regulators, OPEC, AI regulation) that also surface Reuters/Bloomberg stories,
which have no public RSS of their own.

Feeds are fetched concurrently, capped at 25 entries each per fetch, and the
analysis batch is balanced round-robin across sources so one prolific feed
cannot crowd out the rest. Precedence: `PYTHIA_FEEDS` env var >
`PYTHIA_FEEDS_FILE` > `feeds.txt` in the project root > a built-in minimal list.

## The forecasting loop

1. **Scan** — fetch feeds, deduplicate articles by URL, send a source-balanced
   batch to the oracle with the list of already-open forecasts (to avoid
   duplicates).
2. **Forecasts** — each has a falsifiable statement, a probability (0.01–0.99),
   a horizon date, a category, a rationale, and links to the motivating
   articles. Every forecast uses one of three **modes**:
   - **event** — the news says something may happen; will it actually happen?
   - **consequence** — a downstream outcome the articles don't discuss (market
     move, policy response), with the causal chain stated in *basis*.
   - **precedent** — reference-class forecasting: the closest historical
     analogues and their base rate, stated in *basis*
     ("6 of 8 emergency OPEC meetings since 2008 ended in a quota cut").
3. **Resolve** — when a horizon passes ("due" badge), mark the outcome:
   *Happened* / *Did not happen*. Or press **Check resolutions**: the oracle
   re-reads the news (plus archived coverage matched to overdue forecasts)
   and *proposes* resolutions with cited evidence — you accept or dismiss
   each. Proposals are deliberately conservative: an event's absence from a
   small news batch is never treated as proof it didn't happen, and a
   dismissed proposal is not re-raised unless the oracle changes its verdict.
4. **Update** — the self-learning layer. Resolution checks (including the
   per-forecast **Check now** button) can revise an open forecast's
   probability when cited new evidence shifts the odds; every revision is
   kept in an audit history and shown on the card. Once 30 forecasts have
   resolved, the system's measured calibration record is fed back into the
   forecast prompt so it corrects its own biases.
5. **Calibration** — the dashboard tracks the Brier score (0 = perfect,
   0.25 = coin flip), a calibration curve, and a per-mode Brier breakdown.
   Resolved forecasts are scored on their **time-weighted lifetime
   probability**, so late revisions can't game the ledger — it measures
   sustained judgment over the forecast's whole life.

## Headless runs & scheduling

```bash
.venv/bin/python -m pythia.cli scan     # one forecast scan
.venv/bin/python -m pythia.cli check    # one resolution check
.venv/bin/python -m pythia.cli daily    # check, then scan (the morning routine)
.venv/bin/python -m pythia.cli schedule --hour 7 --minute 30
```

`schedule` writes a launchd agent (`~/Library/LaunchAgents/com.pythia.daily.plist`)
that runs `daily` every morning and logs to `~/Library/Logs/pythia.log`; it
prints the `launchctl load -w …` command but never activates anything itself.
Current `PYTHIA_*` variables are captured into the agent; secrets are not.
CLI and server share the database safely: each refuses to start a run while
the other has one in flight.

## Database location

The database lives at `~/Library/Application Support/Pythia/pythia.db` —
deliberately outside Documents, which is often iCloud-synced (iCloud corrupts
SQLite WAL sidecar files). On first start after upgrading, an existing legacy
database at `data/pythia.db` is copied across automatically (WAL-safely); the
legacy file is left untouched and can be deleted once you've confirmed the
move.

## Tests

```bash
.venv/bin/python -m pytest
```

Tests run without network or model access (synthetic feeds, fake oracle).

## Prototype status

This is a runnable end-to-end prototype, not a production deployment. Known gaps
before production use: no authentication or multi-user support, no feed-level
rate limiting, and no monitoring beyond logs. See
[limitations.md](limitations.md) for model- and data-level caveats.
