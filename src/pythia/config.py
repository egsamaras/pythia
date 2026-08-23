"""Central configuration for Pythia.

All settings come from environment variables with the ``PYTHIA_`` prefix;
defaults are chosen so the app runs out of the box on a local machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the database lived before 2026-08-22. Documents is iCloud-synced on
# many Macs, and iCloud fights SQLite's WAL sidecars — so the default moved to
# Application Support; startup copies a legacy DB across once (see server/cli).
LEGACY_DB_PATH = PROJECT_ROOT / "data" / "pythia.db"


def default_db_path() -> Path:
    """Default database location, outside any cloud-synced directory."""
    return Path.home() / "Library" / "Application Support" / "Pythia" / "pythia.db"


def db_path_is_default() -> bool:
    """True when no explicit PYTHIA_DB override is set."""
    return "PYTHIA_DB" not in os.environ

# Minimal fallback used only when no feeds file exists; the curated list
# ships in feeds.txt at the project root. (Al Jazeera was dropped 2026-08-21:
# its feed endpoint resets non-interactive connections.)
DEFAULT_FEEDS: tuple[str, ...] = (
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.arabnews.com/rss.xml",
    "https://techcrunch.com/feed/",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
)

# Geographic scope of a forecast. "global" is reserved for events whose
# consequences are clearly worldwide; everything else names the region
# primarily affected.
REGIONS: tuple[str, ...] = (
    "global",
    "north_america",
    "latin_america",
    "europe",
    "mena",
    "sub_saharan_africa",
    "south_central_asia",
    "east_asia_pacific",
)

# Fixed category list; the dashboard assigns each category a fixed palette slot,
# so the order here is part of the contract (color follows the entity).
CATEGORIES: tuple[str, ...] = (
    "geopolitics",
    "economy",
    "markets",
    "technology",
    "energy",
    "climate",
    "society",
    "security",
)


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PYTHIA_{name}", default)


# GDELT watch queries: named coverage-volume trackers whose escalation ratios
# are computed each scan and shown to the oracle as structured signals.
DEFAULT_GDELT_WATCHES: tuple[tuple[str, str], ...] = (
    ("iran", "iran (military OR nuclear OR sanctions)"),
    ("russia-ukraine", "ukraine russia (offensive OR strike OR mobilization)"),
    ("china-taiwan", "taiwan (military OR blockade OR drills)"),
    ("israel-levant", '(israel OR lebanon OR gaza) (strike OR ceasefire OR escalation)'),
    ("gulf", '("saudi arabia" OR bahrain OR uae OR opec) (security OR sanctions OR attack)'),
    # Negative terms keep weather/civic/figurative uses out of the ratios.
    (
        "coups-emergencies",
        '(coup OR "martial law" OR "state of emergency") -hurricane -storm -flood -wildfire -weather',
    ),
)


def _load_watches_file(path: Path) -> tuple[tuple[str, str], ...]:
    """Parse a watches file: ``name | query`` per line, # comments ignored."""
    watches: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "|" not in stripped:
            continue
        name, _, query = stripped.partition("|")
        if name.strip() and query.strip():
            watches.append((name.strip(), query.strip()))
    return tuple(watches)


def _resolve_gdelt_watches() -> tuple[tuple[str, str], ...]:
    path = Path(os.environ.get("PYTHIA_GDELT_WATCHES_FILE", PROJECT_ROOT / "gdelt.txt"))
    if path.is_file():
        watches = _load_watches_file(path)
        if watches:
            return watches
    return DEFAULT_GDELT_WATCHES


def _load_feeds_file(path: Path) -> tuple[str, ...]:
    """Parse a feeds file: one URL per line, blank lines and # comments ignored."""
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            urls.append(stripped)
    return tuple(urls)


def _resolve_feeds() -> tuple[str, ...]:
    """Feed list precedence: PYTHIA_FEEDS env > feeds file > built-in defaults."""
    env = os.environ.get("PYTHIA_FEEDS", "")
    if env.strip():
        return tuple(u.strip() for u in env.split(",") if u.strip())
    file_env = os.environ.get("PYTHIA_FEEDS_FILE")
    feeds_path = Path(file_env) if file_env else PROJECT_ROOT / "feeds.txt"
    if feeds_path.is_file():
        feeds = _load_feeds_file(feeds_path)
        if feeds:
            return feeds
    return DEFAULT_FEEDS


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    provider: str = "auto"  # auto | anthropic | ollama
    anthropic_model: str = "claude-opus-5"
    anthropic_fallbacks: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    db_path: Path = field(default_factory=default_db_path)
    feeds: tuple[str, ...] = DEFAULT_FEEDS
    feed_timeout_s: float = 15.0
    oracle_timeout_s: float = 1800.0
    # Per-provider article caps: local models have small context windows.
    max_articles_anthropic: int = 80
    max_articles_ollama: int = 30
    max_resolution_forecasts: int = 20
    auto_update_probabilities: bool = True
    gdelt_enabled: bool = True
    gdelt_timeout_s: float = 20.0
    gdelt_watches: tuple[tuple[str, str], ...] = DEFAULT_GDELT_WATCHES


def get_settings() -> Settings:
    """Build a Settings object from the current environment.

    Re-reads the environment on every call so tests and the CLI can override
    values without import-order tricks.
    """
    feeds = _resolve_feeds()
    return Settings(
        provider=_env("PROVIDER", "auto"),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-opus-5"),
        anthropic_fallbacks=_env("ANTHROPIC_FALLBACKS", "1") not in ("0", "false", "no"),
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.1:8b"),
        db_path=Path(_env("DB", str(default_db_path()))),
        feeds=feeds,
        feed_timeout_s=float(_env("FEED_TIMEOUT_S", "15")),
        # 22B-class local models on laptop hardware can legitimately need
        # 15-20 minutes for a schema-constrained batch; 600s proved too tight.
        oracle_timeout_s=float(_env("ORACLE_TIMEOUT_S", "1800")),
        max_articles_anthropic=int(_env("MAX_ARTICLES_ANTHROPIC", "80")),
        max_articles_ollama=int(_env("MAX_ARTICLES_OLLAMA", "30")),
        max_resolution_forecasts=int(_env("MAX_RESOLUTION_FORECASTS", "20")),
        auto_update_probabilities=_env("AUTO_UPDATE", "1") not in ("0", "false", "no"),
        gdelt_enabled=_env("GDELT", "1") not in ("0", "false", "no"),
        gdelt_timeout_s=float(_env("GDELT_TIMEOUT_S", "20")),
        gdelt_watches=_resolve_gdelt_watches(),
    )


def anthropic_credentials_present() -> bool:
    """True when the Anthropic SDK will likely find credentials.

    Covers the env vars plus ``ant auth login`` profiles (stored under
    ~/.config/anthropic), which the SDK resolves automatically. Force
    PYTHIA_PROVIDER to override the heuristic either way.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_PROFILE"):
        return True
    profile_dir = Path.home() / ".config" / "anthropic"
    return profile_dir.is_dir() and any(profile_dir.iterdir())
