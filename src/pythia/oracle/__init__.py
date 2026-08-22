"""Oracle backend selection."""

from __future__ import annotations

from pythia.config import Settings, anthropic_credentials_present
from pythia.oracle.base import ForecastDraft, Oracle, OracleError
from pythia.oracle.ollama_oracle import ollama_available

__all__ = ["ForecastDraft", "Oracle", "OracleError", "get_oracle", "resolve_provider"]


def resolve_provider(settings: Settings) -> str:
    """Decide which backend a scan will use, honoring explicit configuration.

    ``auto`` prefers the Anthropic API when credentials are present, then falls
    back to a reachable local Ollama server.
    """
    if settings.provider in ("anthropic", "ollama"):
        return settings.provider
    if anthropic_credentials_present():
        return "anthropic"
    if ollama_available(settings.ollama_url):
        return "ollama"
    raise OracleError(
        "No oracle backend available: set ANTHROPIC_API_KEY (or run `ant auth login`) "
        "for the Claude API, or start Ollama locally."
    )


def get_oracle(settings: Settings) -> Oracle:
    """Instantiate the configured (or auto-detected) backend."""
    provider = resolve_provider(settings)
    if provider == "anthropic":
        from pythia.oracle.anthropic_oracle import AnthropicOracle

        return AnthropicOracle(settings)
    from pythia.oracle.ollama_oracle import OllamaOracle

    return OllamaOracle(settings)
