"""Anthropic API backend (Claude Opus 5, structured outputs).

Uses the official ``anthropic`` SDK. Credentials resolve from the environment
(ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an ``ant auth login`` profile).
Server-side refusal fallbacks are enabled by default so a safety decline is
retried on a fallback model within the same request; disable with
PYTHIA_ANTHROPIC_FALLBACKS=0.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from pythia.config import Settings
from pythia.oracle.base import (
    FORECAST_JSON_SCHEMA,
    RESOLUTION_JSON_SCHEMA,
    RESOLUTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ForecastDraft,
    OracleError,
    ResolutionDraft,
    build_resolution_prompt,
    build_user_prompt,
    validate_drafts,
    validate_resolution_drafts,
)

logger = logging.getLogger(__name__)


class AnthropicOracle:
    """Claude API backend with guaranteed-JSON responses."""

    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        import anthropic  # imported lazily so the Ollama-only path needs no key

        self.model = settings.anthropic_model
        self._fallbacks = settings.anthropic_fallbacks
        self._client = anthropic.Anthropic(timeout=settings.oracle_timeout_s)
        # Structured outputs accept only a subset of JSON Schema; transform_schema
        # folds unsupported keywords (minimum/maximum, min/maxItems) into
        # descriptions so the API does not reject the request. App-side
        # validation still enforces those bounds.
        self._forecast_schema = anthropic.transform_schema(FORECAST_JSON_SCHEMA)
        self._resolution_schema = anthropic.transform_schema(RESOLUTION_JSON_SCHEMA)

    def _request_json(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        """One structured-output request; returns the parsed JSON object."""
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            # No cache_control: these prompts sit below the minimum cacheable
            # prefix, so annotating them would be a silent no-op.
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if self._fallbacks:
            response = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **request,
            )
        else:
            response = self._client.messages.create(**request)

        if response.stop_reason == "refusal":
            # stop_details fields are not all guaranteed; use what is present.
            sd = getattr(response, "stop_details", None)
            parts = [
                p
                for p in (getattr(sd, "category", None), getattr(sd, "explanation", None))
                if p
            ]
            detail = f" ({'; '.join(parts)})" if parts else ""
            raise OracleError(f"Claude declined the request{detail}")
        if response.stop_reason == "max_tokens":
            raise OracleError("response truncated at max_tokens; raise the limit")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise OracleError("no text block in Claude response")
        logger.info(
            "anthropic call: model=%s input_tokens=%s output_tokens=%s",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise OracleError(f"Claude returned invalid JSON: {exc}") from exc

    def generate(
        self,
        articles: list[dict[str, Any]],
        today: date,
        open_statements: list[str],
        track_record: str | None = None,
        signals: str | None = None,
    ) -> list[ForecastDraft]:
        raw = self._request_json(
            SYSTEM_PROMPT,
            build_user_prompt(articles, today, open_statements, track_record, signals),
            self._forecast_schema,
        )
        return validate_drafts(raw, len(articles), today)

    def resolve(
        self, forecasts: list[dict[str, Any]], articles: list[dict[str, Any]], today: date
    ) -> list[ResolutionDraft]:
        raw = self._request_json(
            RESOLUTION_SYSTEM_PROMPT,
            build_resolution_prompt(forecasts, articles, today),
            self._resolution_schema,
        )
        return validate_resolution_drafts(raw, len(forecasts), len(articles))
