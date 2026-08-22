"""Local Ollama backend — fully on-premise forecasting and resolution.

Uses Ollama's /api/chat with a JSON-schema ``format`` constraint, so the model
is grammar-constrained to the expected schema. Quality depends on the local
model; a 22B-class model is a reasonable floor for useful forecasts.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import httpx

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


def ollama_available(base_url: str, timeout_s: float = 3.0) -> bool:
    """True when an Ollama server answers at base_url."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=timeout_s)
        resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


class OllamaOracle:
    """Forecast generation and resolution via a local Ollama model."""

    provider = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.ollama_model
        self._base_url = settings.ollama_url
        self._timeout_s = settings.oracle_timeout_s

    def _request_json(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        """One schema-constrained chat request; returns the parsed JSON object."""
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {
                # Default Ollama context (2k) is too small for an article batch;
                # 16k leaves headroom for 30 articles + open forecasts + output
                # (overflow would silently truncate the start of the prompt).
                "num_ctx": 16384,
                "temperature": 0.4,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=httpx.Timeout(self._timeout_s, connect=5.0),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OracleError(f"Ollama request failed: {exc}") from exc

        body = resp.json()
        content = (body.get("message") or {}).get("content", "")
        logger.info(
            "ollama call: model=%s eval_count=%s duration_ms=%s",
            body.get("model"),
            body.get("eval_count"),
            round((body.get("total_duration") or 0) / 1e6),
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OracleError(f"Ollama returned invalid JSON: {exc}") from exc

    def generate(
        self,
        articles: list[dict[str, Any]],
        today: date,
        open_statements: list[str],
        track_record: str | None = None,
    ) -> list[ForecastDraft]:
        raw = self._request_json(
            SYSTEM_PROMPT,
            build_user_prompt(articles, today, open_statements, track_record),
            FORECAST_JSON_SCHEMA,
        )
        return validate_drafts(raw, len(articles), today)

    def resolve(
        self, forecasts: list[dict[str, Any]], articles: list[dict[str, Any]], today: date
    ) -> list[ResolutionDraft]:
        raw = self._request_json(
            RESOLUTION_SYSTEM_PROMPT,
            build_resolution_prompt(forecasts, articles, today),
            RESOLUTION_JSON_SCHEMA,
        )
        return validate_resolution_drafts(raw, len(forecasts), len(articles))
