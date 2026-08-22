"""Oracle backend tests with stubbed transports — no network, no real models."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pythia.config import Settings
from pythia.oracle import ollama_oracle
from pythia.oracle.anthropic_oracle import AnthropicOracle
from pythia.oracle.base import (
    FORECAST_JSON_SCHEMA,
    RESOLUTION_JSON_SCHEMA,
    OracleError,
)
from pythia.oracle.ollama_oracle import OllamaOracle

OPEN_FORECASTS = [
    {
        "statement": "Synthetic event happens",
        "horizon_date": "2026-10-01",
        "probability": 0.4,
        "created_at": "2026-08-01T00:00:00+00:00",
    }
]
GOOD_RESOLUTIONS = {
    "resolutions": [
        {
            "forecast_index": 0,
            "justification": "New reporting shifts the odds.",
            "evidence_indices": [0],
            "verdict": "update",
            "new_probability": 0.75,
        }
    ]
}

TODAY = date(2026, 8, 21)
ARTICLES = [
    {"source": "Wire", "title": "Headline", "summary": "s", "published_at": "2026-08-20T00:00:00"}
]
GOOD_OUTPUT = {
    "forecasts": [
        {
            "statement": "Synthetic event happens",
            "mode": "consequence",
            "probability": 0.4,
            "horizon_date": "2026-09-30",
            "category": "markets",
            "rationale": "r",
            "basis": "reported stress -> funding costs rise",
            "evidence_indices": [0],
        }
    ]
}


# --- Ollama -----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._body


def test_ollama_generate_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return _FakeResponse(
            {
                "message": {"content": json.dumps(GOOD_OUTPUT)},
                "model": "llama3.1:8b",
                "eval_count": 100,
                "total_duration": 1_000_000,
            }
        )

    monkeypatch.setattr(ollama_oracle.httpx, "post", fake_post)
    drafts = OllamaOracle(Settings()).generate(ARTICLES, TODAY, [])
    assert len(drafts) == 1
    assert drafts[0].probability == 0.4
    assert captured["url"].endswith("/api/chat")
    payload = captured["payload"]
    assert payload["format"] == FORECAST_JSON_SCHEMA
    assert payload["options"]["num_ctx"] == 16384
    assert payload["stream"] is False


def test_ollama_resolve_carries_resolution_schema_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["payload"] = kwargs["json"]
        return _FakeResponse({"message": {"content": json.dumps(GOOD_RESOLUTIONS)}})

    monkeypatch.setattr(ollama_oracle.httpx, "post", fake_post)
    drafts = OllamaOracle(Settings()).resolve(OPEN_FORECASTS, ARTICLES, TODAY)
    assert captured["payload"]["format"] == RESOLUTION_JSON_SCHEMA
    assert "current p=0.40" in captured["payload"]["messages"][1]["content"]
    assert len(drafts) == 1
    assert drafts[0].verdict == "update"
    assert drafts[0].new_probability == 0.75


def test_anthropic_resolve_transforms_schema(anthropic_oracle: AnthropicOracle) -> None:
    stub = _StubMessages(_response(text=json.dumps(GOOD_RESOLUTIONS)))
    anthropic_oracle._client = SimpleNamespace(beta=SimpleNamespace(messages=stub))
    drafts = anthropic_oracle.resolve(OPEN_FORECASTS, ARTICLES, TODAY)
    assert len(drafts) == 1
    assert drafts[0].new_probability == 0.75

    def keys_used(node: Any) -> set[str]:
        if isinstance(node, dict):
            found = set(node)
            for v in node.values():
                found |= keys_used(v)
            return found
        if isinstance(node, list):
            return set().union(*(keys_used(v) for v in node)) if node else set()
        return set()

    sent_keys = keys_used(stub.kwargs["output_config"]["format"]["schema"])
    for keyword in ("minimum", "maximum", "maxItems"):
        assert keyword not in sent_keys


def test_ollama_connection_error_raises_oracle_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ollama_oracle.httpx, "post", fake_post)
    with pytest.raises(OracleError, match="Ollama request failed"):
        OllamaOracle(Settings()).generate(ARTICLES, TODAY, [])


def test_ollama_invalid_json_raises_oracle_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"message": {"content": "not json"}})

    monkeypatch.setattr(ollama_oracle.httpx, "post", fake_post)
    with pytest.raises(OracleError, match="invalid JSON"):
        OllamaOracle(Settings()).generate(ARTICLES, TODAY, [])


# --- Anthropic --------------------------------------------------------------


class _StubMessages:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return self.response


def _response(
    stop_reason: str = "end_turn",
    text: str | None = None,
    stop_details: SimpleNamespace | None = None,
) -> SimpleNamespace:
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=content,
        usage=SimpleNamespace(input_tokens=10, output_tokens=10),
        model="claude-opus-5",
    )


@pytest.fixture()
def anthropic_oracle(monkeypatch: pytest.MonkeyPatch) -> AnthropicOracle:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return AnthropicOracle(Settings())


def test_anthropic_sends_transformed_schema_with_fallbacks(
    anthropic_oracle: AnthropicOracle,
) -> None:
    stub = _StubMessages(_response(text=json.dumps(GOOD_OUTPUT)))
    anthropic_oracle._client = SimpleNamespace(beta=SimpleNamespace(messages=stub))

    drafts = anthropic_oracle.generate(ARTICLES, TODAY, [])
    assert len(drafts) == 1
    assert stub.kwargs is not None
    # Refusal fallbacks on by default, scalar form + matching beta header.
    assert stub.kwargs["fallbacks"] == "default"
    assert stub.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    # Structured outputs reject these as schema keywords; the transform folds
    # them into description text, so they must not appear as dict keys.
    def keys_used(node: Any) -> set[str]:
        if isinstance(node, dict):
            found = set(node)
            for v in node.values():
                found |= keys_used(v)
            return found
        if isinstance(node, list):
            return set().union(*(keys_used(v) for v in node)) if node else set()
        return set()

    sent_keys = keys_used(stub.kwargs["output_config"]["format"]["schema"])
    for keyword in ("minimum", "maximum", "minItems", "maxItems"):
        assert keyword not in sent_keys, f"unsupported schema keyword sent: {keyword}"


def test_anthropic_refusal_raises_even_without_stop_details(
    anthropic_oracle: AnthropicOracle,
) -> None:
    stub = _StubMessages(_response(stop_reason="refusal", stop_details=None))
    anthropic_oracle._client = SimpleNamespace(beta=SimpleNamespace(messages=stub))
    with pytest.raises(OracleError, match="declined"):
        anthropic_oracle.generate(ARTICLES, TODAY, [])


def test_anthropic_truncation_raises(anthropic_oracle: AnthropicOracle) -> None:
    stub = _StubMessages(_response(stop_reason="max_tokens", text="{}"))
    anthropic_oracle._client = SimpleNamespace(beta=SimpleNamespace(messages=stub))
    with pytest.raises(OracleError, match="max_tokens"):
        anthropic_oracle.generate(ARTICLES, TODAY, [])
