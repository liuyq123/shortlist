"""Claude API client, request options, and session cost tracking."""

import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import anthropic

# Approximate list prices in USD per million tokens (input, output), prompts up to 100K tokens
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Server-side refusal fallback: "default" re-runs a declined request on the model
# Anthropic recommends for that refusal category. Only sent to models that accept it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODEL_PREFIXES = ("claude-fable-5", "claude-opus-5", "claude-sonnet-5-5")

_api_key: Optional[str] = None


def set_api_key(key: str) -> None:
    """Use this key for the rest of the session (kept in memory only)."""
    global _api_key
    _api_key = key.strip() or None


def has_api_key() -> bool:
    return bool(_api_key or os.getenv("ANTHROPIC_API_KEY"))


def get_client(max_retries: int = 2) -> anthropic.Anthropic:
    if _api_key:
        return anthropic.Anthropic(api_key=_api_key, max_retries=max_retries)
    return anthropic.Anthropic(max_retries=max_retries)


def fallback_params(model: str) -> dict[str, Any]:
    """Request params enabling the server-side refusal fallback, if the model supports it."""
    if model.startswith(_FALLBACK_MODEL_PREFIXES):
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    return {}


def describe_error(exc: Exception) -> str:
    """A short, user-facing message for an API error."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Claude API rejected the key. Check your ANTHROPIC_API_KEY."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Try again in a minute."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude API error ({exc.status_code}): {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check your network connection."
    if isinstance(exc, TypeError) and "authentication" in str(exc):
        return "No Claude API key. Add ANTHROPIC_API_KEY to .env or enter it in the app."
    return str(exc)


@dataclass
class Usage:
    """Token usage and approximate cost, summed across requests and threads."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self._lock = Lock()

    def add(self, usage: Any, model: str) -> None:
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_price, output_price = PRICES_PER_MTOK.get(model, (0.0, 0.0))
        cost = (
            input_tokens * input_price
            + cache_write * input_price * 1.25
            + cache_read * input_price * 0.1
            + output_tokens * output_price
        ) / 1_000_000
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_read_tokens += cache_read
            self.cache_write_tokens += cache_write
            self.cost_usd += cost


# Everything spent since the server started, shown in the app header
session_usage = Usage()
