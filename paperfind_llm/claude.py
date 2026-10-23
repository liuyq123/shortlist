"""Model API client, the one wrapper every model request goes through, and session cost tracking."""

import json
import os
import random
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import anthropic

from paperfind_llm import activity

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

MAX_PROMPT_BLOCK_CHARS = 4000  # per message block, in the logged prompt

_api_key: Optional[str] = None
_client: Optional[anthropic.Anthropic] = None


def set_api_key(key: str) -> None:
    """Use this key for the rest of the session (kept in memory only)."""
    global _api_key, _client
    _api_key = key.strip() or None
    _client = None


def has_api_key() -> bool:
    return bool(_api_key or os.getenv("ANTHROPIC_API_KEY"))


def get_client() -> anthropic.Anthropic:
    """A shared client. It doesn't retry by itself: call() retries, so retries show up in the log."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_api_key, max_retries=0) if _api_key else anthropic.Anthropic(max_retries=0)
    return _client


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


def _retry_delay(exc: Exception, attempt: int) -> Optional[float]:
    """Seconds to wait before retrying, or None if the error isn't worth retrying."""
    if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
        return min(8.0, 0.5 * 2**attempt) + random.random() / 2
    if isinstance(exc, anthropic.APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500):
        try:
            return min(30.0, float(exc.response.headers.get("retry-after", "")))
        except ValueError:
            return min(8.0, 0.5 * 2**attempt) + random.random() / 2
    return None


def call(purpose: str, *, model: str, retries: int = 2, output_format: Any = None, **params: Any) -> Any:
    """Make one model request and log it.

    The log records what the request was for, the model, latency, tokens, cost,
    why it stopped, retries, and the full prompt and raw reply. With
    output_format (a Pydantic model), the reply is parsed into it. A refusal is
    returned like any other reply (stop_reason "refusal"), not raised.
    """
    event = activity.begin("llm", purpose, model=model)
    client = get_client()
    attempt = 0
    while True:
        try:
            if output_format is not None:
                response = client.beta.messages.parse(model=model, output_format=output_format, **params)
            else:
                response = client.beta.messages.create(model=model, **params)
            break
        except Exception as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None or attempt >= retries:
                event.finish("error", error=describe_error(exc), retries=attempt, detail=[_prompt_section(params)])
                raise
            attempt += 1
            event.update(retries=attempt, summary=f"retrying in {delay:.0f} s: {describe_error(exc)}")
            time.sleep(delay)

    usage = session_usage.add(response.usage, response.model or model)
    activity.link_response(response.id, event)
    event.finish(
        "refused" if response.stop_reason == "refusal" else "ok",
        model=response.model or model,  # differs from the request if a fallback model answered
        usage=usage,
        stop_reason=response.stop_reason,
        retries=attempt,
        summary="the model declined this request" if response.stop_reason == "refusal" else "",
        detail=[_prompt_section(params), _reply_section(response)],
    )
    return response


# =============================================================================
# Logged prompt and reply
# =============================================================================


def _text(value: Any, limit: int = MAX_PROMPT_BLOCK_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f" … ({len(text) - limit:,} more characters)"


def _block_text(block: Any) -> str:
    if hasattr(block, "model_dump"):
        block = block.model_dump(mode="json")
    if isinstance(block, str):
        return _text(block)
    kind = block.get("type")
    if kind == "text":
        return _text(block.get("text", ""))
    if kind == "tool_use":
        return f"[tool call] {block.get('name')}({_text(block.get('input'), 600)})"
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            content = "\n".join(_block_text(c) for c in content)
        prefix = "[tool error]" if block.get("is_error") else "[tool result]"
        return f"{prefix} {_text(content or '')}"
    if kind in ("thinking", "redacted_thinking"):
        return f"[{kind}]"
    return _text(block)


def _prompt_section(params: dict[str, Any]) -> dict[str, Any]:
    parts = []
    settings = {k: params[k] for k in ("max_tokens", "output_config") if k in params}
    if settings:
        parts.append(f"SETTINGS: {json.dumps(settings)}")
    if params.get("tools"):
        names = [t["name"] if isinstance(t, dict) else getattr(t, "name", "?") for t in params["tools"]]
        parts.append(f"TOOLS: {', '.join(names)}")
    system = params.get("system")
    if system:
        if isinstance(system, list):
            system = "\n".join(_block_text(b) for b in system)
        parts.append(f"SYSTEM:\n{system}")
    for message in params.get("messages", []):
        content = message["content"]
        blocks = [content] if isinstance(content, str) else content
        parts.append(f"{message['role'].upper()}:\n" + "\n".join(_block_text(b) for b in blocks))
    return activity.section("Prompt", "\n\n".join(parts))


def _clean(value: Any) -> Any:
    """Drop opaque or duplicate fields (signatures, parsed copies) and nulls from a raw reply."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None and k not in ("signature", "parsed_output")}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _reply_section(response: Any) -> dict[str, Any]:
    return activity.section("Raw reply", _clean(response.model_dump(mode="json")), format="code")


# =============================================================================
# Cost tracking
# =============================================================================


def usage_numbers(usage: Any, model: str) -> dict[str, Any]:
    """Token counts and approximate cost of one response."""
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
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "cost_usd": cost,
    }


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

    def add(self, usage: Any, model: str) -> dict[str, Any]:
        """Add one response's usage; returns that response's numbers."""
        numbers = usage_numbers(usage, model)
        with self._lock:
            self.calls += 1
            self.input_tokens += numbers["input_tokens"]
            self.output_tokens += numbers["output_tokens"]
            self.cache_read_tokens += numbers["cache_read_tokens"]
            self.cache_write_tokens += numbers["cache_write_tokens"]
            self.cost_usd += numbers["cost_usd"]
        return numbers

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "cost_usd": round(self.cost_usd, 4),
            }


# Everything spent since the server started, shown in the app header
session_usage = Usage()
