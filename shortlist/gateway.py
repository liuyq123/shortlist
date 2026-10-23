"""
The class gateway: the only way this app reaches a model, and the one wrapper every model request goes through.

Inference uses the workshop broker's Chat Completions route with the model and
pricing profile the visitor selected in the portal (read from /v1/bootstrap).
Locally that is a development token from the class app store; in a hosted visit
the platform supplies the broker URL and run token. The broker owns retries, so
this module never resends a request: a failure is logged and shown instead.

call() logs each request for the Agent log: what it was for, the model, latency,
tokens, the reported cost, why it stopped, and the full prompt and raw reply.

While a request runs, the broker holds its worst-case cost against the visit's
allowance: the whole context window at the input price plus max_tokens at the
output price. A small allowance fits only one such hold at a time, so call()
waits for room instead of sending a request the broker would refuse.
"""

import json
import re
import uuid
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Optional, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from shortlist import activity, config

CONTRACT_VERSION = "workshop.app.v1"
TIMEOUT_SECONDS = 90  # hosted requests have a 90-second deadline
MAX_PROMPT_BLOCK_CHARS = 4000  # per message, in the logged prompt
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "workshop-app.json"
DEFAULT_MAX_TOKENS = 12000  # room for models that reason before answering; only tokens written are charged

OutputT = TypeVar("OutputT", bound=BaseModel)


class GatewayError(Exception):
    def __init__(self, message: str, code: str = "", status: int = 0):
        super().__init__(message)
        self.code = code
        self.status = status


class ReplyError(Exception):
    """The model answered, but not in a usable form (unreadable JSON, a refusal, or cut off)."""


@dataclass(frozen=True)
class Selection:
    model: str
    profile: str
    funding: Optional[str]
    label: str = ""
    context_hold_usd: Optional[float] = None  # the context window at the input price
    output_usd_per_token: Optional[float] = None

    def hold_usd(self, max_tokens: int) -> Optional[float]:
        """What the broker holds while a request with this max_tokens runs, if the profile is priced."""
        if self.context_hold_usd is None or self.output_usd_per_token is None:
            return None
        return self.context_hold_usd + max_tokens * self.output_usd_per_token


_selection: Optional[Selection] = None
_selection_lock = Lock()


def configured() -> bool:
    return bool(config.GATEWAY_URL and config.GATEWAY_TOKEN.strip())


def _redact(text: str) -> str:
    return text.replace(config.GATEWAY_TOKEN, "[redacted]") if config.GATEWAY_TOKEN else text


def _request(method: str, path: str, body: Optional[dict[str, Any]] = None, profile: str = "") -> dict[str, Any]:
    if not configured():
        raise GatewayError(
            "Not connected to the class gateway. Put WORKSHOP_DEVELOPMENT_URL and "
            "WORKSHOP_DEVELOPMENT_TOKEN in .env and restart.",
            code="SETUP",
        )
    url = urlsplit(config.GATEWAY_URL)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise GatewayError("The gateway URL must start with http:// or https://.", code="SETUP")
    connection = (HTTPSConnection if url.scheme == "https" else HTTPConnection)(url.hostname, url.port, timeout=TIMEOUT_SECONDS)
    headers = {
        "Authorization": f"Bearer {config.GATEWAY_TOKEN}",
        "Content-Type": "application/json",
        "X-Workshop-Contract-Version": CONTRACT_VERSION,
    }
    if body is not None:
        headers.update({
            "X-Workshop-Operation-ID": str(uuid.uuid4()),
            "X-Workshop-Expected-Profile": profile,
            "X-Workshop-Attempt-Limit": "1",
        })
    try:
        connection.request(method, url.path.rstrip("/") + path, None if body is None else json.dumps(body), headers)
        response = connection.getresponse()
        raw = response.read()
    except (HTTPException, OSError) as exc:
        raise GatewayError(_redact(f"Could not reach the class gateway: {exc}"), code="NETWORK") from None
    finally:
        connection.close()
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    if response.status != 200:
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        code = error.get("code") or f"HTTP_{response.status}"
        if code == "MODEL_NOT_ALLOWED" and (error.get("details") or {}).get("selectionChanged"):
            forget_selection()
        message = error.get("message") or f"The class gateway returned HTTP {response.status}."
        raise GatewayError(_redact(str(message)), code=code, status=response.status)
    return data


def _manifest_models() -> list[str]:
    try:
        return json.loads(MANIFEST_PATH.read_text())["inference"]["models"]
    except (OSError, ValueError, KeyError):
        return []


def bootstrap() -> Selection:
    """Read the visit's model, pricing profile and funding from the gateway. Free: no inference."""
    global _selection
    boot = _request("GET", "/v1/bootstrap")
    expected_mode = "hosted" if config.HOSTED else "development"
    if (
        boot.get("contractVersion") != CONTRACT_VERSION
        or boot.get("mode") != expected_mode
        or boot.get("simulated")
        or (config.HOSTED and boot.get("runId") != config.RUN_ID)
    ):
        raise GatewayError("The workshop bootstrap does not match this app's runtime.", code="CONTRACT")
    inference = boot.get("inference") or {}
    chosen = inference.get("selection") or {}
    if not chosen.get("modelId") or not chosen.get("profileId"):
        raise GatewayError("The gateway didn't supply a model. Choose one in the class app store.", code="CONTRACT")
    allowed = _manifest_models()
    if allowed and chosen["modelId"] not in allowed:
        raise GatewayError(f"The selected model {chosen['modelId']} isn't one this app declares.", code="MODEL")
    profile = next((p for p in inference.get("profiles") or [] if p.get("id") == chosen["profileId"]), {})
    context_tokens = _number(profile.get("contextTokens"))
    input_price = _number(profile.get("maxInputUsdPerToken", profile.get("inputUsdPerToken")))
    output_price = _number(profile.get("maxOutputUsdPerToken", profile.get("outputUsdPerToken")))
    allowance.refresh(boot.get("usage") or {})
    with _selection_lock:
        _selection = Selection(
            chosen["modelId"],
            chosen["profileId"],
            inference.get("fundingMode"),
            label=profile.get("label") or "",
            context_hold_usd=context_tokens * input_price if context_tokens is not None and input_price is not None else None,
            output_usd_per_token=output_price,
        )
        return _selection


def _number(value: Any) -> Optional[float]:
    """A count or USD amount from the bootstrap (prices arrive as decimal strings)."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class Allowance:
    """The visit's model allowance, as the gateway reported it, less the holds of requests in flight."""

    def __init__(self) -> None:
        self._changed = Condition()
        self.limit_usd: Optional[float] = None
        self.available_usd: Optional[float] = None  # not counting our own holds
        self._held = 0.0
        self._running = 0

    def refresh(self, usage: dict[str, Any]) -> None:
        available = _number(usage.get("availableUsd"))
        with self._changed:
            self.limit_usd = _number(usage.get("limitUsd"))
            # The gateway's figure already excludes the holds of our requests still running
            self.available_usd = None if available is None else available + self._held
            self._changed.notify_all()

    def acquire(self, hold: Optional[float], model_label: str) -> None:
        """Wait until the allowance has room for this hold. Raises if it can't fit even on its own."""
        if hold is None:
            return
        with self._changed:
            while self.available_usd is not None and self._held + hold > self.available_usd + 1e-9:
                if not self._running:
                    raise GatewayError(
                        f"Only ${max(0.0, self.available_usd):.2f} of model allowance is left, but each {model_label or 'model'} request "
                        f"needs a temporary ${hold:.2f} hold while it runs, so no more requests can start. "
                        + ("Choose a model with a smaller hold in the class app store, or start a new visit."
                           if config.HOSTED else
                           "Choose a model with a smaller hold, or create a token with a higher ceiling, in the class app store (Account)."),
                        code="ALLOWANCE_LOW",
                    )
                self._changed.wait()
            self._held += hold
            self._running += 1

    def parallel(self, max_tokens: int, most: int) -> int:
        """How many requests of this size to run at once (1 to `most`): as many holds as the allowance fits now."""
        chosen = _selection
        hold = chosen.hold_usd(max_tokens) if chosen else None
        with self._changed:
            if hold is None or self.available_usd is None:
                return most
            return max(1, min(most, int((self.available_usd - self._held) // hold)))

    def release(self, hold: Optional[float], cost_usd: Optional[float]) -> None:
        if hold is None:
            return
        with self._changed:
            self._held -= hold
            self._running -= 1
            if self.available_usd is not None and cost_usd:
                self.available_usd -= cost_usd
            self._changed.notify_all()

    def to_dict(self) -> dict[str, Any]:
        hold = _selection.hold_usd(DEFAULT_MAX_TOKENS) if _selection else None
        with self._changed:
            return {
                "limit_usd": self.limit_usd,
                "available_usd": None if self.available_usd is None else round(max(0.0, self.available_usd), 4),  # holds aside
                "hold_usd": None if hold is None else round(hold, 4),
            }


allowance = Allowance()


def selection() -> Selection:
    with _selection_lock:
        current = _selection
    return current or bootstrap()


def forget_selection() -> None:
    global _selection
    with _selection_lock:
        _selection = None


def describe_error(exc: Exception) -> str:
    """A short, user-facing message for a failed request."""
    if isinstance(exc, GatewayError):
        if exc.code in ("SESSION_EXPIRED", "SESSION_REVOKED") or exc.status == 401:
            return "The gateway token expired or was revoked. Create a new one in the class app store (Account), or start a new visit."
        if exc.code == "ALLOWANCE_EXHAUSTED" or exc.status == 402:
            return f"Not enough model allowance left. {exc}"
        if exc.code == "RATE_LIMITED":
            return f"The gateway is limiting requests right now: {exc}"
        if exc.code == "CONCURRENCY_LIMIT":
            return "The gateway refused a request because too many were running at once. Try again."
        if exc.code == "MODEL_NOT_ALLOWED":
            return "The selected model changed in the portal. Try again."
    return _redact(str(exc))


def call(
    purpose: str,
    *,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Make one model request through the gateway and log it. Returns the Chat Completions response."""
    chosen = selection()
    body: dict[str, Any] = {"model": chosen.model, "messages": messages, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    hold = chosen.hold_usd(max_tokens)
    activity.check_stop()  # a stopped run starts no new requests; ones already sent still finish
    try:
        allowance.acquire(hold, chosen.label)  # waits while other requests' holds fill the allowance
    except GatewayError as exc:
        activity.begin("llm", purpose, model=chosen.model).finish("error", error=describe_error(exc), detail=[_prompt_section(body)])
        raise
    try:
        activity.check_stop()  # the person may have pressed Stop while this request waited
    except activity.Stopped:
        allowance.release(hold, None)
        raise
    response: dict[str, Any] = {}
    try:
        response = _send(purpose, body, chosen)
        return response
    finally:
        allowance.release(hold, usage_numbers(response.get("usage") or {})["cost_usd"])


def _send(purpose: str, body: dict[str, Any], chosen: Selection) -> dict[str, Any]:
    event = activity.begin("llm", purpose, model=chosen.model)
    try:
        response = _request("POST", "/v1/llm/chat/completions", body, chosen.profile)
        choice = response["choices"][0]
        choice["message"]  # noqa: B018 (check the shape)
    except GatewayError as exc:
        event.finish("error", error=describe_error(exc), detail=[_prompt_section(body)])
        raise
    except (KeyError, IndexError, TypeError):
        event.finish("error", error="The gateway's reply had no message.", detail=[_prompt_section(body)])
        raise GatewayError("The gateway's reply had no message.", code="CONTRACT") from None

    response.setdefault("id", f"local-{uuid.uuid4().hex}")
    usage = session_usage.add(response.get("usage") or {})
    refused = bool(choice["message"].get("refusal"))
    activity.link_response(response["id"], event)
    event.finish(
        "refused" if refused else "ok",
        model=response.get("model") or chosen.model,
        usage=usage,
        stop_reason=choice.get("finish_reason"),
        summary="the model declined this request" if refused else "",
        detail=[_prompt_section(body), activity.section("Raw reply", _clean(response), format="code")],
    )
    return response


def reply_text(response: dict[str, Any]) -> str:
    return (response["choices"][0]["message"].get("content") or "").strip()


def call_json(purpose: str, *, system: str, user: str, output: type[OutputT], max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[dict[str, Any], OutputT]:
    """One request whose reply must be a JSON object matching `output`. Raises ReplyError otherwise."""
    response = call(
        purpose,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    choice = response["choices"][0]
    if choice["message"].get("refusal"):
        raise ReplyError("The model declined this request.")
    if choice.get("finish_reason") == "length":
        raise ReplyError("The reply was cut off at the length limit.")
    try:
        return response, parse_json(reply_text(response), output)
    except ReplyError as exc:
        activity.annotate(response, summary=str(exc))
        raise


def parse_json(text: str, output: type[OutputT]) -> OutputT:
    """Read a JSON object from a model reply, allowing code fences or text around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ReplyError("The reply contained no JSON object.")
    try:
        data = json.loads(text[start : end + 1], strict=False)  # tolerate raw newlines inside strings
    except ValueError as exc:
        raise ReplyError(f"The reply's JSON couldn't be read ({exc.msg}).") from None
    try:
        return output.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ReplyError(f"The reply's JSON didn't match the expected format ({where}: {first['msg']}).") from None


# =============================================================================
# Logged prompt and reply
# =============================================================================


def _text(value: Any, limit: int = MAX_PROMPT_BLOCK_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f" … ({len(text) - limit:,} more characters)"


def _prompt_section(body: dict[str, Any]) -> dict[str, Any]:
    parts = [f"MODEL: {body['model']}", f"SETTINGS: {json.dumps({k: body[k] for k in ('max_tokens', 'tool_choice') if k in body})}"]
    if body.get("tools"):
        parts.append("TOOLS: " + ", ".join(t["function"]["name"] for t in body["tools"]))
    for message in body["messages"]:
        lines = []
        if message.get("content"):
            lines.append(_text(message["content"]))
        for call_ in message.get("tool_calls") or []:
            lines.append(f"[tool call] {call_['function']['name']}({_text(call_['function'].get('arguments', ''), 600)})")
        label = message["role"].upper() + (" (tool result)" if message["role"] == "tool" else "")
        parts.append(f"{label}:\n" + "\n".join(lines))
    return activity.section("Prompt", "\n\n".join(parts))


PRIVATE_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")


def _clean(value: Any) -> Any:
    """Drop nulls from a raw reply, for reading. The model's private reasoning is left out."""
    if isinstance(value, dict):
        return {k: "(not shown)" if k in PRIVATE_FIELDS else _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


# =============================================================================
# Usage and cost
# =============================================================================


def usage_numbers(usage: dict[str, Any]) -> dict[str, Any]:
    """Token counts and the cost the gateway reported for one response (None if it reported none)."""
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    cost = next((usage[k] for k in ("cost", "cost_usd", "total_cost") if isinstance(usage.get(k), (int, float))), None)
    return {
        "input_tokens": max(0, prompt - cached - written),
        "cache_read_tokens": cached,
        "cache_write_tokens": written,
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cost_usd": float(cost) if cost is not None else None,
    }


@dataclass
class Usage:
    """Token usage and reported cost, summed across requests and threads."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    unpriced: int = 0  # requests whose cost the gateway didn't report

    def __post_init__(self) -> None:
        self._lock = Lock()

    def add(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Add one response's usage; returns that response's numbers."""
        numbers = usage_numbers(usage)
        with self._lock:
            self.calls += 1
            self.input_tokens += numbers["input_tokens"]
            self.output_tokens += numbers["output_tokens"]
            self.cache_read_tokens += numbers["cache_read_tokens"]
            self.cache_write_tokens += numbers["cache_write_tokens"]
            if numbers["cost_usd"] is None:
                self.unpriced += 1
            else:
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
                "unpriced": self.unpriced,
            }


# Everything spent since the server started, shown in the app header
session_usage = Usage()
