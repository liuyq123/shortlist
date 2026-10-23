"""
Activity log: what each agent did, for the app's Agent log view.

A run is one job: screening papers, building a profile, or one assistant
message. Its events are the model requests, tool calls, and other steps it made,
with timing, tokens, cost, and (fetched on demand) the full prompt and raw reply.
Problems that don't stop a run, such as a failed batch, are recorded as issues.
The last MAX_RUNS runs are kept in memory.

    with activity.run("screener", "Screen today's papers", emit):
        with activity.span("group", "Batch 1 of 3: 150 titles") as group:
            ...                      # model requests made here become its children
            group.finish(summary="kept 48 of 150 titles; scored 48, 41 at 4+")
"""

import contextvars
import itertools
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Iterator, Optional

MAX_RUNS = 30
MAX_SECTION_CHARS = 60_000

_lock = Lock()
_ids = itertools.count(1)
_version = 0  # bumped on every change, so the page can ask for changes only
_runs: dict[int, "Run"] = {}
_events: dict[int, "Event"] = {}
_by_response: dict[str, "Event"] = {}
_current_run: contextvars.ContextVar[Optional["Run"]] = contextvars.ContextVar("current_run", default=None)
_current_event: contextvars.ContextVar[Optional["Event"]] = contextvars.ContextVar("current_event", default=None)


def _changed(run: Optional["Run"]) -> None:
    """Mark a run as changed. Call with _lock held."""
    global _version
    if run is not None:
        _version += 1
        run.version = _version


def section(title: str, body: Any, open: bool = False, format: str = "text") -> dict[str, Any]:
    """One collapsible block of an event's details. format: "text", "code", or "list"."""
    if format == "code" and not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False, indent=1)
    if isinstance(body, str) and len(body) > MAX_SECTION_CHARS:
        body = body[:MAX_SECTION_CHARS] + f"\n… ({len(body) - MAX_SECTION_CHARS:,} more characters)"
    return {"title": title, "body": body, "open": open, "format": format}


@dataclass(eq=False)
class Event:
    id: int
    run: Optional["Run"]
    parent: Optional[int]
    kind: str  # "group", "llm" (a model request), "tool", or "step"
    label: str
    started: float
    ended: Optional[float] = None
    status: str = "running"  # running, ok, refused, or error
    summary: str = ""
    error: str = ""
    model: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    stop_reason: Optional[str] = None
    retries: int = 0
    args: Optional[dict[str, Any]] = None  # tool arguments, shortened
    result: str = ""  # tool result, shortened
    detail: list[dict[str, Any]] = field(default_factory=list)

    def update(self, **fields: Any) -> None:
        with _lock:
            for name, value in fields.items():
                setattr(self, name, value)
            _changed(self.run)

    def finish(self, status: str = "ok", **fields: Any) -> None:
        self.update(status=status, ended=time.time(), **fields)

    def to_dict(self) -> dict[str, Any]:
        start = self.run.started if self.run else self.started
        return {
            "id": self.id,
            "parent": self.parent,
            "kind": self.kind,
            "label": self.label,
            "t0": round(self.started - start, 3),
            "t1": round(self.ended - start, 3) if self.ended else None,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
            "model": self.model,
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "retries": self.retries,
            "args": self.args,
            "result": self.result,
            "has_detail": bool(self.detail),
        }


@dataclass(eq=False)
class Run:
    id: int
    agent: str  # "screener", "assistant", or "builder"
    title: str
    started: float
    input: str = ""
    output: str = ""
    summary: str = ""
    status: str = "running"  # running, done, or error
    error: str = ""
    ended: Optional[float] = None
    keep_empty: bool = True
    events: list[Event] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    version: int = 0

    def update(self, **fields: Any) -> None:
        with _lock:
            for name, value in fields.items():
                setattr(self, name, value)
            _changed(self)

    def to_dict(self) -> dict[str, Any]:
        requests = [e for e in self.events if e.kind == "llm"]
        usage = [e.usage for e in requests if e.usage]
        return {
            "id": self.id,
            "agent": self.agent,
            "title": self.title,
            "started_at": self.started,
            "duration": round((self.ended or time.time()) - self.started, 3),
            "status": self.status,
            "error": self.error,
            "input": self.input,
            "output": self.output,
            "summary": self.summary,
            "issues": self.issues,
            "models": sorted({e.model for e in requests if e.model}),
            "totals": {
                "requests": len(requests),
                "input_tokens": sum(u["input_tokens"] for u in usage),
                "cache_read_tokens": sum(u["cache_read_tokens"] for u in usage),
                "cache_write_tokens": sum(u["cache_write_tokens"] for u in usage),
                "output_tokens": sum(u["output_tokens"] for u in usage),
                "cost_usd": sum(u["cost_usd"] for u in usage if u["cost_usd"] is not None),
                "unpriced": sum(1 for u in usage if u["cost_usd"] is None),
            },
            "events": [e.to_dict() for e in self.events],
        }


def _drop(run_id: int) -> None:
    """Forget a run and its events. Call with _lock held."""
    run = _runs.pop(run_id, None)
    if run is None:
        return
    for event in run.events:
        _events.pop(event.id, None)
    for response_id in [k for k, v in _by_response.items() if v.run is run]:
        del _by_response[response_id]


@contextmanager
def run(
    agent: str,
    title: str,
    emit: Optional[Callable[[dict[str, Any]], None]] = None,
    input: str = "",
    keep_empty: bool = True,
) -> Iterator[Run]:
    """Record everything done inside this block as one run. Emits {"type": "run", "run_id"}."""
    with _lock:
        current = Run(id=next(_ids), agent=agent, title=title, started=time.time(), input=input, keep_empty=keep_empty)
        _runs[current.id] = current
        _changed(current)
        while len(_runs) > MAX_RUNS:
            _drop(next(iter(_runs)))
    if emit:
        emit({"type": "run", "run_id": current.id})
    token = _current_run.set(current)
    try:
        yield current
    except Exception as exc:
        from shortlist.gateway import describe_error

        _end(current, "error", describe_error(exc))
        raise
    else:
        _end(current, "done")
    finally:
        _current_run.reset(token)


def _end(run: Run, status: str, error: str = "") -> None:
    with _lock:
        now = time.time()
        for event in run.events:
            if event.status == "running":  # left open by an exception
                event.status, event.ended = "error", now
        run.status, run.error, run.ended = status, error, now
        _changed(run)
        if not run.events and not run.issues and not run.keep_empty and status == "done":
            _drop(run.id)


def current_run() -> Optional[Run]:
    return _current_run.get()


def begin(kind: str, label: str, parent: Optional[Event] = None, **fields: Any) -> Event:
    """Start an event in the current run, as a child of the innermost open span unless parent is given."""
    run_ = _current_run.get()
    parent = parent or _current_event.get()
    with _lock:
        event = Event(
            id=next(_ids), run=run_, parent=parent.id if parent else None,
            kind=kind, label=label, started=time.time(), **fields,
        )
        if run_ is not None:
            run_.events.append(event)
            _events[event.id] = event
            _changed(run_)
    return event


@contextmanager
def span(kind: str, label: str, **fields: Any) -> Iterator[Event]:
    """An event covering this block. Events begun inside it (in any thread that copied the context) are its children."""
    event = begin(kind, label, **fields)
    token = _current_event.set(event)
    try:
        yield event
    except Exception as exc:
        if event.status == "running":
            event.finish("error", error=event.error or str(exc))
        raise
    else:
        if event.status == "running":
            event.finish()
    finally:
        _current_event.reset(token)


def note(text: str) -> None:
    """Set the summary of the innermost open span, e.g. a tool's plain-language description."""
    event = _current_event.get()
    if event is not None:
        event.update(summary=text)


def issue(kind: str, text: str, count: int = 1) -> None:
    """Record a problem that didn't stop the run, e.g. kind "failed_batch", "skipped", or "lookup_failed"."""
    run_ = _current_run.get()
    if run_ is not None:
        with _lock:
            run_.issues.append({"kind": kind, "text": text, "count": count})
            _changed(run_)


def link_response(response_id: str, event: Event) -> None:
    with _lock:
        _by_response[response_id] = event


def annotate(response: Any, **fields: Any) -> None:
    """Add a summary or detail sections to the event for a model response (see gateway.call)."""
    response_id = response.get("id", "") if isinstance(response, dict) else getattr(response, "id", "")
    event = _by_response.get(response_id)
    if event is None:
        return
    if "detail" in fields:
        fields["detail"] = [*fields["detail"], *event.detail]  # results first, then prompt and raw reply
    event.update(**fields)


def snapshot(since: int = 0) -> dict[str, Any]:
    """Runs changed since a version, newest last, plus the ids of all runs still kept."""
    with _lock:
        return {
            "version": _version,
            "ids": list(_runs),
            "runs": [r.to_dict() for r in _runs.values() if r.version > since],
        }


def event_detail(event_id: int) -> Optional[list[dict[str, Any]]]:
    with _lock:
        event = _events.get(event_id)
        return list(event.detail) if event else None
