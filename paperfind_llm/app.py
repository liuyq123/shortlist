"""
Web app: a FastAPI backend for the single-page UI in static/.

Long operations (building a profile, refreshing the feed, the assistant) stream
newline-delimited JSON events so the page can show progress as it happens:
    {"type": "run" | "step" | "progress" | "test" | "done" | "error", ..., "cost_usd": 0.0123}

The Agent log view polls /api/log for what each run did (see activity.py).
"""

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paperfind_llm import activity, arxiv, assistant, config, screen, store
from paperfind_llm import profile as profiles
from paperfind_llm.claude import describe_error, get_client, has_api_key, session_usage, set_api_key

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("paperfind_llm")

app = FastAPI(title="paperfind")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

Emit = Callable[[dict[str, Any]], None]


def _stream(job: Callable[[Emit], Optional[dict[str, Any]]]) -> StreamingResponse:
    """Run job(emit) in a thread and stream its events, then a final done/error event."""
    events: queue.Queue = queue.Queue()

    def run() -> None:
        try:
            result = job(events.put) or {}
            events.put({"type": "done", **result})
        except Exception as exc:
            logger.exception("Job failed")
            events.put({"type": "error", "message": describe_error(exc)})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            event["cost_usd"] = round(session_usage.cost_usd, 4)
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


def _require_profile() -> dict[str, Any]:
    profile = store.load_profile()
    if profile is None:
        raise HTTPException(400, "Set up your interests first.")
    return profile


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def state() -> dict[str, Any]:
    return {
        "has_key": has_api_key(),
        "profile": store.load_profile(),
        "personas": [
            {"id": p["id"], "label": p["label"], "blurb": p["blurb"]}
            for p in profiles.load_personas().values()
        ],
        "categories": arxiv.CATEGORIES,
        "models": {"screen": config.SCREEN_MODEL, "assistant": config.AGENT_MODEL},
        "window_days": config.WINDOW_DAYS,
        "cost_usd": round(session_usage.cost_usd, 4),
    }


class KeyBody(BaseModel):
    key: str


@app.post("/api/key")
def save_key(body: KeyBody) -> dict[str, Any]:
    set_api_key(body.key)
    try:
        get_client().models.list(limit=1)  # free check that the key works
    except Exception as exc:
        set_api_key("")
        raise HTTPException(400, describe_error(exc))
    return {"ok": True}


@app.post("/api/profile/build")
async def build_profile(
    description: str = Form(""),
    papers: str = Form(""),
    file: Optional[UploadFile] = File(None),
) -> StreamingResponse:
    text = papers
    if file is not None:
        text += "\n" + (await file.read()).decode("utf-8", errors="replace")
    return _stream(lambda emit: {"profile": profiles.build_profile(description, text, emit)})


class PersonaBody(BaseModel):
    id: str


@app.post("/api/profile/persona")
def use_persona(body: PersonaBody) -> dict[str, Any]:
    try:
        return profiles.use_persona(body.id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class ThemeBody(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    examples: list[str] = []


class ProfileBody(BaseModel):
    themes: list[ThemeBody]
    not_interested: list[str]
    categories: list[str]


@app.put("/api/profile")
def update_profile(body: ProfileBody) -> dict[str, Any]:
    profile = _require_profile()
    themes = [t.model_dump() for t in body.themes if t.name.strip() and t.description.strip()]
    if not themes:
        raise HTTPException(400, "Keep at least one theme.")
    categories = [c for c in body.categories if c in arxiv.CATEGORIES]
    if not categories:
        raise HTTPException(400, "Pick at least one arXiv category.")
    for theme in themes:
        if not theme["id"]:
            del theme["id"]
    profile.update({
        "themes": themes,
        "not_interested": [x.strip() for x in body.not_interested if x.strip()],
        "categories": categories,
    })
    return store.save_profile(profile)


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    if config.PROFILE_PATH.exists():
        config.PROFILE_PATH.unlink()
    assistant.reset_history()
    return {"ok": True}


class RefreshBody(BaseModel):
    force: bool = False
    changed: bool = False  # the interests changed since the last screening


@app.post("/api/feed/refresh")
def refresh_feed(body: RefreshBody) -> StreamingResponse:
    profile = _require_profile()
    return _stream(lambda emit: screen.refresh(profile, emit, force_fetch=body.force, changed=body.changed))


@app.get("/api/feed")
def feed() -> dict[str, Any]:
    return screen.get_feed(_require_profile())


@app.get("/api/feed/report")
def feed_report() -> dict[str, Any]:
    """What the screener did with every paper: shown, scored low, dropped by title, or not screened yet."""
    return screen.get_report(_require_profile())


@app.get("/api/log")
def log(since: int = 0) -> dict[str, Any]:
    return {**activity.snapshot(since), "usage": session_usage.to_dict()}


@app.get("/api/log/events/{event_id}")
def log_event(event_id: int) -> dict[str, Any]:
    detail = activity.event_detail(event_id)
    if detail is None:
        raise HTTPException(404, "That log entry is no longer kept.")
    return {"sections": detail}


class VoteBody(BaseModel):
    paper_id: str
    vote: int


@app.post("/api/votes")
def vote(body: VoteBody) -> dict[str, Any]:
    store.set_vote(body.paper_id, body.vote)
    return {"ok": True}


class AskBody(BaseModel):
    message: str


@app.post("/api/assistant")
def ask(body: AskBody) -> StreamingResponse:
    _require_profile()
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Say something first.")
    return _stream(lambda emit: assistant.run(message, emit))


@app.post("/api/proposals/{proposal_id}/apply")
def apply_proposal(proposal_id: str) -> dict[str, Any]:
    try:
        return assistant.apply_proposal(proposal_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/proposals/{proposal_id}/discard")
def discard_proposal(proposal_id: str) -> dict[str, Any]:
    assistant.discard_proposal(proposal_id)
    return {"ok": True}
