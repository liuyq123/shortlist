"""
Local storage: one SQLite database plus profile.json in ~/.paperfind-llm.

Tables:
    papers      arXiv papers fetched for the feed
    screening   the model's verdict per (paper, profile version): passed the title
                screen? abstract score, matching theme, reason
    votes       the user's thumbs up/down
    feedback    assistant conversations and whether their changes were applied
    meta        small key-value state, e.g. when papers were last fetched
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from paperfind_llm import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    categories TEXT,
    published TEXT,
    url TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS papers_published ON papers (published);

CREATE TABLE IF NOT EXISTS screening (
    paper_id TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    passed INTEGER NOT NULL,
    score REAL,
    theme TEXT,
    reason TEXT,
    screened_at TEXT NOT NULL,
    PRIMARY KEY (paper_id, profile_hash)
);

CREATE TABLE IF NOT EXISTS votes (
    paper_id TEXT PRIMARY KEY,
    vote INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    message TEXT,
    reply TEXT,
    changes TEXT,
    applied INTEGER NOT NULL DEFAULT 0
);
"""

_CHUNK = 500


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# =============================================================================
# Papers
# =============================================================================


def upsert_papers(papers: list[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO papers (id, title, authors, abstract, categories, published, url, fetched_at)
            VALUES (:id, :title, :authors, :abstract, :categories, :published, :url, :fetched_at)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title, authors = excluded.authors, abstract = excluded.abstract,
                categories = excluded.categories, published = excluded.published, url = excluded.url
            """,
            [{**p, "fetched_at": now()} for p in papers],
        )


def get_papers_since(since: str, categories: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Papers published on or after `since` (YYYY-MM-DD), newest first, optionally in any of `categories`."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE published >= ? ORDER BY published DESC, id DESC", (since,)
        ).fetchall()
    papers = [dict(row) for row in rows]
    if categories:
        wanted = set(categories)
        papers = [p for p in papers if wanted & set((p["categories"] or "").split())]
    return papers


def get_paper(paper_id: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return dict(row) if row else None


def search_papers(query: str, since: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recent papers whose title or abstract contains every word of the query."""
    words = [w for w in query.lower().split() if len(w) > 1][:6]
    if not words:
        return []
    clause = " AND ".join("(lower(title) LIKE ? OR lower(abstract) LIKE ?)" for _ in words)
    params: list[Any] = [since]
    for word in words:
        params += [f"%{word}%", f"%{word}%"]
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM papers WHERE published >= ? AND {clause} ORDER BY published DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(row) for row in rows]


# =============================================================================
# Screening cache
# =============================================================================


def get_screening(paper_ids: list[str], profile_hash: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with connect() as conn:
        for start in range(0, len(paper_ids), _CHUNK):
            chunk = paper_ids[start : start + _CHUNK]
            marks = ", ".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT * FROM screening WHERE profile_hash = ? AND paper_id IN ({marks})",
                (profile_hash, *chunk),
            ).fetchall()
            results.update({row["paper_id"]: dict(row) for row in rows})
    return results


def save_title_screen(results: list[tuple[str, bool]], profile_hash: str) -> None:
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO screening (paper_id, profile_hash, passed, score, theme, reason, screened_at)
            VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            ON CONFLICT(paper_id, profile_hash) DO UPDATE SET
                passed = excluded.passed, score = NULL, theme = NULL, reason = NULL,
                screened_at = excluded.screened_at
            """,
            [(paper_id, profile_hash, int(passed), now()) for paper_id, passed in results],
        )


def save_scores(results: list[dict[str, Any]], profile_hash: str) -> None:
    with connect() as conn:
        conn.executemany(
            """
            UPDATE screening SET score = :score, theme = :theme, reason = :reason, screened_at = :at
            WHERE paper_id = :paper_id AND profile_hash = :hash
            """,
            [{**r, "hash": profile_hash, "at": now()} for r in results],
        )


# =============================================================================
# Votes and feedback
# =============================================================================


def set_vote(paper_id: str, vote: int) -> None:
    with connect() as conn:
        if vote == 0:
            conn.execute("DELETE FROM votes WHERE paper_id = ?", (paper_id,))
        else:
            conn.execute(
                """
                INSERT INTO votes (paper_id, vote, created_at) VALUES (?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET vote = excluded.vote, created_at = excluded.created_at
                """,
                (paper_id, 1 if vote > 0 else -1, now()),
            )


def get_votes() -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute("SELECT paper_id, vote FROM votes").fetchall()
    return {row["paper_id"]: row["vote"] for row in rows}


def get_recent_votes(limit: int = 30) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT v.paper_id, v.vote, v.created_at, p.title
            FROM votes v LEFT JOIN papers p ON p.id = v.paper_id
            ORDER BY v.created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def log_feedback(message: str, reply: str, changes: list[str], applied: bool) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (created_at, message, reply, changes, applied) VALUES (?, ?, ?, ?, ?)",
            (now(), message, reply, json.dumps(changes), int(applied)),
        )
        return int(cur.lastrowid)


def mark_feedback_applied(feedback_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE feedback SET applied = 1 WHERE id = ?", (feedback_id,))


def get_recent_feedback(limit: int = 8) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT created_at, message, changes, applied FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {**dict(row), "changes": json.loads(row["changes"] or "[]"), "applied": bool(row["applied"])}
        for row in reversed(rows)
    ]


def get_meta(key: str) -> Any:
    with connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else None


def set_meta(key: str, value: Any) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )


# =============================================================================
# Profile
# =============================================================================


def new_theme_id() -> str:
    return uuid.uuid4().hex[:6]


def load_profile() -> Optional[dict[str, Any]]:
    if not config.PROFILE_PATH.exists():
        return None
    try:
        profile = json.loads(config.PROFILE_PATH.read_text())
    except (OSError, ValueError):
        return None
    return profile if isinstance(profile, dict) and profile.get("themes") is not None else None


def save_profile(profile: dict[str, Any]) -> dict[str, Any]:
    for theme in profile.get("themes", []):
        theme.setdefault("id", new_theme_id())
        theme.setdefault("examples", [])
    profile["updated_at"] = now()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.PROFILE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(config.PROFILE_PATH)
    return profile
