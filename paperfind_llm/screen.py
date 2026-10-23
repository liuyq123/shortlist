"""
Two-pass LLM screening of recent arXiv papers.

1. Title screen: numbered titles in batches of 150. The model keeps anything
   plausibly relevant to the profile, generously, since pass 2 reads abstracts.
2. Abstract scoring: survivors in batches of 15 get a 0-10 score, the matching
   theme, and a one-sentence reason.

Both passes use SCREEN_MODEL (default claude-haiku-5-5) and the profile is
cached as the system prompt, so a day of papers costs about a cent. Results are
stored per (paper, profile version), so nothing is screened twice.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from threading import Lock
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from paperfind_llm import arxiv, config, store
from paperfind_llm.claude import fallback_params, get_client, session_usage
from paperfind_llm.profile import profile_hash, render

TITLE_BATCH_SIZE = 150
ABSTRACT_BATCH_SIZE = 15
MAX_ABSTRACT_CHARS = 1500
MAX_WORKERS = 4
MIN_SCORE = 4  # papers below this are hidden from the feed
FETCH_MAX_AGE_SECONDS = 3600

TITLE_SYSTEM_PROMPT = """You screen new research papers for one researcher. Their interest profile:

{profile}

You will get a numbered list of new paper titles. Select every paper that could plausibly match one of their interests. Be generous: a second pass reads the abstracts, so include a paper when in doubt. Leave out papers that clearly match none of the interests or that fall under NOT INTERESTED IN."""

ABSTRACT_SYSTEM_PROMPT = """You score new research papers for one researcher. Their interest profile:

{profile}

Score each paper from 0 to 10 for how much this researcher would want to read it:
- 9-10: squarely within a core interest; they would want to read it today
- 7-8: clearly relevant to one of their interests
- 4-6: related, e.g. shares a method or system but is not their focus
- 0-3: unrelated, or falls under NOT INTERESTED IN

For each paper return its id, the score, the name of the best-matching interest exactly as written in the profile (or "none"), and a reason of at most 25 words saying what in the paper connects to that interest, addressed to the researcher ("Your ...")."""


class TitleScreen(BaseModel):
    selected: list[int]


class PaperScore(BaseModel):
    id: int
    score: int
    theme: str
    reason: str


class AbstractScores(BaseModel):
    papers: list[PaperScore]


OutputT = TypeVar("OutputT", bound=BaseModel)
Emit = Callable[[dict[str, Any]], None]


def _clean_reason(text: str) -> str:
    """Trim stray trailing punctuation the model sometimes leaves, e.g. 'heads.),'."""
    text = text.strip().rstrip(",; ")
    while text.endswith(")") and text.count(")") > text.count("("):
        text = text[:-1].rstrip(",; ")
    return text[:400]


def window_start() -> str:
    return (date.today() - timedelta(days=config.WINDOW_DAYS - 1)).isoformat()


class Screener:
    """Screening calls for one profile."""

    def __init__(self, profile: dict[str, Any]):
        self.profile_text = render(profile)
        self.model = config.SCREEN_MODEL
        self.client = get_client(max_retries=4)

    def _call(self, system_template: str, user: str, output_format: type[OutputT]) -> Optional[OutputT]:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=8000,
            # The profile is the same for every batch, so cache it
            system=[{
                "type": "text",
                "text": system_template.format(profile=self.profile_text),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
            output_config={"effort": "low"},
            **fallback_params(self.model),
        )
        session_usage.add(response.usage, self.model)
        if response.stop_reason == "refusal":
            return None
        return response.parsed_output

    def screen_titles(self, batch: list[dict[str, Any]]) -> list[tuple[str, bool]]:
        lines = [f"{i}. {paper['title']}" for i, paper in enumerate(batch, 1)]
        result = self._call(TITLE_SYSTEM_PROMPT, "Titles:\n" + "\n".join(lines), TitleScreen)
        if result is None:
            return []
        selected = {i for i in result.selected if 1 <= i <= len(batch)}
        return [(paper["id"], i in selected) for i, paper in enumerate(batch, 1)]

    def score_abstracts(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        # The model occasionally skips papers in a batch; ask once more for the missing ones
        for _ in range(2):
            missing = [paper for paper in batch if paper["id"] not in scores]
            if not missing:
                break
            blocks = []
            for i, paper in enumerate(missing, 1):
                abstract = (paper.get("abstract") or "")[:MAX_ABSTRACT_CHARS]
                blocks.append(f"Paper {i}\nTitle: {paper['title']}\nAbstract: {abstract or '(none)'}")
            result = self._call(ABSTRACT_SYSTEM_PROMPT, "\n\n".join(blocks), AbstractScores)
            if result is None:
                break
            for item in result.papers:
                if 1 <= item.id <= len(missing):
                    paper_id = missing[item.id - 1]["id"]
                    scores[paper_id] = {
                        "paper_id": paper_id,
                        "score": float(max(0, min(10, item.score))),
                        "theme": item.theme.strip()[:120],
                        "reason": _clean_reason(item.reason),
                    }
        return list(scores.values())


def _run_batches(fn: Callable, items: list, batch_size: int, on_batch: Callable[[int, int], None]) -> list:
    """Run fn over batches in parallel, reporting (done, total) after each batch.

    A failed batch is skipped (and retried on the next refresh) unless every batch fails.
    """
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results, errors, done = [], [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fn, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                errors.append(exc)
            done += 1
            on_batch(done, len(batches))
    if errors and len(errors) == len(batches):
        raise errors[0]
    return results


_refresh_lock = Lock()


def refresh(profile: dict[str, Any], emit: Emit, force_fetch: bool = False) -> None:
    """Fetch new papers if needed, then screen everything not yet screened under this profile."""
    if not _refresh_lock.acquire(blocking=False):
        raise ValueError("A refresh is already running.")
    try:
        _fetch_if_stale(profile, emit, force_fetch)
        _screen(profile, emit)
    finally:
        _refresh_lock.release()


def _fetch_if_stale(profile: dict[str, Any], emit: Emit, force: bool) -> None:
    categories = sorted(profile.get("categories") or [])
    meta = store.get_meta("last_fetch") or {}
    fresh = (
        meta.get("categories") == categories
        and meta.get("since") == window_start()
        and time.time() - meta.get("at", 0) < FETCH_MAX_AGE_SECONDS
    )
    if fresh and not force:
        return

    emit({"type": "progress", "stage": "fetch", "text": f"Fetching today's arXiv papers in {', '.join(categories)}"})
    papers = arxiv.fetch_announced(categories)
    if not papers:
        # No announcement today (weekend or holiday): search the last few days instead
        emit({"type": "progress", "stage": "fetch", "text": "No announcement today. Searching the last few days"})
        papers = arxiv.fetch_recent(
            categories,
            window_start(),
            on_page=lambda n: emit({"type": "progress", "stage": "fetch", "text": f"Fetched {n} papers"}),
        )
    store.upsert_papers(papers)
    store.set_meta("last_fetch", {"categories": categories, "since": window_start(), "at": time.time(), "count": len(papers)})


def _screen(profile: dict[str, Any], emit: Emit) -> None:
    key = profile_hash(profile)
    papers = store.get_papers_since(window_start(), profile.get("categories"))
    if not papers:
        emit({"type": "progress", "stage": "done", "text": "No new papers in these categories yet."})
        return

    cached = store.get_screening([p["id"] for p in papers], key)
    screener = Screener(profile)

    to_screen = [p for p in papers if p["id"] not in cached]
    if to_screen:
        def on_titles(done: int, total: int) -> None:
            emit({
                "type": "progress", "stage": "titles", "done": done, "total": total,
                "text": f"Reading {len(to_screen)} titles",
            })

        on_titles(0, -(-len(to_screen) // TITLE_BATCH_SIZE))
        outcomes = _run_batches(screener.screen_titles, to_screen, TITLE_BATCH_SIZE, on_titles)
        store.save_title_screen(outcomes, key)
        cached = store.get_screening([p["id"] for p in papers], key)

    to_score = [p for p in papers if cached.get(p["id"], {}).get("passed") and cached[p["id"]]["score"] is None]
    if to_score:
        def on_scores(done: int, total: int) -> None:
            emit({
                "type": "progress", "stage": "abstracts", "done": done, "total": total,
                "text": f"Scoring {len(to_score)} abstracts",
            })

        on_scores(0, -(-len(to_score) // ABSTRACT_BATCH_SIZE))
        scores = _run_batches(screener.score_abstracts, to_score, ABSTRACT_BATCH_SIZE, on_scores)
        store.save_scores(scores, key)


def get_feed(profile: dict[str, Any]) -> dict[str, Any]:
    """The ranked feed under the current profile, from the screening cache."""
    key = profile_hash(profile)
    papers = store.get_papers_since(window_start(), profile.get("categories"))
    cached = store.get_screening([p["id"] for p in papers], key)
    votes = store.get_votes()

    items = []
    for paper in papers:
        result = cached.get(paper["id"])
        if not result or result["score"] is None:
            continue
        items.append({
            **paper,
            "score": result["score"],
            "theme": result["theme"],
            "reason": result["reason"],
            "vote": votes.get(paper["id"], 0),
        })
    items.sort(key=lambda p: (p["score"], p["published"]), reverse=True)

    return {
        "papers": items,
        "stats": {
            "fetched": len(papers),
            "screened": len(cached),
            "passed_titles": sum(1 for r in cached.values() if r["passed"]),
            "shown": sum(1 for p in items if p["score"] >= MIN_SCORE),
            "unscreened": len(papers) - len(cached),
        },
        "min_score": MIN_SCORE,
        "since": window_start(),
    }


def rescore(profile: dict[str, Any], papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Score papers under a (possibly unsaved) profile without caching. Used to test edits."""
    screener = Screener(profile)
    results = _run_batches(screener.score_abstracts, papers, ABSTRACT_BATCH_SIZE, lambda done, total: None)
    return {r["paper_id"]: r for r in results}
