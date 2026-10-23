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

import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from threading import Lock
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, create_model

from paperfind_llm import activity, arxiv, config, store
from paperfind_llm.activity import section
from paperfind_llm.claude import call, describe_error, fallback_params
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

For each paper return its id, the score, the interest it best matches (one of the interest names in the profile, or "none"), and a reason of at most 25 words saying what in the paper connects to that interest, addressed to the researcher ("Your ...")."""


class TitleScreen(BaseModel):
    selected: list[int]


def _scores_format(interests: list[str]) -> type[BaseModel]:
    """The scoring reply for one profile: each paper's interest must be one of its names, or "none".

    With a free-text field named "theme", the model often wrote the paper's topic there (29% of
    papers), and in about 1 batch in 10 it left the reason string open, wrote the next field
    inside it, and ended the list after one paper. With this schema, or the paper count stated
    in the request, or both: 0 short replies in 126 test batches, against 11 in 112 before.
    """
    names = tuple(dict.fromkeys([*interests, "none"]))
    item = create_model("PaperScore", id=(int, ...), score=(int, ...), interest=(Literal[names], ...), reason=(str, ...))
    return create_model("AbstractScores", papers=(list[item], ...))


OutputT = TypeVar("OutputT", bound=BaseModel)
Emit = Callable[[dict[str, Any]], None]


class ScreeningError(Exception):
    """A batch the model declined, or whose reply couldn't be read."""


# Where a reason runs on into JSON, markup, or a link: "…tasks.</reason>", "…','theme':'none", "…},{", "…](Your"
_RUN_ON = re.compile(r"</|\]\(|','|\},\s*\{")


def _clean_reason(text: str) -> str:
    """Cut a reason that ran on into other syntax, and trim stray trailing punctuation, e.g. 'heads.),'."""
    run_on = _RUN_ON.search(text)
    if run_on:
        text = text[: run_on.start()]
    text = text.strip().rstrip(",; ")
    while text.endswith(")") and text.count(")") > text.count("("):
        text = text[:-1].rstrip(",; ")
    return text[:400]


def window_start() -> str:
    return (date.today() - timedelta(days=config.WINDOW_DAYS - 1)).isoformat()


class Screener:
    """Screening calls for one profile."""

    def __init__(self, profile: dict[str, Any], draft: bool = False):
        self.profile_text = render(profile)
        self.model = config.SCREEN_MODEL
        self.draft = draft  # scoring an unsaved draft, for the assistant's test_draft
        self.scores_format = _scores_format([theme["name"] for theme in profile.get("themes", [])])
        self.skipped = 0  # papers the model left out of its scores twice, across batches
        self._lock = Lock()

    def _call(self, purpose: str, system_template: str, user: str, output_format: type[OutputT]) -> tuple[Any, OutputT]:
        response = call(
            purpose,
            model=self.model,
            retries=4,
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
        if response.stop_reason == "refusal":
            raise ScreeningError("The model declined this batch.")
        if response.parsed_output is None:
            raise ScreeningError(f"The model's reply couldn't be read (stop reason: {response.stop_reason}).")
        return response, response.parsed_output

    def screen_titles(self, batch: list[dict[str, Any]], where: str) -> list[tuple[str, bool]]:
        lines = [f"{i}. {paper['title']}" for i, paper in enumerate(batch, 1)]
        response, result = self._call(f"Title screen, {where}", TITLE_SYSTEM_PROMPT, "Titles:\n" + "\n".join(lines), TitleScreen)
        selected = {i for i in result.selected if 1 <= i <= len(batch)}
        kept = [paper["title"] for i, paper in enumerate(batch, 1) if i in selected]
        dropped = [paper["title"] for i, paper in enumerate(batch, 1) if i not in selected]
        activity.annotate(response, summary=f"kept {len(kept)} of {len(batch)} titles", detail=[
            section(f"Kept ({len(kept)})", kept, open=True, format="list"),
            section(f"Dropped ({len(dropped)})", dropped, format="list"),
        ])
        return [(paper["id"], i in selected) for i, paper in enumerate(batch, 1)]

    def score_abstracts(self, batch: list[dict[str, Any]], where: str) -> list[dict[str, Any]]:
        task = "Scoring under the draft" if self.draft else "Abstract scoring"
        scores: dict[str, dict[str, Any]] = {}
        # The model occasionally skips papers in a batch; ask once more for the missing ones
        for attempt in range(2):
            missing = [paper for paper in batch if paper["id"] not in scores]
            if not missing:
                break
            n = len(missing)
            blocks = [f"Score all {n} papers below: return {n} entries, with ids 1 to {n}."]
            for i, paper in enumerate(missing, 1):
                abstract = (paper.get("abstract") or "")[:MAX_ABSTRACT_CHARS]
                blocks.append(f"Paper {i}\nTitle: {paper['title']}\nAbstract: {abstract or '(none)'}")
            purpose = f"{task}, {where}" if attempt == 0 else f"{task}, {where}: retry for {n} skipped"
            try:
                response, result = self._call(purpose, ABSTRACT_SYSTEM_PROMPT, "\n\n".join(blocks), self.scores_format)
            except Exception:
                if attempt == 0:
                    raise
                break  # keep the first attempt's scores; the failed retry is in the log
            new = []
            for item in result.papers:
                if 1 <= item.id <= len(missing) and missing[item.id - 1]["id"] not in scores:
                    paper = missing[item.id - 1]
                    scores[paper["id"]] = {
                        "paper_id": paper["id"],
                        "score": float(max(0, min(10, item.score))),
                        "theme": item.interest,
                        "reason": _clean_reason(item.reason),
                    }
                    new.append((scores[paper["id"]], paper["title"]))
            new.sort(key=lambda pair: pair[0]["score"], reverse=True)
            shown = sum(1 for score, _ in new if score["score"] >= MIN_SCORE)
            best = f", best {new[0][0]['score']:.0f}" if new else ""
            activity.annotate(response, summary=f"scored {len(new)} of {len(missing)}: {shown} at {MIN_SCORE}+{best}", detail=[
                section("Scores", [f"{s['score']:.0f} · {title} · {s['theme']}: {s['reason']}" for s, title in new], open=True, format="list"),
            ])

        with self._lock:
            self.skipped += len(batch) - len(scores)
        return list(scores.values())

    def record_skipped(self) -> None:
        if self.skipped:
            task = "Scoring under the draft" if self.draft else "Abstract scoring"
            later = "They show as ? in the test." if self.draft else "They'll be scored on the next refresh."
            activity.issue("skipped", f"{task}: the model left out {_papers(self.skipped)} twice. {later}", count=self.skipped)


def _run_batches(fn: Callable, items: list, batch_size: int, on_batch: Callable[[int, int], None]) -> tuple[list, list]:
    """Run fn(batch, "batch i of n") over batches in parallel, reporting (done, total) after each batch.

    Returns the combined results and the failed batches as (batch, exception) pairs.
    Raises only if every batch fails. Papers in a failed batch are retried on the next refresh.
    """
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results, failures, done = [], [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Each batch runs in a copy of this context, so its log entries land in the current run
        futures = {
            pool.submit(contextvars.copy_context().run, fn, batch, f"batch {i} of {len(batches)}"): batch
            for i, batch in enumerate(batches, 1)
        }
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                failures.append((futures[future], exc))
            done += 1
            on_batch(done, len(batches))
    if failures and len(failures) == len(batches):
        raise failures[0][1]
    return results, failures


def _papers(n: int) -> str:
    return f"{n} paper{'s' if n != 1 else ''}"


def _failed_note(failures: list) -> str:
    return f"; {len(failures)} batch{'es' if len(failures) != 1 else ''} failed" if failures else ""


def _record_failures(stage: str, failures: list, total_batches: int) -> None:
    if failures:
        papers = sum(len(batch) for batch, _ in failures)
        activity.issue(
            "failed_batch",
            f"{stage}: {len(failures)} of {total_batches} batches failed, so {_papers(papers)} weren't screened "
            f"({describe_error(failures[0][1])})",
            count=len(failures),
        )


_refresh_lock = Lock()


def refresh(profile: dict[str, Any], emit: Emit, force_fetch: bool = False, changed: bool = False) -> dict[str, Any]:
    """Fetch new papers if needed, then screen everything not yet screened under this profile.

    Returns the feed, plus "issues": problems that didn't stop the refresh, such as failed batches.
    """
    if not _refresh_lock.acquire(blocking=False):
        raise ValueError("A refresh is already running.")
    try:
        if changed:
            title = "Re-screen with your new interests"
        elif force_fetch:
            title = "Check arXiv and screen new papers"
        else:
            title = "Screen today's papers"
        with activity.run("screener", title, emit, keep_empty=False) as log:
            fetched = _fetch_if_stale(profile, emit, force_fetch)
            _screen(profile, emit, fetched)
            feed = get_feed(profile)
            if log.events:
                stats = feed["stats"]
                log.update(summary=(
                    f"{stats['fetched']} papers, {stats['passed_titles']} passed the title screen, "
                    f"{stats['shown']} shown"
                ))
        return {"feed": feed, "issues": log.issues}
    finally:
        _refresh_lock.release()


def _fetch_if_stale(profile: dict[str, Any], emit: Emit, force: bool) -> bool:
    """Fetch papers unless the last fetch is recent. Returns whether it fetched."""
    categories = sorted(profile.get("categories") or [])
    meta = store.get_meta("last_fetch") or {}
    fresh = (
        meta.get("categories") == categories
        and meta.get("since") == window_start()
        and time.time() - meta.get("at", 0) < FETCH_MAX_AGE_SECONDS
    )
    if fresh and not force:
        return False

    emit({"type": "progress", "stage": "fetch", "text": f"Fetching today's arXiv papers in {', '.join(categories)}"})
    with activity.span("step", "Fetch today's arXiv announcement (RSS)") as step:
        papers = arxiv.fetch_announced(categories)
        step.finish(summary=f"{len(papers)} new papers in {', '.join(categories)}")
    if not papers:
        # No announcement today (weekend or holiday): search the last few days instead
        emit({"type": "progress", "stage": "fetch", "text": "No announcement today. Searching the last few days"})
        with activity.span("step", f"Search arXiv for the last {config.WINDOW_DAYS} days (no announcement today)") as step:
            papers = arxiv.fetch_recent(
                categories,
                window_start(),
                on_page=lambda n: emit({"type": "progress", "stage": "fetch", "text": f"Fetched {n} papers"}),
            )
            step.finish(summary=f"{len(papers)} papers")
    store.upsert_papers(papers)
    store.set_meta("last_fetch", {"categories": categories, "since": window_start(), "at": time.time(), "count": len(papers)})
    return True


def _screen(profile: dict[str, Any], emit: Emit, fetched: bool) -> None:
    key = profile_hash(profile)
    papers = store.get_papers_since(window_start(), profile.get("categories"))
    if not papers:
        emit({"type": "progress", "stage": "done", "text": "No new papers in these categories yet."})
        return

    cached = store.get_screening([p["id"] for p in papers], key)
    screener = Screener(profile)

    to_screen = [p for p in papers if p["id"] not in cached]
    unscored = [p for p in papers if cached.get(p["id"], {}).get("passed") and cached[p["id"]]["score"] is None]
    if to_screen or unscored or fetched:
        done = len(papers) - len(to_screen) - len(unscored)
        activity.begin(
            "step", "Check the screening cache",
            summary=f"{done} of {len(papers)} papers already screened under these interests",
        ).finish()

    if to_screen:
        batches = -(-len(to_screen) // TITLE_BATCH_SIZE)

        def on_titles(done: int, total: int) -> None:
            emit({
                "type": "progress", "stage": "titles", "done": done, "total": total,
                "text": f"Reading {len(to_screen)} titles",
            })

        on_titles(0, batches)
        with activity.span("group", "Pass 1: title screen") as group:
            outcomes, failures = _run_batches(screener.screen_titles, to_screen, TITLE_BATCH_SIZE, on_titles)
            kept = sum(1 for _, passed in outcomes if passed)
            group.finish(
                "error" if failures else "ok",
                summary=f"read {len(outcomes)} titles, kept {kept}" + _failed_note(failures),
            )
        _record_failures("Title screen", failures, batches)
        store.save_title_screen(outcomes, key)
        cached = store.get_screening([p["id"] for p in papers], key)

    to_score = [p for p in papers if cached.get(p["id"], {}).get("passed") and cached[p["id"]]["score"] is None]
    if to_score:
        batches = -(-len(to_score) // ABSTRACT_BATCH_SIZE)

        def on_scores(done: int, total: int) -> None:
            emit({
                "type": "progress", "stage": "abstracts", "done": done, "total": total,
                "text": f"Scoring {len(to_score)} abstracts",
            })

        on_scores(0, batches)
        with activity.span("group", "Pass 2: abstract scoring") as group:
            scores, failures = _run_batches(screener.score_abstracts, to_score, ABSTRACT_BATCH_SIZE, on_scores)
            shown = sum(1 for s in scores if s["score"] >= MIN_SCORE)
            group.finish(
                "error" if failures else "ok",
                summary=f"scored {len(scores)} abstracts, {shown} at {MIN_SCORE}+" + _failed_note(failures),
            )
        _record_failures("Abstract scoring", failures, batches)
        screener.record_skipped()
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
            "unscored": sum(1 for r in cached.values() if r["passed"] and r["score"] is None),
        },
        "min_score": MIN_SCORE,
        "since": window_start(),
    }


def get_report(profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every paper in the window by what the screener did with it: shown, scored low, dropped by title, or pending."""
    key = profile_hash(profile)
    papers = store.get_papers_since(window_start(), profile.get("categories"))
    cached = store.get_screening([p["id"] for p in papers], key)
    report: dict[str, list[dict[str, Any]]] = {"shown": [], "low": [], "title": [], "pending": []}
    for paper in papers:
        result = cached.get(paper["id"])
        item = {"id": paper["id"], "title": paper["title"], "url": paper["url"], "score": None, "theme": None, "reason": None}
        if result is None or (result["passed"] and result["score"] is None):
            report["pending"].append(item)
        elif not result["passed"]:
            report["title"].append(item)
        else:
            item.update(score=result["score"], theme=result["theme"], reason=result["reason"])
            report["shown" if result["score"] >= MIN_SCORE else "low"].append(item)
    for group in ("shown", "low"):
        report[group].sort(key=lambda p: p["score"], reverse=True)
    return report


def rescore(profile: dict[str, Any], papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Score papers under a (possibly unsaved) profile without caching. Used to test edits."""
    screener = Screener(profile, draft=True)
    results, failures = _run_batches(screener.score_abstracts, papers, ABSTRACT_BATCH_SIZE, lambda done, total: None)
    _record_failures("Scoring under the draft", failures, -(-len(papers) // ABSTRACT_BATCH_SIZE))
    screener.record_skipped()
    return {r["paper_id"]: r for r in results}
