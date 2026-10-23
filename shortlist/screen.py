"""
Two-pass LLM screening of recent arXiv papers.

1. Title screen: numbered titles in batches of 150. The model keeps anything
   plausibly relevant to the profile, generously, since pass 2 reads abstracts.
2. Abstract scoring: survivors in batches of 15 get a 0-10 score, the matching
   interest, and a one-sentence reason.

Each title batch is followed by scoring its survivors, and the feed is sent
after every scored batch, so papers appear while the rest are still screened.
Batches run in parallel only as far as the allowance can hold their requests
(each one holds its worst-case cost while it runs); otherwise one at a time, in order.

Both passes use the model the visitor selected in the class app store, through
the gateway. Replies are JSON checked by this app. Results are stored per
(paper, profile version), so nothing is screened twice.
"""

import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from threading import Lock
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from shortlist import activity, arxiv, config, gateway, store
from shortlist.activity import section
from shortlist.gateway import ReplyError, call_json, describe_error
from shortlist.profile import profile_hash, render

TITLE_BATCH_SIZE = 150
ABSTRACT_BATCH_SIZE = 15
MAX_ABSTRACT_CHARS = 1500
MAX_OUTPUT_TOKENS = 4096
MAX_WORKERS = 3  # parallel batches at most, when the allowance can hold that many requests
MIN_SCORE = 4  # papers below this are hidden from the feed
FETCH_MAX_AGE_SECONDS = 3600

TITLE_SYSTEM_PROMPT = """You screen new research papers for one researcher. Their interest profile:

{profile}

You will get a numbered list of new paper titles. Select every paper that could plausibly match one of their interests. Be generous: a second pass reads the abstracts, so include a paper when in doubt. Leave out papers that clearly match none of the interests or that fall under NOT INTERESTED IN.

Reply with only a JSON object listing the numbers of the titles you select, for example:
{"selected": [3, 8, 15]}"""

ABSTRACT_SYSTEM_PROMPT = """You score new research papers for one researcher. Their interest profile:

{profile}

Score each paper from 0 to 10 for how much this researcher would want to read it:
- 9-10: squarely within a core interest; they would want to read it today
- 7-8: clearly relevant to one of their interests
- 4-6: related, e.g. shares a method or system but is not their focus
- 0-3: unrelated, or falls under NOT INTERESTED IN

For each paper return its id, the score, the interest it best matches, and a reason of at most 25 words saying what in the paper connects to that interest, addressed to the researcher ("Your ...").

For "interest", use one of these names exactly, or "none": {interests}

Reply with only a JSON object with one entry per paper, for example:
{"papers": [{"id": 1, "score": 7, "interest": "<an interest name or none>", "reason": "Your ..."}]}"""


class TitleScreen(BaseModel):
    selected: list[int]


class PaperScore(BaseModel):
    id: int
    score: float
    interest: str = "none"
    reason: str = ""


class AbstractScores(BaseModel):
    papers: list[PaperScore]


OutputT = TypeVar("OutputT", bound=BaseModel)
Emit = Callable[[dict[str, Any]], None]


class ScreeningError(Exception):
    """A batch whose reply couldn't be used: unreadable, declined, or cut off."""


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
        self.interests = [theme["name"] for theme in profile.get("themes", [])]
        self.draft = draft  # scoring an unsaved draft, for the assistant's test_draft
        self.skipped = 0  # papers the model left out of its scores twice, across batches
        self._lock = Lock()

    def _call(self, purpose: str, template: str, user: str, output: type[OutputT]) -> tuple[dict[str, Any], OutputT]:
        system = template.replace("{profile}", self.profile_text).replace(
            "{interests}", ", ".join(f'"{name}"' for name in self.interests)
        )
        try:
            return call_json(purpose, system=system, user=user, output=output, max_tokens=MAX_OUTPUT_TOKENS)
        except ReplyError as exc:
            raise ScreeningError(str(exc)) from None

    def _interest(self, name: str) -> str:
        """The profile's interest name the model meant, or "none" for anything else."""
        wanted = name.strip().lower()
        return next((interest for interest in self.interests if interest.lower() == wanted), "none")

    def screen_titles(self, batch: list[dict[str, Any]], where: str) -> list[tuple[str, bool]]:
        lines = [f"{i}. {paper['title']}" for i, paper in enumerate(batch, 1)]
        user = "Titles:\n" + "\n".join(lines)
        # An unreadable reply gets one fresh request; gateway errors are never resent
        try:
            response, result = self._call(f"Title screen, {where}", TITLE_SYSTEM_PROMPT, user, TitleScreen)
        except ScreeningError:
            response, result = self._call(f"Title screen, {where}: asking again after an unreadable reply", TITLE_SYSTEM_PROMPT, user, TitleScreen)
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
        # If the reply is unreadable or leaves papers out, ask once more for the missing ones
        for attempt in range(2):
            missing = [paper for paper in batch if paper["id"] not in scores]
            if not missing:
                break
            n = len(missing)
            blocks = [f"Score all {n} papers below: return {n} entries, with ids 1 to {n}."]
            for i, paper in enumerate(missing, 1):
                abstract = (paper.get("abstract") or "")[:MAX_ABSTRACT_CHARS]
                blocks.append(f"Paper {i}\nTitle: {paper['title']}\nAbstract: {abstract or '(none)'}")
            purpose = f"{task}, {where}" if attempt == 0 else f"{task}, {where}: asking again for {n} missing"
            try:
                response, result = self._call(purpose, ABSTRACT_SYSTEM_PROMPT, "\n\n".join(blocks), AbstractScores)
            except ScreeningError:
                if attempt == 0:
                    continue
                if not scores:
                    raise
                break  # keep the first attempt's scores; the failed request is in the log
            except Exception:
                if scores:
                    break  # keep the first attempt's scores; gateway errors are not resent
                raise
            new = []
            for item in result.papers:
                if 1 <= item.id <= len(missing) and missing[item.id - 1]["id"] not in scores:
                    paper = missing[item.id - 1]
                    scores[paper["id"]] = {
                        "paper_id": paper["id"],
                        "score": float(max(0, min(10, round(item.score)))),
                        "theme": self._interest(item.interest),
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


def _run_batches(fn: Callable, items: list, batch_size: int) -> tuple[list, list]:
    """Run fn(batch, "batch i of n") over batches in parallel.

    Returns the combined results and the failed batches as (batch, exception) pairs.
    Raises only if every batch fails. Papers in a failed batch are retried on the next refresh.
    """
    batches = _split(items, batch_size)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=gateway.allowance.parallel(MAX_OUTPUT_TOKENS, MAX_WORKERS)) as pool:
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
    if failures and len(failures) == len(batches):
        raise failures[0][1]
    return results, failures


def _split(items: list, size: int) -> list[list]:
    """Split into the fewest batches of at most `size`, evenly (48 -> 12, 12, 12, 12), so no batch is tiny."""
    if not items:
        return []
    count = -(-len(items) // size)
    base, extra = divmod(len(items), count)
    batches, start = [], 0
    for i in range(count):
        end = start + base + (1 if i < extra else 0)
        batches.append(items[start:end])
        start = end
    return batches


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
    to_screen = [p for p in papers if p["id"] not in cached]
    unscored = [p for p in papers if cached.get(p["id"], {}).get("passed") and cached[p["id"]]["score"] is None]
    if to_screen or unscored or fetched:
        done = len(papers) - len(to_screen) - len(unscored)
        activity.begin(
            "step", "Check the screening cache",
            summary=f"{done} of {len(papers)} papers already screened under these interests",
        ).finish()
    if not to_screen and not unscored:
        return

    # Work items: (name, where, titles to screen, papers already past the title screen).
    # Papers that passed the title screen last time but have no score yet go first.
    title_batches = _split(to_screen, TITLE_BATCH_SIZE)
    work = [(f"{_papers(len(unscored))} left unscored last time", "left from last time", [], unscored)] if unscored else []
    work += [
        (f"Batch {i} of {len(title_batches)}: {len(batch)} titles", f"batch {i} of {len(title_batches)}", batch, [])
        for i, batch in enumerate(title_batches, 1)
    ]
    tally = _Tally(profile, emit, total=len(to_screen) + len(unscored))
    screener = Screener(profile)
    tally.progress()

    def screen_batch(name: str, where: str, titles: list[dict[str, Any]], survivors: list[dict[str, Any]]) -> None:
        with activity.span("group", name) as group:
            notes = []
            if titles:
                try:
                    outcomes = screener.screen_titles(titles, where)
                except Exception as exc:
                    tally.failed("titles", titles, exc)
                    group.finish("error", summary=f"title screen failed: {describe_error(exc)}")
                    return
                store.save_title_screen(outcomes, key)
                passed = {paper_id for paper_id, ok in outcomes if ok}
                survivors = survivors + [p for p in titles if p["id"] in passed]
                tally.titles_done(len(titles), len(passed))
                notes.append(f"kept {len(passed)} of {len(titles)} titles")
            parts = _split(survivors, ABSTRACT_BATCH_SIZE)
            scored = shown = failed = 0
            for i, part in enumerate(parts, 1):
                try:
                    scores = screener.score_abstracts(part, f"{where}, part {i} of {len(parts)}" if len(parts) > 1 else where)
                except Exception as exc:
                    tally.failed("scores", part, exc)
                    failed += 1
                    continue
                store.save_scores(scores, key)
                scored += len(scores)
                shown += sum(1 for score in scores if score["score"] >= MIN_SCORE)
                tally.scores_done(part, scores)
            if parts:
                notes.append(f"scored {scored}, {shown} at {MIN_SCORE}+")
            group.finish(
                "error" if failed else "ok",
                summary="; ".join(notes) + (f"; {failed} of {len(parts)} scoring batches failed" if failed else ""),
            )

    # With room for one request at a time, a single worker takes the batches in order
    with ThreadPoolExecutor(max_workers=gateway.allowance.parallel(MAX_OUTPUT_TOKENS, MAX_WORKERS)) as pool:
        # Each batch runs in a copy of this context, so its log entries land in the current run
        futures = [pool.submit(contextvars.copy_context().run, screen_batch, *item) for item in work]
        for future in futures:
            future.result()

    _record_failures("Title screen", tally.title_failures, len(title_batches))
    _record_failures("Abstract scoring", tally.score_failures, tally.score_batches)
    screener.record_skipped()
    if tally.succeeded == 0 and (tally.title_failures or tally.score_failures):
        raise (tally.title_failures or tally.score_failures)[0][1]


class _Tally:
    """Progress across batch threads: papers screened so far, failures, and the feed after each scored batch."""

    def __init__(self, profile: dict[str, Any], emit: Emit, total: int):
        self.profile, self.emit, self.total = profile, emit, total
        self.done = self.shown = self.succeeded = self.score_batches = 0
        self.title_failures: list = []
        self.score_failures: list = []
        self._lock = Lock()

    def progress(self) -> None:
        with self._lock:
            text = f"{_papers(self.shown)} worth a look so far" if self.shown else f"Reading {_papers(self.total)}"
            event = {"type": "progress", "stage": "screen", "unit": "papers", "done": self.done, "total": self.total, "text": text}
        self.emit(event)

    def titles_done(self, read: int, kept: int) -> None:
        with self._lock:
            self.done += read - kept  # dropped titles are finished; kept ones finish when scored
            self.succeeded += 1
        self.progress()

    def scores_done(self, part: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
        with self._lock:
            self.done += len(part)
            self.shown += sum(1 for score in scores if score["score"] >= MIN_SCORE)
            self.succeeded += 1
            self.score_batches += 1
        self.emit({"type": "feed", "feed": get_feed(self.profile)})
        self.progress()

    def failed(self, stage: str, batch: list[dict[str, Any]], exc: Exception) -> None:
        with self._lock:
            self.done += len(batch)
            if stage == "titles":
                self.title_failures.append((batch, exc))
            else:
                self.score_failures.append((batch, exc))
                self.score_batches += 1
        self.progress()


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
    results, failures = _run_batches(screener.score_abstracts, papers, ABSTRACT_BATCH_SIZE)
    _record_failures("Scoring under the draft", failures, len(_split(papers, ABSTRACT_BATCH_SIZE)))
    screener.record_skipped()
    return {r["paper_id"]: r for r in results}
