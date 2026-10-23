"""
The assistant: turns feedback into profile edits and tests them before proposing.

It reads the evidence (your feed, votes, and the screener's reasons), edits a
draft copy of the profile, and re-scores the papers in question under the draft
(test_draft) so the proposal shows what would change. Nothing is saved until you
apply the proposal.
"""

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

from anthropic import beta_tool

from paperfind_llm import activity, arxiv, config, store
from paperfind_llm.activity import section
from paperfind_llm.claude import call, fallback_params
from paperfind_llm.screen import get_feed, rescore, window_start

MAX_TURNS = 16  # model requests per message
MAX_LOGGED_RESULT_CHARS = 300
MAX_TEST_PAPERS = 15
HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are the assistant inside paperfind, a daily arXiv feed. A small model screens each new paper against the researcher's interest profile: first by title, then by abstract, giving a 0-10 score, the best-matching theme, and a reason. Papers scoring 4 or more are shown. The profile (themes, a "not interested" list, and arXiv categories) is the only thing that controls the feed, so you help by editing it.

When the user gives feedback:
1. Look at the evidence: the papers they mean (get_feed, get_votes, get_paper, search_papers) and the screener's score, theme, and reason for each.
2. Work out what in the profile caused it: a theme description that is too broad or too narrow, a missing theme, a missing exclusion, or a category that brings in noise.
3. Make the smallest edit to the draft that fixes it. Prefer sharpening a theme description over piling up exclusions, and keep the user's own wording where it still fits.
4. Check the edit with test_draft on the papers the feedback is about plus a few well-scored papers the user didn't complain about or liked, so you see whether the fix works without collateral damage. If it doesn't, revise and test again, at most twice.

If the user only asks a question, such as why a paper was recommended, answer it without editing. Draft edits are proposals: the user sees them with your test results and decides whether to apply them.

Reply in plain text, under 80 words: what caused the problem, what you changed, and what the test showed (for example, "the antiviral paper drops from 8 to 2, and the three papers you liked stay at 7 or above"). The user sees the full edits and test table next to your reply, so don't list every paper. Refer to papers by short title, not ID."""

_history: list[dict[str, str]] = []  # recent user/assistant messages, for follow-ups
_proposals: dict[str, "Proposal"] = {}


@dataclass
class Proposal:
    id: str
    base_updated_at: Optional[str]
    draft: dict[str, Any]
    changes: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    feedback_id: int


@dataclass
class Session:
    """State for one assistant run."""

    profile: dict[str, Any]
    emit: Callable[[dict[str, Any]], None]
    draft: dict[str, Any] = field(init=False)
    feed: dict[str, dict[str, Any]] = field(init=False)
    tests: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.draft = copy.deepcopy(self.profile)
        self.feed = {p["id"]: p for p in get_feed(self.profile)["papers"]}

    def step(self, text: str) -> None:
        self.emit({"type": "step", "text": text})
        activity.note(text)  # the running tool call's plain-language description in the log

    def theme(self, theme_id: str) -> dict[str, Any]:
        for theme in self.draft["themes"]:
            if theme["id"] == theme_id:
                return theme
        raise ValueError(f"No theme with id {theme_id}. See get_profile.")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)


def _brief(paper: dict[str, Any], screened: Optional[dict[str, Any]], vote: int = 0) -> dict[str, Any]:
    return {
        "paper_id": paper["id"],
        "title": paper["title"],
        "score": screened["score"] if screened else None,
        "theme": screened["theme"] if screened else None,
        "reason": screened["reason"] if screened else None,
        "vote": {1: "liked", -1: "disliked"}.get(vote),
    }


def diff_profiles(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured list of changes from `before` to `after`."""
    changes = []
    old_themes = {t["id"]: t for t in before.get("themes", [])}
    new_themes = {t["id"]: t for t in after.get("themes", [])}
    for theme_id, theme in new_themes.items():
        old = old_themes.get(theme_id)
        if old is None:
            changes.append({"kind": "theme_added", "name": theme["name"], "after": theme["description"]})
        elif (old["name"], old["description"]) != (theme["name"], theme["description"]):
            changes.append({
                "kind": "theme_edited",
                "name": theme["name"],
                "old_name": old["name"],
                "before": old["description"],
                "after": theme["description"],
            })
    for theme_id, theme in old_themes.items():
        if theme_id not in new_themes:
            changes.append({"kind": "theme_removed", "name": theme["name"], "before": theme["description"]})

    old_excl, new_excl = before.get("not_interested", []), after.get("not_interested", [])
    changes += [{"kind": "exclusion_added", "after": x} for x in new_excl if x not in old_excl]
    changes += [{"kind": "exclusion_removed", "before": x} for x in old_excl if x not in new_excl]

    old_cats, new_cats = before.get("categories", []), after.get("categories", [])
    changes += [{"kind": "category_added", "after": c} for c in new_cats if c not in old_cats]
    changes += [{"kind": "category_removed", "before": c} for c in old_cats if c not in new_cats]
    return changes


def summarize(change: dict[str, Any]) -> str:
    kind = change["kind"]
    if kind == "theme_added":
        return f'New theme "{change["name"]}"'
    if kind == "theme_edited":
        return f'Reworded theme "{change["name"]}"'
    if kind == "theme_removed":
        return f'Removed theme "{change["name"]}"'
    if kind == "exclusion_added":
        return f'Not interested: {change["after"]}'
    if kind == "exclusion_removed":
        return f'No longer excluded: {change["before"]}'
    if kind == "category_added":
        return f'Added category {change["after"]}'
    return f'Removed category {change["before"]}'


def _build_tools(session: Session) -> list:
    @beta_tool
    def get_profile() -> str:
        """Show the draft profile (themes with their ids, not-interested list, arXiv categories), the edits made so far in this session, and recent feedback sessions."""
        return _json({
            "themes": [{"id": t["id"], "name": t["name"], "description": t["description"]} for t in session.draft["themes"]],
            "not_interested": session.draft.get("not_interested", []),
            "categories": session.draft.get("categories", []),
            "edits_so_far": [summarize(c) for c in diff_profiles(session.profile, session.draft)],
            "recent_feedback": store.get_recent_feedback(),
        })

    @beta_tool
    def get_feed_papers(limit: int = 25, include_hidden: bool = False) -> str:
        """List papers in the current feed, best first, with score, theme, the screener's reason, and the user's vote.

        Args:
            limit: How many papers to return (max 60).
            include_hidden: Also include scored papers below the display threshold of 4.
        """
        session.step("Reading your feed")
        papers = [p for p in session.feed.values() if include_hidden or p["score"] >= 4]
        papers.sort(key=lambda p: p["score"], reverse=True)
        return _json([_brief(p, p, p["vote"]) for p in papers[: min(limit, 60)]])

    @beta_tool
    def get_votes() -> str:
        """List the papers the user voted on (liked or disliked), most recent first, with their scores and reasons."""
        session.step("Checking your votes")
        votes = store.get_recent_votes()
        if not votes:
            return "The user hasn't voted on any papers yet."
        return _json([
            {**_brief({"id": v["paper_id"], "title": v["title"]}, session.feed.get(v["paper_id"]), v["vote"])}
            for v in votes
        ])

    @beta_tool
    def get_paper(paper_id: str) -> str:
        """Get a paper's full abstract, categories, screening result, and the user's vote.

        Args:
            paper_id: arXiv id as returned by another tool.
        """
        paper = store.get_paper(paper_id)
        if paper is None:
            raise ValueError(f"Paper {paper_id} not found.")
        session.step(f"Reading “{paper['title']}”")
        screened = session.feed.get(paper_id)
        return _json({**_brief(paper, screened, store.get_votes().get(paper_id, 0)), "abstract": paper["abstract"], "categories": paper["categories"]})

    @beta_tool
    def search_papers(query: str) -> str:
        """Find recent papers whose title or abstract contains all the given words. Use it to find a paper the user describes.

        Args:
            query: A few distinctive words, e.g. "antiviral docking".
        """
        session.step(f"Searching recent papers for “{query}”")
        papers = store.search_papers(query, window_start(), limit=12)
        if not papers:
            return "No matching papers. Try fewer or different words."
        return _json([_brief(p, session.feed.get(p["id"])) for p in papers])

    @beta_tool
    def add_theme(name: str, description: str) -> str:
        """Add a theme to the draft.

        Args:
            name: 2-6 words.
            description: 1-2 sentences naming the specific methods, systems, or questions.
        """
        session.step(f"Drafting a new theme: {name}")
        theme = {"id": store.new_theme_id(), "name": name.strip(), "description": description.strip(), "examples": []}
        session.draft["themes"].append(theme)
        return f"Added theme {theme['id']} to the draft."

    @beta_tool
    def edit_theme(theme_id: str, name: str = "", description: str = "") -> str:
        """Change a theme's name and/or description in the draft. Empty arguments are left unchanged.

        Args:
            theme_id: The theme's id from get_profile.
            name: New name, or empty to keep it.
            description: New description, or empty to keep it.
        """
        theme = session.theme(theme_id)
        session.step(f"Rewording the theme “{theme['name']}”")
        if name.strip():
            theme["name"] = name.strip()
        if description.strip():
            theme["description"] = description.strip()
        return "Updated the draft."

    @beta_tool
    def remove_theme(theme_id: str) -> str:
        """Remove a theme from the draft.

        Args:
            theme_id: The theme's id from get_profile.
        """
        theme = session.theme(theme_id)
        session.step(f"Removing the theme “{theme['name']}”")
        session.draft["themes"].remove(theme)
        return "Removed from the draft."

    @beta_tool
    def add_exclusion(text: str) -> str:
        """Add an item to the draft's not-interested list.

        Args:
            text: A short description of papers to deprioritize.
        """
        session.step(f"Adding to “not interested”: {text}")
        if text.strip() and text.strip() not in session.draft["not_interested"]:
            session.draft["not_interested"].append(text.strip())
        return "Updated the draft."

    @beta_tool
    def remove_exclusion(text: str) -> str:
        """Remove an item from the draft's not-interested list.

        Args:
            text: The item exactly as listed in get_profile.
        """
        if text not in session.draft["not_interested"]:
            raise ValueError("Not in the list. Copy it exactly from get_profile.")
        session.step(f"Removing from “not interested”: {text}")
        session.draft["not_interested"].remove(text)
        return "Updated the draft."

    @beta_tool
    def set_categories(categories: list[str]) -> str:
        """Replace the draft's arXiv categories (which papers are fetched at all).

        Args:
            categories: arXiv category codes such as "cs.LG" or "q-bio.BM".
        """
        unknown = [c for c in categories if c not in arxiv.CATEGORIES]
        if unknown or not categories:
            raise ValueError(f"Unknown categories: {unknown}. Valid: {', '.join(arxiv.CATEGORIES)}")
        session.step(f"Setting categories to {', '.join(categories)}")
        session.draft["categories"] = list(dict.fromkeys(categories))
        return "Updated the draft. Note: new categories are fetched after the user applies the change."

    @beta_tool
    def test_draft(paper_ids: list[str]) -> str:
        """Re-score papers under the draft profile and compare with their current scores. Include the papers the feedback is about and a few the user liked or that score well.

        Args:
            paper_ids: Up to 15 arXiv ids.
        """
        papers = [p for p in (store.get_paper(pid) for pid in paper_ids[:MAX_TEST_PAPERS]) if p]
        if not papers:
            raise ValueError("None of those papers were found.")
        session.step(f"Testing the draft on {len(papers)} papers")
        new = rescore(session.draft, papers)
        votes = store.get_votes()
        results = []
        for paper in papers:
            old = session.feed.get(paper["id"])
            after = new.get(paper["id"])
            result = {
                "paper_id": paper["id"],
                "title": paper["title"],
                "vote": {1: "liked", -1: "disliked"}.get(votes.get(paper["id"], 0)),
                "old_score": old["score"] if old else None,
                "new_score": after["score"] if after else None,
                "new_theme": after["theme"] if after else None,
                "new_reason": after["reason"] if after else None,
            }
            session.tests[paper["id"]] = result
            results.append(result)
        session.emit({"type": "test", "results": results})
        return _json({
            "results": results,
            "note": "old_score null means the paper is not in the feed now (filtered by the title screen or not yet screened).",
        })

    return [
        get_profile, get_feed_papers, get_votes, get_paper, search_papers,
        add_theme, edit_theme, remove_theme, add_exclusion, remove_exclusion,
        set_categories, test_draft,
    ]


def _short(value: Any, limit: int = 120) -> Any:
    """A tool argument shortened for the log's one-line view."""
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    if isinstance(value, list) and len(value) > 4:
        return [*value[:3], f"… {len(value) - 3} more"]
    return value


def _run_tool(tools: dict[str, Any], block: Any) -> dict[str, Any]:
    """Run one tool call from the model and log it. Errors go back to the model as the result."""
    with activity.span("tool", block.name, args={k: _short(v) for k, v in (block.input or {}).items()}) as event:
        tool = tools.get(block.name)
        try:
            if tool is None:
                raise ValueError(f"There is no tool named {block.name}.")
            content = tool.call(block.input)
        except Exception as exc:
            event.finish("error", error=str(exc), detail=[section("Arguments", block.input, open=True, format="code")])
            return {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {exc}", "is_error": True}
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        one_line = " ".join(text.split())
        event.finish(
            result=one_line[:MAX_LOGGED_RESULT_CHARS] + ("…" if len(one_line) > MAX_LOGGED_RESULT_CHARS else ""),
            detail=[
                section("Arguments", block.input, open=True, format="code"),
                section("Result", text, open=True, format="code" if text.lstrip().startswith(("{", "[")) else "text"),
            ],
        )
        return {"type": "tool_result", "tool_use_id": block.id, "content": content}


def run(message: str, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Run the assistant on one message. Returns the reply and any proposal."""
    profile = store.load_profile()
    if profile is None:
        raise ValueError("Set up your interests first.")

    title = message if len(message) <= 90 else message[:89] + "…"
    with activity.run("assistant", title, emit, input=message) as log:
        session = Session(profile, emit)
        history = "".join(f"{m['role'].upper()}: {m['text']}\n" for m in _history[-HISTORY_MESSAGES:])
        user_message = f"Today is {date.today().isoformat()}.\n"
        if history:
            user_message += f"Earlier in this conversation:\n{history}\n"
        user_message += f"USER: {message}"

        model = config.AGENT_MODEL
        tools = _build_tools(session)
        by_name = {tool.name: tool for tool in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        response = None
        # Each turn: one model request, then the tool calls it asked for
        for turn in range(1, MAX_TURNS + 1):
            with activity.span("group", f"Turn {turn}") as group:
                response = call(
                    f"Assistant turn {turn}",
                    model=model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=[tool.to_dict() for tool in tools],
                    messages=messages,
                    output_config={"effort": "medium"},
                    cache_control={"type": "ephemeral"},
                    **fallback_params(model),
                )
                calls = [block for block in response.content if block.type == "tool_use"]
                if response.stop_reason != "tool_use" or not calls:
                    group.finish(summary="wrote the reply" if response.stop_reason == "end_turn" else f"stopped: {response.stop_reason}")
                    break
                group.update(summary="called " + ", ".join(block.name for block in calls))
                messages.append(response.to_param())
                messages.append({"role": "user", "content": [_run_tool(by_name, block) for block in calls]})

        if response is None:
            raise ValueError("The assistant returned no response.")
        if response.stop_reason == "refusal":
            raise ValueError("Claude declined this request. Try rephrasing it.")
        reply = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if not reply and response.stop_reason == "tool_use":
            reply = "I ran out of steps before finishing. Here is what I drafted so far."
            activity.issue("out_of_turns", f"Stopped after {MAX_TURNS} turns without a final reply.")
        log.update(output=reply)

        _history.extend([{"role": "user", "text": message}, {"role": "assistant", "text": reply}])
        del _history[:-HISTORY_MESSAGES]

        changes = diff_profiles(profile, session.draft)
        log.update(summary=(
            f"proposed {len(changes)} change{'s' if len(changes) != 1 else ''}, tested on {len(session.tests)} papers"
            if changes else "answered without changing your interests"
        ))
        proposal = None
        if changes:
            feedback_id = store.log_feedback(message, reply, [summarize(c) for c in changes], applied=False)
            proposal = Proposal(
                id=uuid.uuid4().hex[:8],
                base_updated_at=profile.get("updated_at"),
                draft=session.draft,
                changes=changes,
                tests=list(session.tests.values()),
                feedback_id=feedback_id,
            )
            _proposals[proposal.id] = proposal
        else:
            store.log_feedback(message, reply, [], applied=False)

        return {
            "reply": reply,
            "proposal": {
                "id": proposal.id,
                "changes": [{**c, "summary": summarize(c)} for c in proposal.changes],
                "tests": proposal.tests,
            } if proposal else None,
        }


def apply_proposal(proposal_id: str) -> dict[str, Any]:
    proposal = _proposals.pop(proposal_id, None)
    if proposal is None:
        raise ValueError("That proposal is no longer available.")
    current = store.load_profile() or {}
    if current.get("updated_at") != proposal.base_updated_at:
        raise ValueError("Your interests changed since this was proposed. Ask again.")
    profile = store.save_profile(proposal.draft)
    store.mark_feedback_applied(proposal.feedback_id)
    return profile


def discard_proposal(proposal_id: str) -> None:
    _proposals.pop(proposal_id, None)


def reset_history() -> None:
    _history.clear()
