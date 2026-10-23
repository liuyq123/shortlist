"""
The interest profile: what the screener reads to decide which papers you want.

    {
      "description": "free text from the user",
      "themes": [{"id", "name", "description", "examples": [titles]}],
      "not_interested": ["..."],
      "categories": ["cs.LG", ...],      # arXiv categories the feed pulls from
    }

Claude builds it from a description and/or papers you like (pasted titles, arXiv
links, or a BibTeX/RIS export), and the assistant edits it from your feedback.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from paperfind_llm import activity, arxiv, config, store
from paperfind_llm.activity import section
from paperfind_llm.claude import call, fallback_params

MAX_LIBRARY_PAPERS = 800
PROMPT_VERSION = "1"  # part of the screening cache key; bump when screening prompts change

BUILD_SYSTEM_PROMPT = """You write research-interest profiles. A smaller model will use the profile to screen hundreds of new arXiv papers a day for this researcher, deciding from titles and abstracts which ones they would want to read.

Write 3 to 8 themes that cover the researcher's actual interests. For each theme:
- name: 2 to 6 words
- description: 1 or 2 sentences naming the specific methods, systems, or questions, so a screener can tell a relevant paper from one that only shares a buzzword
- example_titles: up to 3 titles copied exactly from the papers given (none if no papers were given)

Group papers by what the researcher studies, not by surface keywords. A topic that appears in only one or two papers is probably a one-off and should not become a theme unless the description mentions it.

List in not_interested the adjacent areas a screener could confuse with these themes but that the researcher likely doesn't want (for example, applications of a method they develop, in fields they don't work in). Keep each item short.

Choose 2 to 6 arxiv_categories from the list provided that will contain most of the papers they want."""


class Theme(BaseModel):
    name: str
    description: str
    example_titles: list[str]


class BuiltProfile(BaseModel):
    themes: list[Theme]
    not_interested: list[str]
    arxiv_categories: list[str]


# =============================================================================
# Parsing what the user gives us
# =============================================================================


def _bibtex_field(entry: str, field: str) -> Optional[str]:
    """Value of a BibTeX field, handling nested braces."""
    match = re.search(rf"\b{field}\s*=\s*([{{\"])", entry, re.IGNORECASE)
    if not match:
        return None
    start = match.end()
    if match.group(1) == '"':
        end = entry.find('"', start)
        value = entry[start:end] if end != -1 else ""
    else:
        depth, i = 1, start
        while i < len(entry) and depth:
            depth += {"{": 1, "}": -1}.get(entry[i], 0)
            i += 1
        value = entry[start : i - 1]
    value = re.sub(r"[{}]", "", value)
    return " ".join(value.split()) or None


def parse_papers(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Papers in pasted text or an uploaded export.

    Returns (papers with a title and maybe an abstract, arXiv IDs to look up).
    Understands BibTeX, RIS, arXiv IDs/links, and plain one-title-per-line lists.
    """
    if re.search(r"@\w+\s*\{", text):
        papers = []
        for entry in re.split(r"(?=@\w+\s*\{)", text):
            title = _bibtex_field(entry, "title")
            if title:
                papers.append({"title": title, "abstract": _bibtex_field(entry, "abstract")})
        return papers, []

    if re.search(r"^(TI|T1)  - ", text, re.MULTILINE):
        papers = []
        for record in re.split(r"^ER  -.*$", text, flags=re.MULTILINE):
            title = re.search(r"^(?:TI|T1)  - (.+)$", record, re.MULTILINE)
            abstract = re.search(r"^AB  - (.+)$", record, re.MULTILINE)
            if title:
                papers.append({"title": title.group(1).strip(), "abstract": abstract.group(1).strip() if abstract else None})
        return papers, []

    ids = arxiv.parse_ids(text)
    titles = []
    for line in text.splitlines():
        line = line.strip(" -*\t")
        if not line or arxiv.parse_ids(line) or line.startswith("http"):
            continue
        if len(line.split()) >= 3:
            titles.append({"title": line, "abstract": None})
    return titles, ids


# =============================================================================
# Building
# =============================================================================


def _build_prompt(description: Optional[str], papers: list[dict[str, Any]]) -> str:
    parts = []
    if description:
        parts.append(f"The researcher describes their interests:\n{description.strip()}")
    if papers:
        lines = []
        for paper in papers[:MAX_LIBRARY_PAPERS]:
            line = f"- {paper['title']}"
            if paper.get("abstract"):
                line += f"\n  {paper['abstract'][:300]}"
            lines.append(line)
        parts.append(f"Papers they like or have saved ({len(lines)}):\n" + "\n".join(lines))
    catalog = "\n".join(f"{code}: {name}" for code, name in arxiv.CATEGORIES.items())
    parts.append(f"arXiv categories to choose from:\n{catalog}")
    return "\n\n".join(parts)


def build_profile(
    description: Optional[str],
    papers_text: str = "",
    emit: Callable[[dict[str, Any]], None] = lambda event: None,
) -> dict[str, Any]:
    """Build and save a profile from a description and/or papers. Emits progress events."""
    description = (description or "").strip() or None
    papers, ids = parse_papers(papers_text) if papers_text.strip() else ([], [])
    given = description or f"{len(papers) + len(ids)} papers"
    with activity.run("builder", "Build your interest profile", emit, input=given) as log:
        if ids:
            emit({"type": "step", "text": f"Looking up {len(ids)} arXiv paper{'s' if len(ids) != 1 else ''}"})
            with activity.span("step", f"Look up {len(ids)} arXiv paper{'s' if len(ids) != 1 else ''}") as step:
                found = arxiv.fetch_by_ids(ids)
                step.finish(summary=f"found {len(found)} of {len(ids)}")
            papers += [{"title": p["title"], "abstract": p["abstract"]} for p in found]
            if len(found) < len(ids):
                missing = len(ids) - len(found)
                emit({"type": "step", "text": f"Couldn't look up {missing} of them; continuing without"})
                activity.issue("lookup_failed", f"Couldn't look up {missing} of {len(ids)} arXiv papers; built the profile without them.", count=missing)
        if not description and not papers:
            raise ValueError("Describe your research or add a few papers you like.")

        if papers:
            emit({"type": "step", "text": f"Reading {min(len(papers), MAX_LIBRARY_PAPERS)} papers"})
        emit({"type": "step", "text": "Claude is writing your interest profile"})

        model = config.AGENT_MODEL
        response = call(
            "Build profile",
            model=model,
            max_tokens=16000,
            system=BUILD_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(description, papers)}],
            output_format=BuiltProfile,
            output_config={"effort": "medium"},
            **fallback_params(model),
        )
        if response.stop_reason == "refusal":
            raise ValueError("Claude declined to build this profile. Try rewording your description.")
        built = response.parsed_output
        if built is None or not built.themes:
            raise ValueError(f"Could not build a profile (stop reason: {response.stop_reason}). Try again.")

        categories = [c for c in built.arxiv_categories if c in arxiv.CATEGORIES] or ["cs.LG"]
        activity.annotate(response, summary=f"{len(built.themes)} themes, {len(built.not_interested)} exclusions", detail=[
            section("Profile", built.model_dump(), open=True, format="code"),
        ])
        log.update(summary=f"{len(built.themes)} themes; categories {', '.join(categories[:8])}")
        return store.save_profile({
            "description": description,
            "themes": [
                {"id": store.new_theme_id(), "name": t.name, "description": t.description, "examples": t.example_titles[:3]}
                for t in built.themes
            ],
            "not_interested": built.not_interested,
            "categories": categories[:8],
            "source": "built",
        })


def load_personas() -> dict[str, dict[str, Any]]:
    text = (Path(__file__).parent / "personas.json").read_text()
    return {persona["id"]: persona for persona in json.loads(text)}


def use_persona(persona_id: str) -> dict[str, Any]:
    persona = load_personas().get(persona_id)
    if persona is None:
        raise ValueError(f"Unknown persona: {persona_id}")
    profile = json.loads(json.dumps(persona["profile"]))  # deep copy
    profile["source"] = f"persona:{persona_id}"
    return store.save_profile(profile)


# =============================================================================
# Rendering for prompts
# =============================================================================


def render(profile: dict[str, Any]) -> str:
    """The profile as plain text for the screening prompts."""
    lines = ["RESEARCH INTERESTS"]
    for i, theme in enumerate(profile.get("themes", []), 1):
        lines.append(f"{i}. {theme['name']}: {theme['description']}")
        examples = [t for t in theme.get("examples", []) if t]
        if examples:
            lines.append("   Examples: " + "; ".join(f'"{t}"' for t in examples))
    if profile.get("not_interested"):
        lines.append("")
        lines.append("NOT INTERESTED IN")
        lines.extend(f"- {item}" for item in profile["not_interested"])
    return "\n".join(lines)


def profile_hash(profile: dict[str, Any]) -> str:
    """Screening cache key: changes when the themes, exclusions, model, or prompts change."""
    text = f"{PROMPT_VERSION}\n{config.SCREEN_MODEL}\n{render(profile)}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]
