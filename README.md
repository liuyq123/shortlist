# paperfind

**Today's arXiv, read for you by Claude.** Tell paperfind what you research, and it screens every new arXiv paper in your fields, shows you the ones worth reading with a one-line reason each, and lets you fix what it gets wrong by talking to an assistant. The assistant finds the cause, edits your interests, and **tests the edit on real papers before you accept it**.

There are no embeddings or vector databases. Every judgment is made by an LLM.

## Quick start

You need Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env     # or paste the key in the app
paperfind-llm                                   # opens http://127.0.0.1:8000
```

Then either describe your research (optionally with a few paper titles, arXiv links, or a BibTeX/RIS export from Zotero), or pick a sample researcher to start immediately.

## What to try

1. **Pick a sample researcher**, e.g. *LLM reasoning & agents*. Within about 30 seconds, today's papers in its arXiv categories are screened and ranked.
2. **Read the reasons.** Each paper card says which of your themes it matches and why. The line above the feed shows the funnel: *557 new papers → 176 passed the title screen → 94 worth a look*.
3. **Disagree with it.** Click **Not for me…** on a paper, or type something like *"Too many speculative decoding papers. For efficiency I only care about quantization."* Watch the assistant:
   - read your feed and the screener's reasons,
   - work out which part of your profile let those papers in,
   - edit a draft of your profile,
   - **re-score the affected papers under the draft** and show a before → after table,
   - revise if the test shows collateral damage, then propose the change.
4. **Apply** the proposal. The feed is re-screened and every card shows how its score moved (↑2, ↓5, *New in your feed*).
5. Vote 👍/👎 on a few papers and click **Learn from my votes**, or ask **Why is #1 ranked first?**

You can also edit themes, exclusions, and categories directly in the left panel.

## How it works

```
arXiv RSS (today's new papers in your categories)
        │
        ▼
Pass 1: title screen      Claude Haiku 5.5, 150 titles per call, keeps anything plausibly relevant
        │
        ▼
Pass 2: abstract scoring  Claude Haiku 5.5, 15 abstracts per call → score 0–10, theme, reason
        │
        ▼
Feed (score ≥ 4), cached per (paper, profile version) so nothing is paid for twice
        ▲
        │ edits, after you approve
Assistant                 Claude Opus 5.5 with 12 tools: inspect feed/votes/papers,
                          edit a draft profile, test_draft (re-score papers under the draft)
```

- **Interest profile.** 3–8 themes, each with a specific description, plus a "not interested" list and arXiv categories. Claude Opus builds it from your description and papers. It is the only thing that controls the feed, which keeps the system explainable: every score cites a theme.
- **Two-pass screening** makes an LLM-only design cheap. The title pass is generous (about 30% survive) and costs a fraction of a cent. Only the survivors get their abstracts read. A day of ~550 papers costs **about 2¢** and takes about 25 seconds.
- **The assistant is an agent, not a single prompt.** It decides what to look at, forms a hypothesis about what in the profile caused the problem, makes an edit, and checks it with `test_draft` before proposing it. Nothing is saved until you click Apply.
- **Costs are visible.** The header shows what this session has spent. A typical assistant conversation costs 10–15¢.

## Configuration

Set these in `.env` (working directory or `~/.paperfind-llm/.env`):

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Required, or paste it in the app (kept in memory only) |
| `SCREEN_MODEL` | `claude-haiku-5-5` | Model for both screening passes |
| `AGENT_MODEL` | `claude-opus-5-5` | Model for the profile builder and the assistant |
| `WINDOW_DAYS` | `3` | How many days of papers the feed covers |
| `PAPERFIND_LLM_DATA_DIR` | `~/.paperfind-llm` | Where the SQLite database and profile live |

## Project layout

```
paperfind_llm/
  app.py          FastAPI server; long operations stream progress as NDJSON
  screen.py       two-pass screening, caching, and the ranked feed
  assistant.py    the feedback agent: tools, draft profile, test_draft, proposals
  profile.py      building the interest profile; parsing BibTeX/RIS/arXiv links; personas
  arxiv.py        arXiv RSS feed, search API fallback, paper lookups
  store.py        SQLite storage and profile.json
  claude.py       Claude client, refusal fallback, cost tracking
  static/         the single-page UI (plain HTML/CSS/JS, no build step)
```

## Limitations

- arXiv only. The RSS feed has one announcement a day, so the feed builds up over the days you use it. On weekends it falls back to arXiv's search API, which is rate-limited.
- LLM scores vary a little between runs, so papers near the cutoff (scores 4–5) can move in or out when the feed is re-screened.
- `test_draft` re-runs only the abstract pass. A draft that changes the title pass is checked when you apply it.
