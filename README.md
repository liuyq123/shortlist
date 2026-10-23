# Shortlist

**Today's arXiv, read for you by an AI agent.** Tell Shortlist what you research, and it screens every new arXiv paper in your fields, shows you the ones worth reading with a one-line reason each, and lets you fix what it gets wrong by talking to an assistant. The assistant finds the cause, edits your interests, and **tests the edit on real papers before you accept it**.

There are no embeddings or vector databases. Every judgment is made by the model you select in the class app store, through the class gateway.

## Run locally

You need Git and [uv](https://docs.astral.sh/uv/). From this folder:

```sh
cp -n .env.example .env
```

In the [class app store](https://nyu-workshop.adityasinghal.com/), open **Account → Local development token**, choose funding, a model and a ceiling, and create a token. Put the gateway and token in `.env` as `WORKSHOP_DEVELOPMENT_URL` and `WORKSHOP_DEVELOPMENT_TOKEN`. Then start the server:

```sh
uv run --locked --env-file .env python app.py
```

It opens `http://localhost:8080`. Requests use the token's model and spend its allowance. Stop the server with Ctrl+C; that does not revoke the token, so keep its Token ID and revoke it in Account when you're done.

## What to try

1. **Pick a sample researcher**, e.g. *LLM reasoning & agents*. Today's papers in its arXiv categories are screened and ranked. The first papers appear in about 15 seconds and the rest fill in as each batch is scored.
2. **Read the reasons.** Each paper card says which of your interests it matches and why. The line above the feed shows the funnel: *442 new papers → 118 passed the title screen → 102 worth a look*.
3. **Disagree with it.** Click **Not for me…** on a paper, or type something like *"Too many papers that just benchmark existing models. I want new methods."* Watch the assistant:
   - read your feed and the screener's reasons,
   - work out which part of your profile let those papers in,
   - edit a draft of your profile,
   - say what each test paper should do (drop out, rise in, or stay), then **re-score them under the draft** and show a before → after table,
   - revise if a paper misses, then propose the change.

   Under the proposal, the app's own check gives a verdict, for example *Passed its check · 5 of 6 papers that don't belong drop below 4; 4 of 4 papers that should stay keep 4+ (2 you liked)*. If the check fails, the button says *Apply anyway*.
4. **Apply** the proposal. The feed is re-screened and every card shows how its score moved (↑2, ↓5, *New in your feed*).
5. Vote 👍/👎 on a few papers and click **Learn from my votes**, or ask **Why is #1 ranked first?**
6. Open the **Agent log** (top bar) to watch every model request and tool call as it happens. Each request shows what it was for (*Title screen, batch 2 of 3*, *Assistant turn 4*), the model, latency, tokens, the cost the gateway reported, and why it stopped. Click a row for the full prompt and the raw reply. Tool calls show their arguments, result, and duration, nested under the model turn that asked for them. Each run also says what ended it, for example *the model answered without requesting a tool (turn 3 of 10)*. A running screen or assistant run has a **Stop** button, which takes effect before its next request.
7. Click **Show what was filtered** above the feed to see every paper the screener dropped: titles cut at pass 1, and abstracts scored below 4 with the screener's reason. **I'd want this** turns a miss into feedback for the assistant.

You can also edit themes, exclusions, and categories directly in the left panel.

Nothing fails silently: a failed batch, a declined or unreadable reply, or papers the model skipped twice are listed as problems on the run in the log, and the feed says how many papers weren't screened, with a Retry button.

## How it works

```
arXiv RSS (today's new papers in your categories)
        │
        ▼
Pass 1: title screen      ~150 titles per request, keeps anything plausibly relevant
        │                 (each title batch goes straight on to pass 2)
        ▼
Pass 2: abstract scoring  ≤15 abstracts per request → score 0–10, interest, reason
        │
        ▼
Feed (score ≥ 4), updated after every scored batch,
cached per (paper, profile version) so nothing is screened twice
        ▲
        │ edits, after you approve
Assistant                 the same model with 12 tools: inspect feed/votes/papers,
                          edit a draft profile, test_draft (re-score papers under the draft)
```

- **Interest profile.** 3–8 themes, each with a specific description, plus a "not interested" list and arXiv categories. The model builds it from your description and papers. It is the only thing that controls the feed, which keeps the system explainable: every score cites an interest.
- **Two-pass screening** makes an LLM-only design cheap. The title pass is generous (about a quarter survive), and only the survivors get their abstracts read. A day of ~450 papers takes about 12 requests.
- **The assistant is an agent, not a single prompt.** It decides what to look at, forms a hypothesis about what in the profile caused the problem, makes an edit, and checks it with `test_draft` before proposing it. Nothing is saved until you click Apply. Its last allowed turn must answer without tools.
- **Proposals are checked by code, not by the assistant.** Before seeing a test result, the assistant sorts each test paper into *should drop*, *should rise* or *should stay*. When it finishes, the app scores the final draft on every one of those papers (re-scoring any it only tested on an earlier draft), always includes papers you liked as *should stay*, and attaches a verdict: passed, partly fixes it, failed, or nothing confirms the fix.
- **Paper text is data.** Titles and abstracts come from third parties, so every prompt tells the model to treat them as data to judge, not instructions.
- **Structured replies are checked by the app.** Screening and profile building ask for JSON, validated before use. An unreadable reply gets one fresh request; a failed gateway request is never resent, because the broker owns retries.
- **One gateway wrapper** (`gateway.call`) makes every model request with the visitor's selected model and profile, and records the prompt, raw reply, tokens, and reported cost for the Agent log. The model's private reasoning is left out of the log.
- **Requests fit the allowance.** While a request runs, the gateway holds its worst-case cost against the allowance: the model's whole context window at the input price plus `max_tokens` at the output price (about $0.56 for Claude Haiku 5.5). The app reads the prices and the allowance left from `/v1/bootstrap` and runs batches in parallel only when the allowance can hold several requests at once; otherwise it runs them one at a time, in order, so the first papers still come quickly. When even one request no longer fits, it says so instead of sending a request the gateway would refuse. Hover over the cost in the top bar to see the allowance left.

## Design, in the workshop's terms

| Dimension | Shortlist |
|---|---|
| Control | Screening is a fixed workflow: its steps are known (fetch, title screen, abstract scoring), so code runs them in order. The assistant is an agent loop: what it reads and how it edits depends on what it finds, and a test that misses sends it back to revise. |
| Trigger | You: picking a profile, clicking Check arXiv, or sending a message. |
| Authority | The assistant edits only a draft. Saving needs your Apply, after the app's check; Apply refuses if you edited your interests after the draft was made. |
| Duration | Bounded: one screen, or one assistant run of at most 10 turns. Stop ends a run before its next request. |
| Environment | Structured tools over the app's own data (feed, votes, papers, draft). Code fetches arXiv; the model never browses. |
| Topology | One agent. The screener is a fixed workflow it reaches through `test_draft`, and the proposal check is code. |
| Memory | Your profile, votes and screening cache, for the current visit. The assistant also sees the last few messages. |
| Interface | Chat for feedback, plus direct actions: vote buttons, *Not for me…*, editable themes and exclusions. |

## Publish to the class app store

1. In **My app**, claim an address and put the same value in `slug` in `workshop-app.json`.
2. Build a fresh ZIP. If your address is `shortlist`:

   ```sh
   uv run --locked python package.py --slug shortlist --output ../shortlist.zip
   ```

   The packager checks the slug, includes only the files the app needs (manifest and lockfile at the root), and leaves out `.env`, `.local/` and caches.
3. In **My app**, select the ZIP, choose **Publish ZIP**, and follow the build status. For later updates, keep the slug and use **Publish replacement ZIP**.

In a hosted visit the platform supplies the gateway, run token, and port. The app listens on `0.0.0.0`, answers `/healthz` without any model or network call, reaches arXiv only through the platform proxy (the arXiv hosts are declared in the manifest), and keeps its data in the visit's temporary directory.

## Configuration

Set these in `.env` for local runs:

| Variable | Default | |
|---|---|---|
| `WORKSHOP_DEVELOPMENT_URL` | (none) | The development gateway from Account, ending in `/api/development` |
| `WORKSHOP_DEVELOPMENT_TOKEN` | (none) | Your local development token. Keep it out of source files and ZIPs |
| `PORT` | `8080` | |
| `WINDOW_DAYS` | `3` | How many days of papers the feed covers |
| `SHORTLIST_DATA_DIR` | `.local` | Where the SQLite database and profile live |

## Project layout

```
app.py              entry point (python app.py), used by the platform and locally
workshop-app.json   the class app manifest: runtime, models, network, storage
package.py          builds the upload ZIP
shortlist/
  gateway.py        the class gateway: bootstrap, the one wrapper for every model request, costs
  app.py            FastAPI server; long operations stream progress as NDJSON; /healthz
  activity.py       the agent log: runs, model requests, tool calls, problems
  screen.py         two-pass screening, caching, the ranked feed, and what was filtered
  assistant.py      the feedback agent: tools, draft profile, test_draft, proposals
  profile.py        building the interest profile; parsing BibTeX/RIS/arXiv links; personas
  arxiv.py          arXiv RSS feed, search API fallback, paper lookups (through the proxy when hosted)
  store.py          SQLite storage and profile.json
  static/           the single-page UI (plain HTML/CSS/JS, no build step)
```

## Limitations

- arXiv only. The RSS feed has one announcement a day, so a visit sees today's papers. On weekends it falls back to arXiv's search API, which is rate-limited.
- One model, the one selected in the class app store, does the screening, profile building, and the assistant. Smaller models give rougher scores and weaker assistant edits.
- A small allowance means one request at a time, so a full day's screen takes about 90 seconds with Claude Haiku 5.5 (about 30 seconds when three fit). A development token's ceiling also has to cover each request's hold, not just its cost: with a $0.65 ceiling, Haiku requests stop once about $0.09 has been spent.
- LLM scores vary a little between runs, so papers near the cutoff (scores 4–5) can move in or out when the feed is re-screened.
- `test_draft` re-runs only the abstract pass. A draft that changes the title pass is checked when you apply it.
- Everything (your interests, votes, the agent log) lasts for one hosted visit; the agent log also clears when a local server restarts.
