"use strict";

// ============================================================================
// State
// ============================================================================

const state = {
  ready: false,
  hasKey: false,
  profile: null,
  personas: [],
  categories: {},
  models: {},
  feed: null,
  feedStale: false,       // interests changed since the feed was screened
  refresh: null,          // latest progress event while a refresh runs
  refreshError: null,
  filterTheme: null,      // theme id, or null for all
  showHidden: false,
  expanded: new Set(),    // paper ids with the abstract open
  editing: null,          // theme id being edited, or "new"
  chat: [],
  busy: false,            // assistant running
  prevScores: null,       // paper id -> score before the last change, to show movement
  prevShown: null,        // ids shown before the last change
  flash: new Set(),       // theme ids to highlight after a change
  colors: new Map(),      // theme id -> color
  tab: "feed",
  build: null,            // {steps, error} while building a profile
};

const ICON_UP = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/></svg>`;
const ICON_DOWN = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/></svg>`;

const PALETTE = ["#4f46e5", "#0d9488", "#d97706", "#db2777", "#2563eb", "#65a30d", "#9333ea", "#dc2626"];

const $ = (selector, root = document) => root.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

// ============================================================================
// Server calls
// ============================================================================

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

// POST to a streaming endpoint; calls onEvent for each event and returns the final "done" event.
async function streamEvents(path, options, onEvent) {
  const response = await fetch(path, { method: "POST", ...options });
  if (!response.ok) throw new Error(await errorDetail(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      const event = JSON.parse(line);
      if (event.cost_usd !== undefined) setCost(event.cost_usd);
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "done") final = event;
      else onEvent(event);
    }
  }
  if (!final) throw new Error("The connection closed early. Try again.");
  return final;
}

const jsonBody = (body) => ({ headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ============================================================================
// Helpers
// ============================================================================

function setCost(usd) {
  $("#cost").textContent = `$${usd.toFixed(3)} spent`;
}

let toastTimer = null;
function toast(message, ms = 5000) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function fmtDate(iso) {
  if (!iso) return "";
  const date = new Date(`${iso}T12:00:00`);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function shortAuthors(authors) {
  const names = (authors || "").split(",").map((a) => a.trim()).filter(Boolean);
  return names.length > 3 ? `${names.slice(0, 3).join(", ")} et al.` : names.join(", ");
}

function shortTitle(title, max = 70) {
  return title.length > max ? `${title.slice(0, max - 1)}…` : title;
}

function setProfile(profile) {
  const old = state.profile;
  state.profile = profile;
  // Stable colors per theme: keep existing ones, give new themes an unused color
  const ids = new Set(profile.themes.map((t) => t.id));
  for (const id of [...state.colors.keys()]) if (!ids.has(id)) state.colors.delete(id);
  for (const theme of profile.themes) {
    if (!state.colors.has(theme.id)) {
      const used = new Set(state.colors.values());
      const color = PALETTE.find((c) => !used.has(c)) || PALETTE[state.colors.size % PALETTE.length];
      state.colors.set(theme.id, color);
    }
  }
  // Highlight themes that changed
  if (old) {
    const before = new Map(old.themes.map((t) => [t.id, `${t.name}|${t.description}`]));
    for (const theme of profile.themes) {
      if (before.get(theme.id) !== `${theme.name}|${theme.description}`) state.flash.add(theme.id);
    }
    if (state.flash.size) setTimeout(() => { state.flash.clear(); renderInterests(); }, 2600);
  }
}

// The screener reports theme names; map them back to the profile's themes
function themeFor(name) {
  if (!state.profile || !name) return null;
  const needle = name.toLowerCase().trim();
  const themes = state.profile.themes;
  return themes.find((t) => t.name.toLowerCase() === needle)
    || themes.find((t) => needle.includes(t.name.toLowerCase()) || t.name.toLowerCase().includes(needle))
    || null;
}

const colorFor = (theme) => (theme ? state.colors.get(theme.id) : null);

function visiblePapers() {
  if (!state.feed) return [];
  const min = state.feed.min_score;
  return state.feed.papers.filter((p) => state.showHidden || p.score >= min);
}

function snapshotScores() {
  if (!state.feed) return;
  state.prevScores = Object.fromEntries(state.feed.papers.map((p) => [p.id, p.score]));
  state.prevShown = new Set(state.feed.papers.filter((p) => p.score >= state.feed.min_score).map((p) => p.id));
}

// ============================================================================
// Top-level rendering
// ============================================================================

function render() {
  $("#reset-btn").hidden = !state.profile;
  if (!state.ready) return;
  if (!state.hasKey) return renderKey();
  if (!state.profile) return renderOnboarding();
  ensureWorkspace();
  renderInterests();
  renderFeed();
  renderAssistant();
}

function renderKey() {
  $("#app").innerHTML = `
    <div class="center-wrap">
      <div class="hero">
        <h1>Connect Claude</h1>
        <p>paperfind reads today's arXiv with your own Anthropic API key. A day of screening costs about 2¢; each assistant conversation about 10–15¢.</p>
      </div>
      <form class="card-box" id="key-form">
        <label class="field-label" for="key-input">Anthropic API key</label>
        <input type="password" id="key-input" placeholder="sk-ant-…" autocomplete="off" required>
        <p class="field-hint">Kept in memory on this machine only. To skip this step, put ANTHROPIC_API_KEY in a .env file and restart.</p>
        <div class="row" style="margin-top:14px"><div class="spacer"></div>
          <button class="btn primary" type="submit">Continue</button>
        </div>
      </form>
    </div>`;
  $("#key-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button");
    button.disabled = true;
    try {
      await api("/api/key", { method: "POST", body: JSON.stringify({ key: $("#key-input").value }) });
      state.hasKey = true;
      render();
      if (state.profile) refreshFeed();
    } catch (err) {
      toast(err.message);
      button.disabled = false;
    }
  });
}

function renderOnboarding() {
  const building = state.build && !state.build.error;
  const steps = state.build ? state.build.steps.map((s, i) => (
    `<li class="${building && i === state.build.steps.length - 1 ? "running" : ""}">${esc(s)}</li>`
  )).join("") : "";
  const keep = (id) => esc($(`#${id}`)?.value || "");
  $("#app").innerHTML = `
    <div class="center-wrap">
      <div class="hero">
        <h1>What do you research?</h1>
        <p>Tell paperfind in a sentence or two and add a few papers you like. Claude turns that into an interest profile, then reads every new arXiv paper for you.</p>
      </div>
      <div class="card-box">
        <label class="field-label" for="desc">Your research</label>
        <textarea id="desc" rows="3" placeholder="e.g. I work on RL fine-tuning of language models for math reasoning, and on making inference cheaper with quantization.">${keep("desc")}</textarea>
        <label class="field-label" for="papers" style="margin-top:16px">Papers you like <span class="muted">(optional)</span></label>
        <textarea id="papers" rows="3" placeholder="Paste titles or arXiv links, one per line">${keep("papers")}</textarea>
        <div class="drop">Or upload a BibTeX / RIS export (Zotero, Mendeley, Google Scholar)
          <input type="file" id="file" accept=".bib,.ris,.txt">
        </div>
        <div class="row" style="margin-top:16px">
          <span class="muted small">Takes about 30 seconds</span>
          <div class="spacer"></div>
          <button class="btn primary" data-action="build" ${building ? "disabled" : ""}>Build my feed →</button>
        </div>
        ${steps ? `<ul class="steps build-steps">${steps}</ul>` : ""}
        ${state.build?.error ? `<div class="banner error" style="margin-top:12px">${esc(state.build.error)}</div>` : ""}
      </div>
      <div class="divider">or try a sample researcher</div>
      <div class="personas">
        ${state.personas.map((p) => `
          <button class="persona" data-action="persona" data-id="${esc(p.id)}" ${building ? "disabled" : ""}>
            <b>${esc(p.label)}</b><span>${esc(p.blurb)}</span>
          </button>`).join("")}
      </div>
    </div>`;
}

function ensureWorkspace() {
  if ($("#workspace")) return;
  $("#app").innerHTML = `
    <div class="workspace" id="workspace" data-tab="${state.tab}">
      <nav class="tabs">
        <button data-action="tab" data-tab="interests">Interests</button>
        <button data-action="tab" data-tab="feed">Papers</button>
        <button data-action="tab" data-tab="assistant">Assistant</button>
      </nav>
      <aside class="panel interests" id="interests"></aside>
      <section class="feed" id="feed"></section>
      <aside class="panel assistant" id="assistant">
        <div class="assistant-head">
          <h2>Assistant</h2>
          <p>Tell it what's off in your feed. It finds the cause, edits a draft of your interests, tests the draft on real papers, and asks before saving.</p>
        </div>
        <div class="chat" id="chat"></div>
        <div class="suggestions" id="suggestions"></div>
        <form class="composer" id="composer">
          <textarea id="composer-input" rows="1" placeholder="e.g. Too many benchmark papers"></textarea>
          <button class="btn primary" type="submit" id="send-btn">Send</button>
        </form>
      </aside>
    </div>`;
  const input = $("#composer-input");
  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || state.busy) return;
    input.value = "";
    autosize(input);
    ask(text);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#composer").requestSubmit();
    }
  });
  input.addEventListener("input", () => autosize(input));
}

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
}

function setTab(tab) {
  state.tab = tab;
  const workspace = $("#workspace");
  if (!workspace) return;
  workspace.dataset.tab = tab;
  for (const button of workspace.querySelectorAll(".tabs button")) {
    button.classList.toggle("active", button.dataset.tab === tab);
  }
}

// ============================================================================
// Interests panel
// ============================================================================

function renderInterests() {
  const el = $("#interests");
  if (!el || !state.profile) return;
  const profile = state.profile;
  const unusedCategories = Object.entries(state.categories).filter(([code]) => !profile.categories.includes(code));

  el.innerHTML = `
    <h2>Your interests</h2>
    <p class="panel-intro">Claude screens every new paper against these. Edit them here or ask the assistant.</p>
    ${profile.themes.map((t) => (state.editing === t.id ? themeEditor(t) : themeCard(t))).join("")}
    ${state.editing === "new"
      ? themeEditor({ id: "", name: "", description: "" })
      : `<button class="btn small" data-action="add-theme">+ Add theme</button>`}

    <h3>Not interested in</h3>
    <ul class="excl-list">
      ${profile.not_interested.map((item, i) => `
        <li><span>${esc(item)}</span><button data-action="remove-excl" data-index="${i}" title="Remove">×</button></li>`).join("")}
    </ul>
    <form class="add-row" id="excl-form">
      <input type="text" id="excl-input" placeholder="Add something to skip">
      <button class="btn small" type="submit">Add</button>
    </form>

    <h3>arXiv categories</h3>
    <div class="chips">
      ${profile.categories.map((code) => `
        <span class="chip" title="${esc(state.categories[code] || code)}">${esc(code)}
          <button data-action="remove-cat" data-code="${esc(code)}" title="Remove">×</button></span>`).join("")}
    </div>
    <div class="add-row">
      <select id="cat-select">
        <option value="">+ Add a category…</option>
        ${unusedCategories.map(([code, name]) => `<option value="${esc(code)}">${esc(code)} — ${esc(name)}</option>`).join("")}
      </select>
    </div>
    ${profile.description ? `<h3>Your description</h3><p class="small muted">${esc(profile.description)}</p>` : ""}`;

  $("#excl-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("#excl-input").value.trim();
    if (text) saveProfile({ ...state.profile, not_interested: [...state.profile.not_interested, text] });
  });
  $("#cat-select").addEventListener("change", (event) => {
    if (event.target.value) saveProfile({ ...state.profile, categories: [...state.profile.categories, event.target.value] });
  });
}

function themeCard(theme) {
  return `
    <div class="theme ${state.flash.has(theme.id) ? "flash" : ""}" style="--c:${colorFor(theme)}">
      <div class="theme-name">${esc(theme.name)}</div>
      <div class="theme-desc">${esc(theme.description)}</div>
      <div class="theme-tools">
        <button class="icon-btn" data-action="edit-theme" data-id="${esc(theme.id)}" title="Edit">✎</button>
        <button class="icon-btn" data-action="delete-theme" data-id="${esc(theme.id)}" title="Remove">×</button>
      </div>
    </div>`;
}

function themeEditor(theme) {
  return `
    <div class="theme theme-edit" style="--c:${colorFor(theme) || "var(--accent)"}">
      <input type="text" id="theme-name" value="${esc(theme.name)}" placeholder="Theme name">
      <textarea id="theme-desc" placeholder="What papers belong here? Name the methods, systems, or questions.">${esc(theme.description)}</textarea>
      <div class="row">
        <button class="btn small" data-action="cancel-edit">Cancel</button>
        <button class="btn small primary" data-action="save-theme" data-id="${esc(theme.id)}">Save</button>
      </div>
    </div>`;
}

async function saveProfile(profile) {
  try {
    if (!state.feedStale) snapshotScores();
    const saved = await api("/api/profile", { method: "PUT", body: JSON.stringify(profile) });
    setProfile(saved);
    state.editing = null;
    state.feedStale = true;
    renderInterests();
    renderFeed();
  } catch (err) {
    toast(err.message);
  }
}

// ============================================================================
// Feed
// ============================================================================

function renderFeed() {
  const el = $("#feed");
  if (!el || !state.profile) return;
  const feed = state.feed;
  const refreshing = state.refresh;

  let html = `
    <div class="feed-head">
      <div>
        <h2>Today's papers</h2>
        <div class="muted small">${esc(state.profile.categories.join(" · "))}${feed ? ` · since ${fmtDate(feed.since)}` : ""}</div>
      </div>
      <div class="spacer"></div>
      <button class="btn small" data-action="check-arxiv" ${refreshing ? "disabled" : ""} title="Fetch the latest arXiv announcement">↻ Check arXiv</button>
    </div>`;

  if (refreshing) html += progressBox(refreshing);
  if (state.refreshError && !refreshing) {
    html += `<div class="banner error">${esc(state.refreshError)}<div class="spacer"></div>
      <button class="btn small" data-action="rescreen">Try again</button></div>`;
  }
  if (state.feedStale && !refreshing) {
    html += `<div class="banner">Your interests changed. Re-screen today's papers to see the effect.
      <span class="muted small">About 30 s and 2¢.</span><div class="spacer"></div>
      <button class="btn small primary" data-action="rescreen">Re-screen</button></div>`;
  }

  if (feed) {
    html += pipeline(feed.stats);
    const papers = visiblePapers();
    html += filterBar(papers);
    const shown = state.filterTheme ? papers.filter((p) => themeFor(p.theme)?.id === state.filterTheme) : papers;
    html += shown.length
      ? shown.map(paperCard).join("")
      : `<div class="empty">${feed.papers.length ? "No papers match this filter." : "No papers scored yet."}</div>`;
  } else if (!refreshing) {
    html += `<div class="empty">No papers yet. <button class="btn small primary" data-action="rescreen">Screen today's papers</button></div>`;
  }
  el.innerHTML = html;
}

function progressBox(event) {
  const determinate = event.total > 0 && event.done !== undefined;
  const pct = determinate ? Math.round((event.done / event.total) * 100) : 0;
  const label = {
    fetch: "Fetching",
    titles: "Pass 1 · titles",
    abstracts: "Pass 2 · abstracts",
  }[event.stage] || "Working";
  return `
    <div class="progress-box">
      <div class="progress-text"><span><b>${label}</b> · ${esc(event.text || "")}</span>
        <span class="muted">${determinate ? `${event.done}/${event.total} batches` : ""}</span></div>
      <div class="bar ${determinate ? "" : "indeterminate"}"><div style="width:${determinate ? pct : 35}%"></div></div>
    </div>`;
}

function pipeline(stats) {
  if (!stats.fetched) return "";
  const pending = stats.unscreened ? ` <span class="muted">(${stats.unscreened} not screened yet)</span>` : "";
  return `
    <div class="pipeline" title="Pass 1 reads titles generously; pass 2 reads abstracts and scores 0–10. Papers scoring 4+ are shown.">
      <b>${stats.fetched}</b> new papers <span class="arrow">→</span>
      <b>${stats.passed_titles}</b> passed the title screen <span class="arrow">→</span>
      <b>${stats.shown}</b> worth a look${pending}
    </div>`;
}

function filterBar(papers) {
  const counts = new Map();
  for (const paper of papers) {
    const theme = themeFor(paper.theme);
    if (theme) counts.set(theme.id, (counts.get(theme.id) || 0) + 1);
  }
  const chips = state.profile.themes.map((theme) => `
    <button class="filter ${state.filterTheme === theme.id ? "active" : ""}" data-action="filter" data-id="${esc(theme.id)}" style="--c:${colorFor(theme)}">
      <span class="dot"></span>${esc(theme.name)} <span class="count">${counts.get(theme.id) || 0}</span>
    </button>`).join("");
  return `
    <div class="filters">
      <button class="filter ${state.filterTheme ? "" : "active"}" data-action="filter" data-id="">All <span class="count">${papers.length}</span></button>
      ${chips}
      <label class="toggle"><input type="checkbox" data-action="toggle-hidden" ${state.showHidden ? "checked" : ""}> Show low scores</label>
    </div>`;
}

function paperCard(paper) {
  const score = Math.round(paper.score);
  const theme = themeFor(paper.theme);
  const prev = state.prevScores ? state.prevScores[paper.id] : undefined;
  let delta = "";
  let isNew = false;
  if (state.prevScores && !state.feedStale) {
    if (prev === undefined) {
      isNew = true;
    } else if (Math.round(prev) !== score) {
      const up = score > prev;
      delta = `<span class="delta ${up ? "up" : "down"}" title="Was ${Math.round(prev)}">${up ? "↑" : "↓"}${Math.abs(score - Math.round(prev))}</span>`;
    }
  }
  const scoreClass = score >= 9 ? "s9" : score >= 7 ? "s7" : "";
  const categories = (paper.categories || "").split(" ");
  const open = state.expanded.has(paper.id);
  const classes = [
    "paper",
    paper.score < state.feed.min_score ? "hidden-score" : "",
    paper.vote < 0 ? "voted-down" : "",
    isNew ? "is-new" : "",
  ].join(" ");

  return `
    <article class="${classes}">
      <div class="score ${scoreClass}" title="Claude's relevance score (0–10)">${score}${delta}</div>
      <div>
        <div class="paper-meta">
          ${theme ? `<span class="theme-tag" style="--c:${colorFor(theme)}">${esc(theme.name)}</span>` : `<span class="theme-tag">${esc(paper.theme || "Other")}</span>`}
          <span>${esc(categories[0] || "")}</span><span>${fmtDate(paper.published)}</span>
          ${isNew ? `<span class="new-badge">New in your feed</span>` : ""}
        </div>
        <h3><a href="${esc(paper.url)}" target="_blank" rel="noopener">${esc(paper.title)}</a></h3>
        <p class="authors">${esc(shortAuthors(paper.authors))}</p>
        <p class="reason">${esc(paper.reason)}</p>
        ${open ? `<p class="abstract">${esc(paper.abstract)}</p>` : ""}
        <div class="paper-actions">
          <button class="icon-btn ${paper.vote > 0 ? "on-up" : ""}" data-action="vote" data-id="${esc(paper.id)}" data-vote="1" title="More like this" aria-label="More like this">${ICON_UP}</button>
          <button class="icon-btn ${paper.vote < 0 ? "on-down" : ""}" data-action="vote" data-id="${esc(paper.id)}" data-vote="-1" title="Less like this" aria-label="Less like this">${ICON_DOWN}</button>
          <button class="icon-btn" data-action="toggle-abstract" data-id="${esc(paper.id)}">${open ? "Hide abstract" : "Abstract"}</button>
          <button class="icon-btn" data-action="ask-why" data-id="${esc(paper.id)}">Why this?</button>
          <button class="icon-btn" data-action="not-for-me" data-id="${esc(paper.id)}">Not for me…</button>
        </div>
      </div>
    </article>`;
}

async function refreshFeed({ force = false, afterChange = false } = {}) {
  if (state.refresh) return;
  state.refresh = { stage: "start", text: "Checking for new papers" };
  state.refreshError = null;
  if (!afterChange && !state.feedStale) state.prevScores = null;
  renderFeed();
  try {
    const final = await streamEvents("/api/feed/refresh", jsonBody({ force }), (event) => {
      if (event.type === "progress") {
        state.refresh = event;
        renderFeed();
      }
    });
    state.feed = final.feed;
    const wasStale = state.feedStale;
    state.feedStale = false;
    if ((afterChange || wasStale) && state.prevShown) summarizeMovement();
  } catch (err) {
    state.refreshError = err.message;
  }
  state.refresh = null;
  renderFeed();
  renderAssistant();
}

function summarizeMovement() {
  const nowShown = new Set(state.feed.papers.filter((p) => p.score >= state.feed.min_score).map((p) => p.id));
  const dropped = [...state.prevShown].filter((id) => !nowShown.has(id)).length;
  const added = [...nowShown].filter((id) => !state.prevShown.has(id)).length;
  const parts = [];
  if (dropped) parts.push(`${dropped} paper${dropped === 1 ? "" : "s"} dropped out`);
  if (added) parts.push(`${added} new one${added === 1 ? "" : "s"} came in`);
  toast(parts.length ? `Feed updated: ${parts.join(", ")}. Arrows show how scores moved.` : "Feed updated. Arrows show how scores moved.");
}

async function vote(paperId, value) {
  const paper = state.feed?.papers.find((p) => p.id === paperId);
  if (!paper) return;
  const next = paper.vote === value ? 0 : value;
  paper.vote = next;
  renderFeed();
  renderAssistant();
  try {
    await api("/api/votes", { method: "POST", body: JSON.stringify({ paper_id: paperId, vote: next }) });
  } catch (err) {
    toast(err.message);
  }
}

// ============================================================================
// Assistant
// ============================================================================

function renderAssistant() {
  const chat = $("#chat");
  if (!chat) return;
  const nearBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;

  if (!state.chat.length) {
    chat.innerHTML = `
      <div class="chat-empty">
        Try telling it things like:
        <ul>
          <li>“Too many papers that only benchmark LLMs on a new domain.”</li>
          <li>“Why is the top paper ranked first?”</li>
          <li>“I'd like more on mixture-of-experts.”</li>
          <li>Or vote on a few papers with ${ICON_UP} / ${ICON_DOWN} and ask it to learn from them.</li>
        </ul>
      </div>`;
  } else {
    chat.innerHTML = state.chat.map((message, i) => (
      message.role === "user" ? `<div class="msg-user">${esc(message.text)}</div>` : botMessage(message, i)
    )).join("");
  }
  if (nearBottom) chat.scrollTop = chat.scrollHeight;

  const votes = state.feed ? state.feed.papers.filter((p) => p.vote !== 0).length : 0;
  const suggestions = [];
  if (votes >= 2) suggestions.push(["Learn from my votes", `Learn from my ${votes} votes: adjust my interests so papers like the ones I liked rank higher and the ones I disliked drop.`]);
  if (state.feed?.papers.length) suggestions.push(["Why is #1 ranked first?", "Why is the top paper in my feed ranked first?"]);
  suggestions.push(["Too many benchmark papers", "Too many papers that just benchmark existing models. I want new methods."]);
  $("#suggestions").innerHTML = state.busy ? "" : suggestions.map(([label, text]) => (
    `<button class="suggestion" data-action="suggest" data-text="${esc(text)}">${esc(label)}</button>`
  )).join("");
  $("#send-btn").disabled = state.busy;
}

function botMessage(message, index) {
  const steps = message.steps.map((step, i) => (
    `<li class="${message.running && i === message.steps.length - 1 ? "running" : ""}">${esc(step)}</li>`
  ));
  if (message.running && !message.steps.length) steps.push(`<li class="running">Thinking</li>`);
  return `
    <div class="msg-bot">
      ${steps.length ? `<ul class="steps">${steps.join("")}</ul>` : ""}
      ${message.tests ? testsTable(message.tests) : ""}
      ${message.reply ? `<div class="reply">${esc(message.reply)}</div>` : ""}
      ${message.error ? `<div class="msg-error">${esc(message.error)}</div>` : ""}
      ${message.proposal ? proposalCard(message, index) : ""}
    </div>`;
}

function testsTable(results) {
  const rows = results.map((r) => {
    const before = r.old_score === null ? "–" : Math.round(r.old_score);
    const after = r.new_score === null ? "?" : Math.round(r.new_score);
    let cls = "same";
    if (r.old_score !== null && r.new_score !== null) cls = r.new_score > r.old_score ? "up" : r.new_score < r.old_score ? "down" : "same";
    const vote = r.vote === "liked" ? `<span class="t-vote up">${ICON_UP}</span>` : r.vote === "disliked" ? `<span class="t-vote down">${ICON_DOWN}</span>` : "";
    return `<tr><td class="t-title" title="${esc(r.new_reason || "")}">${vote}${esc(shortTitle(r.title, 64))}</td>
      <td class="t-score"><span class="${cls}">${before} → ${after}</span></td></tr>`;
  }).join("");
  return `<table class="tests"><caption>Tested the draft on ${results.length} papers</caption>${rows}</table>`;
}

function proposalCard(message, index) {
  const changes = message.proposal.changes.map(changeHtml).join("");
  let footer;
  if (message.proposalState === "applied") footer = `<div class="proposal-done">✓ Applied. Your feed was re-screened.</div>`;
  else if (message.proposalState === "discarded") footer = `<div class="proposal-done">Discarded.</div>`;
  else {
    footer = `
      <div class="proposal-actions">
        <button class="btn small primary" data-action="apply" data-index="${index}">Apply & re-screen</button>
        <button class="btn small" data-action="discard" data-index="${index}">Discard</button>
      </div>`;
  }
  return `<div class="proposal"><div class="proposal-title">Proposed changes to your interests</div>${changes}${footer}</div>`;
}

function changeHtml(change) {
  const kinds = {
    theme_added: "New theme",
    theme_edited: "Theme reworded",
    theme_removed: "Theme removed",
    exclusion_added: "Not interested in",
    exclusion_removed: "No longer excluded",
    category_added: "Category added",
    category_removed: "Category removed",
  };
  let name = change.name ? `<b>${esc(change.name)}</b>` : "";
  if (change.old_name && change.old_name !== change.name) name = `<b>${esc(change.old_name)} → ${esc(change.name)}</b>`;
  return `
    <div class="change">
      <div class="change-kind">${kinds[change.kind] || change.kind}</div>
      ${name}
      ${change.before ? `<div class="before">${esc(change.before)}</div>` : ""}
      ${change.after ? `<div class="after">${esc(change.after)}</div>` : ""}
    </div>`;
}

async function ask(text) {
  if (state.busy) return;
  if (window.matchMedia("(max-width: 820px)").matches) setTab("assistant");
  state.chat.push({ role: "user", text });
  const message = { role: "bot", steps: [], tests: null, reply: null, proposal: null, error: null, running: true };
  state.chat.push(message);
  state.busy = true;
  renderAssistant();
  try {
    const final = await streamEvents("/api/assistant", jsonBody({ message: text }), (event) => {
      if (event.type === "step") message.steps.push(event.text);
      if (event.type === "test") message.tests = event.results;
      renderAssistant();
    });
    message.reply = final.reply;
    message.proposal = final.proposal;
  } catch (err) {
    message.error = err.message;
  }
  message.running = false;
  state.busy = false;
  renderAssistant();
}

async function applyProposal(index) {
  const message = state.chat[index];
  try {
    snapshotScores();
    const profile = await api(`/api/proposals/${message.proposal.id}/apply`, { method: "POST" });
    message.proposalState = "applied";
    setProfile(profile);
    state.feedStale = false;
    renderInterests();
    renderAssistant();
    if (window.matchMedia("(max-width: 820px)").matches) setTab("feed");
    await refreshFeed({ afterChange: true });
  } catch (err) {
    toast(err.message);
  }
}

async function discardProposal(index) {
  const message = state.chat[index];
  message.proposalState = "discarded";
  renderAssistant();
  api(`/api/proposals/${message.proposal.id}/discard`, { method: "POST" }).catch(() => {});
}

// ============================================================================
// Onboarding
// ============================================================================

async function buildProfile() {
  const description = $("#desc").value.trim();
  const papers = $("#papers").value.trim();
  const file = $("#file").files[0];
  if (!description && !papers && !file) {
    toast("Describe your research or add a few papers first.");
    return;
  }
  const form = new FormData();
  form.append("description", description);
  form.append("papers", papers);
  if (file) form.append("file", file);

  state.build = { steps: [], error: null };
  renderOnboarding();
  try {
    const final = await streamEvents("/api/profile/build", { body: form }, (event) => {
      if (event.type === "step") {
        state.build.steps.push(event.text);
        renderOnboarding();
      }
    });
    state.build = null;
    startWorkspace(final.profile);
  } catch (err) {
    state.build.error = err.message;
    renderOnboarding();
  }
}

async function usePersona(id) {
  try {
    startWorkspace(await api("/api/profile/persona", { method: "POST", body: JSON.stringify({ id }) }));
  } catch (err) {
    toast(err.message);
  }
}

function startWorkspace(profile) {
  state.colors.clear();
  setProfile(profile);
  state.flash.clear();
  state.feed = null;
  state.feedStale = false;
  state.prevScores = null;
  state.chat = [];
  render();
  setTab("feed");
  refreshFeed();
}

async function startOver() {
  if (!confirm("Start over with new interests? Your current interests will be cleared.")) return;
  await api("/api/reset", { method: "POST" });
  state.profile = null;
  state.feed = null;
  state.chat = [];
  state.build = null;
  render();
}

// ============================================================================
// Events
// ============================================================================

const actions = {
  build: () => buildProfile(),
  persona: (el) => usePersona(el.dataset.id),
  tab: (el) => setTab(el.dataset.tab),
  "check-arxiv": () => refreshFeed({ force: true }),
  rescreen: () => refreshFeed(),
  filter: (el) => { state.filterTheme = el.dataset.id || null; renderFeed(); },
  "toggle-hidden": (el) => { state.showHidden = el.checked; renderFeed(); },
  "toggle-abstract": (el) => {
    const id = el.dataset.id;
    if (state.expanded.has(id)) state.expanded.delete(id); else state.expanded.add(id);
    renderFeed();
  },
  vote: (el) => vote(el.dataset.id, Number(el.dataset.vote)),
  "ask-why": (el) => {
    const paper = state.feed.papers.find((p) => p.id === el.dataset.id);
    ask(`Why was “${paper.title}” recommended to me?`);
  },
  "not-for-me": (el) => {
    const paper = state.feed.papers.find((p) => p.id === el.dataset.id);
    if (window.matchMedia("(max-width: 820px)").matches) setTab("assistant");
    const input = $("#composer-input");
    input.value = `I don't want papers like “${shortTitle(paper.title, 90)}” because `;
    autosize(input);
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  },
  "add-theme": () => { state.editing = "new"; renderInterests(); $("#theme-name")?.focus(); },
  "edit-theme": (el) => { state.editing = el.dataset.id; renderInterests(); $("#theme-desc")?.focus(); },
  "cancel-edit": () => { state.editing = null; renderInterests(); },
  "save-theme": (el) => {
    const name = $("#theme-name").value.trim();
    const description = $("#theme-desc").value.trim();
    if (!name || !description) return toast("A theme needs a name and a description.");
    const themes = el.dataset.id
      ? state.profile.themes.map((t) => (t.id === el.dataset.id ? { ...t, name, description } : t))
      : [...state.profile.themes, { name, description, examples: [] }];
    saveProfile({ ...state.profile, themes });
  },
  "delete-theme": (el) => {
    const theme = state.profile.themes.find((t) => t.id === el.dataset.id);
    if (!confirm(`Remove the theme “${theme.name}”?`)) return;
    saveProfile({ ...state.profile, themes: state.profile.themes.filter((t) => t.id !== el.dataset.id) });
  },
  "remove-excl": (el) => saveProfile({
    ...state.profile,
    not_interested: state.profile.not_interested.filter((_, i) => i !== Number(el.dataset.index)),
  }),
  "remove-cat": (el) => saveProfile({
    ...state.profile,
    categories: state.profile.categories.filter((c) => c !== el.dataset.code),
  }),
  suggest: (el) => ask(el.dataset.text),
  apply: (el) => applyProposal(Number(el.dataset.index)),
  discard: (el) => discardProposal(Number(el.dataset.index)),
};

document.addEventListener("click", (event) => {
  const el = event.target.closest("[data-action]");
  if (!el || el.tagName === "INPUT") return;
  const action = actions[el.dataset.action];
  if (action) {
    event.preventDefault();
    action(el, event);
  }
});

document.addEventListener("change", (event) => {
  const el = event.target.closest("input[data-action]");
  if (el && actions[el.dataset.action]) actions[el.dataset.action](el, event);
});

$("#reset-btn").addEventListener("click", startOver);

// ============================================================================
// Start
// ============================================================================

(async function init() {
  try {
    const data = await api("/api/state");
    state.hasKey = data.has_key;
    state.personas = data.personas;
    state.categories = data.categories;
    state.models = data.models;
    setCost(data.cost_usd);
    state.ready = true;
    if (data.profile) {
      setProfile(data.profile);
      state.flash.clear();
    }
    render();
    if (state.profile && state.hasKey) {
      setTab("feed");
      refreshFeed();
    }
  } catch (err) {
    $("#app").innerHTML = `<div class="loading">Could not reach the paperfind server: ${esc(err.message)}</div>`;
  }
})();
