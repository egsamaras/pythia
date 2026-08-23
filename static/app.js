/* Pythia dashboard. Plain JS, no dependencies.
   All model/feed-derived strings are untrusted: inserted via textContent only. */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

const CATEGORY_COLORS = {
  geopolitics: "var(--cat-geopolitics)",
  economy: "var(--cat-economy)",
  markets: "var(--cat-markets)",
  technology: "var(--cat-technology)",
  energy: "var(--cat-energy)",
  climate: "var(--cat-climate)",
  society: "var(--cat-society)",
  security: "var(--cat-security)",
};

const MODE_META = {
  event: { icon: "◆", label: "event" },
  consequence: { icon: "→", label: "consequence" },
  precedent: { icon: "⟲", label: "precedent" },
};

const REGION_LABELS = {
  global: "Global",
  north_america: "North America",
  latin_america: "Latin America",
  europe: "Europe",
  mena: "MENA",
  sub_saharan_africa: "Sub-Saharan Africa",
  south_central_asia: "South & Central Asia",
  east_asia_pacific: "East Asia & Pacific",
};

const state = {
  predictions: [],
  stats: null,
  articles: [],
  scans: [],
  filterStatus: "open",
  filterCategory: "all",
  filterMode: "all",
  filterRegion: "all",
  scanPollTimer: null,
  calibrationAsTable: false,
  digestFor: null,       // scan id whose digest is expanded
  digestArticles: [],
  digestInFlight: false,
  runInProgress: false,
  articlesLimit: 12,     // headlines sidebar: 12 by default, 50 expanded
};

/* ---------- helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function safeHref(url) {
  return typeof url === "string" && /^https?:\/\//.test(url) ? url : null;
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function fmtDateTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  return resp.json();
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`;
    let detail;
    try {
      const data = await resp.json();
      detail = data.detail;
      if (typeof detail === "string") message = detail;
    } catch { /* keep default */ }
    const err = new Error(message);
    err.status = resp.status;
    err.detail = detail;
    throw err;
  }
  return resp.json();
}

/* ---------- data loading ---------- */

async function refreshAll() {
  refreshHealth(); // fire-and-forget: recovers badge + filters after outages
  const list = document.getElementById("forecast-list");
  list.classList.add("refreshing"); // hold previous render, no skeleton flash
  try {
    const [preds, stats, articles, scans] = await Promise.all([
      getJSON("/api/predictions"),
      getJSON("/api/stats"),
      getJSON(`/api/articles?limit=${state.articlesLimit}`),
      getJSON("/api/scans"),
    ]);
    state.predictions = preds;
    state.stats = stats;
    state.articles = articles;
    state.scans = scans;
    renderStats();
    renderForecasts();
    renderArticles();
    renderScanStatus();
    renderCalibration();
  } finally {
    list.classList.remove("refreshing");
  }
}

async function refreshHealth() {
  const badge = document.getElementById("provider-badge");
  try {
    const health = await getJSON("/api/health");
    const p = health.provider;
    if (p.available) {
      badge.textContent = `oracle: ${p.provider} · ${p.model}`;
      badge.title = "Active oracle backend"; // drop any stale error tooltip
      badge.classList.remove("offline");
    } else {
      badge.textContent = "⚠ oracle offline"; // status color always ships with an icon
      badge.title = p.error || "";
      badge.classList.add("offline");
    }
    const catSelect = document.getElementById("category-filter");
    if (catSelect.options.length === 1) {
      for (const cat of health.categories) {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        catSelect.appendChild(opt);
      }
    }
    const modeSelect = document.getElementById("mode-filter");
    if (modeSelect.options.length === 1 && health.modes) {
      for (const mode of health.modes) {
        const opt = document.createElement("option");
        opt.value = mode;
        opt.textContent = mode;
        modeSelect.appendChild(opt);
      }
    }
    const regionSelect = document.getElementById("region-filter");
    if (regionSelect.options.length === 1 && health.regions) {
      for (const region of health.regions) {
        const opt = document.createElement("option");
        opt.value = region;
        opt.textContent = REGION_LABELS[region] || region;
        regionSelect.appendChild(opt);
      }
    }
  } catch {
    badge.textContent = "⚠ server unreachable";
    badge.classList.add("offline");
  }
}

/* ---------- scan flow ---------- */

function setRunButtons(running) {
  state.runInProgress = running;
  const scanBtn = document.getElementById("scan-btn");
  const checkBtn = document.getElementById("check-btn");
  scanBtn.disabled = running;
  checkBtn.disabled = running;
  scanBtn.textContent = running ? "Running…" : "Run scan";
  for (const b of document.querySelectorAll(".fc-check-now")) b.disabled = running;
}

async function startRun(url, onError) {
  setRunButtons(true);
  try {
    const res = await postJSON(url);
    pollScan(res.scan_id);
  } catch (err) {
    if (err.status === 409 && err.detail && err.detail.scan_id) {
      pollScan(err.detail.scan_id); // a run is already going: follow it
      return;
    }
    setRunButtons(false);
    const message = String(err.message || err);
    if (onError) onError(message);
    else renderScanError(message);
  }
}

function pollScan(scanId) {
  clearInterval(state.scanPollTimer);
  let failures = 0;
  const stopPolling = () => {
    clearInterval(state.scanPollTimer);
    setRunButtons(false);
  };
  state.scanPollTimer = setInterval(async () => {
    try {
      const scan = await getJSON(`/api/scans/${scanId}`);
      failures = 0;
      renderScanProgress(scan);
      if (scan.status !== "running") {
        stopPolling();
        try {
          await refreshAll();
        } catch (err) {
          renderScanError(`refresh failed: ${String(err.message || err)}`);
        }
      }
    } catch {
      // Tolerate transient failures, but don't poll a dead server forever.
      failures += 1;
      if (failures >= 15) {
        stopPolling();
        renderScanError("Lost contact with the server while a scan was running.");
      }
    }
  }, 2000);
}

function scanErrorEl(message) {
  // Status-critical color always ships with an icon, never color alone.
  return el("p", "scan-error", `⚠ ${message}`);
}

function scanTitle(scan) {
  const kind = scan.kind === "resolution" ? " · resolution check" : "";
  return `#${scan.id}${kind} · ${scan.provider}:${scan.model}`;
}

function renderScanProgress(scan) {
  const box = document.getElementById("scan-status");
  box.replaceChildren();
  box.appendChild(el("p", "scan-line", scanTitle(scan)));
  box.appendChild(el("p", "scan-line", scan.detail || scan.status));
  if (scan.error) box.appendChild(scanErrorEl(scan.error));
}

function renderScanError(message) {
  const box = document.getElementById("scan-status");
  box.replaceChildren(scanErrorEl(message));
}

function renderScanStatus() {
  const last = state.scans[0];
  if (!last) return;
  if (last.status === "running") {
    renderScanProgress(last);
    setRunButtons(true);
    pollScan(last.id);
    return;
  }
  const box = document.getElementById("scan-status");
  box.replaceChildren();
  const head = el("p", "scan-line");
  head.appendChild(el("span", null, `${scanTitle(last)} `));
  head.appendChild(el("span", "when", fmtDateTime(last.finished_at || last.started_at)));
  box.appendChild(head);
  box.appendChild(el("p", "scan-line", last.detail || last.status));
  if (last.error) box.appendChild(scanErrorEl(last.error));

  if (last.article_count > 0) {
    const expanded = state.digestFor === last.id;
    const btn = el(
      "button",
      "btn-ghost digest-toggle",
      expanded ? "Hide digest" : `View digest (${last.article_count} articles)`
    );
    btn.addEventListener("click", () =>
      toggleDigest(last.id).catch((err) => {
        // Keep the card (and the retry button) intact; append the error.
        box.appendChild(scanErrorEl(`digest: ${String(err.message || err)}`));
      }));
    box.appendChild(btn);
    if (expanded) {
      if (last.signals) {
        box.appendChild(el("p", "digest-caption", "Structured signals the oracle read:"));
        box.appendChild(el("pre", "digest-signals", last.signals));
      }
      if (!state.digestArticles.length) {
        box.appendChild(el("p", "digest-caption",
          "No digest was recorded for this scan — it predates digest tracking. "
          + "New scans record exactly what the oracle read."));
      } else {
        box.appendChild(el("p", "digest-caption",
          "Exactly what the oracle read for this run:"));
        const ul = el("ul", "article-list digest-list");
        for (const a of state.digestArticles) ul.appendChild(articleItem(a));
        box.appendChild(ul);
      }
    }
  }
}

async function toggleDigest(scanId) {
  if (state.digestInFlight) return; // double-click guard
  if (state.digestFor === scanId) {
    state.digestFor = null;
    state.digestArticles = [];
    renderScanStatus();
    return;
  }
  state.digestInFlight = true;
  try {
    state.digestArticles = await getJSON(`/api/scans/${scanId}/articles`);
    state.digestFor = scanId;
  } finally {
    state.digestInFlight = false;
  }
  renderScanStatus();
}

/* ---------- stats ---------- */

function renderStats() {
  const s = state.stats;
  if (!s) return;
  document.getElementById("stat-open").textContent = String(s.open);
  document.getElementById("stat-due").textContent = String(s.due);
  document.getElementById("stat-resolved").textContent = String(s.resolved);
  document.getElementById("stat-brier").textContent =
    s.brier === null ? "–" : s.brier.toFixed(3);
  const parts = [];
  for (const [mode, m] of Object.entries(s.by_mode || {})) {
    if (m.brier !== null) parts.push(`${mode} ${m.brier.toFixed(2)}`);
  }
  document.getElementById("stat-brier-modes").textContent = parts.join(" · ");
}

/* ---------- forecasts ---------- */

function visiblePredictions() {
  let preds = state.predictions;
  if (state.filterStatus === "open") preds = preds.filter((p) => p.status === "open");
  else if (state.filterStatus === "due") preds = preds.filter((p) => p.due);
  else if (state.filterStatus === "resolved") preds = preds.filter((p) => p.status === "resolved");
  if (state.filterCategory !== "all") {
    preds = preds.filter((p) => p.category === state.filterCategory);
  }
  if (state.filterMode !== "all") {
    preds = preds.filter((p) => p.mode === state.filterMode);
  }
  if (state.filterRegion !== "all") {
    preds = preds.filter((p) => (p.region || "global") === state.filterRegion);
  }
  const bySoonestHorizon = (a, b) => a.horizon_date.localeCompare(b.horizon_date);
  const byLatestResolved = (a, b) => (b.resolved_at || "").localeCompare(a.resolved_at || "");
  return [...preds].sort(
    state.filterStatus === "resolved" ? byLatestResolved : bySoonestHorizon
  );
}

function renderForecasts() {
  const list = document.getElementById("forecast-list");
  const preds = visiblePredictions();
  document.getElementById("forecast-count").textContent =
    `${preds.length} forecast${preds.length === 1 ? "" : "s"}`;
  list.replaceChildren();
  if (!preds.length) {
    const empty = el("div", "empty-state");
    empty.appendChild(el("p", null, state.predictions.length
      ? "Nothing matches the current filters."
      : "The oracle awaits. Run a scan to generate the first forecasts."));
    list.appendChild(empty);
    return;
  }
  for (const p of preds) list.appendChild(forecastCard(p));
}

function forecastCard(p) {
  const card = el("article", "forecast-card");

  const meta = el("div", "fc-meta");
  const chip = el("span", "cat-chip");
  const dot = el("span", "cat-dot");
  dot.style.background = CATEGORY_COLORS[p.category] || "var(--muted)";
  chip.append(dot, document.createTextNode(p.category));
  meta.appendChild(chip);
  const mm = MODE_META[p.mode] || MODE_META.event;
  meta.appendChild(el("span", "mode-chip", `${mm.icon} ${mm.label}`));
  meta.appendChild(
    el("span", "mode-chip", REGION_LABELS[p.region] || REGION_LABELS.global)
  );
  meta.appendChild(el("span", null, `by ${fmtDate(p.horizon_date)}`));
  if (p.due) meta.appendChild(el("span", "badge-due", "⏱ due"));
  if (p.status === "resolved") {
    meta.appendChild(
      p.outcome
        ? el("span", "badge-hit", "✓ happened")
        : el("span", "badge-miss", "✗ did not happen")
    );
  }
  card.appendChild(meta);

  card.appendChild(el("p", "fc-statement", p.statement));

  const probRow = el("div", "fc-prob-row");
  const pct = Math.round(p.probability * 100);
  probRow.appendChild(el("span", "fc-prob", `${pct}%`));
  const meter = el("div", "meter");
  meter.setAttribute("role", "meter");
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  meter.setAttribute("aria-valuenow", String(pct));
  meter.setAttribute("aria-label", "forecast probability");
  const fill = el("div", "meter-fill");
  fill.style.width = `${pct}%`;
  if (pct >= 99) fill.style.borderRadius = "4px";
  meter.appendChild(fill);
  probRow.appendChild(meter);
  card.appendChild(probRow);

  if (p.revisions && p.revisions.length) {
    const latest = p.revisions[0];
    const from = Math.round(latest.old_probability * 100);
    const older = p.revisions.length > 1 ? ` (+${p.revisions.length - 1} earlier)` : "";
    const rev = el("p", "fc-revision");
    rev.appendChild(el("span", null,
      `↻ revised from ${from}% · ${fmtDateTime(latest.created_at)}${older}`));
    if (latest.note) {
      rev.appendChild(document.createTextNode(` — ${latest.note}`));
    }
    card.appendChild(rev);
  }

  if (p.rationale) card.appendChild(el("p", "fc-rationale", p.rationale));

  if (p.basis && p.mode !== "event") {
    const basis = el("p", "fc-basis");
    basis.appendChild(
      el("strong", null, p.mode === "precedent" ? "Precedent: " : "Chain: ")
    );
    basis.appendChild(document.createTextNode(p.basis));
    card.appendChild(basis);
  }

  if (p.evidence && p.evidence.length) {
    card.appendChild(evidenceList(p.evidence, 4));
  }

  if (p.status === "open" && p.proposal_outcome !== null && p.proposal_outcome !== undefined) {
    card.appendChild(proposalBlock(p, card));
  }
  if (p.status === "resolved" && p.resolution_note) {
    card.appendChild(el("p", "fc-resolution-note", p.resolution_note));
    if (p.resolution_evidence && p.resolution_evidence.length) {
      card.appendChild(evidenceList(p.resolution_evidence, 3));
    }
  }

  const actions = el("div", "fc-actions");
  if (p.status === "open") {
    actions.appendChild(el("span", "hint", "Resolve:"));
    const yes = el("button", "btn-ghost", "Happened");
    yes.addEventListener("click", () =>
      cardAction(card, () => postJSON(`/api/predictions/${p.id}/resolve`, { outcome: true, note: "" })));
    const no = el("button", "btn-ghost", "Did not happen");
    no.addEventListener("click", () =>
      cardAction(card, () => postJSON(`/api/predictions/${p.id}/resolve`, { outcome: false, note: "" })));
    const checkNow = el("button", "btn-ghost fc-check-now", "Check now");
    checkNow.title = "Ask the oracle to check the news for this forecast and update it";
    checkNow.disabled = state.runInProgress; // one model run at a time
    checkNow.addEventListener("click", () =>
      startRun(`/api/predictions/${p.id}/check`, (message) => {
        let msg = card.querySelector(".fc-error");
        if (!msg) {
          msg = el("p", "fc-error");
          card.appendChild(msg);
        }
        msg.textContent = `⚠ ${message}`;
      }));
    actions.append(yes, no, checkNow);
  } else {
    const reopen = el("button", "btn-ghost", "Reopen");
    reopen.addEventListener("click", () =>
      cardAction(card, () => postJSON(`/api/predictions/${p.id}/reopen`)));
    actions.appendChild(reopen);
    if (p.resolved_at) {
      actions.appendChild(el("span", "hint", `resolved ${fmtDateTime(p.resolved_at)}`));
    }
  }
  card.appendChild(actions);
  return card;
}

function evidenceList(items, max) {
  const ul = el("ul", "fc-evidence");
  for (const a of items.slice(0, max)) {
    const li = el("li");
    const href = safeHref(a.url);
    if (href) {
      const link = el("a", null, a.title);
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      li.appendChild(link);
      li.appendChild(document.createTextNode(` — ${a.source}`));
    } else {
      li.textContent = `${a.title} — ${a.source}`;
    }
    ul.appendChild(li);
  }
  return ul;
}

function proposalBlock(p, card) {
  const box = el("div", "fc-proposal");
  const head = el("p", "fc-proposal-head");
  head.appendChild(
    p.proposal_outcome
      ? el("span", "badge-hit", "✓ oracle proposes: happened")
      : el("span", "badge-miss", "✗ oracle proposes: did not happen")
  );
  box.appendChild(head);
  if (p.proposal_note) box.appendChild(el("p", "fc-proposal-note", p.proposal_note));
  if (p.proposal_evidence && p.proposal_evidence.length) {
    box.appendChild(evidenceList(p.proposal_evidence, 3));
  }
  const actions = el("div", "fc-actions");
  const accept = el("button", "btn-ghost", "Accept");
  accept.addEventListener("click", () =>
    cardAction(card, () => postJSON(`/api/predictions/${p.id}/proposal/accept`)));
  const dismiss = el("button", "btn-ghost", "Dismiss");
  dismiss.addEventListener("click", () =>
    cardAction(card, () => postJSON(`/api/predictions/${p.id}/proposal/reject`)));
  actions.append(accept, dismiss);
  box.appendChild(actions);
  return box;
}

async function cardAction(card, action) {
  // Disable the card's buttons for the duration; surface failures on the card.
  const buttons = card.querySelectorAll("button");
  for (const b of buttons) b.disabled = true;
  try {
    await action();
  } catch (err) {
    for (const b of buttons) b.disabled = false;
    let msg = card.querySelector(".fc-error");
    if (!msg) {
      msg = el("p", "fc-error");
      card.appendChild(msg);
    }
    msg.textContent = `⚠ ${String(err.message || err)}`;
    return;
  }
  try {
    await refreshAll(); // the action itself succeeded; report refresh apart
  } catch (err) {
    renderScanError(`refresh failed: ${String(err.message || err)}`);
  }
}

/* ---------- articles ---------- */

function articleItem(a) {
  const li = el("li");
  const href = safeHref(a.url);
  if (href) {
    const link = el("a", null, a.title);
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    li.appendChild(link);
  } else {
    li.appendChild(el("span", null, a.title));
  }
  li.appendChild(el("span", "article-meta", `${a.source} · ${fmtDateTime(a.published_at)}`));
  return li;
}

function renderArticles() {
  const ul = document.getElementById("article-list");
  const moreBtn = document.getElementById("articles-more");
  ul.replaceChildren();
  if (!state.articles.length) {
    const li = el("li");
    li.appendChild(el("p", "empty", "No articles yet — run a scan to fetch the news."));
    ul.appendChild(li);
    moreBtn.hidden = true;
    return;
  }
  for (const a of state.articles) ul.appendChild(articleItem(a));
  // A short default page proves there is nothing more to show.
  moreBtn.hidden = state.articlesLimit === 12 && state.articles.length < 12;
  moreBtn.textContent = state.articlesLimit > 12 ? "Show fewer" : "Show more";
}

async function toggleArticleCount() {
  // Commit state only after the fetch succeeds, so a failure changes nothing.
  const target = state.articlesLimit > 12 ? 12 : 50;
  const articles = await getJSON(`/api/articles?limit=${target}`);
  state.articlesLimit = target;
  state.articles = articles;
  renderArticles();
}

/* ---------- calibration chart ---------- */

const tooltipEl = document.getElementById("tooltip");

function showTooltip(clientX, clientY, valueText, labelText) {
  tooltipEl.replaceChildren(
    el("div", "tt-value", valueText),
    el("div", null, labelText)
  );
  tooltipEl.hidden = false;
  const pad = 12;
  const rect = tooltipEl.getBoundingClientRect();
  let x = clientX + pad;
  let y = clientY - rect.height - pad;
  if (x + rect.width > window.innerWidth - 8) x = clientX - rect.width - pad;
  if (y < 8) y = clientY + pad;
  tooltipEl.style.left = `${x}px`;
  tooltipEl.style.top = `${y}px`;
}

function hideTooltip() {
  tooltipEl.hidden = true;
}

function renderCalibration() {
  const chartBox = document.getElementById("calibration-chart");
  const tableBox = document.getElementById("calibration-table");
  chartBox.replaceChildren();
  tableBox.replaceChildren();
  const bins = (state.stats && state.stats.calibration) || [];
  if (!bins.length) {
    chartBox.appendChild(
      el("p", "empty", "No resolved forecasts yet — calibration appears once outcomes are recorded.")
    );
    tableBox.appendChild(el("p", "empty", "No resolved forecasts yet."));
    return;
  }

  // SVG plot: x = mean forecast probability, y = observed frequency.
  const W = 320, H = 260;
  const m = { top: 10, right: 14, bottom: 42, left: 44 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;
  const sx = (v) => m.left + v * pw;
  const sy = (v) => m.top + (1 - v) * ph;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Calibration: forecast probability versus observed frequency");

  const styles = getComputedStyle(document.body);
  const gridColor = styles.getPropertyValue("--grid").trim();
  const baselineColor = styles.getPropertyValue("--baseline").trim();
  const pointColor = styles.getPropertyValue("--seq-400").trim();
  const surfaceColor = styles.getPropertyValue("--surface").trim();

  // hairline gridlines at 0/25/50/75/100 on both axes, solid, recessive
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const gx = document.createElementNS(SVG_NS, "line");
    gx.setAttribute("x1", sx(0)); gx.setAttribute("x2", sx(1));
    gx.setAttribute("y1", sy(t)); gx.setAttribute("y2", sy(t));
    gx.setAttribute("stroke", gridColor); gx.setAttribute("stroke-width", "1");
    svg.appendChild(gx);
    const gy = document.createElementNS(SVG_NS, "line");
    gy.setAttribute("x1", sx(t)); gy.setAttribute("x2", sx(t));
    gy.setAttribute("y1", sy(0)); gy.setAttribute("y2", sy(1));
    gy.setAttribute("stroke", gridColor); gy.setAttribute("stroke-width", "1");
    svg.appendChild(gy);

    const lx = document.createElementNS(SVG_NS, "text");
    lx.setAttribute("x", sx(t)); lx.setAttribute("y", sy(0) + 16);
    lx.setAttribute("text-anchor", "middle");
    lx.setAttribute("class", "tick-label");
    lx.textContent = `${Math.round(t * 100)}%`;
    svg.appendChild(lx);
    const ly = document.createElementNS(SVG_NS, "text");
    ly.setAttribute("x", sx(0) - 6); ly.setAttribute("y", sy(t) + 3);
    ly.setAttribute("text-anchor", "end");
    ly.setAttribute("class", "tick-label");
    ly.textContent = `${Math.round(t * 100)}%`;
    svg.appendChild(ly);
  }

  // perfect-calibration diagonal (reference baseline)
  const diag = document.createElementNS(SVG_NS, "line");
  diag.setAttribute("x1", sx(0)); diag.setAttribute("y1", sy(0));
  diag.setAttribute("x2", sx(1)); diag.setAttribute("y2", sy(1));
  diag.setAttribute("stroke", baselineColor);
  diag.setAttribute("stroke-width", "1.5");
  svg.appendChild(diag);

  // axis titles
  const xTitle = document.createElementNS(SVG_NS, "text");
  xTitle.setAttribute("x", sx(0.5)); xTitle.setAttribute("y", H - 6);
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("class", "axis-label");
  xTitle.textContent = "Forecast probability";
  svg.appendChild(xTitle);
  const yTitle = document.createElementNS(SVG_NS, "text");
  yTitle.setAttribute("transform", `translate(12 ${sy(0.5)}) rotate(-90)`);
  yTitle.setAttribute("text-anchor", "middle");
  yTitle.setAttribute("class", "axis-label");
  yTitle.textContent = "Observed frequency";
  svg.appendChild(yTitle);

  // data points: 10px dots with a 2px surface ring + 24px invisible hit target
  for (const b of bins) {
    const cx = sx(b.mean_predicted);
    const cy = sy(b.observed_rate);

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
    dot.setAttribute("r", "5");
    dot.setAttribute("fill", pointColor);
    dot.setAttribute("stroke", surfaceColor);
    dot.setAttribute("stroke-width", "2");
    svg.appendChild(dot);

    const hit = document.createElementNS(SVG_NS, "circle");
    hit.setAttribute("cx", cx); hit.setAttribute("cy", cy);
    // 28 viewBox units so the rendered hit area stays >= 24px even when the
    // 320-unit viewBox is scaled down to the narrowest sidebar width.
    hit.setAttribute("r", "14");
    hit.setAttribute("fill", "transparent");
    hit.setAttribute("tabindex", "0");
    const valueText = `observed ${Math.round(b.observed_rate * 100)}%`;
    const labelText =
      `forecast ~${Math.round(b.mean_predicted * 100)}% · n=${b.count}`;
    hit.setAttribute("aria-label", `${valueText}, ${labelText}`);
    hit.addEventListener("pointermove", (ev) =>
      showTooltip(ev.clientX, ev.clientY, valueText, labelText));
    hit.addEventListener("pointerleave", hideTooltip);
    hit.addEventListener("focus", () => {
      const r = hit.getBoundingClientRect();
      showTooltip(r.left + r.width / 2, r.top, valueText, labelText);
    });
    hit.addEventListener("blur", hideTooltip);
    svg.appendChild(hit);
  }
  chartBox.appendChild(svg);

  // table view — the accessibility twin of the chart
  const table = el("table", "cal-table");
  const thead = el("thead");
  const hrow = el("tr");
  for (const h of ["Bin", "Mean forecast", "Observed", "N"]) hrow.appendChild(el("th", null, h));
  thead.appendChild(hrow);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const b of bins) {
    const row = el("tr");
    row.appendChild(el("td", null,
      `${Math.round(b.bin_low * 100)}–${Math.round(b.bin_high * 100)}%`));
    row.appendChild(el("td", null, `${Math.round(b.mean_predicted * 100)}%`));
    row.appendChild(el("td", null, `${Math.round(b.observed_rate * 100)}%`));
    row.appendChild(el("td", null, String(b.count)));
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  tableBox.appendChild(table);
}

function toggleCalibrationView() {
  state.calibrationAsTable = !state.calibrationAsTable;
  const btn = document.getElementById("calibration-toggle");
  btn.setAttribute("aria-pressed", String(state.calibrationAsTable));
  btn.textContent = state.calibrationAsTable ? "Chart" : "Table";
  document.getElementById("calibration-chart").hidden = state.calibrationAsTable;
  document.getElementById("calibration-table").hidden = !state.calibrationAsTable;
}

/* ---------- wiring ---------- */

function initFilters() {
  for (const btn of document.querySelectorAll(".segmented button")) {
    btn.addEventListener("click", () => {
      for (const b of document.querySelectorAll(".segmented button")) {
        b.setAttribute("aria-selected", String(b === btn));
      }
      state.filterStatus = btn.dataset.status;
      renderForecasts();
    });
  }
  document.getElementById("category-filter").addEventListener("change", (ev) => {
    state.filterCategory = ev.target.value;
    renderForecasts();
  });
  document.getElementById("mode-filter").addEventListener("change", (ev) => {
    state.filterMode = ev.target.value;
    renderForecasts();
  });
  document.getElementById("region-filter").addEventListener("change", (ev) => {
    state.filterRegion = ev.target.value;
    renderForecasts();
  });
}

document.getElementById("scan-btn").addEventListener("click", () => startRun("/api/scan"));
document.getElementById("check-btn").addEventListener("click", () => startRun("/api/resolution-check"));
document.getElementById("calibration-toggle").addEventListener("click", toggleCalibrationView);
document.getElementById("articles-more").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  btn.disabled = true;
  try {
    await toggleArticleCount();
  } catch {
    btn.textContent = "⚠ retry"; // next successful render restores the label
  }
  btn.disabled = false;
});
initFilters();
refreshAll().catch((err) => renderScanError(String(err.message || err)));
