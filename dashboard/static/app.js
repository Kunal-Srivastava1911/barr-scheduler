"use strict";

/* ================================================================ helpers */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const SVGNS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
const isNum = x => x !== null && x !== undefined && Number.isFinite(Number(x));
const fmt = (x, d = 2) => (isNum(x) ? Number(x).toFixed(d) : "–");
const PCT_METRICS = new Set(["cs_share", "util", "first_q", "pred_under"]);
function fmtMetric(k, v) {
  if (!isNum(v)) return "–";
  if (k === "n_cs") return String(Math.round(v));
  if (PCT_METRICS.has(k)) return (v * 100).toFixed(1) + " %";
  if (k === "jain") return Number(v).toFixed(3);
  if (k === "throughput") return Number(v).toFixed(1);
  return Number(v).toFixed(2);
}
function niceTicks(max, count = 6) {
  if (!(max > 0)) return [0];
  const raw = max / count, mag = 10 ** Math.floor(Math.log10(raw)), e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
// horizontal bar with square baseline end and 4px rounded data end
function hbarPath(x, y, w, h, r = 4) {
  r = Math.min(r, w, h / 2);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}
function vbarPath(x, y0, w, h, r = 4) {   // y0 = baseline, grows upward
  r = Math.min(r, h, w / 2);
  const top = y0 - h;
  return `M${x},${y0}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${y0}Z`;
}
async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function stream(id, onEvent) {
  const es = new EventSource(`/api/jobs/${id}/events`);
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    onEvent(ev);
    if (ev.type === "done" || ev.type === "error") es.close();
  };
  es.onerror = () => es.close();
  return es;
}
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

/* tooltip */
const tip = $("#tip");
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14, r = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { tip.hidden = true; };

/* ================================================================ theme, tabs */
(function theme() {
  const saved = store.get("barr-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("barr-theme", next);
  });
})();

function selectTab(name) {
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + name));
  store.set("barr-tab", name);
  if (name === "sim" && SIM) layoutLanes();
}
$$(".tabs button").forEach(b => b.addEventListener("click", () => selectTab(b.dataset.tab)));

/* ================================================================ meta, status */
let META = null;
let TRAIN_ES = null;

function renderStatus() {
  const s = $("#status");
  const trained = META.trained;
  const training = !!META.train_job;
  const pred = training ? `<span class="chip"><span class="dot busy"></span>Training predictors…</span>`
    : trained ? `<span class="chip"><span class="dot good"></span>Predictors ready</span>`
      : `<span class="chip"><span class="dot bad"></span>Predictors not trained</span>`;
  const backend = META.report ? `<span class="chip">GBRT: ${esc(META.report.gbrt_backend || "numpy")}</span>` : "";
  const linux = `<span class="chip">Linux traces: ${META.linux_mixes.length ? META.linux_mixes.length + " mixes" : "none collected"}</span>`;
  s.innerHTML = pred + backend + linux;
}

async function loadMeta() {
  META = await api("/api/meta");
  renderStatus();
}

/* ================================================================ simulator */
const KIND_SLOT = { interactive: "--s1", compute: "--s2", mixed: "--s3" };
const kindColor = k => `var(${KIND_SLOT[k] || "--s4"})`;
let SIM = null;

function initSimForm() {
  const fam = $("#s-family");
  fam.innerHTML = META.families.map(f =>
    `<option value="${f}" ${f === "bimodal" ? "selected" : ""}>${f === "demo" ? "demo (4 processes, Fig. 3)" : f}</option>`).join("");
  const defaults = META.trained ? ["RR-5", "BARR-GBRT", "BARR-S-GBRT"] : ["RR-5", "BARR-EMA", "BARR-S-EMA"];
  const names = META.schedulers.map(s => s.name).concat(["BARR-LINEAR", "BARR-KNN", "BARR-S-KNN"]);
  if (META.report && META.report.models && META.report.models.lstm) names.push("BARR-LSTM", "BARR-S-LSTM");
  $("#s-scheds").innerHTML = names.map(n =>
    `<label><input type="checkbox" value="${n}" ${defaults.includes(n) ? "checked" : ""}> ${esc(n)}</label>`).join("");
  $("#s-scheds").addEventListener("change", e => {
    const on = $$("#s-scheds input:checked");
    if (on.length > 4) { e.target.checked = false; $("#s-msg").textContent = "Up to four schedulers at once."; }
  });
  fam.addEventListener("change", () => {
    const demo = fam.value === "demo";
    ["#s-n", "#s-load", "#s-bmin", "#s-bmax", "#s-seed"].forEach(id => { $(id).disabled = demo; });
  });
  $("#s-run").addEventListener("click", runSim);
  $("#kind-legend").innerHTML = ["interactive", "compute", "mixed"].map(k =>
    `<span><i style="background:${kindColor(k)}"></i>${k}</span>`).join("") +
    `<span><i style="background:var(--wait)"></i>ready, waiting</span><span><i style="background:var(--cs)"></i>context switch</span>`;
}

async function runSim() {
  const btn = $("#s-run");
  const schedulers = $$("#s-scheds input:checked").map(i => i.value);
  if (!schedulers.length) { $("#s-msg").textContent = "Select at least one scheduler."; return; }
  const body = {
    family: $("#s-family").value, n_procs: +$("#s-n").value, load: +$("#s-load").value,
    seed: +$("#s-seed").value, bursts: [+$("#s-bmin").value, +$("#s-bmax").value],
    cs_cost: +$("#s-cs").value, schedulers,
    barr: { delta: +$("#b-delta").value, beta: +$("#b-beta").value, q_min: +$("#b-qmin").value,
            q_max: +$("#b-qmax").value, aging: +$("#b-aging").value },
    barr_s: { delta: +$("#b-delta").value, beta: +$("#b-beta").value, q_min: +$("#b-qmin").value,
              q_max: +$("#bs-qmax").value, aging: +$("#b-aging").value },
  };
  btn.disabled = true;
  $("#s-msg").textContent = "Simulating…";
  try {
    const data = await api("/api/simulate", body);
    $("#s-msg").textContent = `${data.procs.length} processes, ${data.procs.reduce((a, p) => a + p.n_bursts, 0)} bursts, c = ${data.cs_cost} ms.`;
    startSim(data);
  } catch (e) {
    $("#s-msg").textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

function prepResult(r, procs) {
  const sl = r.slices.slice().sort((a, b) => a[2] - b[2]);
  const sw = [];
  let last = null;
  for (const s of sl) {
    if (last !== null && s[0] !== last) sw.push(s[2]);
    last = s[0];
  }
  const done = procs.map(p => {
    const c = r.completion[p.pid];
    return [c, c - p.arrival - p.cpu - p.io];
  }).sort((a, b) => a[0] - b[0]);
  const prefix = [0];
  for (const d of done) prefix.push(prefix[prefix.length - 1] + d[1]);
  return { ...r, sl, sw, done, prefix };
}
function upperBound(arr, t, key = x => x) {    // number of items with key <= t
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (key(arr[m]) <= t) lo = m + 1; else hi = m; }
  return lo;
}

function startSim(data) {
  if (SIM && SIM.raf) cancelAnimationFrame(SIM.raf);
  const tEnd = Math.max(...data.results.map(r => r.end), 1e-3);
  SIM = {
    data, tEnd, t: 0, playing: true, last: null,
    results: data.results.map(r => prepResult(r, data.procs)),
    lanes: [],
  };
  $("#p-time").max = 1000;
  buildLanes();
  renderCompare();
  $("#p-play").textContent = "❚❚";
  const at = parseFloat(new URLSearchParams(location.search).get("at"));
  if (at >= 0 && at <= 1) {            // ?at=0.6 opens paused at 60 % of the run
    SIM.playing = false;
    SIM.t = at * tEnd;
    $("#p-play").textContent = "▶";
    setTime(SIM.t);
  }
  SIM.raf = requestAnimationFrame(tick);
}

function tick(now) {
  if (!SIM) return;
  if (SIM.playing) {
    if (SIM.last !== null) {
      const speed = +$("#p-speed").value;
      SIM.t = Math.min(SIM.tEnd, SIM.t + (now - SIM.last) / 1000 * (SIM.tEnd / 14) * speed);
    }
    SIM.last = now;
    setTime(SIM.t);
    if (SIM.t >= SIM.tEnd) { SIM.playing = false; $("#p-play").textContent = "▶"; }
  } else {
    SIM.last = null;
  }
  SIM.raf = requestAnimationFrame(tick);
}

$("#p-play").addEventListener("click", () => {
  if (!SIM) return;
  if (SIM.t >= SIM.tEnd) SIM.t = 0;
  SIM.playing = !SIM.playing;
  SIM.last = null;
  $("#p-play").textContent = SIM.playing ? "❚❚" : "▶";
});
$("#p-restart").addEventListener("click", () => {
  if (!SIM) return;
  SIM.t = 0; SIM.playing = true; SIM.last = null;
  $("#p-play").textContent = "❚❚";
});
$("#p-time").addEventListener("input", e => {
  if (!SIM) return;
  SIM.playing = false;
  $("#p-play").textContent = "▶";
  SIM.t = e.target.value / 1000 * SIM.tEnd;
  setTime(SIM.t);
});

function buildLanes() {
  const host = $("#lanes");
  host.innerHTML = "";
  SIM.lanes = SIM.results.map((r, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="lane-head">
        <h3>${esc(r.label)}</h3>${r.oracle ? '<span class="badge">oracle: knows true bursts</span>' : ""}
        <div class="kpis">
          <div class="kpi"><b class="num" data-k="cpu">–</b><span>on CPU</span></div>
          <div class="kpi"><b class="num" data-k="done">0</b><span>completed</span></div>
          <div class="kpi"><b class="num" data-k="cs">0</b><span>context switches</span></div>
          <div class="kpi"><b class="num" data-k="wt">–</b><span>mean WT so far (ms)</span></div>
        </div>
      </div>
      <div class="queue" data-k="queue"></div>
      <div class="gantt-host"></div>`;
    host.appendChild(card);
    return { card, r, idx: i };
  });
  layoutLanes();
}

function layoutLanes() {
  if (!SIM) return;
  for (const lane of SIM.lanes) drawGantt(lane);
  setTime(SIM.t);
}

function drawGantt(lane) {
  const { r } = lane;
  const host = lane.card.querySelector(".gantt-host");
  host.innerHTML = "";
  const procs = SIM.data.procs;
  const n = procs.length;
  const W = Math.max(280, host.clientWidth || 800);
  const labW = 46, rightPad = 8, axisH = 20;
  const rowH = Math.max(9, Math.min(22, Math.round(320 / n)));
  const barH = rowH - 3;
  const H = n * rowH + axisH + 4;
  const plotW = W - labW - rightPad;
  const x = t => labW + (t / SIM.tEnd) * plotW;
  const row = {};
  procs.forEach((p, i) => { row[p.pid] = i; });
  const s = svg("svg", { class: "gantt", width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Gantt chart for ${r.label}` });
  const ticks = niceTicks(SIM.tEnd, Math.max(3, Math.floor(plotW / 90)));
  for (const t of ticks) {
    svg("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: n * rowH }, s);
    const lab = svg("text", { x: x(t), y: n * rowH + 14, "text-anchor": "middle" }, s);
    lab.textContent = t === 0 ? "0 ms" : String(t);
  }
  procs.forEach((p, i) => {
    const lab = svg("text", { class: "rowlab", x: labW - 6, y: i * rowH + barH / 2 + 4, "text-anchor": "end" }, s);
    lab.textContent = "P" + p.pid;
  });
  const clipId = `clip-${lane.idx}-${Date.now()}`;
  const defs = svg("defs", {}, s);
  const clip = svg("clipPath", { id: clipId }, defs);
  const clipRect = svg("rect", { x: labW, y: 0, width: 0, height: n * rowH }, clip);
  const g = svg("g", { "clip-path": `url(#${clipId})` }, s);
  r.bursts.forEach((b, j) => {
    const y = row[b[0]] * rowH;
    svg("rect", { class: "wait", x: x(b[2]), y, width: Math.max(0.5, x(b[4]) - x(b[2])), height: barH,
      fill: "var(--wait)", rx: 2, "data-b": j }, g);
  });
  const kind = {};
  procs.forEach(p => { kind[p.pid] = p.kind; });
  r.sl.forEach((sl, j) => {
    const y = row[sl[0]] * rowH;
    if (sl[3] > sl[2] + 1e-9) {
      svg("rect", { x: x(sl[2]), y, width: Math.max(0.6, x(sl[3]) - x(sl[2])), height: barH, fill: "var(--cs)" }, g);
    }
    if (sl[4] > sl[3]) {
      svg("rect", { class: "seg", x: x(sl[3]), y, width: Math.max(0.8, x(sl[4]) - x(sl[3])), height: barH,
        fill: kindColor(kind[sl[0]]), rx: 1.5, "data-s": j }, g);
    }
  });
  const cursor = svg("line", { class: "cursor", x1: labW, x2: labW, y1: 0, y2: n * rowH }, s);
  host.appendChild(s);
  lane.g = { clipRect, cursor, x, labW };

  const byPid = {};
  procs.forEach(p => { byPid[p.pid] = p; });
  s.addEventListener("mousemove", ev => {
    const t = ev.target;
    if (t.dataset && t.dataset.s !== undefined) {
      const sl = r.sl[+t.dataset.s];
      const p = byPid[sl[0]];
      showTip(`<b>P${sl[0]}</b> <span class="muted">${esc(p.kind)}</span><br>burst ${sl[1] + 1} of ${p.n_bursts}<br>` +
        `runs ${fmt(sl[3])} → ${fmt(sl[4])} ms (${fmt(sl[4] - sl[3])} ms)` +
        (sl[3] > sl[2] ? `<br><span class="muted">after a ${fmt(sl[3] - sl[2])} ms context switch</span>` : ""), ev);
    } else if (t.dataset && t.dataset.b !== undefined) {
      const b = r.bursts[+t.dataset.b];
      const pred = b[6] > 0 ? `<br>predicted ${fmt(b[6])} ms · actual ${fmt(b[5])} ms` : `<br>burst ${fmt(b[5])} ms`;
      showTip(`<b>P${b[0]}</b> burst ${b[1] + 1}<br>ready ${fmt(b[2])} ms · first run ${fmt(b[3])} ms<br>` +
        `response ${fmt(b[3] - b[2])} ms · done ${fmt(b[4])} ms${pred}` +
        `<br><span class="muted">${b[7] ? "preempted at least once" : "completed in its first quantum"} (q₁ = ${fmt(b[8])} ms)</span>`, ev);
    } else hideTip();
  });
  s.addEventListener("mouseleave", hideTip);
}

function setTime(t) {
  if (!SIM) return;
  $("#p-time").value = Math.round(t / SIM.tEnd * 1000);
  $("#p-readout").textContent = `t = ${t.toFixed(1)} / ${SIM.tEnd.toFixed(1)} ms`;
  for (const lane of SIM.lanes) {
    const { r, card } = lane;
    if (lane.g) {
      const xx = lane.g.x(t);
      lane.g.clipRect.setAttribute("width", Math.max(0, xx - lane.g.labW));
      lane.g.cursor.setAttribute("x1", xx);
      lane.g.cursor.setAttribute("x2", xx);
    }
    const k = q => card.querySelector(`[data-k="${q}"]`);
    const nDone = upperBound(r.done, t, d => d[0]);
    k("done").textContent = `${nDone}/${r.done.length}`;
    k("cs").textContent = upperBound(r.sw, t);
    k("wt").textContent = nDone ? (r.prefix[nDone] / nDone).toFixed(2) : "–";
    const i = upperBound(r.sl, t, s => s[2]) - 1;
    let cpu = "idle";
    if (i >= 0) {
      const s = r.sl[i];
      if (t < s[3]) cpu = "switch";
      else if (t < s[4]) cpu = "P" + s[0];
    }
    if (t >= SIM.tEnd) cpu = "done";
    k("cpu").textContent = cpu;
    const qi = upperBound(r.queue, t, q => q[0]) - 1;
    const queue = qi >= 0 && t < SIM.tEnd ? r.queue[qi][2] : [];
    const running = cpu.startsWith("P") ? `<span class="qpill run">${cpu}</span>` : "";
    k("queue").innerHTML = `<span>CPU</span>${running || `<span class="qpill">${cpu}</span>`}<span>ready queue →</span>` +
      (queue.length ? queue.slice(0, 24).map(p => `<span class="qpill">P${p}</span>`).join("") +
        (queue.length > 24 ? `<span>+${queue.length - 24}</span>` : "") : `<span>empty</span>`);
  }
}

const COMPARE = [["mean_wt", "Mean waiting time (ms)"], ["p95_resp", "P95 response time (ms)"],
  ["n_cs", "Context switches"], ["max_wt", "Maximum waiting time (ms)"]];

function renderCompare() {
  $("#compare-card").hidden = false;
  const host = $("#compare-charts");
  host.innerHTML = "";
  const boxes = COMPARE.map(([, title]) => {
    const box = document.createElement("div");
    box.innerHTML = `<h4 style="margin-bottom:6px">${title}</h4>`;
    host.appendChild(box);
    return box;
  });
  // measure only after every box is in the grid, so each gets its own column width
  COMPARE.forEach(([key], i) => hbarChart(boxes[i],
    SIM.results.map(r => ({ label: r.label, value: r.metrics[key], fmt: fmtMetric(key, r.metrics[key]) }))));
  const keys = Object.keys(META.metrics).filter(k => SIM.results.some(r => isNum(r.metrics[k])));
  let html = `<thead><tr><th>Metric</th>${SIM.results.map(r => `<th>${esc(r.label)}</th>`).join("")}</tr></thead><tbody>`;
  for (const k of keys) {
    const m = META.metrics[k];
    const vals = SIM.results.map(r => r.metrics[k]);
    const finite = vals.filter(isNum);
    const best = finite.length > 1 ? (m.better === "higher" ? Math.max(...finite) : Math.min(...finite)) : null;
    html += `<tr><td>${esc(m.label)}${m.unit ? ` <span class="muted">(${m.unit})</span>` : ""}${m.better === "higher" ? " ↑" : ""}</td>` +
      vals.map(v => `<td style="${v === best ? "font-weight:700" : ""}">${fmtMetric(k, v)}</td>`).join("") + "</tr>";
  }
  $("#compare-table").innerHTML = html + "</tbody>";
}

function hbarChart(host, items) {
  const W = Math.max(220, host.clientWidth || 280);
  const labW = Math.min(120, Math.max(...items.map(i => i.label.length)) * 6.6 + 8);
  const valW = 56, rowH = 26, barH = 14;
  const H = items.length * rowH + 4;
  const max = Math.max(...items.map(i => (isNum(i.value) ? i.value : 0)), 1e-9);
  const finite = items.filter(i => isNum(i.value)).map(i => i.value);
  const best = Math.min(...finite);
  const s = svg("svg", { class: "bars", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  const plotW = W - labW - valW;
  svg("line", { x1: labW, x2: labW, y1: 0, y2: H, stroke: "var(--axis)" }, s);
  items.forEach((it, i) => {
    const y = i * rowH + (rowH - barH) / 2;
    const lab = svg("text", { x: labW - 8, y: y + barH / 2 + 4, "text-anchor": "end" }, s);
    lab.textContent = it.label;
    const w = isNum(it.value) ? Math.max(1, it.value / max * plotW) : 0;
    const bar = svg("path", { d: hbarPath(labW, y, w, barH), fill: "var(--s1)" }, s);
    const val = svg("text", { class: "val" + (it.value === best && finite.length > 1 ? " best" : ""), x: labW + w + 6, y: y + barH / 2 + 4 }, s);
    val.textContent = it.fmt + (it.value === best && finite.length > 1 ? " ★" : "");
    const hit = svg("rect", { x: 0, y: i * rowH, width: W, height: rowH, fill: "transparent" }, s);
    hit.addEventListener("mousemove", ev => showTip(`<b>${esc(it.label)}</b><br>${it.fmt}`, ev));
    hit.addEventListener("mouseleave", hideTip);
    void bar;
  });
  host.appendChild(s);
}

/* ================================================================ predictors */
const PRED_LABEL = { ema: "Exponential average", last: "Last value", linear: "Linear (ridge)",
  knn: "kNN (k = 10)", gbrt: "GBRT", lstm: "LSTM" };
const PRED_SLOT = { gbrt: "--s1", lstm: "--s2", linear: "--s3", knn: "--s4", last: "--s5" };

function initPredictors() {
  $("#t-lstm").disabled = !META.lstm_available;
  $("#t-lstm").checked = META.lstm_available;
  $("#t-lstm-note").textContent = META.lstm_available ? "" : "(needs PyTorch)";
  $("#t-run").addEventListener("click", async () => {
    try {
      const { id } = await api("/api/jobs", { kind: "train", params: { n_procs: +$("#t-n").value, lstm: $("#t-lstm").checked } });
      META.train_job = id;
      renderStatus();
      attachTrain(id, true);
    } catch (e) { $("#t-log").textContent = e.message; }
  });
  if (META.report) renderReport(META.report);
  if (META.train_job) attachTrain(META.train_job, true);
}

function attachTrain(id, reset) {
  const log = $("#t-log");
  if (reset) log.textContent = "";
  $("#t-run").disabled = true;
  if (TRAIN_ES) TRAIN_ES.close();
  TRAIN_ES = stream(id, ev => {
    if (ev.type === "log") { log.textContent += ev.msg + "\n"; log.scrollTop = log.scrollHeight; }
    if (ev.type === "done") {
      META.trained = true; META.train_job = null; META.report = ev.report;
      $("#t-run").disabled = false;
      log.textContent += `\nDone in ${ev.report.seconds.toFixed(1)} s. Learned BARR variants are now available.\n`;
      renderStatus(); renderReport(ev.report);
    }
    if (ev.type === "error") {
      META.train_job = null; $("#t-run").disabled = false;
      log.textContent += "\nERROR: " + ev.msg + "\n"; renderStatus();
    }
  });
}

function renderReport(rep) {
  const models = rep.models || {};
  const names = ["ema", "last", "linear", "knn", "gbrt", "lstm"].filter(n => models[n]);
  $("#p-meta").textContent = `${rep.n_train} train / ${rep.n_val} val / ${rep.n_test} test bursts · EMA α = ${rep.ema_alpha} · GBRT ${rep.gbrt_backend}, ${rep.gbrt_loss} loss`;
  let html = `<thead><tr><th>Predictor</th><th>MAE (ms)</th><th>MAE vs EMA</th><th>log error</th><th>under-predicted</th><th>latency median (µs)</th><th>latency p99 (µs)</th></tr></thead><tbody>`;
  const best = Math.min(...names.map(n => models[n].mae));
  for (const n of names) {
    const m = models[n];
    const rel = m.mae_rel_ema;
    html += `<tr class="${m.mae === best ? "hl" : ""}"><td>${PRED_LABEL[n] || n}</td><td>${fmt(m.mae, 3)}</td>` +
      `<td>${fmt(rel, 3)} <span class="muted">(${rel < 1 ? "−" : "+"}${Math.abs((1 - rel) * 100).toFixed(1)} %)</span></td>` +
      `<td>${fmt(m.log_err, 3)}</td><td>${(m.under * 100).toFixed(1)} %</td><td>${fmt(m.lat_med_us, 1)}</td><td>${fmt(m.lat_p99_us, 1)}</td></tr>`;
  }
  $("#p-table").innerHTML = html + "</tbody>";

  // grouped bars: MAE relative to EMA, overall + per family
  const series = ["gbrt", "lstm", "linear", "knn", "last"].filter(n => models[n]);
  const fams = Object.keys(models.ema.family || {});
  const groups = [{ name: "all", get: n => models[n].mae / models.ema.mae }].concat(
    fams.map(f => ({ name: f, get: n => models[n].family[f].mae / models.ema.family[f].mae })));
  groupedBars($("#p-chart"), groups, series);
}

function groupedBars(host, groups, series) {
  host.innerHTML = "";
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.style.marginBottom = "8px";
  legend.innerHTML = series.map(n => `<span><i style="background:var(${PRED_SLOT[n]})"></i>${PRED_LABEL[n]}</span>`).join("") +
    `<span><i style="background:none;border-top:2px dashed var(--ink-2);height:0;border-radius:0"></i>EMA = 1.0</span>`;
  host.appendChild(legend);
  const W = Math.max(300, host.clientWidth || 500), H = 260;
  const m = { l: 38, r: 8, t: 10, b: 28 };
  const vals = groups.flatMap(g => series.map(n => g.get(n)));
  const ymax = Math.max(1.15, ...vals) * 1.05;
  const y = v => m.t + (1 - v / ymax) * (H - m.t - m.b);
  const s = svg("svg", { class: "bars", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  for (const t of niceTicks(ymax, 5)) {
    svg("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)" }, s);
    const lab = svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, s);
    lab.textContent = t.toFixed(1);
  }
  const gw = (W - m.l - m.r) / groups.length;
  const bw = Math.min(18, (gw - 16) / series.length - 2);
  groups.forEach((g, gi) => {
    const x0 = m.l + gi * gw + (gw - series.length * (bw + 2)) / 2;
    series.forEach((n, si) => {
      const v = g.get(n);
      const x = x0 + si * (bw + 2);
      svg("path", { d: vbarPath(x, y(0), bw, y(0) - y(v)), fill: `var(${PRED_SLOT[n]})` }, s);
      const hit = svg("rect", { x: x - 1, y: m.t, width: bw + 2, height: H - m.t - m.b, fill: "transparent" }, s);
      hit.addEventListener("mousemove", ev => showTip(`<b>${PRED_LABEL[n]}</b> · ${esc(g.name)}<br>MAE / EMA = ${v.toFixed(3)}`, ev));
      hit.addEventListener("mouseleave", hideTip);
    });
    const lab = svg("text", { x: m.l + gi * gw + gw / 2, y: H - 8, "text-anchor": "middle" }, s);
    lab.textContent = g.name;
  });
  svg("line", { x1: m.l, x2: W - m.r, y1: y(1), y2: y(1), stroke: "var(--ink-2)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, s);
  svg("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: "var(--axis)" }, s);
  host.appendChild(s);
}

/* ================================================================ experiments */
function initExperiments() {
  const presets = Object.keys(META.configs);
  $("#e-preset").innerHTML = presets.map(p => `<option value="${p}">${p}</option>`).join("") + `<option value="custom">custom</option>`;
  $("#e-fams").innerHTML = META.families.filter(f => f !== "demo").map(f =>
    `<label><input type="checkbox" value="${f}" checked> ${f}</label>`).join("");
  $("#e-scheds").innerHTML = META.schedulers.map(s =>
    `<label><input type="checkbox" value="${s.name}"> ${esc(s.name)}</label>`).join("");
  $("#e-preset").addEventListener("change", () => applyPreset($("#e-preset").value));
  applyPreset(presets.includes("draft") ? "draft" : presets[0]);
  $("#e-run").addEventListener("click", runExperiment);
  refreshResultList();
  $("#e-load").addEventListener("change", async () => {
    const name = $("#e-load").value;
    if (!name) return;
    const { summary } = await api(`/api/results/${encodeURIComponent(name)}`);
    renderSummary(summary, name);
  });
}

function refreshResultList(select) {
  const opts = META.results;
  $("#e-load").innerHTML = `<option value="">${opts.length ? "choose…" : "none yet"}</option>` +
    opts.map(r => `<option value="${esc(r)}" ${r === select ? "selected" : ""}>${esc(r)}</option>`).join("");
  if (!select && opts.length && !$("#e-results").children.length) {
    const first = opts.includes("draft") ? "draft" : opts[0];
    $("#e-load").value = first;
    api(`/api/results/${encodeURIComponent(first)}`).then(d => renderSummary(d.summary, first)).catch(() => {});
  }
}

function applyPreset(name) {
  const c = META.configs[name];
  if (!c) return;
  $("#e-preset").value = name;
  $("#e-name").value = c.name || name;
  $$("#e-fams input").forEach(i => { i.checked = (c.families || []).includes(i.value); });
  $("#e-loads").value = (c.loads || [0.7]).join(", ");
  $("#e-seeds").value = (c.seeds || [1]).length;
  $("#e-n").value = c.n_procs || 60;
  $("#e-cs").value = c.cs_cost ?? 0.1;
  $$("#e-scheds input").forEach(i => { i.checked = c.schedulers.includes(i.value); });
}

async function runExperiment() {
  const base = META.configs[$("#e-preset").value] || META.configs.draft || {};
  const cfg = {
    ...base,
    name: ($("#e-name").value || "experiment").replace(/[^\w.-]/g, "_"),
    families: $$("#e-fams input:checked").map(i => i.value),
    loads: $("#e-loads").value.split(/[ ,]+/).filter(Boolean).map(Number),
    seeds: Array.from({ length: Math.max(1, +$("#e-seeds").value) }, (_, i) => i + 1),
    n_procs: +$("#e-n").value,
    cs_cost: +$("#e-cs").value,
    schedulers: $$("#e-scheds input:checked").map(i => i.value),
  };
  if (!cfg.schedulers.length) { $("#e-count").textContent = "Select schedulers first."; return; }
  if (!cfg.schedulers.some(s => s.startsWith("RR-"))) { $("#e-count").textContent = "Include at least one fixed RR (the reference)."; return; }
  const btn = $("#e-run");
  btn.disabled = true;
  const log = $("#e-log");
  log.textContent = "";
  $("#e-bar").style.width = "0";
  const live = $("#e-live");
  live.innerHTML = `<thead><tr><th>Workload</th><th>Scheduler</th><th>mean WT</th><th>switches</th><th>P95 resp</th><th>max WT</th></tr></thead><tbody></tbody>`;
  const tbody = live.querySelector("tbody");
  try {
    const { id } = await api("/api/jobs", { kind: "experiment", params: cfg });
    stream(id, ev => {
      if (ev.type === "log") { log.textContent += ev.msg + "\n"; log.scrollTop = log.scrollHeight; }
      if (ev.type === "progress") {
        $("#e-bar").style.width = (ev.done / ev.total * 100).toFixed(1) + "%";
        $("#e-count").textContent = `${ev.done} / ${ev.total} runs · ${ev.t.toFixed(1)} s`;
        const r = ev.row;
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${esc(r.workload)}</td><td>${esc(r.scheduler)}</td><td>${fmt(r.mean_wt)}</td><td>${r.n_cs}</td><td>${fmt(r.p95_resp)}</td><td>${fmt(r.max_wt)}</td>`;
        tbody.prepend(tr);
        while (tbody.children.length > 40) tbody.lastChild.remove();
      }
      if (ev.type === "done") {
        btn.disabled = false;
        if (!META.results.includes(ev.name)) META.results.push(ev.name);
        refreshResultList(ev.name);
        renderSummary(ev.summary, ev.name);
      }
      if (ev.type === "error") { btn.disabled = false; log.textContent += "\nERROR: " + ev.msg; $("#e-count").textContent = "failed: " + ev.msg; }
    });
  } catch (e) {
    btn.disabled = false;
    $("#e-count").textContent = e.message;
  }
}

const LABEL = n => { const s = (META.schedulers.find(x => x.name === n) || {}).label; return s || n; };
const IS_ORACLE = n => !!(META.schedulers.find(x => x.name === n) || {}).oracle;

function heat(v) {
  if (!isNum(v)) return "";
  const z = Math.max(-1, Math.min(1, Math.log2(v) / 1.5));
  const pct = Math.round(Math.abs(z) * 55);
  const pole = z < 0 ? "var(--div-neg)" : "var(--div-pos)";
  return `background:color-mix(in oklab, ${pole} ${pct}%, var(--div-mid))`;
}

function renderSummary(summary, name) {
  const host = $("#e-results");
  host.innerHTML = "";
  const groups = summary.groups || {};
  for (const [g, d] of Object.entries(groups)) {
    const card = document.createElement("div");
    card.className = "card";
    const bestRR = Object.entries(d.best_rr || {}).map(([s, c]) => `${LABEL(s)} on ${c}`).join(", ");
    card.innerHTML = `
      <div class="card-head"><h3>${esc(name)} · ${g === "synthetic" ? "synthetic workloads" : "Linux trace windows"}</h3>
        <span class="muted small">${d.n_workloads} workloads</span><span class="spacer"></span>
        <a class="small" href="/results/${encodeURIComponent(name)}/runs.csv" download>runs.csv</a></div>
      <p class="small muted">Ratios to the best fixed RR of each workload (lowest mean WT; ${esc(bestRR)}), geometric mean over workloads. Blue cells are better than the best RR, red cells worse.</p>
      <div class="two-col">
        <div class="table-wrap"><table class="data heat-table"></table></div>
        <div><h4 style="margin-bottom:6px">Trade-off: context switches vs mean waiting time</h4><div class="scatter"></div></div>
      </div>
      <div class="two-col" style="margin-top:16px">
        <div><h4 style="margin-bottom:6px">Planned success thresholds</h4><div class="table-wrap"><table class="data thr"></table></div></div>
        <div><h4 style="margin-bottom:6px">Wilcoxon signed-rank vs the strongest non-oracle baseline</h4><div class="table-wrap" style="max-height:260px;overflow:auto"><table class="data wil"></table></div></div>
      </div>`;
    host.appendChild(card);

    const metrics = [["mean_wt", "WT"], ["n_cs", "CS"], ["p95_resp", "P95"], ["max_wt", "max WT"]];
    let html = `<thead><tr><th>Scheduler</th>${metrics.map(m => `<th>${m[1]}</th>`).join("")}<th>mean WT ms [95 % CI]</th></tr></thead><tbody>`;
    for (const [s, r] of Object.entries(d.ratios)) {
      const raw = (d.raw[s] || {}).mean_wt || [];
      const barr = s.startsWith("BARR");
      html += `<tr class="${barr ? "hl" : ""}"><td>${esc(LABEL(s))}${IS_ORACLE(s) ? ' <span class="badge">oracle</span>' : ""}</td>` +
        metrics.map(([k]) => `<td class="heat" style="${heat(r[k])}">${fmt(r[k])}</td>`).join("") +
        `<td>${fmt(raw[0], 1)} <span class="muted">[${fmt(raw[1], 1)}, ${fmt(raw[2], 1)}]</span></td></tr>`;
    }
    card.querySelector(".heat-table").innerHTML = html + "</tbody>";

    scatter(card.querySelector(".scatter"), Object.entries(d.ratios).map(([s, r]) => ({
      name: s, label: LABEL(s), x: r.n_cs, y: r.mean_wt, barr: s.startsWith("BARR"), oracle: IS_ORACLE(s),
    })).filter(p => isNum(p.x) && isNum(p.y)));

    const mark = ok => ok ? `<span class="status-ok">✓ met</span>` : `<span class="status-bad">✗ not met</span>`;
    let th = `<thead><tr><th>Scheduler</th><th>CS ≥ 20 % fewer</th><th>P95 within 10 %</th><th>max WT ≤ RR q_max</th></tr></thead><tbody>`;
    for (const t of d.thresholds || []) {
      th += `<tr><td>${esc(LABEL(t.scheduler))}</td><td>${mark(t.cs_met)} <span class="muted">${fmt(t.cs_ratio)}×</span></td>` +
        `<td>${mark(t.p95_met)} <span class="muted">${fmt(t.p95_ratio)}×</span></td>` +
        `<td>${t.maxwt_ratio_qmax !== undefined ? mark(t.maxwt_met) + ` <span class="muted">${fmt(t.maxwt_ratio_qmax)}×</span>` : '<span class="muted">n/a</span>'}</td></tr>`;
    }
    card.querySelector(".thr").innerHTML = th + "</tbody>";

    let wl = `<thead><tr><th>Scheduler</th><th>metric</th><th>change</th><th>p</th><th>n</th></tr></thead><tbody>`;
    const tests = d.tests || [];
    if (tests.length) {
      wl += `<tr><td colspan="5" class="muted" style="text-align:left">baseline: ${esc(LABEL(tests[0].baseline))}</td></tr>`;
    }
    for (const t of tests) {
      const sig = t.p < 0.05;
      wl += `<tr><td>${esc(LABEL(t.scheduler))}</td><td>${t.metric}</td><td>${t.change_pct > 0 ? "+" : ""}${fmt(t.change_pct, 1)} %</td>` +
        `<td style="${sig ? "font-weight:700" : ""}">${t.p < 0.001 ? "<0.001" : fmt(t.p, 3)}${sig ? " *" : ""}</td><td>${t.n}</td></tr>`;
    }
    card.querySelector(".wil").innerHTML = wl + "</tbody>";
  }
  if (!Object.keys(groups).length) host.innerHTML = `<div class="card empty">No results in this summary.</div>`;
}

function scatter(host, pts) {
  const W = Math.max(300, host.clientWidth || 460), H = 300;
  const m = { l: 44, r: 12, t: 10, b: 34 };
  const lx = pts.map(p => Math.log(p.x)), ly = pts.map(p => Math.log(p.y));
  const pad = 0.12;
  const x0 = Math.min(...lx, 0) - pad, x1 = Math.max(...lx, 0) + pad;
  const y0 = Math.min(...ly, 0) - pad, y1 = Math.max(...ly, 0) + pad;
  const X = v => m.l + (Math.log(v) - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => m.t + (1 - (Math.log(v) - y0) / (y1 - y0)) * (H - m.t - m.b);
  const s = svg("svg", { class: "bars", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  const tickVals = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12];
  for (const v of tickVals) {
    const lv = Math.log(v);
    if (lv > x0 && lv < x1) {
      svg("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: "var(--grid)" }, s);
      svg("text", { x: X(v), y: H - m.b + 14, "text-anchor": "middle" }, s).textContent = v + "×";
    }
    if (lv > y0 && lv < y1) {
      svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)" }, s);
      svg("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, s).textContent = v + "×";
    }
  }
  svg("line", { x1: X(1), x2: X(1), y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", "stroke-dasharray": "4 4" }, s);
  svg("line", { x1: m.l, x2: W - m.r, y1: Y(1), y2: Y(1), stroke: "var(--ink-3)", "stroke-dasharray": "4 4" }, s);
  svg("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }, s).textContent = "context switches (ratio, log) →";
  svg("text", { x: m.l + 4, y: m.t + 12 }, s).textContent = "↑ mean WT (ratio, log)";
  svg("text", { x: X(1) + 4, y: Y(1) - 5, class: "val small" }, s).textContent = "best RR";
  // greedy direct labels: right of the point, else left, else hover only
  const placed = pts.map(p => ({ x0: X(p.x) - 6, x1: X(p.x) + 6, y0: Y(p.y) - 6, y1: Y(p.y) + 6 }));
  placed.push({ x0: X(1) + 2, x1: X(1) + 52, y0: Y(1) - 16, y1: Y(1) - 2 });   // "best RR" note
  const overlaps = b => placed.some(o => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  const order = pts.map((p, i) => i).sort((a, b) => (pts[b].barr - pts[a].barr) || (pts[a].oracle - pts[b].oracle));
  const labelAt = {};
  for (const i of order) {
    const p = pts[i], w = p.label.length * 6.2, cx = X(p.x), cy = Y(p.y);
    for (const side of [1, -1]) {
      const b = side > 0 ? { x0: cx + 8, x1: cx + 8 + w, y0: cy - 7, y1: cy + 6 }
        : { x0: cx - 8 - w, x1: cx - 8, y0: cy - 7, y1: cy + 6 };
      if (b.x0 < m.l || b.x1 > W - m.r || overlaps(b)) continue;
      placed.push(b);
      labelAt[i] = side;
      break;
    }
  }
  pts.forEach((p, i) => {
    const col = p.barr ? "var(--s2)" : "var(--s1)";
    const c = svg("circle", { cx: X(p.x), cy: Y(p.y), r: 5, fill: p.oracle ? "var(--surface-1)" : col,
      stroke: p.oracle ? col : "var(--surface-1)", "stroke-width": 2 }, s);
    if (labelAt[i]) {
      const lab = svg("text", { x: X(p.x) + 8 * labelAt[i], y: Y(p.y) + 4,
        "text-anchor": labelAt[i] > 0 ? "start" : "end" }, s);
      lab.textContent = p.label;
    }
    const hit = svg("circle", { cx: X(p.x), cy: Y(p.y), r: 11, fill: "transparent" }, s);
    hit.addEventListener("mousemove", ev => showTip(`<b>${esc(p.label)}</b>${p.oracle ? ' <span class="muted">oracle</span>' : ""}<br>` +
      `switches ${p.x.toFixed(2)}× best RR<br>mean WT ${p.y.toFixed(2)}× best RR`, ev));
    hit.addEventListener("mouseleave", hideTip);
    void c;
  });
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.style.marginBottom = "6px";
  legend.innerHTML = `<span><i style="background:var(--s1);border-radius:50%;width:10px;height:10px"></i>baselines</span>` +
    `<span><i style="background:var(--s2);border-radius:50%;width:10px;height:10px"></i>BARR variants</span>` +
    `<span><i style="border:2px solid var(--ink-2);border-radius:50%;width:10px;height:10px;background:none"></i>oracle (not achievable online)</span>` +
    `<span class="muted">lower-left is better</span>`;
  host.appendChild(legend);
  host.appendChild(s);
}

/* ================================================================ tests */
function initTests() {
  $("#x-run").addEventListener("click", async () => {
    const log = $("#x-log");
    log.innerHTML = "";
    $("#x-run").disabled = true;
    $("#x-summary").innerHTML = `<span class="chip"><span class="dot busy"></span>running</span>`;
    const { id } = await api("/api/jobs", { kind: "tests" });
    stream(id, ev => {
      if (ev.type === "log") {
        const line = document.createElement("div");
        line.textContent = ev.msg;
        if (/PASSED|\d+ passed/.test(ev.msg) && !/failed/.test(ev.msg)) line.className = "pass";
        if (/FAILED|ERROR|\d+ failed/.test(ev.msg)) line.className = "fail";
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;
        const m = ev.msg.match(/=+ (.*(passed|failed).*) =+/);
        if (m) $("#x-summary").innerHTML = `<span class="chip"><span class="dot ${/failed|error/.test(m[1]) ? "bad" : "good"}"></span>${esc(m[1])}</span>`;
      }
      if (ev.type === "done" || ev.type === "error") $("#x-run").disabled = false;
    });
  });
}

/* ================================================================ boot */
let resizeTimer = null;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (SIM && $("#tab-sim").classList.contains("active")) { layoutLanes(); renderCompare(); }
    if (META && META.report && $("#tab-pred").classList.contains("active")) renderReport(META.report);
  }, 150);
});

(async function boot() {
  try {
    await loadMeta();
  } catch (e) {
    $("#lanes").innerHTML = `<div class="card empty">Cannot reach the dashboard server: ${esc(e.message)}</div>`;
    return;
  }
  initSimForm();
  initPredictors();
  initExperiments();
  initTests();
  const tab = location.hash.slice(1) || store.get("barr-tab");
  if (tab && $("#tab-" + tab)) selectTab(tab);
  $$(".tabs button").forEach(b => b.addEventListener("click", () => {
    if (b.dataset.tab === "pred" && META.report) setTimeout(() => renderReport(META.report), 0);
  }));
  runSim();
})();
