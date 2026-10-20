/* app.js — shell, state, shared visual components. Chapters live in chapters.js. */
"use strict";
const D = window.__DATA__;
const MICRO = D.micro, TRAINED = D.trained, CFG = MICRO.config;
const MV = MICRO.vocab;                     // micro vocabulary (id -> char)
const MSTOI = Object.fromEntries(MV.map((c, i) => [c, i]));
const LS_KEY = "transformer-atlas-v1";

/* ---------------- state ---------------- */
const DEFAULTS = () => ({
  ch: "map", dims: "micro", view: "all", op: null, codeFile: null, nav: false, pane: "visual",
  text: MICRO.excerpt.slice(0, 9), causal: true, head: 0, qi: 7, kj: 6, pos: 7, stage8: 5, speed: 1,
  tokText: "First Citizen:\nBefore we proceed, hear me speak.", tokStrict: true, bpeMerges: 40, tokHover: null,
  prepLower: false, prepPunct: false, prepStop: false, prepNl: false, splitWrong: false,
  winT: 8, winStart: 0, winB: 2, winMode: "seq", winStride: 8, winSeed: 1,
  ln: "ln1", qkv: "q", qkvCell: [7, 1], ffnPos: 7, lossPos: 7, lossView: "logits",
  step11: 6, paramSel: "blocks.0.attn.q_proj.weight", paramIdx: [1, 2], noZero: false, accumScaled: true,
  sysNode: "attn", trL: 0, trH: 0, trQ: 47, curveView: "loss", sampleIdx: 0,
  genPrompt: "First", genTemp: 0, genSeed: 3, genOut: null, genStep: 0,
  overrides: {}, insEl: null, inspect: null, codeW: null,
  mem: { preset: "train", V: 66, Tmax: 128, T: 128, L: 4, H: 4, D: 128, ff: 4, B: 32, tie: false, pb: 4, ab: 4, ob: 4,
    master: false, accum: 1, ckpt: false, attn: "explicit", drop: true, overhead: 10, gpu: 16 },
  progress: { visited: {}, quiz: {} },
});
let S = DEFAULTS();
function load() {
  try {
    const s = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    if (s) for (const k of ["ch", "dims", "progress", "codeW", "text"]) if (s[k] !== undefined) S[k] = s[k];
  } catch (e) { /* storage unavailable: run without it */ }
  if (location.hash && location.hash.length > 1) S.ch = location.hash.slice(1);
}
function save() {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ ch: S.ch, dims: S.dims, progress: S.progress, codeW: S.codeW, text: S.text })); } catch (e) { /* ignore */ }
}

/* ---------------- micro computation (real, in browser) ---------------- */
const BASEP = Eng.deepNum(MICRO.params);
const M = {};
function currentParams() {
  const P = {};
  for (const k in BASEP) P[k] = Eng.clone(BASEP[k]);
  for (const k in S.overrides) for (const key in S.overrides[k]) {
    const idx = key.split(",").map(Number);
    if (idx.length === 1) P[k][idx[0]] = S.overrides[k][key]; else P[k][idx[0]][idx[1]] = S.overrides[k][key];
  }
  return P;
}
function textIds(t) { return Array.from(t).map((c) => MSTOI[c]); }
function validText(t) { return Array.from(t).length === 9 && Array.from(t).every((c) => c in MSTOI && c !== "<unk>"); }
function compute() {
  const ch = Array.from(S.text);
  M.P = currentParams();
  M.ids = textIds(ch.slice(0, 8).join(""));
  M.tg = textIds(ch.slice(1, 9).join(""));
  M.fw = Eng.forward(M.P, CFG, M.ids, M.tg, { causal: S.causal });
  M.fwMasked = S.causal ? M.fw : Eng.forward(M.P, CFG, M.ids, M.tg, { causal: true });
  M.bw = Eng.backward(M.P, CFG, M.fw);
  M.clip = Eng.clip(M.bw.G, MICRO.optimizer.grad_clip);
  M.adam = Eng.adamw(M.P, M.clip.G, MICRO.adam_before, MICRO.optimizer);
  M.canonical = S.text === MICRO.excerpt.slice(0, 9) && !Object.keys(S.overrides).length && S.causal;
}
function verifyMicro() {
  const P = BASEP, f = Eng.forward(P, CFG, MICRO.input_ids, MICRO.target_ids);
  const b = Eng.backward(P, CFG, f), tr = MICRO.trace, rows = [];
  const add = (what, js, rec) => rows.push({ what, diff: Eng.maxAbsDiff(js, rec) });
  add("x0 = tok_emb + pos_emb", f.x0, tr.x0);
  add("LayerNorm 1", f.ln1.y, tr.ln1);
  add("attention scores (raw)", f.S, tr.scores_raw);
  add("attention probabilities", f.P, tr.probs);
  add("merged head outputs", f.merged, tr.merged);
  add("h1 (after attention residual)", f.h1, tr.h1);
  add("FFN GELU activations", f.fact, tr.ffn_act);
  add("h2 (block output)", f.h2, tr.h2);
  add("logits", f.logits, tr.logits);
  rows.push({ what: "loss", diff: Math.abs(f.loss - tr.loss) });
  let gmax = 0, gworst = "";
  for (const k in b.G) { const d = Eng.maxAbsDiff(b.G[k], MICRO.grads[k]); if (d > gmax) { gmax = d; gworst = k; } }
  rows.push({ what: "all parameter gradients (max over " + Object.keys(b.G).length + " tensors)", diff: gmax, note: gworst });
  rows.push({ what: "total gradient norm", diff: Math.abs(Eng.totalNorm(b.G) - MICRO.grad_norm_total) });
  const a = Eng.adamw(P, Eng.deepNum(MICRO.grads_clipped), MICRO.adam_before, MICRO.optimizer);
  let amax = 0;
  for (const k in P) amax = Math.max(amax, Eng.maxAbsDiff(a.after[k], MICRO.params_after[k]));
  rows.push({ what: "AdamW update (from PyTorch's clipped gradients)", diff: amax });
  const u = Eng.forward(P, CFG, MICRO.input_ids, MICRO.target_ids, { causal: false });
  rows.push({ what: "unmasked forward loss (leakage demo)", diff: Math.abs(u.loss - MICRO.unmasked.loss) });
  M.verify = rows;
  M.verifyMax = Math.max(...rows.map((r) => r.diff));
  return rows;
}

/* ---------------- formatting & small components ---------------- */
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
function fmt(v, d = 3) {
  if (v === undefined || v === null) return "";
  if (typeof v === "string") v = Eng.num(v);
  if (v === -Infinity) return "−∞";
  if (v === Infinity) return "∞";
  if (Number.isNaN(v)) return "NaN";
  const a = Math.abs(v);
  let s = a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : v.toFixed(d);
  return s.replace("-", "−");
}
const fmtInt = (n) => Number(n).toLocaleString("en-US");
function fmtBytes(b) {
  if (b === null || b === undefined) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"]; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return (i ? b.toFixed(b < 10 ? 2 : 1) : b) + " " + u[i];
}
const vis = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c === "<unk>" ? "∅" : c === "\t" ? "⇥" : c);
const visStr = (s) => Array.from(s).map((c) => (c === "\n" ? "↵" : c === " " ? "·" : c)).join("");
const KINDS = {
  in: ["k-in", "▸", "input"], par: ["k-par", "θ", "parameter"], act: ["k-act", "a", "activation"],
  grad: ["k-grad", "∇", "gradient"], tgt: ["k-tgt", "y", "target / loss"],
};
const kindTag = (k) => `<span class="kind ${KINDS[k][0]}"><b>${KINDS[k][1]}</b>${KINDS[k][2]}</span>`;
const PROV = {
  browser: ["prov-browser", "⚙", "Computed in your browser"],
  recorded: ["prov-recorded", "●", "Recorded from PyTorch"],
  analytic: ["prov-analytic", "≈", "Analytical estimate"],
  illus: ["prov-illus", "◇", "Illustrative"],
};
const prov = (p, extra = "") => `<span class="prov ${PROV[p][0]}" title="${esc(PROV[p][2])}">${PROV[p][1]} ${PROV[p][2]}${extra}</span>`;

const DIMS = {
  micro: { B: 1, T: 8, T_max: 8, V: MV.length, D: 8, H: 2, Dh: 4, L: 1, Dff: 32 },
  train: { B: 32, T: 128, T_max: 128, V: TRAINED.config.vocab_size, D: 128, H: 4, Dh: 32, L: 4, Dff: 512 },
};
// shape(["B","T","D"]) -> symbolic or concrete according to the dims toggle
function shape(dims) {
  if (S.dims === "sym") return `<span class="shape">[${dims.join(", ")}]</span>`;
  const t = DIMS[S.dims];
  return `<span class="shape" title="[${dims.join(", ")}]">[${dims.map((d) => (typeof d === "number" ? d : d.split("*").reduce((a, x) => a * (t[x] ?? NaN), 1))).join(", ")}]</span>`;
}

/* registries rebuilt every render */
let HEAT = {}, INS = {}, CHARTS = {}, ACT = {};
let uid = 0;
const nid = (p = "x") => p + ++uid;

function colorFor(kind, t) {
  const base = { in: [12, 138, 123], par: [109, 70, 224], act: [27, 111, 192], grad: [194, 37, 92], tgt: [217, 83, 11], neg: [89, 98, 122] };
  const c = t >= 0 ? base[kind] : base.neg, a = Math.min(1, Math.abs(t)) * 0.82;
  return { bg: `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`, fg: a > 0.5 ? "#fff" : "#12141a" };
}
/* heatmap: M = 2-D array */
function heat(Mx, o = {}) {
  const id = o.id || nid("h");
  const flatv = Mx.flat().map(Eng.num).filter((v) => Number.isFinite(v));
  const mx = o.max || Math.max(1e-12, ...flatv.map(Math.abs));
  const cls = o.size === "compact" ? "hm compact" : o.size === "mid" ? "hm mid" : "hm";
  if (o.onCell) HEAT[id] = o.onCell;
  if (o.inspect) INS[id] = o.inspect;
  const cols = Mx[0].length;
  let h = `<div class="hm-wrap"><table class="${cls}" data-heat="${id}" ${o.op ? `data-op="${o.op}"` : ""}>`;
  if (o.colLabels) h += `<tr><th></th>${o.colLabels.map((c, j) => `<th class="${o.colSel === j ? "colsel" : ""}">${esc(c)}</th>`).join("")}</tr>`;
  Mx.forEach((row, r) => {
    h += `<tr class="${o.rowSel === r ? "rowsel" : ""}">`;
    if (o.rowLabels) h += `<th class="rl">${esc(o.rowLabels[r])}</th>`;
    row.forEach((v0, c) => {
      const v = Eng.num(v0);
      const masked = o.masked && o.masked(r, c);
      const col = Number.isFinite(v) ? colorFor(o.kind || "act", v / mx) : { bg: "transparent", fg: "#6a7080" };
      const sel = o.sel && o.sel[0] === r && o.sel[1] === c ? " sel" : "";
      const hl = o.hl && o.hl(r, c) ? " hl" : "";
      const dim = o.dim && o.dim(r, c) ? " dim" : "";
      const txt = o.size === "compact" ? "" : o.values === false ? "" : fmt(v, o.digits ?? 2);
      h += `<td class="${masked ? "masked" : ""}${sel}${hl}${dim}" data-r="${r}" data-c="${c}" style="background:${col.bg};color:${col.fg}" title="[${r}, ${c}] = ${fmt(v, 5)}">${masked && !Number.isFinite(v) ? "−∞" : txt}</td>`;
    });
    h += "</tr>";
  });
  h += "</table></div>";
  return h;
}
function tensor(name, dims, kind, inner, o = {}) {
  const insKey = o.inspectKey;
  return `<div class="tensor" ${o.op ? `data-op="${o.op}"` : ""}>
    <div class="tt"><span class="nm">${name}</span>${shape(dims)}${kindTag(kind)}${insKey ? `<button class="btn small ghost" data-act="inspect" data-arg="${insKey}" title="Open in inspector">Inspect</button>` : ""}</div>${inner}</div>`;
}
function bars(values, o = {}) {
  const w = o.w || 560, h = o.h || 140, n = values.length, pad = { l: 34, r: 6, t: 8, b: o.labels ? 26 : 8 };
  const vals = values.map(Eng.num);
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (o.min !== undefined) lo = o.min;
  if (o.max !== undefined) hi = o.max;
  if (hi === lo) hi = lo + 1;
  const bw = (w - pad.l - pad.r) / n, y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / (hi - lo));
  const id = o.id || nid("b");
  if (o.onBar) HEAT[id] = (r, c) => o.onBar(c);
  let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.aria || "bar chart")}" data-bars="${id}">`;
  const ticks = niceTicks(lo, hi, 4);
  ticks.forEach((t) => { s += `<line class="grid" x1="${pad.l}" x2="${w - pad.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${pad.l - 4}" y="${y(t) + 3}" text-anchor="end">${fmt(t, Math.abs(t) < 1 ? 2 : 1)}</text>`; });
  s += `<line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${y(0)}" y2="${y(0)}"/>`;
  vals.forEach((v, i) => {
    const isHi = o.hi && o.hi(i), kind = isHi ? (o.hiKind || "tgt") : (o.kind || "act");
    const col = getComputedStyle(document.documentElement).getPropertyValue(`--k-${{ in: "in", par: "par", act: "act", grad: "grad", tgt: "tgt" }[kind]}`) || "#1b6fc0";
    const y0 = y(Math.max(0, v)), y1 = y(Math.min(0, v));
    s += `<rect x="${pad.l + i * bw + 1}" y="${y0}" width="${Math.max(1, bw - 2)}" height="${Math.max(0.5, y1 - y0)}" fill="${col.trim()}" opacity="${isHi ? 1 : o.dimOthers ? 0.45 : 0.85}" data-r="0" data-c="${i}" style="cursor:${o.onBar ? "pointer" : "default"}"><title>${esc(o.labels ? o.labels[i] : i)}: ${fmt(v, 4)}</title></rect>`;
    if (o.labels && (n <= 40)) s += `<text x="${pad.l + i * bw + bw / 2}" y="${h - pad.b + 12}" text-anchor="middle" style="font-family:var(--f-mono);font-size:${n > 24 ? 8.5 : 10}px;${isHi ? "fill:#12141a;font-weight:700" : ""}">${esc(o.labels[i])}</text>`;
  });
  return s + "</svg>";
}
function niceTicks(lo, hi, n) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0)), err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
/* line chart with hover readout */
function lineChart(series, o = {}) {
  const id = o.id || nid("c"), w = o.w || 640, h = o.h || 240, pad = { l: 46, r: 14, t: 10, b: 32 };
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).filter(Number.isFinite);
  const x0 = o.xMin ?? Math.min(...xs), x1 = o.xMax ?? Math.max(...xs);
  let y0 = o.yMin ?? Math.min(...ys), y1 = o.yMax ?? Math.max(...ys);
  if (y1 === y0) y1 = y0 + 1;
  const X = (v) => pad.l + (w - pad.l - pad.r) * (v - x0) / (x1 - x0 || 1);
  const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - y0) / (y1 - y0));
  CHARTS[id] = { series, X, Y, x0, x1, w, pad, fmtX: o.fmtX || ((v) => fmtInt(Math.round(v))), fmtY: o.fmtY || ((v) => fmt(v, 3)) };
  let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" data-chart="${id}" role="img" aria-label="${esc(o.aria || "line chart")}">`;
  niceTicks(y0, y1, 4).forEach((t) => { s += `<line class="grid" x1="${pad.l}" x2="${w - pad.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${pad.l - 6}" y="${Y(t) + 3}" text-anchor="end">${o.fmtTickY ? o.fmtTickY(t) : fmt(t, t < 10 ? 2 : 0)}</text>`; });
  niceTicks(x0, x1, 6).forEach((t) => { s += `<text x="${X(t)}" y="${h - pad.b + 14}" text-anchor="middle">${o.fmtTickX ? o.fmtTickX(t) : fmtInt(t)}</text>`; });
  s += `<line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${h - pad.b}" y2="${h - pad.b}"/><line class="axis" x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${h - pad.b}"/>`;
  if (o.xLabel) s += `<text x="${(pad.l + w - pad.r) / 2}" y="${h - 4}" text-anchor="middle">${esc(o.xLabel)}</text>`;
  if (o.yLabel) s += `<text x="12" y="${pad.t + (h - pad.t - pad.b) / 2}" text-anchor="middle" transform="rotate(-90 12 ${pad.t + (h - pad.t - pad.b) / 2})">${esc(o.yLabel)}</text>`;
  series.forEach((sr) => {
    const d = sr.points.filter((p) => Number.isFinite(p[1])).map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    s += `<path d="${d}" fill="none" stroke="${sr.color}" stroke-width="${sr.width || 1.8}" ${sr.dash ? `stroke-dasharray="${sr.dash}"` : ""} opacity="${sr.opacity || 1}"/>`;
    if (sr.dots) sr.points.forEach((p) => { s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.6" fill="${sr.color}"/>`; });
  });
  s += `<line class="hover-line" x1="0" x2="0" y1="${pad.t}" y2="${h - pad.b}" stroke="#12141a" stroke-width="1" opacity="0" stroke-dasharray="3 3"/>`;
  s += `</svg><div class="chart-tip" id="tip-${id}">${o.tip || "Hover the chart to read values."}</div>`;
  if (o.legend !== false) s = `<div class="legend" style="margin-bottom:6px">${series.map((sr) => `<span><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" stroke="${sr.color}" stroke-width="2.5" ${sr.dash ? `stroke-dasharray="${sr.dash}"` : ""}/></svg> ${esc(sr.name)}</span>`).join("")}</div>` + s;
  return s;
}

/* ---------------- quiz ---------------- */
function quiz(q) {
  const st = S.progress.quiz[q.id];
  let h = `<div class="quiz" id="quiz-${q.id}"><div class="q">${q.q}</div>`;
  if (q.options) {
    h += `<div class="opts">${q.options.map((o, i) => {
      const cls = st !== undefined ? (i === q.answer ? "right" : i === st ? "wrong" : "") : "";
      return `<button class="${cls}" data-act="quiz" data-arg="${q.id}|${i}">${o}</button>`;
    }).join("")}</div>`;
    if (st !== undefined) h += `<div class="ans">${st === q.answer ? '<span class="pill ok">Correct</span>' : '<span class="pill bad">Not quite</span>'} ${q.explain}</div>`;
  } else {
    h += `<textarea rows="2" id="pred-${q.id}" placeholder="Write your prediction first (stays on this page only)"></textarea><div style="margin-top:6px"><button class="btn small" data-act="quiz" data-arg="${q.id}|reveal">Reveal answer</button></div>`;
    if (st !== undefined) h += `<div class="ans">${q.explain}</div>`;
  }
  return h + "</div>";
}

/* ---------------- shell rendering ---------------- */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
let OPS = {};          // resolved op -> {file, start, end, label}
let OP_ERRORS = [];

function resolveOps() {
  OPS = {}; OP_ERRORS = [];
  for (const o of OPS_DEF) {
    const src = D.sources[o.file];
    if (!src) { OP_ERRORS.push(o.id + ": missing file " + o.file); continue; }
    const lines = src.split("\n");
    let from = 0;
    if (o.after) { from = lines.findIndex((l) => l.includes(o.after)); if (from < 0) { OP_ERRORS.push(o.id + ": anchor"); continue; } }
    const i = lines.findIndex((l, k) => k >= from && l.includes(o.find));
    if (i < 0) { OP_ERRORS.push(o.id + ": not found: " + o.find); continue; }
    OPS[o.id] = { ...o, start: i + 1, end: i + (o.n || 1) };
  }
  window.__opErrors = OP_ERRORS;
}
const chapter = () => CHAPTERS.find((c) => c.id === S.ch) || CHAPTERS[0];

function renderNav() {
  const groups = {};
  CHAPTERS.forEach((c) => (groups[c.group] = groups[c.group] || []).push(c));
  const nVisited = CHAPTERS.filter((c) => S.progress.visited[c.id]).length;
  let h = `<div class="progress"><div class="bar"><i style="width:${(100 * nVisited / CHAPTERS.length).toFixed(0)}%"></i></div><span>${nVisited} of ${CHAPTERS.length} sections visited · ${Object.keys(S.progress.quiz).length} ${Object.keys(S.progress.quiz).length === 1 ? "check" : "checks"} answered</span></div>`;
  for (const g in groups) {
    h += `<h6>${g}</h6>`;
    groups[g].forEach((c) => {
      h += `<a href="#${c.id}" data-act="go" data-arg="${c.id}" ${c.id === S.ch ? 'aria-current="page"' : ""}><span class="n">${c.num || "·"}</span><span>${c.short || c.title}</span><i class="done ${S.progress.visited[c.id] ? "v" : ""}" aria-label="${S.progress.visited[c.id] ? "visited" : "not visited"}"></i></a>`;
    });
  }
  $("#nav").innerHTML = h;
  $("#nav").classList.toggle("open", S.nav);
}
function renderCode() {
  const c = chapter();
  const files = c.files && c.files.length ? c.files : ["model.py"];
  if (!S.codeFile || !Object.keys(D.sources).includes(S.codeFile)) S.codeFile = files[0];
  const op = S.op && OPS[S.op];
  const chOps = (c.ops || []).map((id) => OPS[id]).filter(Boolean);
  const allFiles = [...new Set([...files, S.codeFile])];
  $("#code-tabs").innerHTML = allFiles.map((f) => `<button class="tab" role="tab" aria-selected="${f === S.codeFile}" data-act="file" data-arg="${f}">${f}${chOps.some((o) => o.file === f) ? '<span class="dot" aria-hidden="true"></span>' : ""}</button>`).join("") +
    `<span style="flex:1"></span><select id="all-files" aria-label="Open any source file">${Object.keys(D.sources).map((f) => `<option ${f === S.codeFile ? "selected" : ""}>${f}</option>`).join("")}</select>`;
  $("#code-tools").innerHTML = `<span class="grow">${op ? `<b style="color:var(--ink)">${esc(op.label)}</b> · ${op.file}:${op.start}${op.end > op.start ? "–" + op.end : ""}` : "Click a highlighted line, or a tensor on the right, to link code and computation."}</span>
    ${chOps.length ? `<select id="op-select" aria-label="Operations in this chapter"><option value="">Operations in this chapter…</option>${chOps.map((o) => `<option value="${o.id}" ${o.id === S.op ? "selected" : ""}>${esc(o.label)}</option>`).join("")}</select>` : ""}
    <button class="btn small" data-act="copy-op" ${op ? "" : "disabled"}>Copy snippet</button><button class="btn small" data-act="copy-file">Copy file</button>`;
  const lines = PyHL.highlight(D.sources[S.codeFile]);
  const ranges = Object.values(OPS).filter((o) => o.file === S.codeFile && (c.ops || []).includes(o.id));
  $("#code").innerHTML = lines.map((l, i) => {
    const n = i + 1, inOp = ranges.find((o) => n >= o.start && n <= o.end);
    const sel = op && op.file === S.codeFile && n >= op.start && n <= op.end;
    return `<div class="ln${inOp ? " op" : ""}${sel ? " sel" : ""}" ${inOp ? `data-act="code-op" data-arg="${inOp.id}" title="${esc(inOp.label)}"` : ""} id="L${n}"><span class="no">${n}</span><span class="tx">${l || " "}</span></div>`;
  }).join("");
}
function scrollCodeToOp() {
  const op = S.op && OPS[S.op];
  if (!op || op.file !== S.codeFile) return;
  const el = $("#L" + op.start);
  if (el) el.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}
const STRIP = [
  ["map", "System map"], ["text", "Corpus & split"], ["tok", "Tokenizer"], ["win", "Windows"], ["emb", "Embeddings"],
  ["block", "Block"], ["qkv", "Q K V"], ["attn", "Attention"], ["ffn", "Heads + FFN"], ["loss", "Logits & loss"],
  ["backprop", "Backprop"], ["train", "Training"], ["memory", "Memory"], ["ckpt", "Checkpoint"], ["infer", "Inference"],
];
const KIND_COLOR = { data: "var(--k-in)", forward: "var(--k-act)", train: "var(--k-grad)", infer: "var(--k-tgt)", ref: "var(--muted)" };
function renderContent(keepScroll) {
  HEAT = {}; INS = {}; CHARTS = {}; uid = 0;
  const c = chapter(), sc = $("#scroll"), top = sc.scrollTop;
  const focusId = document.activeElement && document.activeElement.id;
  let h = `<div class="page">`;
  if (c.num) h += `<nav class="strip" aria-label="Pipeline position">${STRIP.map(([id, l]) => { const cc = CHAPTERS.find((x) => x.id === id); return `<button class="${id === c.id ? "cur" : ""}" style="--sc:${KIND_COLOR[cc.kind]}" data-act="go" data-arg="${id}">${l}</button>`; }).join("")}</nav>`;
  h += `<div class="eyebrow">${c.num ? `<span>Chapter ${c.num}</span>` : `<span>${c.group}</span>`}<span class="kind-pill kp-${c.kind}">${{ data: "Dataset transformation", forward: "Model forward computation", train: "Training-only operation", infer: "Inference & persistence", ref: "Reference" }[c.kind]}</span></div>`;
  h += `<h1 class="ch-title">${c.title}</h1><p class="thesis">${c.thesis}</p>`;
  if (c.card) {
    const labels = [["problem", "Problem it solves"], ["ml", "Relation to familiar ML"], ["op", "Exact operation"], ["shapes", "Input → output shapes"], ["params", "Trainable parameters"], ["changes", "Changes during training"], ["fixed", "Stays fixed"], ["failures", "Failure modes"], ["verify", "How to verify"]];
    h += `<details class="stage-wrap" open><summary>Stage summary</summary><dl class="stage">${labels.filter(([k]) => c.card[k]).map(([k, l]) => `<div><dt>${l}</dt><dd>${typeof c.card[k] === "function" ? c.card[k]() : c.card[k]}</dd></div>`).join("")}</dl></details>`;
  }
  const secs = c.sections();
  const tags = [...new Set(secs.map((s) => s.tag).filter(Boolean))];
  if (tags.length > 1) {
    h += `<div class="views" role="toolbar" aria-label="Views"><span class="small muted">View</span><div class="seg">${["all", ...["pipeline", "tensor", "numerical", "math", "engineering"].filter((t) => tags.includes(t))].map((t) => `<button aria-pressed="${S.view === t || (t === "all" && !tags.includes(S.view))}" data-act="view" data-arg="${t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}</div>
      <span class="legend hide-sm" style="margin-left:auto">${Object.keys(KINDS).map(kindTag).join("")}</span></div>`;
  }
  const activeView = tags.includes(S.view) ? S.view : "all";
  RENDERED = [];
  secs.forEach((s, i) => {
    if (activeView !== "all" && s.tag && s.tag !== activeView) return;
    RENDERED.push(i);
    h += `<section class="card" id="sec-${i}" aria-labelledby="sh-${i}"><header><h2 id="sh-${i}">${s.title}</h2>${s.tag ? `<span class="tag">${s.tag}</span>` : ""}<span style="flex:1"></span>${s.prov ? (Array.isArray(s.prov) ? s.prov : [s.prov]).map((p) => prov(p)).join(" ") : ""}</header>${s.ctrl ? `<div class="ctrl">${s.ctrl()}</div>` : ""}<div class="body">${s.body()}</div></section>`;
  });
  if (c.quiz && c.quiz.length) h += `<section class="card"><header><h2>Knowledge check</h2><span class="tag">predict, then reveal</span></header><div class="body">${c.quiz.map(quiz).join("")}</div></section>`;
  const idx = CHAPTERS.indexOf(c);
  h += `<div style="display:flex;justify-content:space-between;gap:10px;margin-top:20px;flex-wrap:wrap">${idx > 0 ? `<button class="btn" data-act="go" data-arg="${CHAPTERS[idx - 1].id}">← ${CHAPTERS[idx - 1].short || CHAPTERS[idx - 1].title}</button>` : "<span></span>"}${idx < CHAPTERS.length - 1 ? `<button class="btn primary" data-act="go" data-arg="${CHAPTERS[idx + 1].id}">${CHAPTERS[idx + 1].short || CHAPTERS[idx + 1].title} →</button>` : ""}</div>`;
  h += `</div>`;
  $("#scroll").innerHTML = h;
  if (c.after) c.after();
  if (keepScroll) sc.scrollTop = top;
  if (focusId) { const f = document.getElementById(focusId); if (f) f.focus({ preventScroll: true }); }
  markOp();
}
let RENDERED = [];
function renderBodies() {
  HEAT = {}; INS = {}; CHARTS = {}; uid = 0;
  const secs = chapter().sections();
  RENDERED.forEach((i) => { const el = document.querySelector(`#sec-${i} > .body`); if (el) el.innerHTML = secs[i].body(); });
  if (chapter().after) chapter().after();
  markOp();
}
function markOp() {
  $$(".op-focus").forEach((e) => e.classList.remove("op-focus"));
  if (S.op) $$(`[data-op="${S.op}"]`).forEach((e) => e.classList.add("op-focus"));
}
function renderInspector() {
  const el = $("#ins-body"), o = S.inspect && INS[S.inspect] ? INS[S.inspect] : S.inspect && LAST_INS[S.inspect];
  if (o) LAST_INS[S.inspect] = o;
  if (!o) { el.innerHTML = `<div class="ins-empty">Select a tensor (the <b>Inspect</b> buttons, or any heatmap cell) to see its shape, dtype, provenance, statistics and values. Parameters also show their gradient and AdamW update, and can be edited to see downstream effects.</div>`; $("#ins-title").textContent = "Inspector"; return; }
  $("#ins-title").textContent = "Inspector · " + o.name;
  const data = typeof o.data === "function" ? o.data() : o.data;
  const fl = Eng.flat(data).map(Eng.num).filter(Number.isFinite);
  const mean = fl.reduce((a, v) => a + v, 0) / (fl.length || 1);
  const std = Math.sqrt(fl.reduce((a, v) => a + (v - mean) ** 2, 0) / (fl.length || 1));
  const is2d = Array.isArray(data[0]);
  const grid = is2d ? data : [data];
  let h = `<div class="ins-grid"><div><dl class="kv"><dt>name</dt><dd>${esc(o.name)}</dd><dt>kind</dt><dd>${kindTag(o.kind)}</dd><dt>shape</dt><dd>${o.shape}</dd><dt>dtype</dt><dd>${o.dtype || "float32 in PyTorch · float64 in browser"}</dd><dt>source</dt><dd>${prov(o.prov || "browser")}</dd><dt>min / max</dt><dd>${fmt(Math.min(...fl), 4)} / ${fmt(Math.max(...fl), 4)}</dd><dt>mean ± std</dt><dd>${fmt(mean, 4)} ± ${fmt(std, 4)}</dd><dt>elements</dt><dd>${fl.length}</dd></dl>${o.note ? `<p class="small muted" style="margin-top:8px">${o.note}</p>` : ""}`;
  if (o.param && S.insEl && S.insEl[0] === o.param) {
    const idx = S.insEl.slice(1), get = (m) => (idx.length === 1 ? m[idx[0]] : m[idx[0]][idx[1]]);
    const v = get(M.P[o.param]), g = get(M.bw.G[o.param]), gc = get(M.clip.G[o.param]);
    h += `<h3>Element [${idx.join(", ")}]</h3><dl class="kv"><dt>value</dt><dd>${fmt(v, 6)}</dd><dt>∂loss/∂θ</dt><dd>${fmt(g, 6)}</dd><dt>after clipping</dt><dd>${fmt(gc, 6)}</dd><dt>after AdamW</dt><dd>${fmt(get(M.adam.after[o.param]), 6)}</dd></dl>
      <label class="small" style="display:flex;gap:6px;align-items:center;margin-top:8px">Set value <input type="number" step="0.05" id="ins-edit" value="${(+v).toFixed(4)}" style="width:110px"></label>
      <div style="display:flex;gap:6px;margin-top:6px"><button class="btn small" data-act="ins-apply">Apply & recompute</button><button class="btn small ghost" data-act="ins-revert">Revert all edits</button></div>`;
  } else if (o.param) h += `<p class="small muted">Click a cell to inspect and edit one element.</p>`;
  h += `</div><div style="min-width:0">${heat(grid, { kind: o.kind, size: grid[0].length > 16 ? "mid" : undefined, values: grid[0].length <= 16, digits: 3, rowLabels: o.rowLabels || grid.map((_, i) => i), colLabels: o.colLabels || grid[0].map((_, j) => j), id: "ins-grid", sel: S.insEl && S.insEl[0] === o.param ? (is2d ? S.insEl.slice(1) : [0, S.insEl[1]]) : null, onCell: o.param ? (r, c) => { S.insEl = is2d ? [o.param, r, c] : [o.param, c]; renderInspector(); } : null })}</div></div>`;
  el.innerHTML = h;
}
const LAST_INS = {};
function paramInspect(name, extra = {}) {
  return { name, kind: "par", param: name, shape: "[" + MICRO.shapes[name].join(", ") + "]", data: () => M.P[name], prov: "browser", note: "Micro-model parameter (snapshot recorded from PyTorch right before the inspected training step; edits are applied in the browser).", ...extra };
}
function actInspect(name, dims, data, extra = {}) {
  return { name, kind: extra.kind || "act", shape: `[${dimsConcrete(dims, data)}]`, data, prov: extra.prov || "browser", ...extra };
}
function dimsConcrete(dims, data) {
  const s = []; let d = data;
  while (Array.isArray(d)) { s.push(d.length); d = d[0]; }
  return s.join(", ") + "  (" + dims.join(", ") + ")";
}

/* ---------------- refresh ---------------- */
function refresh(opts = {}) {
  if (opts.compute !== false) compute();
  if (opts.full) renderContent(true); else renderBodies();
  renderInspector();
  if (opts.code) renderCode();
  save();
}
function go(id, push = true) {
  if (!CHAPTERS.find((c) => c.id === id)) id = "map";
  S.ch = id; S.op = null; S.codeFile = null; S.view = "all"; S.nav = false;
  S.progress.visited[id] = true;
  if (push) { try { history.replaceState(null, "", "#" + id); } catch (e) { /* sandbox */ } }
  renderNav(); renderCode(); renderContent(false); renderInspector(); save();
  $("#scroll").scrollTop = 0;
  $("#code").scrollTop = 0;
}
function setOp(id, fromCode) {
  S.op = id;
  const op = OPS[id];
  if (op) S.codeFile = op.file;
  renderCode(); scrollCodeToOp(); markOp();
  if (fromCode) { const t = $(`[data-op="${id}"]`); if (t) t.scrollIntoView({ block: "center", behavior: "smooth" }); }
  if (S.pane === "code" && !fromCode && innerWidth <= 900) { /* stay */ }
}
function toast(msg) {
  const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; t.setAttribute("role", "status");
  document.body.appendChild(t); setTimeout(() => t.remove(), 2200);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch (e) { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); toast("Copied"); } catch (e2) { toast("Select and copy manually"); } ta.remove(); }
}
async function downloadSource() {
  const files = Object.entries(D.sources).map(([name, text]) => ({ name: "transformer-atlas/" + name, text }));
  const blob = Zip.make(files);
  try {
    const dl = window.claude && window.claude.use ? await window.claude.use("downloads") : null;
    if (dl) { await dl.save({ filename: "transformer-atlas-source.zip", data: blob }); toast("Saved"); return; }
  } catch (e) { if (e && e.code === "declined") return; }
  try {
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "transformer-atlas-source.zip";
    document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    toast("Download started");
  } catch (e) { toast("Downloads are blocked here — the files are in your project folder"); }
}

/* ---------------- search palette ---------------- */
function openPalette() {
  const idx = [];
  CHAPTERS.forEach((c) => { idx.push({ t: "Chapter", label: c.title, sub: c.thesis.replace(/<[^>]+>/g, ""), go: () => go(c.id) }); });
  CHAPTERS.forEach((c) => { const saved = S.ch; S.ch = c.id; try { c.sections().forEach((s, i) => idx.push({ t: "Section", label: s.title, sub: c.title, go: () => { go(c.id); setTimeout(() => { const e = $("#sec-" + i); if (e) e.scrollIntoView(); }, 30); } })); } catch (e) { /* ignore */ } S.ch = saved; });
  GLOSSARY.forEach((g) => idx.push({ t: "Glossary", label: g[0], sub: g[1].replace(/<[^>]+>/g, ""), go: () => { go("glossary"); setTimeout(() => { const e = document.getElementById("g-" + slug(g[0])); if (e) { e.scrollIntoView({ block: "center" }); e.classList.add("op-focus"); } }, 40); } }));
  Object.values(OPS).forEach((o) => idx.push({ t: "Code", label: o.label, sub: `${o.file}:${o.start}`, go: () => { const c = CHAPTERS.find((c) => (c.ops || []).includes(o.id)); if (c) go(c.id); setOp(o.id, true); } }));
  const bg = document.createElement("div"); bg.className = "palette-bg";
  bg.innerHTML = `<div class="palette" role="dialog" aria-label="Search"><input id="pal-in" placeholder="Search chapters, operations, glossary…" autocomplete="off"><ul id="pal-list" role="listbox"></ul></div>`;
  document.body.appendChild(bg);
  let sel = 0, res = [];
  const draw = () => {
    const q = $("#pal-in").value.toLowerCase().trim();
    res = (q ? idx.filter((x) => (x.label + " " + x.sub).toLowerCase().includes(q)) : idx.filter((x) => x.t === "Chapter")).slice(0, 40);
    sel = Math.min(sel, Math.max(0, res.length - 1));
    $("#pal-list").innerHTML = res.map((r, i) => `<li class="${i === sel ? "on" : ""}" data-i="${i}" role="option"><span class="t">${r.t}</span><span>${esc(r.label)}</span><span class="s">${esc(r.sub)}</span></li>`).join("") || `<li><span class="s">No matches</span></li>`;
  };
  const close = () => bg.remove();
  bg.addEventListener("click", (e) => { if (e.target === bg) close(); const li = e.target.closest("li[data-i]"); if (li) { close(); res[+li.dataset.i].go(); } });
  $("#pal-in").addEventListener("input", () => { sel = 0; draw(); });
  $("#pal-in").addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(res.length - 1, sel + 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); }
    else if (e.key === "Enter" && res[sel]) { close(); res[sel].go(); }
    else if (e.key === "Escape") close();
  });
  draw(); $("#pal-in").focus();
}
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-");

/* ---------------- events ---------------- */
function bind() {
  document.addEventListener("click", (e) => {
    const cell = e.target.closest("[data-heat] td[data-c], [data-bars] rect[data-c]");
    if (cell) {
      const host = cell.closest("[data-heat],[data-bars]"), id = host.dataset.heat || host.dataset.bars;
      if (HEAT[id]) { HEAT[id](+cell.dataset.r, +cell.dataset.c); }
      if (INS[id]) { S.inspect = id; $("#inspector").classList.remove("collapsed"); renderInspector(); }
      const opHost = host.closest("[data-op]");
      if (opHost) setOp(opHost.dataset.op);
      return;
    }
    const a = e.target.closest("[data-act]");
    if (a) {
      const fn = ACTIONS[a.dataset.act] || ACT[a.dataset.act];
      if (fn) { e.preventDefault(); fn(a.dataset.arg, a, e); }
      return;
    }
    const opEl = e.target.closest("[data-op]");
    if (opEl && !e.target.closest("input,select,textarea")) setOp(opEl.dataset.op);
  });
  document.addEventListener("change", (e) => {
    const t = e.target;
    if (t.id === "all-files") { S.codeFile = t.value; S.op = null; renderCode(); return; }
    if (t.id === "op-select" && t.value) { setOp(t.value, true); return; }
    const fn = t.dataset.bind && BINDS[t.dataset.bind];
    if (fn) fn(t);
  });
  document.addEventListener("input", (e) => {
    const t = e.target;
    if (t.matches("input[type=range][data-bind], input[type=text][data-bind], textarea[data-bind], input[type=number][data-bind]")) {
      const fn = BINDS[t.dataset.bind]; if (fn) fn(t);
    }
  });
  document.addEventListener("mousemove", (e) => {
    const svg = e.target.closest && e.target.closest("svg[data-chart]");
    if (!svg) return;
    const c = CHARTS[svg.dataset.chart]; if (!c) return;
    const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * c.w;
    const xv = c.x0 + (x - c.pad.l) / (c.w - c.pad.l - c.pad.r) * (c.x1 - c.x0);
    const parts = c.series.map((s) => {
      let best = null; for (const p of s.points) if (!best || Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      return best ? `${s.name}: ${c.fmtY(best[1])} @ ${c.fmtX(best[0])}` : "";
    });
    const line = svg.querySelector(".hover-line");
    line.setAttribute("x1", c.X(Math.max(c.x0, Math.min(c.x1, xv)))); line.setAttribute("x2", line.getAttribute("x1")); line.setAttribute("opacity", 0.5);
    const tip = document.getElementById("tip-" + svg.dataset.chart); if (tip) tip.textContent = parts.filter(Boolean).join("   ·   ");
  });
  document.addEventListener("keydown", (e) => {
    const typing = e.target.closest("input,textarea,select");
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); return; }
    if (typing) return;
    if (e.key === "/") { e.preventDefault(); openPalette(); }
    else if (e.key === "]" || (e.altKey && e.key === "ArrowRight")) { const i = CHAPTERS.indexOf(chapter()); if (i < CHAPTERS.length - 1) go(CHAPTERS[i + 1].id); }
    else if (e.key === "[" || (e.altKey && e.key === "ArrowLeft")) { const i = CHAPTERS.indexOf(chapter()); if (i > 0) go(CHAPTERS[i - 1].id); }
    else if (chapter().keys) chapter().keys(e);
  });
  // resizable split
  const g = $("#gutter");
  g.addEventListener("pointerdown", (e) => {
    g.setPointerCapture(e.pointerId); g.classList.add("drag");
    const work = $("#work").getBoundingClientRect();
    const mv = (ev) => { const pct = Math.min(70, Math.max(22, (ev.clientX - work.left) / work.width * 100)); S.codeW = pct; $("#code-pane").style.setProperty("--code-w", pct + "%"); $("#code-pane").style.width = pct + "%"; };
    const up = () => { g.classList.remove("drag"); g.removeEventListener("pointermove", mv); g.removeEventListener("pointerup", up); save(); };
    g.addEventListener("pointermove", mv); g.addEventListener("pointerup", up);
  });
  g.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    S.codeW = Math.min(70, Math.max(22, (S.codeW || 42) + (e.key === "ArrowLeft" ? -3 : 3)));
    $("#code-pane").style.width = S.codeW + "%"; save();
  });
}
const BINDS = {};
const ACTIONS = {
  go: (id) => go(id),
  view: (v) => { S.view = v; renderContent(false); },
  file: (f) => { S.codeFile = f; renderCode(); scrollCodeToOp(); },
  "code-op": (id) => setOp(id, true),
  "copy-op": () => { const o = OPS[S.op]; if (o) copyText(D.sources[o.file].split("\n").slice(o.start - 1, o.end).join("\n")); },
  "copy-file": () => copyText(D.sources[S.codeFile]),
  download: () => downloadSource(),
  inspect: (k) => { S.inspect = k; S.insEl = null; $("#inspector").classList.remove("collapsed"); renderInspector(); },
  "ins-toggle": () => { $("#inspector").classList.toggle("collapsed"); },
  "ins-apply": () => {
    const v = parseFloat($("#ins-edit").value); if (!Number.isFinite(v) || !S.insEl) return;
    const [name, ...idx] = S.insEl; S.overrides[name] = S.overrides[name] || {}; S.overrides[name][idx.join(",")] = v;
    refresh(); toast("Recomputed every dependent tensor");
  },
  "ins-revert": () => { S.overrides = {}; refresh(); toast("Parameters restored"); },
  quiz: (arg) => {
    const [id, v] = arg.split("|");
    S.progress.quiz[id] = v === "reveal" ? "seen" : +v;
    const pred = $("#pred-" + id), keep = pred ? pred.value : "";
    renderContent(true); renderNav(); save();
    const p2 = $("#pred-" + id); if (p2) p2.value = keep;
  },
  dims: (v) => { S.dims = v; $$("#dims button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.arg === v)); renderContent(true); renderInspector(); save(); },
  search: () => openPalette(),
  menu: () => { S.nav = !S.nav; renderNav(); },
  pane: (p) => { S.pane = p; $("#work").dataset.pane = p; $$("#mswitch button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.arg === p)); },
  reset: () => { const keep = S.progress, ch = S.ch; S = DEFAULTS(); S.progress = keep; S.ch = ch; $("#ex-in").value = S.text; $("#ex-in").classList.remove("invalid"); refresh({ code: true, full: true }); ACTIONS.dims(S.dims); toast("Interactive state reset"); },
  "reset-progress": () => { S.progress = { visited: {}, quiz: {} }; S.progress.visited[S.ch] = true; save(); renderNav(); renderContent(true); toast("Progress cleared"); },
};

function boot() {
  load();
  resolveOps();
  verifyMicro();
  compute();
  S.progress.visited[S.ch] = true;
  $("#ex-in").value = S.text;
  $("#ex-in").addEventListener("input", (e) => {
    const v = e.target.value;
    if (validText(v)) { e.target.classList.remove("invalid"); S.text = v; refresh({ full: true }); }
    else e.target.classList.add("invalid");
    e.target.title = validText(v) ? "" : `Exactly 9 characters from the micro vocabulary: ${MV.slice(1).map(vis).join(" ")}`;
  });
  $$("#dims button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.arg === S.dims));
  if (S.codeW) $("#code-pane").style.width = S.codeW + "%";
  $("#work").dataset.pane = S.pane;
  bind();
  go(S.ch, false);
  window.__ready = true;
}
