/* chapters2.js — chapters 9–15, extensions, reference */
"use strict";

/* ================= Chapter 9: heads + FFN ================= */
ACT.ffnPos = (t) => { S.ffnPos = +t; refresh({ compute: false, full: true }); };
function geluCurve(pre) {
  const w = 300, h = 170, x0 = -4, x1 = 4, y0 = -0.5, y1 = 4;
  const X = (v) => 30 + (w - 40) * (v - x0) / (x1 - x0), Y = (v) => 10 + (h - 30) * (1 - (v - y0) / (y1 - y0));
  let d = "";
  for (let v = x0; v <= x1 + 1e-9; v += 0.05) d += `${d ? "L" : "M"}${X(v).toFixed(1)},${Y(Eng.gelu(v)).toFixed(1)}`;
  let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="GELU curve with this position's pre-activations"><line class="axis" x1="30" x2="${w - 10}" y1="${Y(0)}" y2="${Y(0)}"/><line class="axis" x1="${X(0)}" x2="${X(0)}" y1="10" y2="${h - 20}"/>`;
  [-4, -2, 0, 2, 4].forEach((t) => (s += `<text x="${X(t)}" y="${h - 6}" text-anchor="middle">${t}</text>`));
  [0, 2, 4].forEach((t) => (s += `<text x="24" y="${Y(t) + 3}" text-anchor="end">${t}</text>`));
  s += `<path d="${d}" fill="none" stroke="var(--k-act)" stroke-width="2"/>`;
  pre.forEach((v) => { const c = Math.max(x0, Math.min(x1, v)); s += `<circle cx="${X(c)}" cy="${Y(Eng.gelu(c))}" r="3" fill="var(--k-tgt)" opacity=".8"><title>${fmt(v, 3)} → ${fmt(Eng.gelu(v), 3)}</title></circle>`; });
  return s + "</svg>";
}
ch({
  id: "ffn", num: 9, short: "Heads + FFN", title: "Merging heads and the feed-forward network", group: "The model", kind: "forward",
  thesis: "Head outputs are concatenated back to width D and mixed by W_O; then a two-layer MLP with GELU transforms every position independently, with the same weights at every position.",
  files: ["model.py"], ops: ["merge", "out_proj", "res_attn", "ffn_def", "ffn_fwd", "res_ffn"],
  card: {
    problem: "Combine what the heads retrieved, then apply per-position non-linear computation.",
    ml: "A Dense(Dff, GELU) → Dense(D) block applied to each position, as Keras TimeDistributed would.",
    op: "attn = merge(O) W_Oᵀ + b_O;  FFN(x) = W2 · GELU(W1 x + b1) + b2",
    shapes: () => `O ${shape(["B", "H", "T", "Dh"])} → ${shape(["B", "T", "D"])};  FFN ${shape(["B", "T", "D"])} → ${shape(["B", "T", "Dff"])} → ${shape(["B", "T", "D"])}`,
    params: () => `W_O: D²+D; FFN: 2·D·Dff + Dff + D = ${2 * 8 * 32 + 32 + 8} (micro), ${fmtInt(2 * 128 * 512 + 512 + 128)} (training)`,
    changes: "W_O, W1, W2 and biases.",
    fixed: "Dff = 4D and the activation function.",
    failures: "view() on a non-contiguous tensor; merging heads in the wrong order; using the tanh-approximate GELU when comparing to exact numbers.",
    verify: "Running the FFN on one position alone gives the same output as that row of the batched computation.",
  },
  sections: () => {
    const f = M.fw, L = posLabels(), t = S.ffnPos;
    INS.merged = actInspect("merged heads", ["T", "D"], f.merged, { rowLabels: L });
    INS.attnout = actInspect("attention output", ["T", "D"], f.attn, { rowLabels: L });
    INS.h1 = actInspect("h1", ["T", "D"], f.h1, { rowLabels: L });
    const W1 = M.P["blocks.0.ffn.fc1.weight"], b1 = M.P["blocks.0.ffn.fc1.bias"], W2 = M.P["blocks.0.ffn.fc2.weight"], b2 = M.P["blocks.0.ffn.fc2.bias"];
    const solo = Eng.linear(Eng.linear([f.ln2.y[t]], W1, b1).map((r) => r.map(Eng.gelu)), W2, b2)[0];
    return [
      {
        title: "Concatenate heads, project, add", tag: "tensor", prov: "browser", body: () => `<div class="tensors">${f.O.map((o, h) => tensor(`O<sub>h=${h}</sub>`, ["T", "Dh"], "act", heat(o, { kind: "act", rowLabels: L, digits: 2 }))).join("")}<span class="op-sym">⇒</span>
        ${tensor("merged = transpose(1,2).view(B,T,D)", ["T", "D"], "act", heat(f.merged, { kind: "act", rowLabels: L, colLabels: ["h0·0", "h0·1", "h0·2", "h0·3", "h1·0", "h1·1", "h1·2", "h1·3"], digits: 2, id: "merged", inspect: INS.merged }), { op: "merge", inspectKey: "merged" })}</div>
        <div class="tensors" style="margin-top:14px">${tensor("x0 (residual)", ["T", "D"], "act", heat(f.x0, { kind: "act", rowLabels: L, digits: 2 }))}<span class="op-sym">+</span>${tensor("merged · W_Oᵀ + b_O", ["T", "D"], "act", heat(f.attn, { kind: "act", rowLabels: L, digits: 2, id: "attnout", inspect: INS.attnout }), { op: "out_proj", inspectKey: "attnout" })}<span class="op-sym">=</span>${tensor("h1", ["T", "D"], "act", heat(f.h1, { kind: "act", rowLabels: L, digits: 2, id: "h1", inspect: INS.h1 }), { op: "res_attn", inspectKey: "h1" })}</div>
        <p style="margin-top:8px">${recordedCheck(f.merged, MICRO.trace.merged, "merged vs PyTorch:")} ${recordedCheck(f.h1, MICRO.trace.h1, "h1:")}</p>`,
      },
      {
        title: "The FFN at one position", tag: "numerical", prov: "browser",
        ctrl: () => `<span class="small muted">position</span>${inputChips("ffnPos", t).replace('class="chips"', 'class="chips" style="display:inline-flex"')}`,
        body: () => `<div class="grid2"><div>
          <h3 style="margin-top:0">LN2(h1)[${t}] ${shape(["D"])}</h3>${heat([f.ln2.y[t]], { kind: "act", digits: 2 })}
          <h3>W1·x + b1 ${shape(["Dff"])} — pre-activation</h3>${bars(f.fpre[t], { kind: "act", h: 110, w: 460, aria: "FFN pre-activations" })}
          <h3>GELU ${shape(["Dff"])}</h3>${bars(f.fact[t], { kind: "act", h: 110, w: 460, aria: "FFN activations" })}
          <h3>W2·GELU + b2 ${shape(["D"])} — added to h1</h3>${heat([f.fout[t]], { kind: "act", digits: 3 })}</div>
          <div><h3 style="margin-top:0">This position's 32 pre-activations on the GELU curve</h3>${geluCurve(f.fpre[t])}
          <p class="small">Negative inputs are mostly suppressed, positive ones pass nearly unchanged; unlike ReLU the curve is smooth and slightly negative around −0.75.</p>
          <div class="callout ${Eng.maxAbsDiff(solo, f.fout[t]) < 1e-12 ? "ok" : "warn"}"><b>Position independence, checked</b>Running the FFN on row ${t} alone gives max |Δ| = ${fmt(Eng.maxAbsDiff(solo, f.fout[t]), 2)} against the batched result. No information moves between positions here; that is attention's job.</div>
          <p>${recordedCheck(f.fact, MICRO.trace.ffn_act, "GELU activations vs PyTorch:")}</p></div></div>`,
      },
      {
        title: "GELU, exactly", tag: "math", body: () => `<div class="eq">GELU(x) = x · Φ(x) = ½ x (1 + erf(x / √2))          ← nn.GELU() default, used here
GELU(x) ≈ ½ x (1 + tanh(√(2/π) (x + 0.044715 x³)))   ← nn.GELU(approximate="tanh"), used by GPT-2
GELU'(x) = Φ(x) + x · φ(x)</div><p>The browser engine uses an erf approximation with fractional error below 1.2·10⁻⁷, which is why its GELU values agree with PyTorch's float32 results to about 10⁻⁷.</p>`,
      },
      {
        title: "Where information mixes", tag: "engineering", body: () => `<div class="tbl-wrap"><table class="data"><tr><th></th><th>attention</th><th>FFN</th></tr>
        <tr><td>mixes across positions</td><td>yes, along T (causally)</td><td>no</td></tr><tr><td>weights shared across positions</td><td>yes</td><td>yes</td></tr><tr><td>weights shared across blocks</td><td>no: each of the L blocks has its own</td><td>no</td></tr>
        <tr><td>parameters per block (training)</td><td class="num">${fmtInt(4 * (128 * 128 + 128))}</td><td class="num">${fmtInt(2 * 128 * 512 + 512 + 128)}</td></tr><tr><td>FLOPs per token per block</td><td>8D² + 4·T·D</td><td>4·D·Dff = 16D²</td></tr></table></div>
        <p class="small" style="margin-top:8px">The FFN holds about two thirds of each block's parameters. The concatenation is free (a reshape); W_O is what actually lets heads combine.</p>`,
      },
    ];
  },
  quiz: [
    { id: "q-ffn-1", q: "True or false: the FFN in block 2 uses the same weights as the FFN in block 1, just as all positions share weights.", options: ["True", "False"], answer: 1, explain: "Weights are shared across positions within a block, but each block is a separate module with its own parameters." },
  ],
});

/* ================= Chapter 10: logits and loss ================= */
ACT.lossPos = (t) => { S.lossPos = +t; refresh({ compute: false, full: true }); };
segAct("lossView", "lossView", false, { compute: false });
ch({
  id: "loss", num: 10, short: "Logits & loss", title: "Vocabulary logits and cross-entropy", group: "The model", kind: "train",
  thesis: "A final LayerNorm and one linear map score every vocabulary entry at every position. Cross-entropy then reads off the negative log-probability of the true next character and averages over all B·T positions.",
  files: ["model.py", "tests/test_llm.py"], ops: ["lnf", "lm_head", "logits", "ce"],
  card: {
    problem: "Convert hidden states into a training signal that rewards probability on the correct next token.",
    ml: "Softmax classification with V classes, done at every position.",
    op: "loss = −1/(B·T) Σ_{b,t} log softmax(z_{b,t})[y_{b,t}]",
    shapes: () => `${shape(["B", "T", "D"])} → logits ${shape(["B", "T", "V"])} → view ${shape(["B*T", "V"])} with targets ${shape(["B*T"])} → scalar`,
    params: () => `ln_f (2D) + lm_head V·D = ${2 * 8 + MV.length * 8} (micro), ${fmtInt(256 + 66 * 128)} (training); 0 extra if tied to E`,
    changes: "ln_f γ/β and lm_head.",
    fixed: "The targets.",
    failures: "Softmax before F.cross_entropy; mismatched flattening of logits and targets; ignore_index not set when padding is used.",
    verify: "At initialisation loss ≈ ln V; F.cross_entropy equals the manual −mean(log_softmax[y]).",
  },
  sections: () => {
    const f = M.fw, t = S.lossPos, z = f.logits[t], p = f.probs[t], y = f.targets[t];
    const m = Math.max(...z), se = z.reduce((a, v) => a + Math.exp(v - m), 0), lse = m + Math.log(se);
    const pSoft = p, ceOnProbs = Math.log(pSoft.reduce((a, v) => a + Math.exp(v), 0)) - pSoft[y];
    INS.logits = actInspect("logits", ["T", "V"], f.logits, { rowLabels: posLabels(), colLabels: MV.map(vis) });
    INS.Wh = paramInspect("lm_head.weight", { rowLabels: MV.map(vis) });
    return [
      {
        title: "Hidden states to logits", tag: "tensor", prov: "browser", body: () => `<div class="tensors">${tensor("LN_f(h2)", ["T", "D"], "act", heat(f.lnf.y, { kind: "act", rowLabels: posLabels(), digits: 2, rowSel: t }), { op: "lnf" })}<span class="op-sym">·</span>${tensor("lm_head.weightᵀ", ["D", "V"], "par", `<button class="btn small" data-act="inspect" data-arg="Wh">Inspect [${MV.length} × 8]</button>`, { op: "lm_head" })}<span class="op-sym">=</span></div>
        <div style="margin-top:10px">${tensor("logits z", ["T", "V"], "act", heat(f.logits, { kind: "act", size: "mid", values: false, rowLabels: posLabels(), colLabels: MV.map(vis), rowSel: t, hl: (r, c) => c === f.targets[r], id: "logits", inspect: INS.logits, onCell: (r) => { S.lossPos = r; refresh({ compute: false, full: true }); } }), { op: "logits", inspectKey: "logits" })}</div>
        <p class="small muted">Outlined cells are each position's target. ${recordedCheck(f.logits, MICRO.trace.logits, "logits vs PyTorch:")}</p>`,
      },
      {
        title: "One position: logits, probabilities, NLL", tag: "numerical", prov: "browser",
        ctrl: () => `<span class="small muted">position</span>${inputChips("lossPos", t).replace('class="chips"', 'class="chips" style="display:inline-flex"')}${segC("lossView", [["logits", "logits z"], ["probs", "softmax(z)"]], S.lossView)}`,
        body: () => `<p>Input so far “${esc(visStr(M.ids.slice(0, t + 1).map((i) => MV[i]).join("")))}” → target '<b style="color:var(--k-tgt)">${esc(mvis(y))}</b>' (id ${y}).</p>
        ${bars(S.lossView === "logits" ? z : p, { labels: MV.map(vis), hi: (i) => i === y, kind: "act", hiKind: "tgt", h: 170, w: 680, aria: "Scores over the vocabulary" })}
        <div class="calc" style="margin-top:10px">stable log-sum-exp:  m = max(z) = ${fmt(m, 4)}
log Σ exp(z) = m + log Σ exp(z − m) = ${fmt(m, 4)} + log(${fmt(se, 5)}) = ${fmt(lse, 5)}
log p(y) = z[y] − logsumexp(z) = ${fmt(z[y], 4)} − ${fmt(lse, 4)} = ${fmt(z[y] - lse, 5)}
NLL = −log p(y) = ${fmt(f.nll[t], 5)}      p(y) = ${fmt(p[y], 5)}</div>
        <div class="tbl-wrap" style="margin-top:10px"><table class="data"><tr><th>t</th><th>context end</th><th>target</th><th class="num">p(target)</th><th class="num">NLL</th></tr>${f.nll.map((n, i) => `<tr class="${i === t ? "sel" : ""}"><td class="mono">${i}</td><td class="mono">${esc(mvis(M.ids[i]))}</td><td class="mono" style="color:var(--k-tgt)">${esc(mvis(f.targets[i]))}</td><td class="num">${fmt(f.probs[i][f.targets[i]], 4)}</td><td class="num">${fmt(n, 4)}</td></tr>`).join("")}<tr><td colspan="4"><b>mean = loss</b></td><td class="num"><b>${fmt(f.loss, 5)}</b></td></tr></table></div>
        <p style="margin-top:8px">${M.canonical ? `PyTorch recorded loss ${fmt(MICRO.trace.loss, 5)} · ${okPill(Math.abs(f.loss - MICRO.trace.loss))}` : recordedCheck(0, 0)} <span class="small muted">Uniform guessing would give ln V = ${fmt(Math.log(MV.length), 4)}.</span></p>`,
      },
      {
        title: "Flatten, then cross-entropy", tag: "math", body: () => `<div class="eq">logits.view(B*T, V)  ${shape(["B*T", "V"])}      targets.view(B*T)  ${shape(["B*T"])}
loss = −1/(B·T) · Σ<sub>b,t</sub> log p( <span class="y">y<sub>b,t</sub></span> | x<sub>b,0..t</sub> ),    log p = z<sub>y</sub> − logsumexp(z)</div>
        <p><code>F.cross_entropy</code> combines <code>log_softmax</code> and negative log-likelihood in one numerically stable kernel. Its input must be <b>raw logits</b>. Gradient with respect to the logits is simply (softmax(z) − one_hot(y)) / (B·T), which chapter 11 uses.</p>`,
      },
      {
        title: "The softmax-before-cross-entropy bug", tag: "numerical", prov: "browser", body: () => `<p>If you pass probabilities instead of logits, PyTorch applies softmax a second time. For position ${t}:</p>
        <div class="calc">correct:  cross_entropy(z, y)          = ${fmt(f.nll[t], 5)}
buggy:    cross_entropy(softmax(z), y) = ${fmt(ceOnProbs, 5)}</div>
        <p>The buggy value can never fall below log(V − 1 + e) − 1 = ${fmt(Math.log(MV.length - 1 + Math.E) - 1, 4)} because its "logits" are trapped in [0, 1]. Training stalls with tiny, misleading gradients. <code>test_cross_entropy_equals_manual_nll</code> pins the correct behaviour.</p>`,
      },
      {
        title: "Padding and ignored targets", tag: "engineering", body: () => `<p>Every window here has exactly T real targets, so no mask is needed. With variable-length sequences you pad to a common length and set padded targets to <code>ignore_index</code> (−100 by default): those positions contribute neither to the sum nor to the count in the mean. The attention mask must then also hide padded keys. Keep the two masks conceptually separate: the causal mask enforces "no future", the padding mask enforces "not real text".</p>`,
      },
    ];
  },
  quiz: [
    { id: "q-loss-1", q: "Your loss starts at 0.98 on a V = 66 character model and barely moves. You find `F.cross_entropy(logits.softmax(-1), y)`. Diagnose it.", explain: "Softmax was applied twice. cross_entropy expects raw logits and applies log_softmax itself; feeding probabilities squashes the inputs into [0,1], bounds the achievable loss and shrinks gradients. Remove the explicit softmax. (Untrained loss should be near ln 66 ≈ 4.19.)" },
  ],
});

/* ================= Chapter 11: backprop ================= */
const STEP11 = [
  ["Clear gradients", "zero", "opt.zero_grad(set_to_none=True) sets every parameter's .grad to None. Without it, PyTorch would add the new gradients to the previous step's."],
  ["Forward pass", "fwd_loss", "Builds the computation graph: every operation records what it needs for its backward function (inputs, softmax outputs, LayerNorm statistics)."],
  ["Compute loss", "ce", "Mean cross-entropy over B·T = 8 targets. The loss is the root of the graph."],
  ["Backward pass", "backward", "Autograd walks the graph in reverse, applying the chain rule and accumulating ∂loss/∂θ into each parameter's .grad."],
  ["Inspect gradient norms", "clip", "‖g‖ over all parameters is a cheap health signal: spikes precede divergence; zeros reveal dead paths."],
  ["Clip the gradient", "clip", "clip_grad_norm_ rescales all gradients together if ‖g‖ exceeds the threshold, preserving direction."],
  ["Optimizer step", "opt_step", "AdamW updates every parameter from its gradient and its running moment estimates."],
];
ACT.step11 = (v) => { S.step11 = Math.max(0, Math.min(6, v === "next" ? S.step11 + 1 : v === "prev" ? S.step11 - 1 : +v)); refresh({ compute: false }); setOp(STEP11[S.step11][1]); };
bindStr("paramSel", () => { const sh = MICRO.shapes[S.paramSel]; S.paramIdx = sh.length === 2 ? [0, 0] : [0]; refresh({ compute: false }); });
bindBool("noZero", { compute: false });
segAct("accum", "accumScaled", false, { compute: false });
const BACK_ORDER = ["lm_head.weight", "ln_f.weight", "ln_f.bias", "blocks.0.ffn.fc2.weight", "blocks.0.ffn.fc2.bias", "blocks.0.ffn.fc1.weight", "blocks.0.ffn.fc1.bias", "blocks.0.ln2.weight", "blocks.0.ln2.bias", "blocks.0.attn.out_proj.weight", "blocks.0.attn.out_proj.bias", "blocks.0.attn.v_proj.weight", "blocks.0.attn.v_proj.bias", "blocks.0.attn.k_proj.weight", "blocks.0.attn.k_proj.bias", "blocks.0.attn.q_proj.weight", "blocks.0.attn.q_proj.bias", "blocks.0.ln1.weight", "blocks.0.ln1.bias", "pos_emb.weight", "tok_emb.weight"];
const norm = (m) => Math.sqrt(Eng.flat(m).reduce((a, v) => a + v * v, 0));
ch({
  id: "backprop", num: 11, short: "Backprop & update", title: "Backpropagation and the AdamW update", group: "Training", kind: "train",
  thesis: "One training step: clear gradients, run the forward pass, compute the loss, let autograd push ∂loss/∂θ to every parameter, optionally clip, and let AdamW move each parameter. Every number below is recomputed in your browser by a hand-written backward pass.",
  files: ["train.py", "tests/test_llm.py"], ops: ["zero", "fwd_loss", "ce", "micro_loop", "backward", "clip", "opt_step", "make_opt", "t_fd", "t_tie"],
  keys: (e) => { if (e.key === "ArrowRight") ACT.step11("next"); else if (e.key === "ArrowLeft") ACT.step11("prev"); },
  card: {
    problem: "Compute how the loss changes with every parameter and move the parameters downhill.",
    ml: "Identical to backprop in your Keras ANNs; only the graph contains attention and residual branches.",
    op: "g = ∇θ loss;  ĝ = g·min(1, c/‖g‖);  θ ← AdamW(θ, ĝ, m, v)",
    shapes: () => "Each .grad has exactly its parameter's shape. Activations are not updated; they are recomputed every forward pass.",
    params: "All of them receive gradients (k_proj.bias receives exactly zero; see below).",
    changes: "θ, AdamW's m and v, and its step count.",
    fixed: "Token ids, targets, the mask, hyperparameters.",
    failures: "Missing zero_grad (gradients accumulate); unscaled accumulation (effective lr × k); clipping after step; weight decay on LayerNorm and biases.",
    verify: "Finite-difference check in float64; loss on the same batch decreases after a small step; tied gradient equals the sum of both paths.",
  },
  sections: () => {
    const G = M.bw.G, GC = M.clip.G, A = M.adam, opt = MICRO.optimizer, st = S.step11;
    const after = Eng.forward(A.after, CFG, M.ids, M.tg);
    const pn = S.paramSel, sh = MICRO.shapes[pn], idx = S.paramIdx, get = (mx) => (sh.length === 2 ? mx[idx[0]][idx[1]] : mx[idx[0]]);
    const ab = MICRO.adam_before[pn], b1 = opt.betas[0], b2 = opt.betas[1], tstep = ab.step + 1, wd = opt.weight_decay[pn], lr = opt.lr;
    const th = get(M.P[pn]), g = get(G[pn]), gc = get(GC[pn]), m0 = get(ab.exp_avg), v0 = get(ab.exp_avg_sq), m1 = get(A.m[pn]), v1 = get(A.v[pn]);
    const mh = m1 / (1 - b1 ** tstep), vh = v1 / (1 - b2 ** tstep), th1 = get(A.after[pn]);
    let stepBody = `<div class="steps">${STEP11.map((s, i) => `<button class="${i === st ? "cur" : i < st ? "done" : ""}" data-act="step11" data-arg="${i}">${i + 1}. ${s[0]}</button>`).join("")}</div>
      <div class="stepper" style="margin-bottom:10px"><button class="btn small" data-act="step11" data-arg="prev">◀ Back</button><button class="btn small primary" data-act="step11" data-arg="next">Next ▶</button></div>
      <p>${STEP11[st][2]}</p>`;
    if (st === 0) stepBody += `<div class="calc">.grad is None for all ${Object.keys(G).length} parameter tensors (${fmtInt(Eng.flat(Object.values(M.P)).length)} scalars)</div>`;
    if (st === 1) stepBody += `<div class="calc">tokens  x = [${M.ids.join(", ")}]   "${visStr(M.ids.map((i) => MV[i]).join(""))}"
logits  ${S.dims === "sym" ? "[B, T, V]" : `[1, 8, ${MV.length}]`}   first row: [${M.fw.logits[0].slice(0, 6).map((v) => fmt(v, 3)).join(", ")}, …]</div>`;
    if (st === 2) stepBody += `<div class="calc">loss = mean NLL over 8 targets = ${fmt(M.fw.loss, 6)}${M.canonical ? `      (PyTorch: ${fmt(MICRO.trace.loss, 6)})` : ""}</div>`;
    if (st >= 3) stepBody += `<h3>‖∂loss/∂θ‖ per parameter, in the order autograd produces them</h3>${bars(BACK_ORDER.map((n) => norm(G[n])), { labels: BACK_ORDER.map((n) => n.replace("blocks.0.", "").replace(".weight", ".W").replace(".bias", ".b").replace("attn.", "").replace("ffn.", "")), kind: "grad", h: 190, w: 720, aria: "Gradient norm per parameter", onBar: (i) => { S.paramSel = BACK_ORDER[i]; const s2 = MICRO.shapes[S.paramSel]; S.paramIdx = s2.length === 2 ? [0, 0] : [0]; refresh({ compute: false, full: true }); } })}<p class="small muted">Select a bar to inspect that parameter's update below.</p>`;
    if (st >= 4) stepBody += `<div class="calc">‖g‖ = √(Σ over all parameters of Σ g²) = ${fmt(M.clip.norm, 6)}${M.canonical ? `      (PyTorch clip_grad_norm_ returned ${fmt(MICRO.grad_norm_total, 6)})` : ""}</div>`;
    if (st >= 5) stepBody += `<div class="calc">max_norm = ${opt.grad_clip};  coef = max_norm / (‖g‖ + 1e-6) = ${fmt(M.clip.coef, 6)}  ${M.clip.coef < 1 ? "→ every gradient is multiplied by coef; the step direction is unchanged" : "→ no clipping needed"}</div>`;
    if (st >= 6) stepBody += `<div class="calc">loss on the same window: before ${fmt(M.fw.loss, 6)}  →  after ${fmt(after.loss, 6)}${M.canonical ? `      (PyTorch after step: ${fmt(MICRO.loss_after_step, 6)})` : ""}</div>`;
    const pOptions = Object.keys(MICRO.shapes).map((n) => [n, `${n}  [${MICRO.shapes[n].join("×")}]`]);
    return [
      { title: "One training step", tag: "pipeline", prov: "browser", body: () => stepBody },
      {
        title: "One parameter, one update", tag: "numerical", prov: "browser",
        ctrl: () => selectC("paramSel", "parameter", pOptions, S.paramSel),
        body: () => `<div class="grid2"><div>${tensor(`∂loss/∂ ${pn}`, sh.map(String), "grad", heat(sh.length === 2 ? G[pn] : [G[pn]], { kind: "grad", size: sh[0] * (sh[1] || 1) > 120 ? "mid" : undefined, values: (sh[1] || sh[0]) <= 12, digits: 3, sel: sh.length === 2 ? idx : [0, idx[0]], onCell: (r, c) => { S.paramIdx = sh.length === 2 ? [r, c] : [c]; refresh({ compute: false }); } }))}<p class="small muted">Select an element.</p></div>
        <div><div class="calc">θ (before)                      = ${fmt(th, 7)}
g = ∂loss/∂θ                     = ${fmt(g, 7)}
ĝ = g · coef (clipped)           = ${fmt(gc, 7)}
step t                           = ${tstep}  (${ab.step} earlier steps)
m_prev, v_prev                   = ${fmt(m0, 6)}, ${fmt(v0, 6)}
m = β1·m_prev + (1−β1)·ĝ          = ${fmt(m1, 7)}     β1 = ${b1}
v = β2·v_prev + (1−β2)·ĝ²         = ${fmt(v1, 7)}     β2 = ${b2}
m̂ = m/(1−β1^t), v̂ = v/(1−β2^t)   = ${fmt(mh, 6)}, ${fmt(vh, 6)}
decay:  θ·(1 − lr·λ)             = ${fmt(th * (1 - lr * wd), 7)}   lr = ${lr}, λ = ${wd}
update: lr · m̂ / (√v̂ + ε)        = ${fmt(lr * mh / (Math.sqrt(vh) + opt.eps), 7)}   ε = ${opt.eps}
θ' (after)                       = ${fmt(th1, 7)}
plain SGD would give θ − lr·ĝ     = ${fmt(th - lr * gc, 7)}</div>
        <p style="margin-top:8px">${M.canonical ? `PyTorch θ' = ${fmt(get(MICRO.params_after[pn]), 7)} · ${okPill(Math.abs(th1 - get(MICRO.params_after[pn])), pn.endsWith("k_proj.bias") ? 1 : 1e-5)}` : recordedCheck(0, 0)}</p></div></div>
        ${pn.endsWith("k_proj.bias") ? `<div class="callout warn"><b>The key bias has an exactly zero gradient</b>Adding the same vector b_K to every key adds q·b_K to every score in a row; softmax ignores constant shifts, so loss cannot depend on b_K. Both PyTorch (float32) and this page (float64) produce only rounding noise here (~10⁻⁸ and ~10⁻¹⁸). Adam divides m by √v, so even noise produces a visible step; that is why the two engines' updates for this one tensor disagree. Weight decay (λ = 0 for biases) is not what moves it.</div>` : ""}`,
      },
      {
        title: "Embeddings receive a scatter-add", tag: "numerical", prov: "browser", body: () => {
          const counts = {}; M.ids.forEach((i, p) => (counts[i] = counts[i] || []).push(p));
          const rows = Object.keys(counts).map(Number).sort((a, b) => a - b);
          const rep = rows.find((r) => counts[r].length > 1);
          return `<p>Only rows of E that appear in the window receive gradient; the other ${MV.length - rows.length} rows get exactly 0. Token ids are integers used as indices, so they have no gradient at all.</p>
          <div class="tbl-wrap"><table class="data"><tr><th>id</th><th>char</th><th>positions</th><th class="num">‖∂loss/∂E[id]‖</th></tr>${rows.map((r) => `<tr><td class="mono">${r}</td><td class="mono">${esc(mvis(r))}</td><td class="mono">${counts[r].join(", ")}</td><td class="num">${fmt(norm(M.bw.G["tok_emb.weight"][r]), 5)}</td></tr>`).join("")}</table></div>
          ${rep !== undefined ? `<div class="calc" style="margin-top:10px">∂loss/∂E[${rep}] = ${counts[rep].map((p) => `∂loss/∂x0[${p}]`).join(" + ")}
${counts[rep].map((p) => `  [${M.bw.dx0[p].map((v) => fmt(v, 4)).join(", ")}]`).join("\n+ ")}
= [${M.bw.G["tok_emb.weight"][rep].map((v) => fmt(v, 4)).join(", ")}]</div>` : ""}`;
        },
      },
      {
        title: "Gradient accumulation and zero_grad", tag: "numerical", prov: ["recorded", "browser"],
        ctrl: () => `${segC("accum", [["true", "loss / k (correct)"], ["false", "no scaling"]], String(S.accumScaled), "accumulation")}${checkC("noZero", "forget zero_grad (2 backward calls)", S.noZero)}`,
        body: () => {
          const ga = MICRO.grad_accum, sel = S.accumScaled === true || S.accumScaled === "true";
          const acc = sel ? ga.accum_scaled : ga.accum_unscaled;
          return `<p>Two windows (“${esc(visStr(ga.windows[0]))}”, “${esc(visStr(ga.windows[1]))}”), lm_head.weight gradient, recorded from PyTorch:</p>
          <div class="calc">full batch B=2, one backward:        ‖g‖ = ${fmt(norm(ga.full), 6)}
two microbatches, ${sel ? "each loss / 2" : "loss NOT divided"}:     ‖g‖ = ${fmt(norm(acc), 6)}
max |full − accumulated| = ${fmt(Eng.maxAbsDiff(ga.full, acc), 3)}${sel ? "" : `   (ratio ${fmt(ga.ratio_unscaled, 4)})`}</div>
          <p>${sel ? "Dividing each microbatch loss by k = 2 reproduces the full-batch mean, because gradients add across backward calls. Effective batch = k × microbatch, with activation memory of one microbatch." : "Without the division the accumulated gradient is the <b>sum</b> of k means, k = 2 times too large. With SGD that silently doubles the learning rate; with clipping or Adam it changes the step in less obvious ways."}</p>
          ${S.noZero ? `<div class="callout bad"><b>Forgetting zero_grad</b>Two backward passes on the same window without clearing: ∂loss/∂θ becomes 2g for every parameter (browser: ‖g‖ ${fmt(M.clip.norm, 4)} → ${fmt(2 * M.clip.norm, 4)}). Across steps, stale gradients from previous batches keep adding up, so the model follows a running sum instead of the current gradient.</div>` : ""}`;
        },
      },
      {
        title: "AdamW, precisely", tag: "math", body: () => `<div class="eq">m<sub>t</sub> = β1 m<sub>t−1</sub> + (1 − β1) <span class="g">g<sub>t</sub></span>                first moment (momentum)
v<sub>t</sub> = β2 v<sub>t−1</sub> + (1 − β2) <span class="g">g<sub>t</sub></span>²               second moment (per-element scale)
m̂<sub>t</sub> = m<sub>t</sub> / (1 − β1<sup>t</sup>),   v̂<sub>t</sub> = v<sub>t</sub> / (1 − β2<sup>t</sup>)      bias correction (m, v start at 0)
<span class="w">θ</span> ← <span class="w">θ</span> − lr·λ·<span class="w">θ</span> − lr · m̂<sub>t</sub> / (√v̂<sub>t</sub> + ε)       decoupled weight decay (Loshchilov & Hutter)</div>
        <div class="prose"><ul><li><b>Decoupled</b> means decay is applied to θ directly, not added to the gradient (which Adam would then rescale). In <code>make_optimizer</code>, matrices and embeddings get λ = ${TRAINED.train_config.weight_decay}; biases and LayerNorm parameters get λ = 0.</li><li><b>ε</b> = ${MICRO.optimizer.eps} prevents division by zero where v̂ ≈ 0 and caps the step for parameters with tiny gradients.</li><li>Each parameter carries m and v of its own shape: <b>2 extra values per parameter</b>, which chapter 13 counts as optimizer-state memory.</li><li><b>Parameters vs activations vs gradients.</b> Parameters persist and are updated; activations are recomputed every forward pass and freed after backward; gradients live only between backward and the optimizer step (or across accumulation microbatches).</li></ul></div>`,
      },
      {
        title: "Weight tying changes the gradient", tag: "math", body: () => `<div class="eq">tied:  <span class="w">W<sub>head</sub></span> ≡ <span class="w">E</span>      ⇒   ∂loss/∂E = ∂loss/∂E |<sub>lookup path</sub> + ∂loss/∂W<sub>head</sub> |<sub>output path</sub></div>
        <p>One tensor used twice receives the sum of both uses' gradients. Every row of E then gets gradient from the output layer, even characters absent from the batch. Tying removes V·D parameters (${fmtInt(66 * 128)} in the training model, about 1%; for a 50k-token vocabulary it is far more). The baseline leaves tying off (<code>--tie_weights</code> enables it); <code>test_weight_tying_counts_and_gradient_sum</code> verifies both the count and the gradient sum.</p>`,
      },
    ];
  },
  quiz: [
    { id: "q-bp-1", q: "Which parameters receive a gradient from the logit of position 3 only (ignoring other positions)?", options: ["Only lm_head", "lm_head, ln_f and the FFN, but not attention", "Everything that influences h2[3]: lm_head, ln_f, FFN, LN2, W_O, W_V/W_K/W_Q, LN1, P[0..3] and the E rows of tokens 0..3", "All E rows"], answer: 2, explain: "Position 3's output depends on tokens 0–3 through attention, so gradient reaches the projections, position rows 0–3 and the embedding rows of those four tokens, never later positions." },
    { id: "q-bp-2", q: "You accumulate gradients over 4 microbatches but call loss.backward() without dividing by 4. What changes?", explain: "The accumulated gradient is the sum of 4 means, 4× the intended mean. With plain SGD that is a 4× learning rate. Clipping will likely rescale it (masking the bug); Adam is largely scale-invariant per element, so the error mostly shows up in ε-sensitive and weight-decay terms. Divide each microbatch loss by grad_accum, as train.py does." },
  ],
});

/* ================= Chapter 12: training ================= */
segAct("curveView", "curveView", false, { compute: false });
bindNum("sampleIdx");
ch({
  id: "train", num: 12, short: "Training & validation", title: "Training and validating the real model", group: "Training", kind: "train",
  thesis: "The same step, repeated thousands of times on random windows, with periodic evaluation on held-out text. Everything on this page was recorded from the run that produced the checkpoint in out/.",
  files: ["train.py"], ops: ["set_seed", "device", "lr_at", "make_opt", "estimate", "eval_block", "zero", "micro_loop", "clip", "opt_step", "resume", "log"],
  card: {
    problem: "Find parameters with low loss on unseen text, reproducibly, and keep enough state to resume.",
    ml: "A standard fit loop with validation checkpoints; there is no early-stopping callback, but every eval writes a checkpoint.",
    op: "for step: lr ← schedule(step); eval every 250; sample batch; forward; backward; clip; AdamW",
    shapes: () => `x, y ${shape(["B", "T"])} per microbatch; ${fmtInt(TRAINED.num_params)} parameters`,
    params: "All.",
    changes: "Parameters, optimizer state, RNG states, step counter.",
    fixed: "Architecture, data split, vocabulary, schedule.",
    failures: "Evaluating in train mode; evaluating on training data; learning rate too high (loss spikes); no warmup with large lr.",
    verify: "Overfit one tiny batch to near-zero loss; resume reproduces identical weights; eval batches are fixed across evals.",
  },
  sections: () => {
    const log = TRAINED.log, tr = log.filter((r) => r.type === "train"), ev = log.filter((r) => r.type === "eval");
    const last = ev[ev.length - 1], tc = TRAINED.train_config;
    const smooth = (a, k) => a.map((p, i) => [p[0], a.slice(Math.max(0, i - k), i + 1).reduce((s, q) => s + q[1], 0) / Math.min(i + 1, k + 1)]);
    let chart = "";
    if (S.curveView === "loss") chart = lineChart([{ name: "train loss (batch, smoothed)", color: "var(--k-act)", opacity: 0.6, width: 1.2, points: smooth(tr.map((r) => [r.step, r.loss]), 5) }, { name: "train loss (eval)", color: "var(--k-in)", dots: true, points: ev.map((r) => [r.step, r.train_loss]) }, { name: "validation loss", color: "var(--k-tgt)", dots: true, points: ev.map((r) => [r.step, r.val_loss]) }], { xLabel: "optimizer step", yLabel: "cross-entropy (nats/char)", h: 260, yMin: 1 });
    if (S.curveView === "lr") chart = lineChart([{ name: "learning rate", color: "var(--k-par)", points: tr.map((r) => [r.step, r.lr]) }], { xLabel: "optimizer step", yLabel: "lr", h: 220, fmtY: (v) => v.toExponential(2), fmtTickY: (v) => v.toExponential(1) });
    if (S.curveView === "gn") chart = lineChart([{ name: "‖g‖ before clipping", color: "var(--k-grad)", width: 1.2, points: tr.map((r) => [r.step, r.grad_norm]) }, { name: "clip threshold", color: "var(--muted)", dash: "4 4", points: [[0, tc.grad_clip], [tc.max_steps, tc.grad_clip]] }], { xLabel: "optimizer step", yLabel: "global gradient norm", h: 220, yMin: 0 });
    const si = Math.min(S.sampleIdx, ev.length - 1), sm = ev[si];
    return [
      {
        title: "The recorded run", tag: "numerical", prov: "recorded", body: () => `<div class="grid3">
        <dl class="kv"><dt>steps</dt><dd>${fmtInt(TRAINED.step)}</dd><dt>batch × context</dt><dd>${tc.batch_size} × ${TRAINED.config.block_size}</dd><dt>targets seen</dt><dd>${fmtInt(TRAINED.step * tc.batch_size * TRAINED.config.block_size)}</dd><dt>wall time</dt><dd>${(last.elapsed_s / 60).toFixed(1)} min</dd><dt>device</dt><dd>${TRAINED.env.device}, ${TRAINED.env.threads} threads</dd></dl>
        <dl class="kv"><dt>final train loss</dt><dd>${fmt(last.train_loss, 4)}</dd><dt>final val loss</dt><dd>${fmt(last.val_loss, 4)}</dd><dt>val, exhaustive</dt><dd>${fmt(TRAINED.val_loss_exhaustive, 4)} (${fmtInt(TRAINED.val_targets)} targets)</dd><dt>test, exhaustive</dt><dd>${fmt(TRAINED.test_loss_exhaustive, 4)} (${fmtInt(TRAINED.test_targets)} targets)</dd></dl>
        <dl class="kv"><dt>val perplexity</dt><dd>${fmt(Math.exp(TRAINED.val_loss_exhaustive), 3)}</dd><dt>bits per character</dt><dd>${fmt(TRAINED.val_loss_exhaustive / Math.LN2, 3)}</dd><dt>uniform baseline</dt><dd>ln 66 = ${fmt(Math.log(66), 3)}</dd><dt>torch</dt><dd>${TRAINED.env.torch}</dd></dl></div>
        <p class="small muted" style="margin-top:8px">Periodic eval uses 50 fixed random batches per split (train mode off, no_grad). "Exhaustive" losses cover every non-overlapping 128-character window of the split once and were computed after training; the test split was not used before that.</p>`,
      },
      {
        title: "Curves", tag: "numerical", prov: "recorded",
        ctrl: () => segC("curveView", [["loss", "loss"], ["lr", "learning rate"], ["gn", "gradient norm"]], S.curveView),
        body: () => `${chart}<p class="small" style="margin-top:6px">${S.curveView === "loss" ? `The gap between train and validation loss (${fmt(last.val_loss - last.train_loss, 3)} nats at the end) is the generalisation gap. Validation is the later part of the file, so part of the gap is distribution shift rather than overfitting. A widening gap with flat validation loss is the overfitting signal.` : S.curveView === "lr" ? "Linear warmup over 100 steps, then cosine decay from 1e-3 to 1e-4 by the final step." : "Values above the dashed line were clipped to norm 1.0 before the update. Early steps clip almost always; later steps rarely."}</p>`,
      },
      {
        title: "Samples during training", tag: "numerical", prov: "recorded",
        ctrl: () => rangeC("sampleIdx", "eval #", 0, ev.length - 1, si),
        body: () => `<p class="small muted">Step ${fmtInt(sm.step)} · val loss ${fmt(sm.val_loss, 3)} · temperature 0.8 · seed 0 · prompt "\\n"</p><pre class="calc" style="white-space:pre-wrap">${esc(sm.sample)}</pre>`,
      },
      {
        title: "Reading the numbers", tag: "math", body: () => `<div class="eq">perplexity = exp(loss)          bits/char = loss / ln 2
untrained: loss ≈ ln V = ${fmt(Math.log(66), 3)}  →  perplexity ≈ V = 66</div>
        <div class="prose"><ul><li>Perplexity ${fmt(Math.exp(TRAINED.val_loss_exhaustive), 2)} means the model is, on average, as uncertain as a uniform choice among that many characters.</li><li><b>Not comparable across tokenizers.</b> A BPE model's per-token perplexity measures uncertainty over much larger units. Compare per-character or per-byte quantities (bits per byte) instead.</li><li><b>Lower loss is not reliability.</b> The model learns the statistics of the text: spelling, verse layout, speaker tags. It has no mechanism for facts, instructions or truthfulness, and loss says nothing about those.</li></ul></div>`,
      },
      {
        title: "Hyperparameters and their usual effects", tag: "engineering", prov: "illus", body: () => `<p class="small muted">General guidance from practice and the literature; only the configuration in the first row was measured here.</p><div class="tbl-wrap"><table class="data"><tr><th>knob</th><th>this run</th><th>raising it usually…</th></tr>
        <tr><td>batch size B</td><td class="num">${tc.batch_size}</td><td>lowers gradient noise; allows a higher lr; costs activation memory linearly</td></tr><tr><td>context T</td><td class="num">${TRAINED.config.block_size}</td><td>gives each prediction more context; attention cost grows as T²</td></tr><tr><td>depth L / width D</td><td class="num">${TRAINED.config.n_layer} / ${TRAINED.config.n_embd}</td><td>adds capacity; needs more data and steps; D² terms dominate parameters</td></tr><tr><td>learning rate</td><td class="num">${tc.lr}</td><td>speeds early progress until it causes spikes or divergence</td></tr><tr><td>dropout</td><td class="num">${TRAINED.config.dropout}</td><td>reduces overfitting on small data; slows fitting</td></tr><tr><td>weight decay</td><td class="num">${tc.weight_decay}</td><td>shrinks weights toward 0; mild regularisation</td></tr></table></div>`,
      },
      {
        title: "Reproducibility and resuming", tag: "engineering", body: () => `<div class="prose"><ul><li><code>set_seed</code> seeds Python, NumPy and torch. A <b>dedicated</b> <code>torch.Generator</code> drives batch sampling, and its state is saved in the checkpoint, so a resumed run draws exactly the batches the original would have.</li><li><code>test_resume_is_bit_exact</code> trains 6 steps straight and 3 + 3 with a resume in between, and asserts every weight is identical. It passed (see Validation).</li><li>Limits: different hardware, thread counts, CUDA kernels or library versions can change floating-point summation order. For GPU determinism add <code>torch.use_deterministic_algorithms(True)</code> and accept the speed cost.</li></ul></div><div class="calc">python train.py                               # fresh run → out/ckpt.pt, out/log.jsonl, out/vocab.json
python train.py --resume                      # continue from out/ckpt.pt
python train.py --max_steps 8000 --resume     # extend (note: changes the cosine schedule's length)</div>`,
      },
    ];
  },
  quiz: [
    { id: "q-train-1", q: "A colleague reports character-level perplexity 3.9 and BPE perplexity 24 on the same text and concludes the character model is better. What is wrong?", explain: "Perplexity is per token, and the tokens differ. A BPE token covers several characters, so its perplexity is naturally higher. Convert both to bits per character (or per byte) before comparing." },
  ],
});

/* ================= Chapter 13: memory ================= */
const PRESETS = {
  micro: { V: 36, Tmax: 8, T: 8, L: 1, H: 2, D: 8, ff: 4, B: 1 },
  train: { V: 66, Tmax: 128, T: 128, L: 4, H: 4, D: 128, ff: 4, B: 32 },
  gpt2s: { V: 50257, Tmax: 1024, T: 1024, L: 12, H: 12, D: 768, ff: 4, B: 8 },
  b1: { V: 50257, Tmax: 2048, T: 2048, L: 24, H: 16, D: 2048, ff: 4, B: 4 },
};
function memEst(m) {
  const { V, Tmax, T, L, H, D, B } = m, Dff = m.ff * D, ab = m.ab, pb = m.pb, ob = m.ob;
  const perLayer = 4 * D + 4 * (D * D + D) + (D * Dff + Dff) + (Dff * D + D);
  const P = V * D + Tmax * D + L * perLayer + 2 * D + (m.tie ? 0 : V * D);
  const BTD = B * T * D, att = B * H * T * T;
  const layerAct = (8 * BTD + 2 * B * T * Dff) * ab + (m.attn === "explicit" ? att * ab + (m.drop ? att * ab + att : 0) : B * H * T * 4) + (m.drop ? 2 * BTD : 0) + 4 * B * T * 4;
  const headAct = (3 * BTD) * ab + B * T * V * ab + B * T * V * 4 + (m.drop ? BTD : 0) + B * T * 8;
  const acts = m.ckpt ? L * BTD * ab + layerAct + headAct : L * layerAct + headAct;
  const temp = Math.max(m.attn === "explicit" ? 2 * att * ab : 0, 2 * B * T * V * 4);
  const items = [
    { k: "Parameters", b: P * pb, f: `P × ${pb} B`, kind: "par" },
    { k: "Gradients", b: P * pb, f: `P × ${pb} B (same dtype as parameters)`, kind: "grad" },
    { k: "AdamW states (m, v)", b: 2 * P * ob, f: `2 × P × ${ob} B`, kind: "grad" },
    { k: "FP32 master weights", b: m.master ? P * 4 : 0, f: m.master ? "P × 4 B" : "off", kind: "par" },
    { k: "Saved activations", b: acts - (m.attn === "explicit" ? (m.ckpt ? 1 : L) * (att * ab + (m.drop ? att * ab + att : 0)) : 0), f: m.ckpt ? "L·BTD (block inputs) + one block recomputed + head" : "per block ≈ (8·BTD + 2·BT·Dff)·a + masks + LN stats; head: 3·BTD·a + BTV·(a+4)", kind: "act" },
    { k: "Attention intermediates", b: m.attn === "explicit" ? (m.ckpt ? 1 : L) * (att * ab + (m.drop ? att * ab + att : 0)) : 0, f: m.attn === "explicit" ? `${m.ckpt ? "1" : "L"} × B·H·T² × (${ab}${m.drop ? ` + ${ab} + 1` : ""}) B` : "not materialised (log-sum-exp stats only)", kind: "act" },
    { k: "Temporary buffers (peak)", b: temp, f: "max(2·B·H·T²·a, 2·B·T·V·4) — rough", kind: "tgt" },
  ];
  const sub = items.reduce((a, i) => a + i.b, 0);
  items.push({ k: `Allocator overhead (${m.overhead}%)`, b: sub * m.overhead / 100, f: "fragmentation and reserved-but-unused blocks; assumption", kind: "in" });
  const total = sub * (1 + m.overhead / 100);
  const flopsTok = 2 * (L * (4 * D * D + 2 * D * Dff) + V * D) + 4 * L * T * D;
  return { P, items, total, perLayer, flopsTrainStep: 3 * flopsTok * B * T * m.accum, attnShare: (4 * L * T * D) / flopsTok, Dff };
}
function memControls(prefix) {
  const m = S.mem;
  const num = (k, label, min, max, step = 1) => `<label for="m-${k}">${label} <input type="number" id="m-${k}" data-bind="mem" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${m[k]}" style="width:${String(max).length + 3}ch"></label>`;
  return `${segC("memPreset", [["micro", "micro"], ["train", "this run"], ["gpt2s", "GPT-2 small-like"], ["b1", "~1.3B-like"]], m.preset, "preset")}
  <div style="flex-basis:100%;height:0"></div>${num("V", "V", 1, 300000)}${num("Tmax", "T_max", 1, 131072)}${num("T", "T", 1, 131072)}${num("L", "L", 1, 200)}${num("H", "H", 1, 256)}${num("D", "D", 1, 32768)}${num("ff", "Dff/D", 1, 16)}${num("B", "B (micro)", 1, 4096)}${num("accum", "accum k", 1, 1024)}
  <div style="flex-basis:100%;height:0"></div>${selectC("mem-pb", "params/grads", [[4, "fp32"], [2, "bf16/fp16"]], m.pb)}${selectC("mem-ab", "activations", [[4, "fp32"], [2, "bf16/fp16"]], m.ab)}${selectC("mem-ob", "optimizer states", [[4, "fp32"], [2, "bf16"], [1, "8-bit"]], m.ob)}${checkC("mem-master", "fp32 master copy", m.master)}${checkC("mem-tie", "tie weights", m.tie)}${checkC("mem-ckpt", "activation checkpointing", m.ckpt)}${checkC("mem-drop", "dropout", m.drop)}${selectC("mem-attn", "attention", [["explicit", "explicit (this code)"], ["efficient", "memory-efficient / flash"]], m.attn)}${selectC("mem-gpu", "GPU", [[8, "8 GB"], [16, "16 GB"], [24, "24 GB"], [40, "40 GB"], [80, "80 GB"]], m.gpu)}${num("overhead", "overhead %", 0, 100)}`;
}
BINDS.mem = (t) => { const v = +t.value; if (Number.isFinite(v) && v >= 0) { S.mem[t.dataset.k] = v; S.mem.preset = "custom"; refresh({ compute: false }); } };
["pb", "ab", "ob", "gpu"].forEach((k) => (BINDS["mem-" + k] = (t) => { S.mem[k] = +t.value; refresh({ compute: false }); }));
BINDS["mem-attn"] = (t) => { S.mem.attn = t.value; refresh({ compute: false }); };
["master", "tie", "ckpt", "drop"].forEach((k) => (BINDS["mem-" + k] = (t) => { S.mem[k] = t.checked; refresh({ compute: false }); }));
ACT.memPreset = (v) => { Object.assign(S.mem, PRESETS[v], { preset: v }); refresh({ compute: false, full: true }); };
function memBar(r, gpuGB) {
  const cap = gpuGB * 1e9, scale = Math.max(r.total, cap);
  const col = { par: "var(--k-par)", grad: "var(--k-grad)", act: "var(--k-act)", tgt: "var(--k-tgt)", in: "var(--k-in)" };
  return `<div style="position:relative;height:30px;border-radius:6px;overflow:hidden;background:var(--surface-2);display:flex">${r.items.filter((i) => i.b > 0).map((i, k) => `<div title="${esc(i.k)}: ${fmtBytes(i.b)}" style="width:${(100 * i.b / scale).toFixed(3)}%;background:${col[i.kind]};opacity:${0.55 + 0.45 * ((k % 2) ? 0.6 : 1)}"></div>`).join("")}<div style="position:absolute;left:${(100 * cap / scale).toFixed(2)}%;top:0;bottom:0;border-left:2px dashed var(--ink)" title="GPU capacity"></div></div><div class="small muted" style="margin-top:4px">Dashed line: ${gpuGB} GB GPU. Total ${fmtBytes(r.total)} ${r.total > cap ? '<span class="pill bad">does not fit</span>' : '<span class="pill ok">fits</span>'}</div>`;
}
function memTable(r) {
  return `<div class="tbl-wrap"><table class="data"><tr><th>component</th><th>assumption</th><th class="num">bytes</th></tr>${r.items.map((i) => `<tr><td>${kindTag(i.kind).replace(/<b>.*<\/b>[^<]*/, `<b>${KINDS[i.kind][1]}</b>`)} ${i.k}</td><td class="small">${i.f}</td><td class="num">${fmtBytes(i.b)}</td></tr>`).join("")}<tr><td colspan="2"><b>Total</b></td><td class="num"><b>${fmtBytes(r.total)}</b></td></tr></table></div>`;
}
ch({
  id: "memory", num: 13, short: "GPU memory", title: "GPU memory and compute", group: "Training", kind: "train",
  thesis: "Training memory is parameters + gradients + optimizer state, which scale with model size, plus activations, which scale with batch × context and, for explicit attention, with context squared.",
  files: ["measure_memory.py", "train.py"], ops: ["mem_measure", "micro_loop"],
  card: {
    problem: "Predict whether a configuration fits on a GPU and which knob relieves the pressure.",
    ml: "The same accounting you would do for a large CNN, plus a T² term that CNNs do not have.",
    op: "bytes ≈ P(p + g + 2o [+4]) + activations(B, T, D, H, L, V) + temporaries + overhead",
    shapes: () => `scores ${shape(["B", "H", "T", "T"])} per block vs activations ${shape(["B", "T", "D"])}`,
    params: "Not applicable.",
    changes: "Nothing; this is an estimate.",
    fixed: "The stated byte-count assumptions.",
    failures: "Assuming a universal bytes-per-parameter constant; forgetting optimizer state; forgetting the T² attention term; reading allocated instead of reserved memory.",
    verify: "Measure with torch.cuda.max_memory_allocated() / max_memory_reserved() for one step and compare.",
  },
  sections: () => {
    const r = memEst(S.mem), m = S.mem;
    const Ts = [128, 256, 512, 1024, 2048, 4096, 8192];
    const series = (attn) => Ts.map((T) => [T, memEst({ ...m, T, Tmax: Math.max(m.Tmax, T), attn }).total / 2 ** 30]);
    return [
      { title: "Estimator", tag: "numerical", prov: "analytic", ctrl: () => memControls(), body: () => `${memBar(r, m.gpu)}<div style="margin-top:12px">${memTable(r)}</div><p class="small muted" style="margin-top:6px">P = ${fmtInt(r.P)} unique parameters${m.tie ? " (tied)" : ""}. Effective batch = B × k = ${fmtInt(m.B * m.accum)}; accumulation does not change the peak, which is set by one microbatch.</p>` },
      { title: "Scaling with context length", tag: "numerical", prov: "analytic", body: () => `${lineChart([{ name: "explicit attention (this code)", color: "var(--k-grad)", dots: true, points: series("explicit") }, { name: "memory-efficient attention", color: "var(--k-act)", dots: true, points: series("efficient") }], { xLabel: "context length T (other settings as above)", yLabel: "estimated total (GiB)", h: 230, fmtY: (v) => fmt(v, 2) + " GiB" })}<p class="small">Explicit attention keeps a B·H·T² probability tensor per block (plus its dropout mask), so doubling T quadruples that term. Memory-efficient kernels (FlashAttention, PyTorch's <code>scaled_dot_product_attention</code> backends) compute softmax in tiles and never store the full matrix in high-bandwidth memory; they still do O(T²) arithmetic, which is ${(100 * r.attnShare).toFixed(1)}% of forward FLOPs at the current settings.</p>` },
      {
        title: "Assumptions, stated", tag: "engineering", body: () => `<div class="prose"><ul>
        <li>Byte counts come only from the dtype selectors: fp32 = 4, bf16/fp16 = 2, 8-bit optimizer = 1. Real mixed-precision recipes differ (some keep fp32 gradients, some keep fp32 master weights and bf16 params). There is no single correct "bytes per parameter"; with the defaults here it is ${(r.items.slice(0, 4).reduce((a, i) => a + i.b, 0) / r.P).toFixed(0)} bytes/parameter before activations.</li>
        <li>Saved activations approximate what autograd keeps for this implementation: LayerNorm inputs/outputs and statistics, Q/K/V, merged heads, the residual h1, FFN pre- and post-GELU, dropout masks (1 byte), the attention probabilities, logits and the fp32 log-softmax inside cross-entropy.</li>
        <li>Activation checkpointing keeps only each block's input and recomputes one block at a time during backward: memory drops from L blocks' activations to about one, for roughly one extra forward pass of compute.</li>
        <li>Temporaries and allocator overhead are rough. The CUDA caching allocator <b>reserves</b> more than it has <b>allocated</b>; reserved memory is what other processes cannot use.</li>
        <li>Compute: training ≈ 3 × forward FLOPs; forward ≈ 2 × (matmul parameters) + 4·L·T·D per token. This step: ${(r.flopsTrainStep / 1e12).toFixed(3)} TFLOP.</li></ul></div>`,
      },
      { title: "Measure instead of estimate", tag: "engineering", body: () => `<p>No GPU was available where this reference was built (2 CPU cores), so no measured CUDA numbers appear on this page. On a CUDA machine, <code>measure_memory.py</code> runs one real training step and reports peak allocated and reserved memory for any configuration:</p><div class="calc">python measure_memory.py --batch_size 32 --block_size 128
python measure_memory.py --batch_size 8 --block_size 1024 --n_layer 12 --n_head 12 --n_embd 768</div><p class="small">Compare its output with the estimator using the same settings. Expect the estimate to be in the right range, not exact.</p>` },
    ];
  },
  quiz: [
    { id: "q-mem-1", q: "Which change halves peak activation memory but keeps the effective batch size?", options: ["Halve T", "Halve the microbatch B and set grad_accum = 2", "Switch AdamW to SGD", "Tie the weights"], answer: 1, explain: "Activations scale with the microbatch; accumulation recovers the effective batch at the cost of two forward/backward passes per step. Halving T changes what the model learns; SGD saves optimizer state, not activations." },
  ],
});

/* ================= Chapter 14: checkpoint ================= */
ch({
  id: "ckpt", num: 14, short: "Checkpoints", title: "Persistence and reproducible reloading", group: "Shipping", kind: "infer",
  thesis: "A checkpoint for inference needs the weights, the architecture and the tokenizer. A checkpoint for resuming training also needs the optimizer's moments, the step, every random-number generator and the data identity.",
  files: ["checkpoint.py", "train.py", "tests/test_llm.py"], ops: ["ckpt_save", "ckpt_rng", "ckpt_load", "ckpt_infer", "ckpt_resume", "resume", "t_resume"],
  card: {
    problem: "Make a trained model reloadable, verifiable and resumable.",
    ml: "Like saving a scikit-learn pipeline with its preprocessor, plus optimizer state for warm restarts.",
    op: "torch.save(dict) atomically; torch.load(weights_only=True); load_state_dict",
    shapes: () => `${fmtInt(TRAINED.num_params)} parameters → ${fmtBytes(TRAINED.ckpt_file_bytes)} file`,
    params: "Serialised, not changed.",
    changes: "Nothing.",
    fixed: "Everything stored.",
    failures: "Saving only weights; losing the vocabulary; loading untrusted pickles; comparing reloaded logits in train mode (dropout).",
    verify: "Logits from original and reloaded model, eval mode, same input, compared with a tolerance; resume reproduces the uninterrupted run.",
  },
  sections: () => [
    {
      title: "What is inside out/ckpt.pt", tag: "tensor", prov: "recorded", body: () => `<div class="tbl-wrap"><table class="data"><tr><th>key</th><th>contents</th><th class="num">tensor bytes</th><th>inference</th><th>resume</th></tr>${TRAINED.ckpt_contents.map((c) => `<tr><td class="mono">${c.key}</td><td class="small" style="max-width:46ch;overflow-wrap:anywhere">${esc(c.what)}</td><td class="num">${c.bytes ? fmtBytes(c.bytes) : "—"}</td><td>${c.infer ? '<span class="pill ok">needed</span>' : '<span class="pill neutral">—</span>'}</td><td>${c.resume ? '<span class="pill ok">needed</span>' : "—"}</td></tr>`).join("")}</table></div>
      <p class="small muted" style="margin-top:6px">File size ${fmtBytes(TRAINED.ckpt_file_bytes)}. Optimizer state is twice the model because AdamW stores m and v for every parameter. Without RNG states, a resumed run draws different batches and dropout masks: it still trains, but it is no longer the same experiment.</p>`,
    },
    {
      title: "Reload verification", tag: "numerical", prov: "recorded", body: () => `<div class="calc">model.eval();  x = first 128 validation tokens
save_checkpoint(...) → load_for_inference(...) → reloaded.eval()
max |logits_original − logits_reloaded| = ${TRAINED.reload_max_abs_diff}</div>
      <p>${TRAINED.reload_max_abs_diff === 0 ? '<span class="pill ok">bitwise identical</span>' : `<span class="pill warn">${TRAINED.reload_max_abs_diff}</span>`} The test suite uses a tolerance of 1e-6 (absolute). Same code, device and dtype should give exactly equal results; across devices or library versions, float32 kernels may sum in a different order, so a tolerance around 1e-5 relative to logit scale is reasonable. Always compare in <code>eval()</code>: in train mode dropout makes two forward passes of the <i>same</i> model differ.</p>`,
    },
    {
      title: "Resume, verified", tag: "engineering", prov: "recorded", body: () => {
        const t = D.tests.find((x) => x.name === "test_resume_is_bit_exact");
        return `<p>${t ? `<span class="pill ${t.outcome === "passed" ? "ok" : "bad"}">${t.outcome}</span>` : ""} <code>test_resume_is_bit_exact</code>: a 6-step run and a 3-step run resumed for 3 more steps produce identical weights for all ${TRAINED.params.length} tensors (<code>torch.equal</code>, no tolerance).</p>
        <div class="callout"><b>Safe loading</b><code>torch.load(..., weights_only=True)</code> only reconstructs tensors and plain containers, never arbitrary Python objects, so a malicious checkpoint cannot run code on load. That is why the checkpoint stores the config and vocabulary as plain dicts and the torch version as a string.</div>`;
      },
    },
  ],
  quiz: [
    { id: "q-ck-1", q: "You saved only model.state_dict() at step 3,000 and resume with a fresh AdamW. What goes wrong?", explain: "AdamW restarts with m = v = 0 and step = 1, so bias correction and the per-parameter scales reset: the first steps are much larger than they would have been (often a loss spike). The learning-rate schedule, sampler RNG and dropout RNG also restart, so the run diverges from the original." },
  ],
});

/* ================= Chapter 15: inference ================= */
bindStr("genPrompt", () => { S.genOut = null; refresh({ compute: false }); });
bindNum("genTemp", () => refresh({ compute: false }));
function genState() {
  if (!S.genOut) {
    const ok = Array.from(S.genPrompt).every((c) => c in MSTOI && c !== "<unk>") && S.genPrompt.length > 0;
    S.genOut = ok ? textIds(S.genPrompt) : null; S.genLast = null;
  }
  return S.genOut;
}
function genStep(n) {
  const ids = genState(); if (!ids) return;
  const r = Eng.rng(S.genSeed * 1000 + ids.length);
  for (let k = 0; k < n; k++) {
    const ctx = ids.slice(-CFG.block_size);
    const f = Eng.forward(M.P, CFG, ctx, null);
    const z = f.logits[ctx.length - 1], temp = +S.genTemp;
    let next;
    if (temp <= 0) next = z.indexOf(Math.max(...z));
    else { const p = Eng.softmax(z.map((v) => v / temp)); let u = r(), c = 0; next = p.length - 1; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) { next = i; break; } } }
    S.genLast = { ctx, z, next, temp, pos: ctx.length - 1 };
    ids.push(next);
  }
}
ACT.gen = (v) => { if (v === "reset") S.genOut = null; else genStep(+v); refresh({ compute: false }); };
ch({
  id: "infer", num: 15, short: "Inference & endpoint", title: "Generating text and serving it", group: "Shipping", kind: "infer",
  thesis: "Generation runs the trained model forward, keeps only the last position's logits, picks one id, appends it, and repeats. Training predicts all positions in parallel; generation cannot, because each new input is the previous output.",
  files: ["generate.py", "serve.py"], ops: ["gen", "gen_crop", "gen_last", "gen_sample", "gen_append", "serve_schema", "serve_life", "serve_ep", "ckpt_infer"],
  card: {
    problem: "Turn a frozen checkpoint into text, locally and over HTTP.",
    ml: "model.predict() in a loop, feeding each prediction back as input.",
    op: "repeat: z = model(ids[-T_max:])[:, −1]; id ~ softmax(z/τ); ids.append(id)",
    shapes: () => `ids ${shape(["B", "T"])} → logits ${shape(["B", "T", "V"])} → last ${shape(["B", "V"])} → next ${shape(["B", "1"])}`,
    params: "Frozen; eval() mode, no_grad().",
    changes: "Only the sequence.",
    fixed: "Weights, tokenizer, T_max.",
    failures: "Forgetting eval() (dropout during generation); prompts with unseen characters; contexts longer than T_max without cropping.",
    verify: "Output length = prompt + max_new_tokens; all ids in [0, V); greedy output is deterministic.",
  },
  sections: () => {
    const ids = genState(), last = S.genLast;
    let gen = "";
    if (!ids) gen = `<div class="callout bad"><b>Prompt uses characters the micro model does not know</b>Allowed: ${MV.slice(1).map((c) => `<code>${esc(vis(c))}</code>`).join(" ")}</div>`;
    else {
      const T0 = Array.from(S.genPrompt).length, ctxStart = Math.max(0, ids.length - CFG.block_size);
      gen = `<div class="chips">${ids.map((id, i) => `<span class="chip ${i >= T0 ? "tgt" : ""} ${i < ctxStart ? "ghost" : ""}" title="${i < ctxStart ? "outside the context window (cropped)" : "inside context"}"><span class="c">${esc(mvis(id))}</span><span class="i">${id}</span></span>`).join("")}</div>
      <p class="small muted" style="margin-top:6px">Faded tokens fall outside the last T_max = ${CFG.block_size} positions and are no longer seen. Orange ids were generated.</p>`;
      if (last) gen += `<h3>Last step: final position logits → next id</h3>${bars(last.temp > 0 ? Eng.softmax(last.z.map((v) => v / last.temp)) : last.z, { labels: MV.map(vis), hi: (i) => i === last.next, hiKind: "tgt", kind: "act", h: 140, w: 680 })}<p class="small">${last.temp > 0 ? `softmax(z / ${last.temp}) sampled → '${esc(mvis(last.next))}'` : `greedy argmax of the logits → '${esc(mvis(last.next))}'`} (context “${esc(visStr(last.ctx.map((i) => MV[i]).join("")))}”, read at position ${last.pos}).</p>`;
    }
    return [
      {
        title: "Generate with the micro model, here", tag: "numerical", prov: "browser",
        ctrl: () => `<label>prompt <input type="text" id="g-prompt" data-bind="genPrompt" value="${esc(S.genPrompt)}" style="width:14ch;font-family:var(--f-mono)"></label>${rangeC("genTemp", "temperature", 0, 1.5, S.genTemp, 0.1)}<button class="btn small primary" data-act="gen" data-arg="1">Next token</button><button class="btn small" data-act="gen" data-arg="20">+20</button><button class="btn small ghost" data-act="gen" data-arg="reset">Reset</button>`,
        body: () => `${gen}<p class="small muted">The micro model memorised a 250-character excerpt, so it mostly continues that excerpt. Temperature 0 is greedy and deterministic.</p>`,
      },
      {
        title: "Why generation is sequential", tag: "math", body: () => `<div class="eq">training:   one forward pass → logits for all T positions at once (teacher forcing)
generation: id<sub>n+1</sub> = sample( softmax( f(id<sub>1..n</sub>)<sub>last</sub> / τ ) )   — needs id<sub>n</sub> before it can compute id<sub>n+1</sub></div>
        <p>This implementation recomputes the whole (cropped) context for every new token, so producing n tokens costs n full forward passes. Caching each layer's keys and values (a KV cache) avoids recomputing past positions; it is a later topic. With learned absolute positions, the model has no embedding for position T_max, so <code>generate</code> keeps only the last T_max tokens: earlier text is simply forgotten, and the remaining tokens shift to new positions each step.</p>`,
      },
      {
        title: "Samples from the trained model", tag: "numerical", prov: "recorded", body: () => TRAINED.samples.map((s) => `<h3>prompt ${JSON.stringify(s.prompt)} · τ = ${s.temperature}${s.temperature === 0 ? " (greedy)" : ""} · seed ${s.seed}</h3><pre class="calc" style="white-space:pre-wrap">${esc(s.text)}</pre>`).join("") + `<p class="small">Recognisable structure (speaker tags, line lengths, archaic word shapes) with no meaning. Greedy decoding tends to loop; sampling at lower temperature trades diversity for fewer misspellings. This is what an ${fmtInt(TRAINED.num_params)}-parameter character model trained for ${(TRAINED.log.filter((r) => r.type === "eval").pop().elapsed_s / 60).toFixed(0)} CPU-minutes does; it is not an assistant.</p>`,
      },
      {
        title: "A minimal HTTP endpoint", tag: "engineering", prov: "recorded", body: () => {
          const sv = TRAINED.serve;
          return `<div class="calc">pip install -r requirements.txt
CKPT=out/ckpt.pt uvicorn serve:app --port 8000

curl -s localhost:8000/generate -H 'content-type: application/json' \\
  -d '${JSON.stringify(sv.request)}'</div>
          <div class="grid2" style="margin-top:10px"><div><h3 style="margin-top:0">Recorded response (200)</h3><pre class="calc" style="white-space:pre-wrap">${esc(JSON.stringify(sv.response, null, 2))}</pre></div><div><h3 style="margin-top:0">Unknown character (${sv.bad_response.status})</h3><pre class="calc" style="white-space:pre-wrap">${esc(JSON.stringify(sv.bad_response.body, null, 2))}</pre></div></div>
          <div class="prose"><ul><li>The model loads once in the <code>lifespan</code> handler, not per request.</li><li>Pydantic validates lengths and ranges: prompt 1–1,000 characters, 1–1,000 new tokens, temperature 0–2; violations return 422 automatically.</li><li>Strict tokenization turns unknown characters into a clear 422 listing them, instead of silently generating from <code>&lt;unk&gt;</code>.</li><li>Educational only: one request at a time on CPU, no batching, no streaming, no auth, no rate limiting.</li></ul></div>`;
        },
      },
      {
        title: "Deliberately deferred", tag: "engineering", body: () => `<p>KV caching, request batching and continuous batching, quantization (int8/int4 weights), speculative decoding, and advanced sampling (top-p, repetition penalties) all build directly on the loop above. They change speed and cost, not the model's mathematics.</p>`,
      },
    ];
  },
  quiz: [
    { id: "q-inf-1", q: "Why does generate() slice logits[:, -1, :]?", options: ["To save memory", "Only the final position's prediction is the next token; earlier positions predict tokens we already have", "Because the first positions are masked", "To apply temperature"], answer: 1, explain: "Position t predicts token t+1. All positions except the last predict tokens already in the sequence." },
  ],
});

/* ================= Beyond the baseline ================= */
ch({
  id: "alts", short: "Beyond the baseline", title: "Beyond the baseline", group: "Extensions", kind: "ref",
  thesis: "Modern LLMs change several components. Each alternative below is kept separate from the baseline: different equations, different code, none of it used anywhere else in this reference.",
  files: ["tests/test_llm.py", "model.py"], ops: ["scores", "softmax", "av"],
  sections: () => [
    { title: "Tokenization: characters → BPE / unigram", tag: "math", body: () => `<div class="eq">baseline: V = 66, one token per character
BPE (Sennrich et al., 2016): start from symbols, repeatedly merge the most frequent adjacent pair
Unigram LM (Kudo, 2018): start large, prune tokens whose removal least hurts corpus likelihood</div><p>Effects: sequences become 3–4× shorter for English, V grows to 32k–200k, so the embedding table and the output projection (V·D each) become a large share of parameters and the softmax over V dominates the head's compute. Byte-level variants guarantee every string is representable. The tokenizer chapter shows a working BPE on this corpus.</p>` },
    { title: "Positions: learned absolute → sinusoidal → RoPE", tag: "math", body: () => `<div class="eq">baseline:   x0 = E[x] + P[t],  P learned, t &lt; T_max
sinusoidal: P[t, 2i] = sin(t / 10000<sup>2i/D</sup>),  P[t, 2i+1] = cos(t / 10000<sup>2i/D</sup>)     (no parameters)
RoPE:       rotate each pair (q<sub>2i</sub>, q<sub>2i+1</sub>) of q and k by angle t·θ<sub>i</sub>,  θ<sub>i</sub> = 10000<sup>−2i/Dh</sup>
            ⇒ q<sub>i</sub>·k<sub>j</sub> depends on positions only through (i − j)</div><div class="calc"># NOT part of the baseline: a RoPE sketch applied to q, k of shape [B, H, T, Dh]
def rope(x, theta=10000.0):
    B, H, T, Dh = x.shape
    freqs = theta ** (-torch.arange(0, Dh, 2, device=x.device) / Dh)       # [Dh/2]
    ang = torch.arange(T, device=x.device)[:, None] * freqs[None, :]       # [T, Dh/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)</div><p>RoPE adds no parameters, injects position inside attention instead of into the residual stream, and is the default in most current open models (Su et al., 2021). Requires an even Dh.</p>` },
    { title: "Normalisation: LayerNorm → RMSNorm", tag: "math", body: () => `<div class="eq">LayerNorm: y = (x − μ) / √(σ² + ε) ⊙ γ + β
RMSNorm:   y = x / √(mean(x²) + ε) ⊙ γ          (no mean subtraction, no β; Zhang & Sennrich, 2019)</div><p>Cheaper and empirically as good in large models. Parameters per norm: D instead of 2D.</p>` },
    { title: "Feed-forward: GELU MLP → SwiGLU", tag: "math", body: () => `<div class="eq">baseline: FFN(x) = W2 · GELU(W1 x + b1) + b2                   2·D·Dff weights
SwiGLU:   FFN(x) = W2 · ( SiLU(W1 x) ⊙ (W3 x) )                 3·D·Dff weights  (Shazeer, 2020)
          Dff is usually set to ≈ 8D/3 so the parameter count matches 4D·2</div>` },
    { title: "Attention: multi-head → grouped-query", tag: "math", body: () => `<div class="eq">MHA (baseline): H query heads, H key heads, H value heads      W_K, W_V ∈ ℝ<sup>D×D</sup>
GQA:  H query heads share G &lt; H key/value heads (Ainslie et al., 2023)  W_K, W_V ∈ ℝ<sup>(G·Dh)×D</sup>
MQA:  G = 1</div><p>Mostly an inference optimisation: the KV cache shrinks by H/G. Quality stays close to MHA for moderate G.</p>` },
    { title: "Mapping the explicit attention to an optimised kernel", tag: "engineering", body: () => `<div class="calc"># explicit (model.py)                         # optimised equivalent
scores = (q @ k.transpose(-2, -1)) / sqrt(Dh)  y = F.scaled_dot_product_attention(
scores = scores.masked_fill(~mask, -inf)            q, k, v, is_causal=True,
probs  = softmax(scores, dim=-1)                    dropout_p=p if self.training else 0.0)
probs  = attn_dropout(probs)
y      = probs @ v</div><p><code>test_matches_pytorch_sdpa</code> confirms the two agree to 1e-6 in eval mode. Dropout masks are drawn differently, so training runs will not be bit-identical. On CUDA, SDPA dispatches to FlashAttention or memory-efficient kernels that never materialise the T×T matrix.</p>` },
    { title: "What this baseline teaches, and what it defers", tag: "engineering", body: () => `<div class="grid2"><div><h3 style="margin-top:0">Teaches well</h3><ul class="small"><li>Every tensor and shape of a decoder-only Transformer</li><li>Causal masking, the shift, and why leakage happens</li><li>Exact gradients and the AdamW update</li><li>Checkpoint completeness and verification</li><li>Memory accounting and the T² term</li></ul></div><div><h3 style="margin-top:0">Defers</h3><ul class="small"><li>Subword tokenizers and huge vocabularies</li><li>Mixed precision, loss scaling, distributed training (DDP/FSDP/tensor parallel)</li><li>Data pipelines at TB scale, deduplication, filtering</li><li>Scaling-law-driven sizing, long-context methods</li><li>Instruction tuning, preference optimisation, evaluation suites</li><li>KV cache, batching, quantization</li></ul></div></div>` },
  ],
});

/* ================= Architecture lab ================= */
ch({
  id: "lab", short: "Architecture lab", title: "Architecture laboratory", group: "Extensions", kind: "ref",
  thesis: "Change any dimension and see whether the configuration is valid, how many unique parameters it has, how much training memory it needs and which term dominates compute. These numbers update analytically; whether the model would be any good requires training it.",
  files: ["config.py", "model.py"], ops: ["num_params", "qkv_def", "ffn_def"],
  sections: () => {
    const m = S.mem, r = memEst(m), errs = [], warns = [];
    if (m.D % m.H !== 0) errs.push(`D = ${m.D} is not divisible by H = ${m.H}: the head split needs D = H × Dh (ModelConfig.validate raises).`);
    if (m.T > m.Tmax) errs.push(`T = ${m.T} exceeds T_max = ${m.Tmax}: the position table has no row for t ≥ T_max (GPT.forward raises).`);
    ["V", "Tmax", "T", "L", "H", "D", "ff", "B"].forEach((k) => { if (!(m[k] >= 1) || !Number.isInteger(m[k])) errs.push(`${k} must be a positive integer.`); });
    const Dh = m.D / m.H;
    if (!errs.length && Dh < 8) warns.push(`Head width Dh = ${Dh} is very small; each head's dot products have little room.`);
    if (!errs.length && Dh % 2) warns.push(`Dh = ${Dh} is odd; that is fine for this baseline but incompatible with RoPE.`);
    if (r.total > m.gpu * 1e9) warns.push(`Estimated ${fmtBytes(r.total)} exceeds the ${m.gpu} GB budget. Try activation checkpointing, memory-efficient attention, a smaller microbatch with accumulation, or bf16.`);
    const rows = [["tok_emb", "V·D", m.V * m.D], ["pos_emb", "T_max·D", m.Tmax * m.D], ["L × LayerNorm ×2", "L·4D", m.L * 4 * m.D], ["L × W_Q,W_K,W_V,W_O", "L·4(D²+D)", m.L * 4 * (m.D * m.D + m.D)], ["L × FFN", "L·(2·D·Dff + Dff + D)", m.L * (2 * m.D * r.Dff + r.Dff + m.D)], ["ln_f", "2D", 2 * m.D], ["lm_head", m.tie ? "tied to tok_emb: 0" : "V·D", m.tie ? 0 : m.V * m.D]];
    return [
      { title: "Configuration", tag: "numerical", prov: "analytic", ctrl: () => memControls(), body: () => `${errs.length ? errs.map((e) => `<div class="callout bad"><b>Invalid</b>${e}</div>`).join("") : '<div class="callout ok"><b>Valid configuration</b>All shape constraints of the baseline are satisfied.</div>'}${warns.map((w) => `<div class="callout warn"><b>Note</b>${w}</div>`).join("")}` },
      { title: "Unique parameters", tag: "tensor", prov: "analytic", body: () => `<div class="tbl-wrap"><table class="data"><tr><th>component</th><th>formula</th><th class="num">count</th><th class="num">share</th></tr>${rows.map(([a, b, c]) => `<tr><td>${a}</td><td class="mono small">${b}</td><td class="num">${fmtInt(c)}</td><td class="num">${(100 * c / r.P).toFixed(1)}%</td></tr>`).join("")}<tr><td colspan="2"><b>total</b></td><td class="num"><b>${fmtInt(r.P)}</b></td><td></td></tr></table></div><p class="small muted" style="margin-top:6px">With the "this run" preset this reproduces <code>model.num_params()</code> = ${fmtInt(TRAINED.num_params)} exactly.</p>` },
      { title: "Memory and compute", tag: "numerical", prov: "analytic", body: () => `${memBar(r, m.gpu)}<div class="grid2" style="margin-top:12px"><dl class="kv"><dt>train FLOPs / step</dt><dd>${(r.flopsTrainStep / 1e12).toFixed(3)} TFLOP</dd><dt>attention share of forward FLOPs</dt><dd>${(100 * r.attnShare).toFixed(1)}%</dd><dt>dominant memory term</dt><dd>${[...r.items].sort((a, b) => b.b - a.b)[0].k}</dd></dl>
      <p class="small">Matmul FLOPs grow with D² per token; attention FLOPs with T·D per token. Attention dominates when T is large relative to D (roughly T &gt; 6D for this layout). Quality claims ("deeper is better") cannot be computed here; they require retraining.</p></div>` },
    ];
  },
});

/* ================= Glossary ================= */
const GLOSSARY = [
  ["Activation", "A value computed in the forward pass (e.g. h1, logits). Not a parameter; recomputed every pass and kept only as long as backward needs it. Also used for the non-linearity itself (GELU)."],
  ["AdamW", "Adam with decoupled weight decay: per-parameter step lr·m̂/(√v̂+ε), plus θ ← θ(1 − lr·λ). Stores two moments per parameter."],
  ["Attention probabilities", "softmax over keys of the masked, scaled scores; shape [B, H, T, T]; each row sums to 1 before dropout."],
  ["Autograd", "PyTorch's reverse-mode automatic differentiation; builds a graph during the forward pass and applies the chain rule in backward()."],
  ["Batch size (B)", "Number of independent windows processed together. Distinct from sequence length T."],
  ["Bias correction", "Division of Adam's moments by (1 − β^t) to undo their initialisation at zero."],
  ["Bits per character", "loss / ln 2. Comparable across models that use the same characters, unlike per-token perplexity."],
  ["Block size (T_max)", "Maximum context length; number of rows in the learned position table."],
  ["Causal mask", "Lower-triangular boolean matrix; scores for keys j > i are set to −∞ so position i cannot see the future."],
  ["Checkpoint", "Serialised training state. For inference: weights + config + tokenizer. For resuming: also optimizer, step and RNG states."],
  ["Context length (T)", "Positions per sequence; the axis attention mixes along."],
  ["Cross-entropy", "−log p(target); averaged over all B·T positions. F.cross_entropy expects raw logits."],
  ["Decoder-only", "A Transformer with causal self-attention and no encoder; trained to predict the next token."],
  ["Dropout", "Randomly zeroes activations with probability p during training and scales the rest by 1/(1−p). Disabled by model.eval()."],
  ["Embedding", "A trainable table; row i is the vector for id i. Lookup is equivalent to one-hot × table."],
  ["Effective batch", "Microbatch size × gradient-accumulation steps."],
  ["Feed-forward network (FFN)", "Two linear layers with GELU between them, applied independently at every position."],
  ["GELU", "x·Φ(x); smooth activation used in GPT models."],
  ["Gradient", "∂loss/∂θ, stored in θ.grad with θ's shape."],
  ["Gradient accumulation", "Summing gradients over several microbatches (each loss divided by k) before one optimizer step."],
  ["Gradient clipping", "Rescaling all gradients together when their global norm exceeds a threshold."],
  ["Head (attention)", "One of H parallel attention computations on a Dh-wide slice of Q, K, V."],
  ["Key", "Per-position vector compared against queries to produce scores."],
  ["KV cache", "Storing past keys and values during generation so they are not recomputed. Not implemented here."],
  ["LayerNorm", "Normalises each position's D features to mean 0, variance 1, then applies γ, β. Independent of other examples."],
  ["Leakage", "Information from targets or held-out data reaching the model's inputs, making loss look better than it is."],
  ["Logits", "Unnormalised scores over the vocabulary, shape [B, T, V]."],
  ["Microbatch", "The batch processed in one forward/backward call; activation memory scales with it."],
  ["Parameter", "A trainable tensor (weights, biases, embeddings, γ, β) updated by the optimizer."],
  ["Perplexity", "exp(mean cross-entropy). Only comparable between models with the same tokenizer."],
  ["Pre-LN", "LayerNorm applied to each sub-layer's input rather than after the residual add."],
  ["Query", "Per-position vector that scores the keys it may attend to."],
  ["Residual stream", "The running [B, T, D] tensor that each block reads from and adds to."],
  ["Scaled dot-product attention", "softmax(QKᵀ/√Dh + mask) V."],
  ["Teacher forcing", "Conditioning each training prediction on the true previous tokens, enabling all positions to train in parallel."],
  ["Temperature", "Divides logits before sampling; < 1 sharpens, > 1 flattens, 0 means greedy."],
  ["Token id", "Integer index of a vocabulary entry. Categorical, never treated as a magnitude."],
  ["Value", "Per-position vector that is averaged with the attention weights."],
  ["Weight tying", "Using the token embedding matrix as the output projection; the shared tensor receives both gradients."],
  ["weights_only loading", "torch.load mode that only reconstructs tensors and plain containers, preventing code execution from untrusted files."],
];
ch({
  id: "glossary", short: "Glossary", title: "Glossary", group: "Reference", kind: "ref",
  thesis: "Precise definitions as used in this reference. Press / to search them along with chapters and code.",
  files: ["config.py"], ops: [],
  sections: () => [{ title: `${GLOSSARY.length} terms`, body: () => `<dl class="kv" style="grid-template-columns:minmax(150px,max-content) 1fr;gap:8px 18px">${GLOSSARY.map(([t, d]) => `<dt id="g-${slug(t)}" style="font-weight:600;color:var(--ink)">${t}</dt><dd style="font-family:var(--f-ui);font-size:13.5px">${d}</dd>`).join("")}</dl>` }],
});

/* ================= Sources ================= */
const BIB = [
  ["Vaswani et al. (2017). Attention Is All You Need.", "https://arxiv.org/abs/1706.03762", "Ch. 6–9"],
  ["Radford et al. (2018). Improving Language Understanding by Generative Pre-Training.", "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf", "Ch. 1, 10"],
  ["Radford et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "Ch. 5, 6, 11 (init, Pre-LN placement)"],
  ["Ba, Kiros & Hinton (2016). Layer Normalization.", "https://arxiv.org/abs/1607.06450", "Ch. 6"],
  ["Xiong et al. (2020). On Layer Normalization in the Transformer Architecture.", "https://arxiv.org/abs/2002.04745", "Ch. 6"],
  ["He et al. (2016). Deep Residual Learning for Image Recognition.", "https://arxiv.org/abs/1512.03385", "Ch. 6"],
  ["Hendrycks & Gimpel (2016). Gaussian Error Linear Units (GELUs).", "https://arxiv.org/abs/1606.08415", "Ch. 9"],
  ["Srivastava et al. (2014). Dropout.", "https://jmlr.org/papers/v15/srivastava14a.html", "Ch. 6, 8"],
  ["Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.", "https://arxiv.org/abs/1412.6980", "Ch. 11"],
  ["Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization.", "https://arxiv.org/abs/1711.05101", "Ch. 11"],
  ["Pascanu, Mikolov & Bengio (2013). On the difficulty of training recurrent neural networks.", "https://arxiv.org/abs/1211.5063", "Ch. 11 (clipping)"],
  ["Press & Wolf (2017). Using the Output Embedding to Improve Language Models.", "https://arxiv.org/abs/1608.05859", "Ch. 11 (tying)"],
  ["Sennrich, Haddow & Birch (2016). Neural Machine Translation of Rare Words with Subword Units.", "https://arxiv.org/abs/1508.07909", "Ch. 3, extensions"],
  ["Kudo (2018). Subword Regularization (unigram LM tokenizer).", "https://arxiv.org/abs/1804.10959", "Ch. 3, extensions"],
  ["Su et al. (2021). RoFormer: Rotary Position Embedding.", "https://arxiv.org/abs/2104.09864", "Extensions"],
  ["Zhang & Sennrich (2019). Root Mean Square Layer Normalization.", "https://arxiv.org/abs/1910.07467", "Extensions"],
  ["Shazeer (2020). GLU Variants Improve Transformer.", "https://arxiv.org/abs/2002.05202", "Extensions"],
  ["Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models.", "https://arxiv.org/abs/2305.13245", "Extensions"],
  ["Dao et al. (2022). FlashAttention.", "https://arxiv.org/abs/2205.14135", "Ch. 13"],
  ["Rabe & Staats (2021). Self-attention Does Not Need O(n²) Memory.", "https://arxiv.org/abs/2112.05682", "Ch. 13"],
  ["Chen et al. (2016). Training Deep Nets with Sublinear Memory Cost.", "https://arxiv.org/abs/1604.06174", "Ch. 13 (checkpointing)"],
  ["Kaplan et al. (2020). Scaling Laws for Neural Language Models.", "https://arxiv.org/abs/2001.08361", "Ch. 13 (FLOP approximation)"],
  ["Jain & Wallace (2019). Attention is not Explanation.", "https://arxiv.org/abs/1902.10186", "Ch. 8"],
  ["Karpathy, char-rnn (Tiny Shakespeare dataset).", "https://github.com/karpathy/char-rnn", "Ch. 2"],
  ["PyTorch: torch.nn.Linear", "https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html", "Ch. 7"],
  ["PyTorch: torch.nn.Embedding", "https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html", "Ch. 5"],
  ["PyTorch: torch.nn.LayerNorm", "https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html", "Ch. 6"],
  ["PyTorch: torch.nn.functional.cross_entropy", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html", "Ch. 10"],
  ["PyTorch: torch.nn.functional.scaled_dot_product_attention", "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", "Extensions"],
  ["PyTorch: torch.optim.AdamW", "https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html", "Ch. 11"],
  ["PyTorch: torch.nn.utils.clip_grad_norm_", "https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html", "Ch. 11"],
  ["PyTorch: Saving and loading models / torch.load weights_only", "https://docs.pytorch.org/docs/stable/generated/torch.load.html", "Ch. 14"],
  ["PyTorch: Reproducibility notes", "https://docs.pytorch.org/docs/stable/notes/randomness.html", "Ch. 12"],
  ["PyTorch: CUDA memory management", "https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management", "Ch. 13"],
  ["FastAPI: Lifespan events", "https://fastapi.tiangolo.com/advanced/events/", "Ch. 15"],
];
ch({
  id: "sources", short: "Sources", title: "Sources", group: "Reference", kind: "ref",
  thesis: "Primary papers for architectural claims and official documentation for API behaviour. API behaviour was checked against the installed PyTorch version listed in Validation.",
  files: ["model.py"], ops: [],
  sections: () => [{ title: "Bibliography", body: () => `<div class="tbl-wrap"><table class="data"><tr><th>source</th><th>used in</th></tr>${BIB.map(([t, u, c]) => `<tr><td><a href="${u}" target="_blank" rel="noopener">${esc(t)}</a></td><td class="small">${c}</td></tr>`).join("")}</table></div>` }],
});

/* ================= Validation ================= */
ch({
  id: "validate", short: "Validation report", title: "Validation report", group: "Reference", kind: "ref",
  thesis: "What was actually executed, what matched, and what could not be verified here.",
  files: ["tests/test_llm.py", "export_micro.py"], ops: ["t_tok", "t_win", "t_causal", "t_fd", "t_tie", "t_resume"],
  sections: () => [
    { title: "Browser engine vs PyTorch (live, computed now)", prov: "browser", body: () => `<div class="tbl-wrap"><table class="data"><tr><th>quantity (canonical micro example)</th><th class="num">max |Δ|</th><th></th></tr>${M.verify.map((r) => `<tr><td>${r.what}${r.note ? ` <span class="small muted">(worst: ${r.note})</span>` : ""}</td><td class="num">${fmt(r.diff, 2)}</td><td>${r.diff < 1e-5 ? '<span class="pill ok">pass</span>' : '<span class="pill warn">check</span>'}</td></tr>`).join("")}</table></div><p class="small muted" style="margin-top:6px">PyTorch computes in float32 (≈7 significant digits); the browser in float64. Differences of 10⁻⁷–10⁻⁶ are float32 rounding. Source operations resolved: ${Object.keys(OPS).length}/${OPS_DEF.length}${OP_ERRORS.length ? ` — <span class="pill bad">${OP_ERRORS.length} missing</span> ${esc(OP_ERRORS.join("; "))}` : ' <span class="pill ok">all</span>'}.</p>` },
    { title: "Python test suite", prov: "recorded", body: () => `<p class="small">${D.tests.filter((t) => t.outcome === "passed").length} of ${D.tests.length} passed · pytest · torch ${TRAINED.env.torch} · Python ${TRAINED.env.python} · CPU</p><div class="tbl-wrap"><table class="data"><tr><th>test</th><th>checks</th><th></th></tr>${D.tests.map((t) => `<tr><td class="mono small">${t.name}</td><td class="small">${esc(t.doc || "")}</td><td><span class="pill ${t.outcome === "passed" ? "ok" : "bad"}">${t.outcome}</span></td></tr>`).join("")}</table></div>` },
    { title: "Micro-model checks recorded in Python", prov: "recorded", body: () => `<div class="tbl-wrap"><table class="data"><tr><th>finite-difference check (float64)</th><th class="num">numeric</th><th class="num">autograd</th></tr>${MICRO.finite_difference.map((f) => `<tr><td class="mono small">${f.param}[${f.index.join(",")}]</td><td class="num">${f.numeric.toFixed(9)}</td><td class="num">${f.autograd.toFixed(9)}</td></tr>`).join("")}</table></div>` },
    { title: "HTML checks", prov: "recorded", body: () => D.htmlChecks ? `<p class="small">Automated with headless Chromium ${esc(D.htmlChecks.browser)} on the previous build of this file (identical except for this table).</p><div class="tbl-wrap"><table class="data">${D.htmlChecks.items.map((i) => `<tr><td>${esc(i.name)}</td><td class="small">${esc(i.detail || "")}</td><td><span class="pill ${i.ok ? "ok" : "bad"}">${i.ok ? "pass" : "fail"}</span></td></tr>`).join("")}</table></div>` : `<p>Not yet recorded.</p>` },
    { title: "Not verified here", body: () => `<div class="prose"><ul><li>No CUDA GPU was available: memory figures are analytical estimates, and <code>measure_memory.py</code> was not executed on a GPU.</li><li>Training ran once on CPU; run-to-run variance across seeds was not measured.</li><li>Screen-reader behaviour was not tested with assistive technology.</li></ul></div>` },
    { title: "Progress", body: () => `<p>Visited sections and answered checks are stored in this browser only.</p><button class="btn" data-act="reset-progress">Clear learning progress</button>` },
  ],
});

/* ================= Run locally ================= */
ch({
  id: "run", short: "Run it locally", title: "Run it locally", group: "Reference", kind: "ref",
  thesis: "The interactive page is one self-contained HTML file. The Python project next to it reproduces every recorded number.",
  files: ["README.md", "requirements.txt"], ops: [],
  sections: () => [
    { title: "Commands", body: () => `<div class="calc">python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
curl -L -o data/input.txt ${D.corpus.url}
shasum -a 256 data/input.txt        # ${D.corpus.sha256.slice(0, 16)}…

python -m pytest -q tests           # ${D.tests.length} checks
python train.py                     # ~${(TRAINED.log.filter((r) => r.type === "eval").pop().elapsed_s / 60).toFixed(0)} min on 2 CPU cores; much faster on a GPU
python train.py --resume            # continue from out/ckpt.pt
python generate.py --prompt "ROMEO:" --temperature 0.8
CKPT=out/ckpt.pt uvicorn serve:app --port 8000
python export_micro.py && python export_trained.py && python build.py   # regenerate this page</div>
    <div class="grid2" style="margin-top:12px"><dl class="kv"><dt>page</dt><dd>TransformerAtlas.html</dd><dt>checkpoint</dt><dd>out/ckpt.pt</dd><dt>training log</dt><dd>out/log.jsonl</dd><dt>vocabulary</dt><dd>out/vocab.json</dd><dt>recorded traces</dt><dd>web/data/*.json</dd></dl>
    <div><p class="small">Offline: every chapter, computation and recorded value works without a network. Only the Geist web fonts load from Google Fonts; without them the page falls back to your system fonts. External links in Sources need a connection.</p><button class="btn primary" data-act="download">Download source (.zip)</button></div></div>` },
  ],
});
