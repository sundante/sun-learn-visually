/* engine.js — real computations performed in the browser (float64).
 * A line-for-line port of model.py for the micro configuration (B=1), plus a
 * hand-written backward pass and torch.optim.AdamW's update rule.
 * On load the app compares these results with the values PyTorch recorded
 * (micro.json) and reports the maximum absolute difference. */
"use strict";
const Eng = (() => {
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const clone = (m) => (Array.isArray(m[0]) ? m.map((r) => r.slice()) : m.slice());
  const num = (v) => (v === "-inf" ? -Infinity : v === "inf" ? Infinity : v);
  const deepNum = (m) => (Array.isArray(m) ? m.map(deepNum) : num(m));

  // y[t][o] = sum_i x[t][i] * W[o][i] + b[o]   (nn.Linear: W is [out, in])
  function linear(X, W, b) {
    return X.map((row) => W.map((w, o) => {
      let s = b ? b[o] : 0;
      for (let i = 0; i < row.length; i++) s += row[i] * w[i];
      return s;
    }));
  }
  function layerNorm(X, g, b, eps) {
    const D = X[0].length, out = { y: [], xhat: [], rstd: [], mu: [], vr: [] };
    for (const x of X) {
      const mu = x.reduce((a, v) => a + v, 0) / D;
      const vr = x.reduce((a, v) => a + (v - mu) ** 2, 0) / D;   // biased variance, as PyTorch
      const rstd = 1 / Math.sqrt(vr + eps);
      const xhat = x.map((v) => (v - mu) * rstd);
      out.xhat.push(xhat); out.rstd.push(rstd); out.mu.push(mu); out.vr.push(vr);
      out.y.push(xhat.map((v, i) => v * g[i] + b[i]));
    }
    return out;
  }
  // erf via the Numerical Recipes erfc Chebyshev fit (fractional error < 1.2e-7)
  function erf(x) {
    const t = 1 / (1 + 0.5 * Math.abs(x));
    const y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? y : -y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const dgelu = (x) => 0.5 * (1 + erf(x / Math.SQRT2)) + x * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  function softmax(row) {
    const m = Math.max(...row);
    const e = row.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
    const s = e.reduce((a, v) => a + v, 0);
    return e.map((v) => v / s);
  }
  function logSumExp(row) {
    const m = Math.max(...row);
    return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0));
  }

  function forward(P, cfg, ids, targets, opts = {}) {
    const causal = opts.causal !== false;
    const T = ids.length, D = cfg.n_embd, H = cfg.n_head, Dh = D / H, eps = cfg.ln_eps;
    const pre = "blocks.0.";
    const tok = ids.map((i) => P["tok_emb.weight"][i].slice());
    const pos = P["pos_emb.weight"].slice(0, T).map((r) => r.slice());
    const x0 = tok.map((r, t) => r.map((v, i) => v + pos[t][i]));
    const ln1 = layerNorm(x0, P[pre + "ln1.weight"], P[pre + "ln1.bias"], eps);
    const q = linear(ln1.y, P[pre + "attn.q_proj.weight"], P[pre + "attn.q_proj.bias"]);
    const k = linear(ln1.y, P[pre + "attn.k_proj.weight"], P[pre + "attn.k_proj.bias"]);
    const v = linear(ln1.y, P[pre + "attn.v_proj.weight"], P[pre + "attn.v_proj.bias"]);
    const split = (M) => Array.from({ length: H }, (_, h) => M.map((r) => r.slice(h * Dh, (h + 1) * Dh)));
    const Q = split(q), K = split(k), Vh = split(v);
    const scale = 1 / Math.sqrt(Dh);
    const S = [], Sm = [], Pr = [], O = [];
    for (let h = 0; h < H; h++) {
      const s = zeros(T, T), sm = zeros(T, T);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        let d = 0;
        for (let c = 0; c < Dh; c++) d += Q[h][i][c] * K[h][j][c];
        s[i][j] = d * scale;
        sm[i][j] = causal && j > i ? -Infinity : s[i][j];
      }
      const p = sm.map(softmax);
      const o = p.map((row) => Array.from({ length: Dh }, (_, c) => row.reduce((a, w, j) => a + w * Vh[h][j][c], 0)));
      S.push(s); Sm.push(sm); Pr.push(p); O.push(o);
    }
    const merged = Array.from({ length: T }, (_, t) => O.flatMap((o) => o[t]));
    const attn = linear(merged, P[pre + "attn.out_proj.weight"], P[pre + "attn.out_proj.bias"]);
    const h1 = x0.map((r, t) => r.map((v2, i) => v2 + attn[t][i]));
    const ln2 = layerNorm(h1, P[pre + "ln2.weight"], P[pre + "ln2.bias"], eps);
    const fpre = linear(ln2.y, P[pre + "ffn.fc1.weight"], P[pre + "ffn.fc1.bias"]);
    const fact = fpre.map((r) => r.map(gelu));
    const fout = linear(fact, P[pre + "ffn.fc2.weight"], P[pre + "ffn.fc2.bias"]);
    const h2 = h1.map((r, t) => r.map((v2, i) => v2 + fout[t][i]));
    const lnf = layerNorm(h2, P["ln_f.weight"], P["ln_f.bias"], eps);
    const logits = linear(lnf.y, P["lm_head.weight"], null);
    const out = { ids, T, tok, pos, x0, ln1, q, k, v, Q, K, V: Vh, S, Sm, P: Pr, O, merged, attn, h1, ln2, fpre, fact, fout, h2, lnf, logits };
    if (targets) {
      out.targets = targets;
      out.lse = logits.map(logSumExp);
      out.nll = logits.map((z, t) => out.lse[t] - z[targets[t]]);
      out.loss = out.nll.reduce((a, v2) => a + v2, 0) / T;
      out.probs = logits.map(softmax);
    }
    return out;
  }

  // ---------- backward pass (mirrors what autograd computes) ----------
  const matTmul = (A, B) => { // A^T @ B : [a_cols x b_cols]
    const R = zeros(A[0].length, B[0].length);
    for (let t = 0; t < A.length; t++) for (let i = 0; i < A[0].length; i++) {
      const a = A[t][i];
      if (a !== 0) for (let j = 0; j < B[0].length; j++) R[i][j] += a * B[t][j];
    }
    return R;
  };
  const matmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, i) => s + v * B[i][j], 0)));
  const colSum = (A) => A[0].map((_, j) => A.reduce((s, r) => s + r[j], 0));
  function linearBack(dY, X, W) { // returns dX, dW, db
    return { dX: matmul(dY, W), dW: matTmul(dY, X), db: colSum(dY) };
  }
  function lnBack(dY, ln, g) {
    const D = g.length, dg = new Array(D).fill(0), db = new Array(D).fill(0), dX = [];
    dY.forEach((dy, t) => {
      const xh = ln.xhat[t];
      const dxh = dy.map((v, i) => v * g[i]);
      dy.forEach((v, i) => { dg[i] += v * xh[i]; db[i] += v; });
      const m1 = dxh.reduce((a, v) => a + v, 0) / D;
      const m2 = dxh.reduce((a, v, i) => a + v * xh[i], 0) / D;
      dX.push(dxh.map((v, i) => ln.rstd[t] * (v - m1 - xh[i] * m2)));
    });
    return { dX, dg, db };
  }
  function backward(P, cfg, f) {
    const G = {}, T = f.T, D = cfg.n_embd, H = cfg.n_head, Dh = D / H, pre = "blocks.0.";
    const dlogits = f.probs.map((p, t) => p.map((v, j) => (v - (j === f.targets[t] ? 1 : 0)) / T));
    G["lm_head.weight"] = matTmul(dlogits, f.lnf.y);
    const dlnf = matmul(dlogits, P["lm_head.weight"]);
    const lf = lnBack(dlnf, f.lnf, P["ln_f.weight"]);
    G["ln_f.weight"] = lf.dg; G["ln_f.bias"] = lf.db;
    const dh2 = lf.dX;
    const f2 = linearBack(dh2, f.fact, P[pre + "ffn.fc2.weight"]);
    G[pre + "ffn.fc2.weight"] = f2.dW; G[pre + "ffn.fc2.bias"] = f2.db;
    const dpre = f2.dX.map((r, t) => r.map((v, i) => v * dgelu(f.fpre[t][i])));
    const f1 = linearBack(dpre, f.ln2.y, P[pre + "ffn.fc1.weight"]);
    G[pre + "ffn.fc1.weight"] = f1.dW; G[pre + "ffn.fc1.bias"] = f1.db;
    const l2 = lnBack(f1.dX, f.ln2, P[pre + "ln2.weight"]);
    G[pre + "ln2.weight"] = l2.dg; G[pre + "ln2.bias"] = l2.db;
    const dh1 = dh2.map((r, t) => r.map((v, i) => v + l2.dX[t][i]));      // residual: gradient copies through
    const op = linearBack(dh1, f.merged, P[pre + "attn.out_proj.weight"]);
    G[pre + "attn.out_proj.weight"] = op.dW; G[pre + "attn.out_proj.bias"] = op.db;
    const dq = zeros(T, D), dk = zeros(T, D), dv = zeros(T, D), scale = 1 / Math.sqrt(Dh);
    for (let h = 0; h < H; h++) {
      const dO = op.dX.map((r) => r.slice(h * Dh, (h + 1) * Dh));
      const p = f.P[h];
      for (let i = 0; i < T; i++) {
        const dP = Array.from({ length: T }, (_, j) => dO[i].reduce((a, v, c) => a + v * f.V[h][j][c], 0));
        const dot = dP.reduce((a, v, j) => a + v * p[i][j], 0);
        for (let j = 0; j < T; j++) {
          const dS = p[i][j] * (dP[j] - dot) * scale;   // softmax Jacobian; masked entries have p=0
          for (let c = 0; c < Dh; c++) {
            dq[i][h * Dh + c] += dS * f.K[h][j][c];
            dk[j][h * Dh + c] += dS * f.Q[h][i][c];
            dv[j][h * Dh + c] += p[i][j] * dO[i][c];
          }
        }
      }
    }
    const lq = linearBack(dq, f.ln1.y, P[pre + "attn.q_proj.weight"]);
    const lk = linearBack(dk, f.ln1.y, P[pre + "attn.k_proj.weight"]);
    const lv = linearBack(dv, f.ln1.y, P[pre + "attn.v_proj.weight"]);
    G[pre + "attn.q_proj.weight"] = lq.dW; G[pre + "attn.q_proj.bias"] = lq.db;
    G[pre + "attn.k_proj.weight"] = lk.dW; G[pre + "attn.k_proj.bias"] = lk.db;
    G[pre + "attn.v_proj.weight"] = lv.dW; G[pre + "attn.v_proj.bias"] = lv.db;
    const da1 = lq.dX.map((r, t) => r.map((v, i) => v + lk.dX[t][i] + lv.dX[t][i]));
    const l1 = lnBack(da1, f.ln1, P[pre + "ln1.weight"]);
    G[pre + "ln1.weight"] = l1.dg; G[pre + "ln1.bias"] = l1.db;
    const dx0 = dh1.map((r, t) => r.map((v, i) => v + l1.dX[t][i]));
    const dE = zeros(P["tok_emb.weight"].length, D), dPos = zeros(P["pos_emb.weight"].length, D);
    f.ids.forEach((id, t) => dx0[t].forEach((v, i) => { dE[id][i] += v; dPos[t][i] += v; }));  // scatter-add
    G["tok_emb.weight"] = dE; G["pos_emb.weight"] = dPos;
    return { G, dx0, dlogits, dh1, dh2 };
  }

  const flat = (m) => (Array.isArray(m) ? m.flat(Infinity) : [m]);
  function totalNorm(G) {
    let s = 0;
    for (const k in G) for (const v of flat(G[k])) s += v * v;
    return Math.sqrt(s);
  }
  const mapDeep = (a, fn, b, c, d) => Array.isArray(a)
    ? a.map((x, i) => mapDeep(x, fn, b && b[i], c && c[i], d && d[i])) : fn(a, b, c, d);
  function clip(G, maxNorm) {
    const n = totalNorm(G), coef = maxNorm / (n + 1e-6);
    if (coef >= 1) return { G, norm: n, coef: 1 };
    const out = {};
    for (const k in G) out[k] = mapDeep(G[k], (v) => v * coef);
    return { G: out, norm: n, coef };
  }
  // torch.optim.AdamW (single-tensor path)
  function adamw(P, G, state, opt) {
    const [b1, b2] = opt.betas, eps = opt.eps, lr = opt.lr;
    const after = {}, m = {}, v = {};
    for (const k in P) {
      const wd = opt.weight_decay[k];
      const t = state[k].step + 1;
      const bc1 = 1 - b1 ** t, bc2 = 1 - b2 ** t;
      m[k] = mapDeep(state[k].exp_avg, (mm, g) => b1 * mm + (1 - b1) * g, G[k]);
      v[k] = mapDeep(state[k].exp_avg_sq, (vv, g) => b2 * vv + (1 - b2) * g * g, G[k]);
      after[k] = mapDeep(P[k], (p, mm, vv) => {
        const decayed = p * (1 - lr * wd);                       // decoupled weight decay
        return decayed - (lr / bc1) * mm / (Math.sqrt(vv) / Math.sqrt(bc2) + eps);
      }, m[k], v[k]);
    }
    return { after, m, v };
  }
  function maxAbsDiff(a, b) {
    const fa = flat(a), fb = flat(b);
    let mx = 0;
    for (let i = 0; i < fa.length; i++) {
      const x = num(fa[i]), y = num(fb[i]);
      if (x === y) continue;
      mx = Math.max(mx, Math.abs(x - y));
    }
    return mx;
  }
  // seeded PRNG (mulberry32) for illustrative sampling
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  return { zeros, clone, num, deepNum, linear, layerNorm, erf, gelu, dgelu, softmax, logSumExp,
    forward, backward, totalNorm, clip, adamw, maxAbsDiff, flat, rng, mapDeep };
})();

/* Character-level BPE trained in the browser (for the tokenizer comparison panel).
 * No byte fallback, no pre-tokenization; merges are learned over the whole excerpt. */
const BPE = {
  train(text, nMerges) {
    let seq = Array.from(text), merges = [];
    for (let m = 0; m < nMerges; m++) {
      const counts = new Map();
      for (let i = 0; i < seq.length - 1; i++) {
        const k = seq[i] + "\u0000" + seq[i + 1];
        counts.set(k, (counts.get(k) || 0) + 1);
      }
      let best = null, bc = 1;
      for (const [k, c] of counts) if (c > bc) { best = k; bc = c; }
      if (!best) break;
      const [a, b] = best.split("\u0000");
      merges.push([a, b, bc]);
      const ns = [];
      for (let i = 0; i < seq.length; i++) {
        if (i < seq.length - 1 && seq[i] === a && seq[i + 1] === b) { ns.push(a + b); i++; } else ns.push(seq[i]);
      }
      seq = ns;
    }
    return merges;
  },
  encode(text, merges) {
    let seq = Array.from(text);
    for (const [a, b] of merges) {
      const ns = [];
      for (let i = 0; i < seq.length; i++) {
        if (i < seq.length - 1 && seq[i] === a && seq[i + 1] === b) { ns.push(a + b); i++; } else ns.push(seq[i]);
      }
      seq = ns;
    }
    return seq;
  },
};

/* Minimal store-only ZIP writer (no compression) for "Download source". */
const Zip = (() => {
  const table = new Uint32Array(256).map((_, n) => {
    let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0;
  });
  const crc32 = (b) => { let c = 0xffffffff; for (const x of b) c = table[(c ^ x) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  function make(files) { // files: [{name, text}]
    const enc = new TextEncoder(), parts = [], central = [];
    let offset = 0;
    for (const f of files) {
      const name = enc.encode(f.name), data = enc.encode(f.text), crc = crc32(data);
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true);
      h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true);
      h.setUint16(26, name.length, true);
      parts.push(new Uint8Array(h.buffer), name, data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
      c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true);
      c.setUint16(28, name.length, true); c.setUint32(42, offset, true);
      central.push(new Uint8Array(c.buffer), name);
      offset += 30 + name.length + data.length;
    }
    const csize = central.reduce((a, p) => a + p.length, 0);
    const e = new DataView(new ArrayBuffer(22));
    e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
    e.setUint32(12, csize, true); e.setUint32(16, offset, true);
    return new Blob([...parts, ...central, new Uint8Array(e.buffer)], { type: "application/zip" });
  }
  return { make };
})();

/* Tiny Python syntax highlighter (line based, tracks triple-quoted strings). */
const PyHL = (() => {
  const KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  const BI = new Set("self super len range int float str list dict tuple set print min max sum zip enumerate isinstance getattr hasattr abs round open type sorted any all".split(" "));
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function line(src, state) {
    let out = "", i = 0;
    if (state.inDoc) {
      const end = src.indexOf(state.inDoc);
      if (end < 0) return { html: `<span class="t-str">${esc(src)}</span>`, state };
      out += `<span class="t-str">${esc(src.slice(0, end + 3))}</span>`; i = end + 3; state = { inDoc: null };
    }
    const re = /(#.*$)|("""|''')|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@\w+)|(\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)|([A-Za-z_]\w*)/g;
    re.lastIndex = i;
    let m, last = i;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      const [tok] = m;
      if (m[1]) out += `<span class="t-com">${esc(tok)}</span>`;
      else if (m[2]) {
        const end = src.indexOf(tok, m.index + 3);
        if (end < 0) { out += `<span class="t-str">${esc(src.slice(m.index))}</span>`; return { html: out, state: { inDoc: tok } }; }
        out += `<span class="t-str">${esc(src.slice(m.index, end + 3))}</span>`; re.lastIndex = end + 3; last = end + 3; continue;
      } else if (m[3]) out += `<span class="t-str">${esc(tok)}</span>`;
      else if (m[4]) out += `<span class="t-dec">${esc(tok)}</span>`;
      else if (m[5]) out += `<span class="t-num">${tok}</span>`;
      else if (KW.has(tok)) out += `<span class="t-kw">${tok}</span>`;
      else if (BI.has(tok)) out += `<span class="t-bi">${tok}</span>`;
      else if (/^(torch|nn|F|math|np)$/.test(tok)) out += `<span class="t-mod">${tok}</span>`;
      else out += esc(tok);
      last = re.lastIndex;
    }
    out += esc(src.slice(last));
    return { html: out, state };
  }
  function highlight(code) {
    let state = { inDoc: null };
    return code.split("\n").map((l) => { const r = line(l, state); state = r.state; return r.html; });
  }
  return { highlight, esc };
})();
