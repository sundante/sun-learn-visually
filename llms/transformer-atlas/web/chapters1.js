/* chapters1.js — operation registry and chapters 1–8 */
"use strict";

/* Every op is resolved against the canonical source at load time; a missing anchor is reported
   in the Validation chapter and fails the automated HTML check. */
const OPS_DEF = [
  // data
  { id: "load_text", file: "data.py", find: "def load_text", n: 5, label: "Read UTF-8 and normalise line endings" },
  { id: "split", file: "data.py", find: "def split_text", n: 7, label: "Contiguous train/val/test split" },
  { id: "train_split", file: "train.py", find: "text = load_text(tc.data_path)", n: 5, label: "Split first, then build a train-only vocabulary" },
  { id: "get_batch", file: "data.py", find: "def get_batch", n: 12, label: "Sample (x, y) windows shifted by one" },
  { id: "seq_windows", file: "data.py", find: "def sequential_windows", n: 4, label: "Deterministic windows with a stride" },
  // tokenizer
  { id: "tok_build", file: "tokenizer.py", find: "def build(cls", n: 2, label: "Build vocabulary from training text" },
  { id: "tok_init", file: "tokenizer.py", find: "self.chars = list(chars)", n: 3, label: "Id 0 = <unk>; characters from id 1" },
  { id: "tok_encode", file: "tokenizer.py", find: "def encode(self", n: 5, label: "encode(): characters → ids" },
  { id: "tok_decode", file: "tokenizer.py", find: "def decode(self", n: 8, label: "decode(): ids → characters" },
  { id: "tok_save", file: "tokenizer.py", find: "def to_dict(self)", n: 2, label: "Vocabulary serialisation" },
  // embeddings
  { id: "emb_def", file: "model.py", find: "self.tok_emb = nn.Embedding", n: 2, label: "Token [V, D] and position [T_max, D] tables" },
  { id: "emb_fwd", file: "model.py", find: "pos = torch.arange(T", n: 3, label: "Row lookup + broadcast position add" },
  // block
  { id: "block_def", file: "model.py", find: "self.ln1 = nn.LayerNorm", n: 4, label: "Block submodules" },
  { id: "ln1_call", file: "model.py", find: "a = self.ln1(x)", n: 1, label: "LayerNorm 1 (pre-attention)" },
  { id: "res_attn", file: "model.py", find: "h1 = x + self.attn(a, trace)", n: 1, label: "Residual add around attention" },
  { id: "ln2_call", file: "model.py", find: "f = self.ln2(h1)", n: 1, label: "LayerNorm 2 (pre-FFN)" },
  { id: "res_ffn", file: "model.py", find: "h2 = h1 + self.ffn(f, trace)", n: 1, label: "Residual add around FFN" },
  { id: "emb_drop", file: "model.py", find: "x = self.drop(tok + self.pos_emb(pos))", n: 1, label: "Dropout on the embedding sum" },
  { id: "blocks_loop", file: "model.py", find: "for i, block in enumerate(self.blocks):", n: 5, label: "Stack of L blocks" },
  // attention
  { id: "qkv_def", file: "model.py", find: "self.q_proj = nn.Linear", n: 4, label: "Q, K, V, O projections ([out, in] storage)" },
  { id: "qkv_proj", file: "model.py", find: "q = self.q_proj(x)", n: 3, label: "Project to Q, K, V" },
  { id: "head_split", file: "model.py", find: "q = q.view(B, T, H, Dh)", n: 3, label: "Split heads: [B,T,D] → [B,H,T,Dh]" },
  { id: "scores", file: "model.py", find: "scores = (q @ k.transpose(-2, -1))", n: 1, label: "Scores S = QKᵀ / √Dh" },
  { id: "mask_buf", file: "model.py", find: "mask = torch.tril", n: 2, label: "Lower-triangular causal mask buffer" },
  { id: "mask_apply", file: "model.py", find: "if self.causal:", n: 3, label: "Mask future keys with −∞" },
  { id: "softmax", file: "model.py", find: "probs = F.softmax(scores, dim=-1)", n: 1, label: "Softmax over keys (last axis)" },
  { id: "attn_drop", file: "model.py", find: "probs = self.attn_dropout(probs)", n: 1, label: "Attention dropout (training only)" },
  { id: "av", file: "model.py", find: "y = probs @ v", n: 1, label: "Weighted sum of values" },
  { id: "merge", file: "model.py", find: "y = y.transpose(1, 2).contiguous().view(B, T, D)", n: 1, label: "Merge heads back to [B,T,D]" },
  { id: "out_proj", file: "model.py", find: "y = self.resid_dropout(self.out_proj(y))", n: 1, label: "Output projection + residual dropout" },
  // ffn
  { id: "ffn_def", file: "model.py", find: "self.fc1 = nn.Linear(cfg.n_embd, cfg.d_ff", n: 4, label: "FFN: D → Dff → D" },
  { id: "ffn_fwd", file: "model.py", find: "h = self.fc1(x)", n: 5, label: "FFN forward with GELU" },
  // head + loss
  { id: "lnf", file: "model.py", find: "x = self.ln_f(x)", n: 1, label: "Final LayerNorm" },
  { id: "lm_head", file: "model.py", find: "self.lm_head = nn.Linear", n: 3, label: "Vocabulary projection (optional tying)" },
  { id: "logits", file: "model.py", find: "logits = self.lm_head(x)", n: 1, label: "Logits [B,T,V]" },
  { id: "ce", file: "model.py", find: "if targets is not None:", n: 3, label: "Flatten and cross-entropy on raw logits" },
  { id: "init", file: "model.py", find: "def _init_weights(m):", n: 8, label: "Weight initialisation" },
  { id: "num_params", file: "model.py", find: "def num_params(self)", n: 3, label: "Unique parameter count" },
  // training
  { id: "set_seed", file: "train.py", find: "def set_seed", n: 6, label: "Seeding (and its limits)" },
  { id: "device", file: "train.py", find: "def pick_device", n: 8, label: "Device selection" },
  { id: "lr_at", file: "train.py", find: "def lr_at", n: 6, label: "Warmup + cosine learning-rate schedule" },
  { id: "make_opt", file: "train.py", find: "def make_optimizer", n: 7, label: "AdamW with decay groups" },
  { id: "estimate", file: "train.py", find: "def estimate_loss", n: 12, label: "Evaluation under no_grad + eval()" },
  { id: "eval_block", file: "train.py", find: "if step % tc.eval_interval == 0", n: 14, label: "Periodic eval, sample, checkpoint" },
  { id: "zero", file: "train.py", find: "opt.zero_grad(set_to_none=True)", n: 1, label: "Clear gradients" },
  { id: "micro_loop", file: "train.py", find: "for _ in range(tc.grad_accum):", n: 5, label: "Forward, loss, scaled backward per microbatch" },
  { id: "fwd_loss", file: "train.py", find: "_, loss = model(x, y)  ", n: 1, label: "Forward pass + loss" },
  { id: "backward", file: "train.py", find: "(loss / tc.grad_accum).backward()", n: 1, label: "Backward pass" },
  { id: "clip", file: "train.py", find: "if tc.grad_clip > 0:", n: 4, label: "Gradient norm + clipping" },
  { id: "opt_step", file: "train.py", find: "opt.step()", n: 1, label: "AdamW step" },
  { id: "resume", file: "train.py", find: "if args.resume:", n: 3, label: "Resume from checkpoint" },
  { id: "log", file: "train.py", find: "if step % tc.log_interval == 0:", n: 4, label: "Logging" },
  // checkpoint
  { id: "ckpt_save", file: "checkpoint.py", find: "def save_checkpoint", n: 22, label: "save_checkpoint()" },
  { id: "ckpt_rng", file: "checkpoint.py", find: "def _rng_state", n: 9, label: "RNG states" },
  { id: "ckpt_load", file: "checkpoint.py", find: "def load_checkpoint", n: 3, label: "Safe loading (weights_only)" },
  { id: "ckpt_infer", file: "checkpoint.py", find: "def load_for_inference", n: 7, label: "Load for inference" },
  { id: "ckpt_resume", file: "checkpoint.py", find: "def restore_for_resume", n: 5, label: "Restore for resume" },
  // generation & serving
  { id: "gen", file: "generate.py", find: "def generate(model", n: 22, label: "Autoregressive generation loop" },
  { id: "gen_crop", file: "generate.py", find: "idx_cond = idx[:, -T_max:]", n: 1, label: "Crop to the last T_max tokens" },
  { id: "gen_last", file: "generate.py", find: "logits = logits[:, -1, :]", n: 1, label: "Keep only the final position" },
  { id: "gen_sample", file: "generate.py", find: "probs = torch.softmax(logits, dim=-1)", n: 2, label: "Sample the next id" },
  { id: "gen_append", file: "generate.py", find: "idx = torch.cat([idx, next_id], dim=1)", n: 1, label: "Append and repeat" },
  { id: "serve_schema", file: "serve.py", find: "class GenerateRequest", n: 14, label: "Request / response schemas" },
  { id: "serve_life", file: "serve.py", find: "async def lifespan", n: 5, label: "Load the model once at startup" },
  { id: "serve_ep", file: "serve.py", find: "def generate_endpoint", n: 14, label: "POST /generate" },
  // memory
  { id: "mem_measure", file: "measure_memory.py", find: "torch.cuda.reset_peak_memory_stats()", n: 10, label: "Measure peak CUDA memory for one step" },
  // tests
  { id: "t_causal", file: "tests/test_llm.py", find: "def test_causal_mask_no_future_leakage", n: 9, label: "Test: future tokens never change earlier logits" },
  { id: "t_fd", file: "tests/test_llm.py", find: "def test_finite_difference_gradient", n: 19, label: "Test: finite-difference gradient check" },
  { id: "t_resume", file: "tests/test_llm.py", find: "def test_resume_is_bit_exact", n: 12, label: "Test: resume is bit-exact" },
  { id: "t_tok", file: "tests/test_llm.py", find: "def test_tokenizer_roundtrip", n: 5, label: "Test: tokenizer round trip" },
  { id: "t_win", file: "tests/test_llm.py", find: "def test_windows_stay_inside_split", n: 10, label: "Test: windows never cross splits" },
  { id: "t_tie", file: "tests/test_llm.py", find: "def test_weight_tying_counts_and_gradient_sum", n: 12, label: "Test: tied gradient = sum of both paths" },
];

/* ---------------- small control helpers ---------------- */
const rangeC = (bind, label, min, max, val, step = 1) =>
  `<label for="r-${bind}">${label} <input type="range" id="r-${bind}" data-bind="${bind}" min="${min}" max="${max}" step="${step}" value="${val}"><output id="o-${bind}">${val}</output></label>`;
const selectC = (bind, label, options, val) =>
  `<label for="s-${bind}">${label} <select id="s-${bind}" data-bind="${bind}">${options.map(([v, l]) => `<option value="${v}" ${String(v) === String(val) ? "selected" : ""}>${l}</option>`).join("")}</select></label>`;
const checkC = (bind, label, val) =>
  `<label for="c-${bind}"><input type="checkbox" id="c-${bind}" data-bind="${bind}" ${val ? "checked" : ""}> ${label}</label>`;
const segC = (act, options, val, label) =>
  `${label ? `<span class="small muted">${label}</span>` : ""}<div class="seg">${options.map(([v, l]) => `<button aria-pressed="${String(v) === String(val)}" data-act="${act}" data-arg="${v}">${l}</button>`).join("")}</div>`;
const setOut = (t) => { const o = document.getElementById("o-" + t.dataset.bind); if (o) o.textContent = t.value; };
function bindNum(key, after) { BINDS[key] = (t) => { S[key] = +t.value; setOut(t); (after || (() => refresh({ compute: false })))(); }; }
function bindBool(key, opts = {}) { BINDS[key] = (t) => { S[key] = t.checked; refresh(opts); }; }
function bindStr(key, after) { BINDS[key] = (t) => { S[key] = t.value; (after || (() => refresh({ compute: false })))(); }; }
function segAct(name, key, num, opts = {}) { ACT[name] = (v) => { S[key] = num ? +v : v; document.querySelectorAll(`[data-act="${name}"]`).forEach((b) => b.setAttribute("aria-pressed", b.dataset.arg === String(v))); refresh(opts); }; }
const mvis = (id) => vis(MV[id]);
const inputChips = (act, selPos, o = {}) => `<div class="chips">${M.ids.map((id, t) => `<button class="chip ${t === selPos ? "sel" : ""}" data-act="${act}" data-arg="${t}" aria-label="position ${t}, character ${esc(MV[id])}, id ${id}"><span class="c">${esc(mvis(id))}</span><span class="i">${id}</span><span class="small muted" style="font-size:9px">t=${t}</span></button>`).join("")}${o.targets ? `<span class="op-sym">→</span>${M.tg.map((id, t) => `<span class="chip tgt ${t === selPos ? "sel" : ""}"><span class="c">${esc(mvis(id))}</span><span class="i">${id}</span><span class="small muted" style="font-size:9px">y${t}</span></span>`).join("")}` : ""}</div>`;
const posLabels = () => M.ids.map((id, t) => `${t}:${mvis(id)}`);
const okPill = (d, tol = 1e-5) => d <= tol ? `<span class="pill ok">matches PyTorch · max |Δ| ${fmt(d, 2)}</span>` : `<span class="pill warn">max |Δ| ${fmt(d, 2)}</span>`;
function recordedCheck(js, rec, label) {
  if (!M.canonical) return `<span class="pill neutral">Recomputed for your edited example — the PyTorch record covers the canonical example only</span>`;
  return `${label ? label + " " : ""}${okPill(Eng.maxAbsDiff(js, rec))}`;
}

const CHAPTERS = [];
const ch = (o) => CHAPTERS.push(o);

/* ================= Chapter 1: System map ================= */
const SYS = [
  { id: "corpus", label: "Raw corpus", sub: "1,115,394 chars", lane: 0, ch: "text", op: "load_text", in: "bytes (UTF-8 file)", out: "str", params: "none", art: "data/input.txt (sha256 pinned)" },
  { id: "split", label: "Split", sub: "90 / 5 / 5 %", lane: 0, ch: "text", op: "split", in: "str", out: "3 × str", params: "none", art: "split sizes stored in checkpoint" },
  { id: "tok", label: "Tokenizer", sub: "V = 66", lane: 0, ch: "tok", op: "tok_encode", in: "str", out: "ids: int64 [N]", params: "none (vocabulary is data)", art: "out/vocab.json + checkpoint['tokenizer']" },
  { id: "win", label: "Windows + batch", sub: "x, y [B, T]", lane: 0, ch: "win", op: "get_batch", in: "ids [N]", out: "x [B,T], y [B,T]", params: "none", art: "sampler RNG state" },
  { id: "emb", label: "Embeddings", sub: "[B, T, D]", lane: 1, ch: "emb", op: "emb_fwd", in: "x [B,T]", out: "[B,T,D]", params: "tok_emb [V,D], pos_emb [T_max,D]", art: "state_dict" },
  { id: "attn", label: "Attention", sub: "[B, H, T, T]", lane: 1, ch: "attn", op: "scores", in: "[B,T,D]", out: "[B,T,D]", params: "W_Q, W_K, W_V, W_O + biases, LN1", art: "state_dict" },
  { id: "ffn", label: "FFN", sub: "[B, T, Dff]", lane: 1, ch: "ffn", op: "ffn_fwd", in: "[B,T,D]", out: "[B,T,D]", params: "W1 [Dff,D], W2 [D,Dff] + biases, LN2", art: "state_dict" },
  { id: "logits", label: "Logits", sub: "[B, T, V]", lane: 1, ch: "loss", op: "logits", in: "[B,T,D]", out: "[B,T,V]", params: "ln_f, lm_head [V,D]", art: "state_dict" },
  { id: "loss", label: "Cross-entropy", sub: "scalar", lane: 2, ch: "loss", op: "ce", in: "logits [B·T,V], y [B·T]", out: "scalar loss", params: "none", art: "log.jsonl" },
  { id: "back", label: "Backward", sub: "∇ for every θ", lane: 2, ch: "backprop", op: "backward", in: "loss", out: ".grad on every parameter", params: "reads all", art: "none" },
  { id: "opt", label: "AdamW step", sub: "θ ← θ − Δ", lane: 2, ch: "backprop", op: "opt_step", in: "θ, ∇θ, m, v", out: "θ', m', v'", params: "updates all", art: "optimizer state" },
  { id: "eval", label: "Evaluate", sub: "val loss", lane: 2, ch: "train", op: "estimate", in: "val windows", out: "mean CE", params: "frozen (no_grad)", art: "log.jsonl" },
  { id: "ckpt", label: "Checkpoint", sub: "ckpt.pt", lane: 3, ch: "ckpt", op: "ckpt_save", in: "model, optimizer, RNG", out: "file", params: "serialised", art: "out/ckpt.pt" },
  { id: "reload", label: "Reload", sub: "logits equal", lane: 3, ch: "ckpt", op: "ckpt_infer", in: "ckpt.pt", out: "model in eval()", params: "restored", art: "—" },
  { id: "gen", label: "Generate", sub: "one id per pass", lane: 3, ch: "infer", op: "gen", in: "prompt ids", out: "ids + new ids", params: "frozen", art: "—" },
  { id: "serve", label: "HTTP endpoint", sub: "POST /generate", lane: 3, ch: "infer", op: "serve_ep", in: "JSON prompt", out: "JSON completion", params: "frozen", art: "—" },
];
const LANES = [["Dataset transformations", "var(--k-in)"], ["Model forward computation", "var(--k-act)"], ["Training-only operations", "var(--k-grad)"], ["Persistence & inference", "var(--k-tgt)"]];
function sysMap() {
  const W = 700, lh = 88, top = 8;
  let s = `<svg viewBox="0 0 ${W} ${top + lh * 4}" role="img" aria-label="System map of the training and inference pipeline">`;
  LANES.forEach(([l, c], i) => {
    const y = top + i * lh;
    s += `<rect x="0" y="${y}" width="${W}" height="${lh - 10}" rx="8" fill="${c}" opacity="0.05"/><rect x="0" y="${y}" width="4" height="${lh - 10}" rx="2" fill="${c}"/>`;
    s += `<text x="12" y="${y + 15}" style="font-size:10px;letter-spacing:.06em;text-transform:uppercase;fill:${c};font-weight:600">${l}</text>`;
  });
  const perLane = [0, 0, 0, 0];
  SYS.forEach((n) => {
    const k = perLane[n.lane]++, x = 14 + k * 172, y = top + n.lane * lh + 24;
    n._x = x; n._y = y;
  });
  SYS.forEach((n, i) => {
    const nx = SYS[i + 1];
    if (nx && nx.lane === n.lane) s += `<path d="M${n._x + 150},${n._y + 20} L${nx._x - 4},${nx._y + 20}" stroke="var(--line-2)" stroke-width="1.5" marker-end="url(#arr)" fill="none"/>`;
  });
  s += `<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#cbd0db"/></marker></defs>`;
  [[3, 4], [7, 8], [11, 12]].forEach(([a, b]) => { const A = SYS[a], B = SYS[b]; s += `<path d="M${A._x + 75},${A._y + 40} C${A._x + 75},${A._y + 62} ${B._x + 75},${B._y - 22} ${B._x + 75},${B._y}" stroke="var(--line-2)" stroke-width="1.5" fill="none" stroke-dasharray="3 3" marker-end="url(#arr)"/>`; });
  SYS.forEach((n) => {
    s += `<g class="node ${S.sysNode === n.id ? "on" : ""}" data-act="sysnode" data-arg="${n.id}" tabindex="0" role="button" aria-label="${n.label}"><rect x="${n._x}" y="${n._y}" width="150" height="40" rx="7"/><text x="${n._x + 10}" y="${n._y + 17}" style="font-weight:600">${n.label}</text><text class="sub" x="${n._x + 10}" y="${n._y + 31}">${n.sub}</text></g>`;
  });
  return s + "</svg>";
}
ACT.sysnode = (id) => { S.sysNode = id; const n = SYS.find((x) => x.id === id); refresh({ compute: false }); if (n) setOp(n.op); };
ch({
  id: "map", num: 1, short: "System map", title: "The complete system", group: "Foundations", kind: "data",
  thesis: "One real text sequence travels the whole pipeline: characters become ids, ids become vectors, vectors become a loss, the loss becomes gradients, gradients become a better checkpoint, and the checkpoint answers HTTP requests.",
  files: ["train.py", "model.py"], ops: SYS.map((n) => n.op),
  sections: () => [
    {
      title: "Pipeline map", tag: "pipeline", body: () => {
        const n = SYS.find((x) => x.id === S.sysNode) || SYS[0];
        return `<div class="sysmap">${sysMap()}</div>
        <div class="grid2" style="margin-top:12px">
          <dl class="kv"><dt>component</dt><dd>${n.label}</dd><dt>input</dt><dd>${n.in}</dd><dt>output</dt><dd>${n.out}</dd><dt>trainable</dt><dd>${n.params}</dd><dt>persists as</dt><dd>${n.art}</dd><dt>code</dt><dd>${OPS[n.op] ? `${OPS[n.op].file}:${OPS[n.op].start}` : "—"}</dd></dl>
          <div><p class="small muted">Select any box. Its source lines are highlighted on the left.</p><button class="btn primary" data-act="go" data-arg="${n.ch}">Open the ${CHAPTERS.find((c) => c.id === n.ch).short} chapter →</button></div>
        </div>`;
      },
    },
    {
      title: "Pretraining, fine-tuning and inference", tag: "engineering", body: () => `<div class="prose">
      <p><b>Pretraining</b> (what this reference implements) starts from random weights and minimises next-token cross-entropy on raw text. There are no labels beyond the text itself: every position's target is the following character.</p>
      <p><b>Fine-tuning</b> continues training a pretrained checkpoint on a narrower distribution (instructions, dialogue, a domain). The code path is the same training loop with a different dataset, usually a smaller learning rate, and often a loss mask so that only response tokens are scored. Preference optimisation and parameter-efficient methods such as LoRA are separate topics.</p>
      <p><b>Inference</b> loads frozen weights and repeats one forward pass per generated token. Nothing is learned; dropout is disabled with <code>model.eval()</code> and autograd is disabled with <code>torch.no_grad()</code>.</p></div>`,
    },
    {
      title: "Two scales, one implementation", tag: "tensor", body: () => {
        const t = TRAINED.config;
        return `<p>Every number you can click comes from the <b>micro</b> configuration. The <b>training</b> configuration is the same <code>GPT</code> class at a size that learns Shakespeare-like text on a CPU. Use the dimension switch in the top bar to rewrite every shape label.</p>
        <div class="tbl-wrap"><table class="data"><tr><th>symbol</th><th>meaning</th><th class="num">micro (inspectable)</th><th class="num">training run</th></tr>
        ${[["B", "batch size", 1, TRAINED.train_config.batch_size], ["T", "sequence length", 8, t.block_size], ["V", "vocabulary size", MV.length, t.vocab_size], ["D", "model width", 8, t.n_embd], ["H", "attention heads", 2, t.n_head], ["Dh", "head width D/H", 4, t.head_dim], ["L", "Transformer blocks", 1, t.n_layer], ["Dff", "FFN hidden width", 32, t.d_ff], ["", "dropout", 0, t.dropout], ["", "unique parameters", fmtInt(Object.values(MICRO.shapes).reduce((a, s) => a + s.reduce((x, y) => x * y, 1), 0)), fmtInt(TRAINED.num_params)]].map((r) => `<tr><td class="mono">${r[0]}</td><td>${r[1]}</td><td class="num">${r[2]}</td><td class="num">${r[3]}</td></tr>`).join("")}</table></div>
        <p class="small muted" style="margin-top:8px">The micro vocabulary (${MV.length} ids) is built from the first ${MICRO.excerpt.length} characters of the corpus; the training vocabulary (${t.vocab_size} ids) from the whole training split. <code>L</code> always means the number of blocks; the loss is written <code>loss</code>.</p>`;
      },
    },
    {
      title: "How to read this reference", tag: "engineering", body: () => `<div class="grid2"><div>
        <h3 style="margin-top:0">Tensor kinds</h3><div class="legend" style="flex-direction:column;align-items:flex-start;gap:6px">${kindTag("in")} token ids and raw data ${kindTag("par")} trainable weights ${kindTag("act")} values computed in the forward pass ${kindTag("grad")} ∂loss/∂(something) ${kindTag("tgt")} next-token targets and the loss</div>
        <h3>Where numbers come from</h3><div class="legend" style="flex-direction:column;align-items:flex-start;gap:6px">${prov("browser")}<span class="small muted" style="margin-top:-4px">JavaScript port of the model, checked against PyTorch on load</span>${prov("recorded")}<span class="small muted" style="margin-top:-4px">Saved from an executed Python run in this project</span>${prov("analytic")}<span class="small muted" style="margin-top:-4px">Formula-based estimate with stated assumptions</span>${prov("illus")}<span class="small muted" style="margin-top:-4px">Teaching sketch, not a measurement</span></div></div>
        <div><h3 style="margin-top:0">Browser engine check</h3><p class="small">On load, the in-browser forward pass, backward pass and AdamW step were recomputed and compared with the PyTorch record.</p>
        <p>${M.verifyMax < 1e-4 ? `<span class="pill ok">All ${M.verify.length} checks agree · worst |Δ| ${fmt(M.verifyMax, 2)}</span>` : `<span class="pill warn">worst |Δ| ${fmt(M.verifyMax, 2)}</span>`} <a href="#validate" data-act="go" data-arg="validate">details</a></p>
        <h3>Keyboard</h3><dl class="kv"><dt><kbd>/</kbd> or <kbd>⌘K</kbd></dt><dd>search</dd><dt><kbd>[</kbd> <kbd>]</kbd></dt><dd>previous / next chapter</dd><dt><kbd>Space</kbd> <kbd>←</kbd> <kbd>→</kbd></dt><dd>attention stepper (chapter 8)</dd><dt><kbd>←</kbd> <kbd>→</kbd> on divider</dt><dd>resize panes</dd></dl></div></div>`,
    },
  ],
  quiz: [
    { id: "q-map-1", q: "Which of these is a training-only operation that never runs during inference?", options: ["Embedding lookup", "Final LayerNorm", "Cross-entropy backward pass", "Softmax over attention scores"], answer: 2, explain: "Inference runs the same forward computation (including attention softmax and LayerNorm) but never computes a loss gradient." },
  ],
});

/* ================= Chapter 2: Text ================= */
const STOP = new Set("the a an and or of to in is it that this i you he she we they me my thy thou be are was not with for as on at by his her".split(" "));
function prep(text) {
  let t = text;
  if (S.prepLower) t = t.toLowerCase();
  if (S.prepPunct) t = t.replace(/[^\w\s]/g, "");
  if (S.prepStop) t = t.split(/(\s+)/).filter((w) => !STOP.has(w.toLowerCase())).join("");
  if (S.prepNl) t = t.replace(/\s+/g, " ");
  return t;
}
["prepLower", "prepPunct", "prepStop", "prepNl", "splitWrong"].forEach((k) => bindBool(k, { compute: false }));
ch({
  id: "text", num: 2, short: "Reading text", title: "Reading and preparing the corpus", group: "Foundations", kind: "data",
  thesis: "A language model learns the distribution of exactly the characters you give it. Preparation therefore means keeping the text faithful and splitting it so that no validation character can leak into training.",
  files: ["data.py", "train.py"], ops: ["load_text", "split", "train_split"],
  card: {
    problem: "Turn a file of bytes into a faithful character stream and three non-overlapping partitions.",
    ml: "Like a train/test split for tabular data, but rows are overlapping windows of one long sequence, so the split must happen on the sequence first.",
    op: "decode UTF-8 → normalise \\r\\n to \\n → text[:0.9N], text[0.9N:0.95N], text[0.95N:]",
    shapes: () => "file bytes → str (N = 1,115,394) → 3 strings",
    params: "None.",
    changes: "Nothing; this is fixed preprocessing.",
    fixed: "Everything, including case, punctuation and whitespace.",
    failures: "Invalid UTF-8; stray \\r; splitting after windowing (leakage); verbatim duplicate lines across splits.",
    verify: "Checksum the file; assert split lengths sum to N; assert every sampled window is a substring of its own split.",
  },
  sections: () => [
    {
      title: "The dataset", tag: "engineering", prov: "recorded", body: () => {
        const c = D.corpus;
        return `<dl class="kv"><dt>name</dt><dd>Tiny Shakespeare</dd><dt>source</dt><dd><a href="https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare" target="_blank" rel="noopener">karpathy/char-rnn · data/tinyshakespeare/input.txt</a></dd>
        <dt>acquire</dt><dd>curl -L -o data/input.txt ${c.url}</dd><dt>sha256</dt><dd>${c.sha256}</dd><dt>size</dt><dd>${fmtInt(c.bytes)} bytes · ${fmtInt(c.chars)} characters · ${fmtInt(c.lines)} lines</dd>
        <dt>encoding</dt><dd>UTF-8 (pure ASCII in practice: ${c.non_ascii} non-ASCII characters) · ${c.cr} carriage returns</dd><dt>distinct chars</dt><dd>${c.unique}</dd>
        <dt>usage terms</dt><dd style="font-family:var(--f-ui)">${c.license}</dd></dl>`;
      },
    },
    {
      title: "Before / after preprocessing", tag: "numerical", prov: "browser",
      ctrl: () => `${checkC("prepLower", "lowercase", S.prepLower)}${checkC("prepPunct", "strip punctuation", S.prepPunct)}${checkC("prepStop", "remove stop words", S.prepStop)}${checkC("prepNl", "collapse whitespace", S.prepNl)}`,
      body: () => {
        const raw = D.corpus.head, out = prep(raw), vRaw = new Set(raw).size, vOut = new Set(out).size;
        const changed = S.prepLower || S.prepPunct || S.prepStop || S.prepNl;
        return `<div class="grid2"><div><h3 style="margin-top:0">Baseline (what train.py uses)</h3><div class="calc" style="white-space:pre-wrap">${esc(visStr(raw))}</div></div>
        <div><h3 style="margin-top:0">${changed ? "With the selected transforms" : "Pick a transform above"}</h3><div class="calc" style="white-space:pre-wrap">${esc(visStr(out))}</div></div></div>
        <p class="small" style="margin-top:8px">${raw.length} → ${out.length} characters · ${vRaw} → ${vOut} distinct characters on this excerpt. <span class="muted">· marks a space, ↵ a newline.</span></p>
        <div class="callout ${changed ? "warn" : ""}"><b>${changed ? "What the model can no longer learn" : "Why the baseline keeps everything"}</b>${changed ? [S.prepLower && "capitalised speaker names and sentence starts", S.prepPunct && "the colon that ends every speaker tag, question marks, and verse punctuation", S.prepStop && "grammar: function words carry most of the syntax the model must reproduce", S.prepNl && "line and speech structure, which is how the plays are laid out"].filter(Boolean).join("; ") + ". A generative model can only produce characters it was trained to predict, so anything removed here can never appear in its output." : "The model's job is to reproduce text exactly. Stop-word removal and stemming help a bag-of-words classifier ignore noise; for next-character prediction, that “noise” is the signal. The only normalisation applied is converting Windows (\\r\\n) and old-Mac (\\r) line endings to \\n, which this file does not contain but other corpora do."}</div>`;
      },
    },
    {
      title: "Split first, then make windows", tag: "tensor", prov: "recorded",
      ctrl: () => `${checkC("splitWrong", "show what goes wrong if windows are made first", S.splitWrong)}`,
      body: () => {
        const c = D.corpus, n = c.chars, sp = c.split;
        const seg = (len, col, l) => `<div style="flex:${len};background:${col};color:#fff;padding:6px 8px;font-size:12px;min-width:0;overflow:hidden;white-space:nowrap">${l}</div>`;
        let h = `<div style="display:flex;border-radius:6px;overflow:hidden;margin-bottom:8px">${seg(sp[0], "var(--k-in)", `train · ${fmtInt(sp[0])} chars`)}${seg(sp[1] * 3, "var(--k-act)", `val · ${fmtInt(sp[1])}`)}${seg(sp[2] * 3, "var(--k-tgt)", `test · ${fmtInt(sp[2])}`)}</div><p class="small muted">Validation and test segments drawn 3× wider than scale so they are readable.</p>`;
        h += `<div class="grid2">${c.boundaries.map((b) => `<div><h3 style="margin-top:0">${b.name} at character ${fmtInt(b.index)}</h3><div class="calc" style="white-space:pre-wrap">${esc(visStr(b.before))}<span style="background:var(--hl)">|</span>${esc(visStr(b.after))}</div></div>`).join("")}</div>`;
        h += `<p class="small">Both boundaries fall mid-speech because the split is by character position. Each split becomes its own tensor, so <code>get_batch</code> can only draw windows that lie entirely inside one split. The test <code>test_windows_stay_inside_split</code> checks this on 1,280 random validation windows.</p>`;
        if (S.splitWrong) {
          const T = 128, pVal = 0.05;
          h += `<div class="callout bad"><b>The leak, quantified</b>Suppose you first cut every start position into a window of T+1 = ${T + 1} characters (stride 1), then randomly send 5% of windows to validation. A validation window starting at position s shares ${T} of its ${T + 1} characters with the window starting at s+1, which lands in training with probability ${1 - pVal}. The chance that <i>neither</i> neighbour is in training is ${pVal}² = ${(pVal * pVal).toFixed(4)}, so about ${(100 * (1 - pVal * pVal)).toFixed(2)}% of validation windows are near-copies of training windows. Validation loss then measures memorisation, not generalisation. <span class="muted">(Analytical, assuming independent random assignment.)</span></div>`;
        }
        const dd = c.dup_val_lines, pct = (x) => (100 * x.dups / x.lines).toFixed(1);
        h += `<div class="callout"><b>Verbatim repeats across the split, measured</b>${pct(dd.any)}% of non-empty validation lines (${fmtInt(dd.any.dups)} of ${fmtInt(dd.any.lines)}) also occur verbatim in training, but 98% of them are speaker tags such as "PETRUCHIO:". Only ${dd.gt10.dups} of ${fmtInt(dd.gt10.lines)} lines longer than 10 characters repeat, and none longer than 20. Repeated tags make those few characters easy to predict on both sides; they are part of the text's real distribution rather than leakage, but they do lower validation loss slightly. Document-level deduplication is out of scope for this baseline.</div>`;
        return h;
      },
    },
    {
      title: "Choosing a split strategy", tag: "engineering", body: () => `<div class="tbl-wrap"><table class="data"><tr><th>strategy</th><th>what it guarantees</th><th>trade-off here</th></tr>
      <tr><td>Contiguous character ranges <span class="pill neutral">baseline</span></td><td>No shared characters between splits; simple and reproducible.</td><td>Validation is the end of the file (later plays), so it tests mild distribution shift. Boundaries cut mid-word.</td></tr>
      <tr><td>Document-level (per play)</td><td>Whole works held out; closer to "new document" generalisation.</td><td>Needs play boundaries this file does not mark; sizes become uneven.</td></tr>
      <tr><td>Chronological</td><td>Mimics deployment on future data.</td><td>Meaningful for logs or news; for Shakespeare it reduces to the contiguous split.</td></tr>
      <tr><td>Random windows, then split <span class="pill bad">invalid</span></td><td>Nothing.</td><td>Near-duplicate windows on both sides (see above).</td></tr></table></div>`,
    },
  ],
  quiz: [
    { id: "q-text-1", q: "Why is stop-word removal inappropriate for training a generative character model?", options: ["It makes the vocabulary too large", "The model must learn to generate those words; removing them removes the targets", "It slows down tokenization", "It breaks UTF-8 decoding"], answer: 1, explain: "Every character the model should produce must appear in its training targets. Removing function words deletes most of the syntax the model is supposed to learn." },
    { id: "q-text-2", q: "You build windows over the whole corpus and then randomly assign 10% to validation. Your validation loss is suspiciously close to training loss. What happened?", explain: "Adjacent windows overlap by T of their T+1 characters, so almost every validation window has a near-identical twin in training (probability 1 − 0.1² = 99%). Split the character stream first, then window each split." },
  ],
});

/* ================= Chapter 3: Tokenizer ================= */
const TVOCAB = D.corpus.vocab;      // training vocabulary (id -> char), id 0 = <unk>
const TSTOI = Object.fromEntries(TVOCAB.map((c, i) => [c, i]));
let BPE_CACHE = null;
bindStr("tokText");
bindNum("bpeMerges");
ACT.tokStrict = (v) => { S.tokStrict = v === "1"; document.querySelectorAll('[data-act="tokStrict"]').forEach((b) => b.setAttribute("aria-pressed", b.dataset.arg === v)); refresh({ compute: false }); };
ACT.tokpick = (i) => { S.tokHover = +i; refresh({ compute: false }); };
ch({
  id: "tok", num: 3, short: "Tokenization", title: "A reversible character tokenizer", group: "Foundations", kind: "data",
  thesis: "The tokenizer is a fixed lookup table between characters and integer ids. Ids are names for table rows, not quantities: the model never does arithmetic on them.",
  files: ["tokenizer.py", "train.py"], ops: ["tok_build", "tok_init", "tok_encode", "tok_decode", "tok_save", "train_split", "t_tok"],
  card: {
    problem: "Map text to a finite set of categorical ids and back, without losing information.",
    ml: "Exactly categorical encoding of a nominal feature, except every position in the sequence is such a feature.",
    op: "encode(s)[t] = stoi[s[t]];  decode(ids) = ''.join(itos[i] for i in ids)",
    shapes: () => "str of length N → list[int] of length N",
    params: "None. The vocabulary is data built from the training split.",
    changes: "Nothing during training.",
    fixed: "The id assignment. Changing it after training scrambles the meaning of every embedding row.",
    failures: "Characters unseen in training; building the vocabulary from all splits; reordering the vocabulary between save and load.",
    verify: "decode(encode(s)) == s for every split; unknown characters raise or map to id 0 by explicit policy.",
  },
  sections: () => [
    {
      title: "Encode and decode", tag: "numerical", prov: "browser",
      ctrl: () => `<div style="flex:1 1 100%"><textarea id="tok-in" data-bind="tokText" rows="2" aria-label="Text to tokenize">${esc(S.tokText)}</textarea></div>${segC("tokStrict", [["1", "strict (raise)"], ["0", "replace with <unk>"]], S.tokStrict ? "1" : "0", "Unknown characters")}`,
      body: () => {
        const chars = Array.from(S.tokText), unknown = chars.filter((c) => !(c in TSTOI));
        const ids = chars.map((c) => TSTOI[c] ?? 0);
        let h = `<div class="chips" style="margin-bottom:10px">${chars.map((c, i) => `<button class="chip ${c in TSTOI ? "" : "unk"} ${S.tokHover === ids[i] ? "sel" : ""}" data-act="tokpick" data-arg="${ids[i]}" data-op="tok_encode" title="'${esc(c)}' → ${c in TSTOI ? ids[i] : "unknown"}"><span class="c">${esc(vis(c))}</span><span class="i">${c in TSTOI ? ids[i] : "?"}</span></button>`).join("")}</div>`;
        if (unknown.length && S.tokStrict) {
          h += `<div class="callout bad"><b>UnknownCharacterError</b>characters not in vocabulary: [${[...new Set(unknown)].map((c) => `'${esc(c)}'`).join(", ")}]. Strict encoding refuses rather than silently changing your text. The HTTP endpoint turns this into a 422 response.</div>`;
        } else {
          const dec = ids.map((i) => (i === 0 ? "\ufffd" : TVOCAB[i])).join("");
          h += `<dl class="kv"><dt>ids</dt><dd>[${ids.join(", ")}]</dd><dt>decode(ids)</dt><dd>${esc(visStr(dec))}</dd><dt>round trip</dt><dd>${dec === S.tokText ? '<span class="pill ok">identical</span>' : `<span class="pill warn">differs: ${unknown.length} character(s) became U+FFFD</span>`}</dd><dt>length</dt><dd>${chars.length} characters → ${ids.length} tokens</dd></dl>`;
        }
        return h;
      },
    },
    {
      title: `Training vocabulary · V = ${TVOCAB.length}`, tag: "tensor", prov: "recorded", body: () => {
        const used = new Set(Array.from(S.tokText).map((c) => TSTOI[c]).filter((x) => x !== undefined));
        return `<div class="vocab-grid" data-op="tok_init">${TVOCAB.map((c, i) => `<div class="${S.tokHover === i ? "on" : used.has(i) ? "used" : ""}" title="id ${i}: ${c === "<unk>" ? "<unk> (reserved)" : JSON.stringify(c)}"><b>${esc(vis(c))}</b><span>${i}</span></div>`).join("")}</div>
        <p class="small muted" style="margin-top:8px">Id 0 is reserved for <code>&lt;unk&gt;</code>. Ids 1–${TVOCAB.length - 1} follow Unicode code-point order of the ${TVOCAB.length - 1} characters seen in the training split, so the mapping is deterministic. Select a token above to locate it; outlined cells occur in your text. Validation contains ${D.corpus.unk_val} and test ${D.corpus.unk_test} characters outside this vocabulary.</p>`;
      },
    },
    {
      title: "Character tokens versus learned subwords", tag: "engineering", prov: "browser",
      ctrl: () => rangeC("bpeMerges", "BPE merges", 0, 300, S.bpeMerges, 5),
      body: () => {
        if (!BPE_CACHE) BPE_CACHE = BPE.train(D.corpus.bpe_text, 300);
        const merges = BPE_CACHE.slice(0, S.bpeMerges), segs = BPE.encode(S.tokText, merges);
        return `<p>A byte-pair encoder repeatedly merges the most frequent adjacent pair into a new symbol. This one was trained <b>in your browser</b> on the first ${fmtInt(D.corpus.bpe_text.length)} characters of the training text, with no byte fallback and no word pre-splitting. It illustrates the mechanism; it is not GPT-2's tokenizer.</p>
        <div class="chips" style="margin:8px 0">${segs.map((s, i) => `<span class="chip" style="background:${i % 2 ? "var(--k-in-soft)" : "var(--surface)"}"><span class="c" style="font-size:13px">${esc(visStr(s))}</span></span>`).join("")}</div>
        <dl class="kv"><dt>character tokens</dt><dd>${Array.from(S.tokText).length}</dd><dt>BPE tokens</dt><dd>${segs.length} (${S.bpeMerges} merges → base vocabulary + ${merges.length} symbols)</dd><dt>first merges</dt><dd>${merges.slice(0, 12).map(([a, b, n]) => `${esc(visStr(a))}+${esc(visStr(b))}→${esc(visStr(a + b))} (${n})`).join("  ") || "—"}</dd></dl>
        <div class="callout"><b>Trade-off</b>Subwords shorten sequences (cheaper attention, longer effective context) at the cost of a much larger embedding table and output softmax. Characters keep V tiny and make every string representable, which is why the baseline uses them. Unigram tokenization (Kudo, 2018) is the main probabilistic alternative to BPE.</div>`;
      },
    },
    {
      title: "Ids are categories, not magnitudes", tag: "math", body: () => `<div class="eq">id('e') = ${TSTOI.e},  id('E') = ${TSTOI.E}
'e' is <i>not</i> ${(TSTOI.e / TSTOI.E).toFixed(2)}× 'E' — the integers only name rows.

one_hot(i) ∈ {0,1}<sup>V</sup>,   <span class="w">E</span> ∈ ℝ<sup>V×D</sup>
embedding(i) = one_hot(i) · <span class="w">E</span> = <span class="w">E</span>[i, :]</div><p>Feeding raw ids into a linear layer would impose a false ordering (as if <code>'z'</code> were "more" than <code>'a'</code>). The embedding table gives each id an independent learned vector instead. Chapter 5 shows the lookup numerically.</p>`,
    },
  ],
  quiz: [
    { id: "q-tok-1", q: "Why is the vocabulary built from the training split only?", options: ["It is faster", "So no information from validation/test text shapes the model's input space", "Because validation has no characters", "PyTorch requires it"], answer: 1, explain: "Building it from all splits is a small but real form of leakage, and it hides the question of what to do with unseen characters, which then surfaces only in production." },
    { id: "q-tok-2", q: "Explain why an embedding lookup does not make token ids continuous inputs to the network.", explain: "The id is used only as an index: the forward pass selects row i of E. No gradient flows to the id, and nearby ids have unrelated rows. Mathematically the lookup is one_hot(i)·E, and a one-hot vector carries no ordering." },
  ],
});

/* ================= Chapter 4: Windows ================= */
["winT", "winStart", "winB", "winStride"].forEach((k) => bindNum(k));
segAct("winMode", "winMode", false, { compute: false });
ACT.resample = () => { S.winSeed++; refresh({ compute: false }); };
function windowsNow() {
  const ids = D.corpus.strip_ids, N = ids.length, T = S.winT, out = [];
  const maxStart = N - T - 1;
  const r = Eng.rng(S.winSeed * 7919);
  for (let b = 0; b < S.winB; b++) {
    const s = S.winMode === "seq" ? Math.min(maxStart, S.winStart + b * S.winStride) : Math.floor(r() * (maxStart + 1));
    out.push(s);
  }
  return out;
}
ch({
  id: "win", num: 4, short: "Windows & targets", title: "Sliding windows, sampling and targets", group: "Foundations", kind: "data",
  thesis: "One window of T + 1 consecutive tokens yields T supervised examples at once: position t is asked to predict token t + 1 having seen tokens 0…t.",
  files: ["data.py", "train.py"], ops: ["get_batch", "seq_windows", "t_win"],
  card: {
    problem: "Turn one long id stream into a batch of fixed-shape (input, target) pairs.",
    ml: "Like building supervised rows from a time series with a lag-1 target, except each row carries T targets.",
    op: "x = ids[s : s+T];  y = ids[s+1 : s+T+1]",
    shapes: () => `ids ${shape(["N"])} → x ${shape(["B", "T"])}, y ${shape(["B", "T"])}`,
    params: "None.",
    changes: "The sampled start positions every step.",
    fixed: "T (the context length) and the one-token shift.",
    failures: "Off-by-one (targets equal inputs); windows crossing split boundaries; T larger than the position table.",
    verify: "y[:, :-1] == x[:, 1:]; with ids = arange, y == x + 1.",
  },
  sections: () => [
    {
      title: "Window sampler", tag: "tensor", prov: "browser",
      ctrl: () => `${rangeC("winT", "T", 2, 16, S.winT)}${rangeC("winB", "B", 1, 4, S.winB)}${segC("winMode", [["seq", "sequential"], ["random", "random"]], S.winMode, "starts")}${rangeC("winStart", "first start", 0, 50, S.winStart)}${rangeC("winStride", "stride", 1, 16, S.winStride)}<button class="btn small" data-act="resample">Resample</button>`,
      body: () => {
        const ids = D.corpus.strip_ids, starts = windowsNow(), T = S.winT, cols = ["var(--k-in)", "var(--k-act)", "var(--k-par)", "var(--k-tgt)"];
        let strip = `<div class="hm-wrap" id="win-strip" data-op="get_batch"><div style="display:grid;grid-template-columns:repeat(${ids.length},30px);gap:2px;width:max-content;padding-bottom:${S.winB * 8 + 4}px;position:relative">`;
        ids.forEach((id, i) => { strip += `<div class="chip" data-win="${i}" style="min-width:0;padding:2px 0;${starts[0] <= i && i < starts[0] + T ? "background:var(--k-in-soft)" : ""}"><span class="c" style="font-size:13px">${esc(vis(TVOCAB[id]))}</span><span class="i" style="font-size:9.5px">${id}</span><span style="font-size:8.5px;color:var(--muted)">${i}</span></div>`; });
        starts.forEach((s, b) => {
          strip += `<div style="position:absolute;left:${s * 32}px;width:${T * 32 - 2}px;top:${52 + b * 8}px;height:3px;background:${cols[b]};border-radius:2px" title="x[${b}]"></div><div style="position:absolute;left:${(s + 1) * 32}px;width:${T * 32 - 2}px;top:${54 + b * 8}px;height:3px;background:${cols[b]};opacity:.4;border-radius:2px" title="y[${b}]"></div>`;
        });
        strip += `</div></div><p class="small muted">Drag across the strip to move the first window (sequential mode). Solid bars are inputs x[b], faint bars the targets y[b], shifted one to the right.</p>`;
        let tbl = `<div class="tbl-wrap"><table class="data"><tr><th>b</th><th>start</th><th>x[b] (input)</th><th>y[b] (target)</th></tr>`;
        starts.forEach((s, b) => {
          const x = ids.slice(s, s + T), y = ids.slice(s + 1, s + T + 1);
          tbl += `<tr><td class="mono">${b}</td><td class="num">${s}</td><td class="mono">${x.map((i) => `<span title="id ${i}">${esc(vis(TVOCAB[i]))}</span>`).join(" ")}<br><span class="muted">[${x.join(", ")}]</span></td><td class="mono" style="color:var(--k-tgt)">${y.map((i) => esc(vis(TVOCAB[i]))).join(" ")}<br><span class="muted">[${y.join(", ")}]</span></td></tr>`;
        });
        tbl += `</table></div>`;
        const s0 = starts[0];
        const pairs = Array.from({ length: Math.min(T, 8) }, (_, t) => `t=${t}: sees "${visStr(ids.slice(s0, s0 + t + 1).map((i) => TVOCAB[i]).join(""))}" → predicts '${vis(TVOCAB[ids[s0 + t + 1]])}'`).join("\n");
        return `${strip}${tbl}<div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0">x ${shape(["B", "T"]).replace(/\[.*\]/, `[${S.winB}, ${T}]`)} ${kindTag("in")} y ${shape(["B", "T"]).replace(/\[.*\]/, `[${S.winB}, ${T}]`)} ${kindTag("tgt")}</div><div class="calc">${esc(pairs)}${T > 8 ? "\n…" : ""}</div>`;
      },
    },
    {
      title: "Teacher forcing and the one-token shift", tag: "math", body: () => `<div class="eq">x<sub>b</sub> = ( <span class="x">w<sub>s</sub></span>, …, <span class="x">w<sub>s+T−1</sub></span> )      y<sub>b</sub> = ( <span class="y">w<sub>s+1</sub></span>, …, <span class="y">w<sub>s+T</sub></span> )

loss = −1/(B·T) · Σ<sub>b</sub> Σ<sub>t</sub> log p( <span class="y">y<sub>b,t</sub></span> | <span class="x">x<sub>b,0..t</sub></span> )</div>
      <div class="prose"><p><b>Teacher forcing</b>: during training, position t is conditioned on the true previous tokens, never on the model's own earlier predictions. That is what lets all T predictions be computed in one parallel forward pass. The causal mask (chapter 8) is what makes this honest: position t cannot look at x<sub>t+1</sub>, which is exactly y<sub>t</sub>.</p>
      <p><b>Two different axes.</b> B counts independent windows; nothing flows between rows. T is positions within one window; attention mixes information along it. Doubling B doubles independent examples; doubling T lengthens the context each prediction can use and quadruples explicit attention storage.</p></div>`,
    },
    {
      title: "Steps, not epochs", tag: "engineering", prov: "analytic", body: () => {
        const tc = TRAINED.train_config, N = D.corpus.split[0], k = tc.batch_size * tc.max_steps, T = TRAINED.config.block_size;
        return `<p>Random sampling draws start positions with replacement, so "epoch" is ambiguous: some windows are seen several times, others never as a start. Training length is therefore counted in optimizer steps.</p>
        <dl class="kv"><dt>training tokens</dt><dd>N = ${fmtInt(N)}</dd><dt>targets per step</dt><dd>B·T = ${tc.batch_size}·${T} = ${fmtInt(tc.batch_size * T)}</dd><dt>steps in the recorded run</dt><dd>${fmtInt(tc.max_steps)}</dd><dt>targets processed</dt><dd>${fmtInt(k * T)} ≈ ${(k * T / N).toFixed(1)} × N</dd><dt>starts never drawn</dt><dd>≈ e^(−${fmtInt(k)}/${fmtInt(N)}) = ${(100 * Math.exp(-k / N)).toFixed(1)}% of positions</dd><dt>tokens never covered</dt><dd>≈ e^(−${(k * T / N).toFixed(1)}) ≈ ${Math.exp(-k * T / N).toExponential(1)}</dd></dl>
        <p class="small muted" style="margin-top:6px">Most start positions are never used, yet virtually every token is covered because each window spans T tokens.</p>`;
      },
    },
  ],
  after: () => {
    const strip = document.getElementById("win-strip"); if (!strip) return;
    let down = false;
    const setFrom = (e) => { const c = e.target.closest("[data-win]"); if (!c || S.winMode !== "seq") return; const v = Math.min(50, +c.dataset.win); if (v !== S.winStart) { S.winStart = v; const r = document.getElementById("r-winStart"); if (r) { r.value = v; setOut(r); } refresh({ compute: false }); } };
    strip.addEventListener("pointerdown", (e) => { down = true; setFrom(e); });
    strip.addEventListener("pointerover", (e) => { if (down) setFrom(e); });
    window.addEventListener("pointerup", () => (down = false), { once: true });
  },
  quiz: [
    { id: "q-win-1", q: "A window has T = 128 inputs. How many next-token predictions contribute to its loss?", options: ["1", "127", "128", "129"], answer: 2, explain: "Every input position t = 0…127 has a target (the token at t + 1), so one window gives T = 128 terms." },
  ],
});

/* ================= Chapter 5: Embeddings ================= */
ACT.pos = (t) => { S.pos = +t; refresh({ compute: false, full: true }); };
ch({
  id: "emb", num: 5, short: "Embeddings", title: "Token and position embeddings", group: "The model", kind: "forward",
  thesis: "Two trainable tables turn ids into vectors: a token table indexed by character and a position table indexed by slot. Their sum is the first activation of the residual stream.",
  files: ["model.py"], ops: ["emb_def", "emb_fwd", "emb_drop"],
  card: {
    problem: "Give each categorical id a learned dense vector and tell the model where in the window it is.",
    ml: "The Keras Embedding layer, plus a second embedding indexed by position.",
    op: "x0[b,t] = E[x[b,t]] + P[t]",
    shapes: () => `x ${shape(["B", "T"])} → E[x] ${shape(["B", "T", "D"])} + P[:T] ${shape(["T", "D"])} → x0 ${shape(["B", "T", "D"])}`,
    params: () => `E: V·D = ${DIMS.micro.V * 8} (micro), ${fmtInt(66 * 128)} (training). P: T_max·D = 64 / ${fmtInt(128 * 128)}.`,
    changes: "Rows of E selected in the batch, and rows 0…T−1 of P.",
    fixed: "The ids themselves and the maximum position T_max.",
    failures: "T > T_max raises; ids ≥ V index out of range; forgetting position information makes the model permutation-blind.",
    verify: "E[x] equals one_hot(x) @ E; rows for unused ids receive exactly zero gradient.",
  },
  sections: () => {
    const f = M.fw, t = S.pos, id = M.ids[t];
    INS.E = paramInspect("tok_emb.weight", { rowLabels: MV.map(vis) });
    INS.P = paramInspect("pos_emb.weight");
    INS.x0 = actInspect("x0 = E[x] + P", ["T", "D"], f.x0, { rowLabels: posLabels() });
    return [
      {
        title: "Lookup, then broadcast add", tag: "tensor", prov: "browser", body: () => `${inputChips("pos", t)}
        <div class="tensors" style="margin-top:12px">
          ${tensor("tok_emb.weight  E", ["V", "D"], "par", heat(M.P["tok_emb.weight"], { kind: "par", size: "mid", rowLabels: MV.map(vis), rowSel: id, hl: (r) => r === id, dim: (r) => !M.ids.includes(r), digits: 2, inspect: INS.E, id: "E", onCell: (r) => { S.inspect = "E"; } }), { op: "emb_def", inspectKey: "E" })}
          <div style="display:flex;flex-direction:column;gap:14px">
          ${tensor("pos_emb.weight  P", ["T_max", "D"], "par", heat(M.P["pos_emb.weight"], { kind: "par", rowLabels: [0, 1, 2, 3, 4, 5, 6, 7], rowSel: t, hl: (r) => r === t, digits: 2, id: "P", inspect: INS.P }), { op: "emb_def", inspectKey: "P" })}
          ${tensor("x0", ["T", "D"], "act", heat(f.x0, { kind: "act", rowLabels: posLabels(), rowSel: t, hl: (r) => r === t, digits: 2, id: "x0", inspect: INS.x0 }), { op: "emb_fwd", inspectKey: "x0" })}
          </div></div>
        <h3>Position t = ${t}: '${esc(mvis(id))}' (id ${id})</h3>
        <div class="tensors">${tensor(`E[${id}]`, ["D"], "par", heat([f.tok[t]], { kind: "par", digits: 3 }))}<span class="op-sym">+</span>${tensor(`P[${t}]`, ["D"], "par", heat([f.pos[t]], { kind: "par", digits: 3 }))}<span class="op-sym">=</span>${tensor(`x0[${t}]`, ["D"], "act", heat([f.x0[t]], { kind: "act", digits: 3 }), { op: "emb_fwd" })}</div>
        <p style="margin-top:10px">${recordedCheck(f.x0, MICRO.trace.x0, "x0 vs PyTorch:")} <span class="small muted">Dimmed rows of E are characters absent from this window.</span></p>`,
      },
      {
        title: "The lookup is a one-hot matrix product", tag: "math", prov: "browser", body: () => {
          const oh = MV.map((_, i) => (i === id ? 1 : 0));
          const prod = M.P["tok_emb.weight"][0].map((_, j) => oh.reduce((a, v, i) => a + v * M.P["tok_emb.weight"][i][j], 0));
          return `<div class="eq">one_hot(${id}) ∈ ℝ<sup>${MV.length}</sup> = [${oh.join(" ")}]
one_hot(${id}) · <span class="w">E</span> = [${prod.map((v) => fmt(v, 3)).join(", ")}]
<span class="w">E</span>[${id}, :]       = [${M.P["tok_emb.weight"][id].map((v) => fmt(v, 3)).join(", ")}]</div>
          <p>Identical, ${Eng.maxAbsDiff(prod, M.P["tok_emb.weight"][id]) === 0 ? "bit for bit" : "to rounding"}. <code>nn.Embedding</code> performs the indexing without ever building the V-wide one-hot vector, which matters when V is 50,000+. The backward pass is the matching scatter-add: the gradient for row i is the sum of the upstream gradients at every position where id i occurs (chapter 11).</p>`;
        },
      },
      {
        title: "Same character, different positions", tag: "numerical", prov: "browser", body: () => {
          const counts = {}; M.ids.forEach((i, p) => (counts[i] = counts[i] || []).push(p));
          const rep = Object.entries(counts).filter(([, ps]) => ps.length > 1);
          if (!rep.length) return `<p>No character repeats in the current window. Edit the running example in the top bar (e.g. "First Cit" repeats 'i').</p>`;
          const [rid, ps] = rep[0];
          return `<p>'${esc(mvis(+rid))}' occurs at positions ${ps.join(" and ")}. Both select the same row E[${rid}]; only the position rows differ, so x0 differs:</p>
          <div class="tensors">${ps.map((p) => tensor(`x0[${p}]`, ["D"], "act", heat([M.fw.x0[p]], { kind: "act", digits: 3 }))).join("")}</div>
          <p class="small" style="margin-top:8px">After attention, each copy's vector also depends on the different preceding characters, so representations become context-dependent even though the token table is shared.</p>`;
        },
      },
      {
        title: "What this design commits to", tag: "engineering", body: () => `<div class="prose"><ul>
        <li><b>Both tables are parameters</b> updated by AdamW. Position information is part of the model computation, not an offline feature.</li>
        <li><b>Learned absolute positions</b> exist only for t &lt; T_max (${TRAINED.config.block_size} in training). The forward pass raises for longer inputs; generation crops the context to the last T_max tokens (chapter 15). Sinusoidal encodings and RoPE are compared in “Beyond the baseline”.</li>
        <li><b>Initialisation</b>: N(0, 0.02²) for both tables, as in GPT-2.</li>
        <li><b>Dropout</b> is applied to the sum (p = ${TRAINED.config.dropout} in training, 0 in the micro model).</li></ul></div>`,
      },
    ];
  },
  quiz: [
    { id: "q-emb-1", q: "In one batch the id for 'e' appears 37 times. What does the gradient of E['e'] equal?", options: ["The gradient at the first occurrence", "The average over occurrences", "The sum of the upstream gradients at all 37 positions", "Zero, because ids are integers"], answer: 2, explain: "The lookup reads the same row 37 times, so by the multivariate chain rule the contributions add (index_add / scatter-add in the backward pass)." },
  ],
});

/* ================= Chapter 6: Transformer block ================= */
segAct("lnSel", "ln", false, { compute: false });
function blockDiagram() {
  const W = 560, H = 430, cx = 120;
  const box = (y, label, sub, op, col, x = 220, w = 200) => `<g class="blk ${S.op === op ? "on" : ""}" data-op="${op}" tabindex="0" role="button" aria-label="${label}"><rect x="${x}" y="${y}" width="${w}" height="38" rx="7" fill="${col}" fill-opacity=".1" stroke="${col}"/><text x="${x + 12}" y="${y + 16}" style="font-weight:600">${label}</text><text class="shape-t" x="${x + 12}" y="${y + 30}">${sub}</text></g>`;
  const sh = (d) => shape(d).replace(/<[^>]+>/g, "");
  let s = `<svg viewBox="0 0 ${W} ${H}" class="diagram" role="img" aria-label="Pre-LayerNorm Transformer block">`;
  s += `<defs><marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#3f4452"/></marker></defs>`;
  s += `<line x1="${cx}" y1="20" x2="${cx}" y2="${H - 14}" stroke="var(--k-act)" stroke-width="5" opacity=".35"/>`;
  s += `<text x="${cx - 108}" y="34">residual stream x</text><text class="shape-t" x="${cx - 108}" y="48">${sh(["B", "T", "D"])}</text>`;
  const add = (y, op) => `<g class="blk ${S.op === op ? "on" : ""}" data-op="${op}"><circle cx="${cx}" cy="${y}" r="13" fill="#fff" stroke="#3f4452" stroke-width="1.5"/><text x="${cx - 5}" y="${y + 5}" style="font-size:16px">+</text></g>`;
  s += `<path d="M${cx},70 H200" stroke="#3f4452" fill="none" marker-end="url(#ar2)"/>`;
  s += box(52, "LayerNorm 1", "normalise over D · γ, β", "ln1_call", "#6d46e0");
  s += `<path d="M320,90 V104" stroke="#3f4452" marker-end="url(#ar2)"/>`;
  s += box(106, "Causal self-attention", `mixes positions · ${sh(["B", "H", "T", "T"])}`, "scores", "#1b6fc0");
  s += `<path d="M320,144 V158" stroke="#3f4452" marker-end="url(#ar2)"/>`;
  s += box(160, "Dropout (resid)", "p in training, off in eval()", "out_proj", "#59627a");
  s += `<path d="M220,180 H${cx + 16}" stroke="#3f4452" marker-end="url(#ar2)"/>` + add(180, "res_attn");
  s += `<text x="${cx + 20}" y="210" class="shape-t">h1 = x + Attn(LN1(x))  ${sh(["B", "T", "D"])}</text>`;
  s += `<path d="M${cx},240 H200" stroke="#3f4452" fill="none" marker-end="url(#ar2)"/>`;
  s += box(222, "LayerNorm 2", "normalise over D · γ, β", "ln2_call", "#6d46e0");
  s += `<path d="M320,260 V274" stroke="#3f4452" marker-end="url(#ar2)"/>`;
  s += box(276, "Feed-forward (GELU)", `per position · ${sh(["B", "T", "Dff"])} inside`, "ffn_fwd", "#1b6fc0");
  s += `<path d="M320,314 V328" stroke="#3f4452" marker-end="url(#ar2)"/>`;
  s += box(330, "Dropout (FFN)", "p in training, off in eval()", "ffn_fwd", "#59627a");
  s += `<path d="M220,350 H${cx + 16}" stroke="#3f4452" marker-end="url(#ar2)"/>` + add(350, "res_ffn");
  s += `<text x="${cx + 20}" y="384" class="shape-t">h2 = h1 + FFN(LN2(h1))  ${sh(["B", "T", "D"])}</text>`;
  s += `<text x="${cx - 108}" y="${H - 20}">to next block</text>`;
  return s + "</svg>";
}
ch({
  id: "block", num: 6, short: "Transformer block", title: "Anatomy of a pre-LayerNorm block", group: "The model", kind: "forward",
  thesis: "A block reads the residual stream through LayerNorm, computes an update with attention or the FFN, and adds the update back. The stream itself is never normalised in place, so information and gradients can pass straight through.",
  files: ["model.py"], ops: ["block_def", "ln1_call", "res_attn", "ln2_call", "res_ffn", "attn_drop", "out_proj", "ffn_fwd", "emb_drop", "blocks_loop"],
  card: {
    problem: "Repeatedly refine every position's vector using context (attention) and per-position computation (FFN).",
    ml: "A residual MLP block (ResNet-style skip connections), with one sub-layer that mixes across positions.",
    op: "h1 = x + Attn(LN1(x));  h2 = h1 + FFN(LN2(h1))",
    shapes: () => `${shape(["B", "T", "D"])} → ${shape(["B", "T", "D"])} (shape-preserving)`,
    params: () => `per block 4(D²+D) + 2·D·Dff + Dff + D + 4D = ${fmtInt(4 * (64 + 8) + 2 * 8 * 32 + 32 + 8 + 32)} (micro), ${fmtInt(198272)} (training)`,
    changes: "All projection weights, biases and both LayerNorm γ/β.",
    fixed: "The wiring, epsilon (1e-5) and dropout probability.",
    failures: "Shape mismatch in residual add; Post-LN with no warmup diverging; forgetting model.eval() keeps dropout on.",
    verify: "Output shape equals input shape; LayerNorm outputs have mean 0 and variance ≈ 1 per position before γ/β.",
  },
  sections: () => [
    { title: "Block diagram", tag: "pipeline", body: () => `<div style="overflow-x:auto"><div style="min-width:480px;max-width:620px">${blockDiagram()}</div></div><p class="small muted">Select a box to highlight its source line. Dropout also appears inside attention, on the probabilities (chapter 8), and once on the embedding sum before the first block.</p>` },
    {
      title: "LayerNorm, numerically", tag: "numerical", prov: "browser",
      ctrl: () => `${segC("lnSel", [["ln1", "LN1 (before attention)"], ["ln2", "LN2 (before FFN)"], ["lnf", "final LN"]], S.ln)}<span class="small muted">position</span>${inputChips("pos", S.pos).replace('class="chips"', 'class="chips" style="display:inline-flex"')}`,
      body: () => {
        const f = M.fw, t = S.pos, which = S.ln, ln = f[which], src = which === "ln1" ? f.x0 : which === "ln2" ? f.h1 : f.h2;
        const g = M.P[which === "lnf" ? "ln_f.weight" : `blocks.0.${which}.weight`], b = M.P[which === "lnf" ? "ln_f.bias" : `blocks.0.${which}.bias`];
        const x = src[t], mu = ln.mu[t], vr = ln.vr[t];
        const rec = which === "lnf" ? MICRO.trace.lnf : MICRO.trace[which];
        return `<div class="tensors">${tensor(`input[${t}]`, ["D"], "act", heat([x], { kind: "act", digits: 3 }))}<span class="op-sym">→</span>${tensor("x̂ = (x − μ)/√(σ²+ε)", ["D"], "act", heat([ln.xhat[t]], { kind: "act", digits: 3 }))}<span class="op-sym">→</span>${tensor("y = γ ⊙ x̂ + β", ["D"], "act", heat([ln.y[t]], { kind: "act", digits: 3 }), { op: which === "ln1" ? "ln1_call" : which === "ln2" ? "ln2_call" : "lnf" })}</div>
        <div class="calc" style="margin-top:10px">μ   = mean(x)            = ${fmt(mu, 6)}
σ²  = mean((x − μ)²)      = ${fmt(vr, 6)}   (biased: divides by D = 8, as PyTorch does)
1/√(σ²+ε) with ε = 1e-5   = ${fmt(ln.rstd[t], 6)}
γ = [${g.map((v) => fmt(v, 3)).join(", ")}]
β = [${b.map((v) => fmt(v, 3)).join(", ")}]
check: mean(x̂) = ${fmt(ln.xhat[t].reduce((a, v) => a + v, 0) / 8, 6)},  var(x̂) = ${fmt(ln.xhat[t].reduce((a, v) => a + v * v, 0) / 8, 6)}</div>
        <p style="margin-top:8px">${recordedCheck(ln.y, rec, `${which} output vs PyTorch:`)}</p>`;
      },
    },
    {
      title: "LayerNorm is not BatchNorm", tag: "math", body: () => `<div class="eq">LayerNorm:  μ<sub>b,t</sub> = (1/D) Σ<sub>d</sub> x<sub>b,t,d</sub>      — one mean per (b, t), reduced over features
BatchNorm:  μ<sub>d</sub>   = (1/(B·T)) Σ<sub>b,t</sub> x<sub>b,t,d</sub>  — one mean per feature, reduced over examples</div>
      <div class="tbl-wrap"><table class="data"><tr><th></th><th>LayerNorm (used here)</th><th>BatchNorm (CNN habit)</th></tr>
      <tr><td>reduces over</td><td>D, within one position</td><td>batch (and positions)</td></tr><tr><td>depends on other examples</td><td>no</td><td>yes</td></tr><tr><td>train vs eval behaviour</td><td>identical</td><td>batch stats vs running stats</td></tr><tr><td>causal-safe</td><td>yes: never mixes positions</td><td>statistics over T would leak future positions</td></tr><tr><td>parameters</td><td>γ, β ∈ ℝ<sup>D</sup></td><td>γ, β ∈ ℝ<sup>D</sup> + running mean/var buffers</td></tr></table></div>`,
    },
    {
      title: "Residuals, Pre-LN, and dropout placement", tag: "engineering", body: () => `<div class="prose"><p><b>Residual addition needs identical shapes.</b> <code>x + self.attn(a)</code> is element-wise on ${shape(["B", "T", "D"])}; every sub-layer must therefore project back to width D. That is why attention ends with W<sub>O</sub> ∈ ℝ<sup>D×D</sup> and the FFN with W<sub>2</sub> ∈ ℝ<sup>D×Dff</sup>.</p>
      <p><b>Gradient highway.</b> ∂h2/∂x contains an identity term from each skip connection, so gradients reach early layers without passing through every sub-layer's Jacobian.</p>
      <p><b>Pre-LN vs Post-LN.</b> The original Transformer normalised after the add (Post-LN). Normalising the branch input instead keeps the residual stream un-normalised and is markedly more stable to train without careful warmup (Xiong et al., 2020). Pre-LN needs a final LayerNorm before the vocabulary projection, which this model has.</p></div>
      <div class="tbl-wrap"><table class="data"><tr><th>dropout site</th><th>applies to</th><th>code</th></tr>
      <tr><td>embedding</td><td>E[x] + P</td><td><a href="#" data-act="code-op" data-arg="emb_drop">model.py:${OPS.emb_drop?.start}</a></td></tr><tr><td>attention probabilities</td><td>[B,H,T,T] after softmax</td><td><a href="#" data-act="code-op" data-arg="attn_drop">model.py:${OPS.attn_drop?.start}</a></td></tr><tr><td>attention output</td><td>after W<sub>O</sub>, before the add</td><td><a href="#" data-act="code-op" data-arg="out_proj">model.py:${OPS.out_proj?.start}</a></td></tr><tr><td>FFN output</td><td>after W<sub>2</sub>, before the add</td><td><a href="#" data-act="code-op" data-arg="ffn_fwd">model.py:${OPS.ffn_fwd?.start + 4}</a></td></tr></table></div>`,
    },
  ],
  quiz: [
    { id: "q-block-1", q: "Validation loss looks noisy and higher than expected. You find the eval function never calls model.eval(). What is happening?", explain: "Dropout stays active, so each evaluation randomly zeroes activations and rescales the rest. Validation measures a noisier, weaker model and varies between calls. estimate_loss() calls model.eval() and wraps everything in torch.no_grad(); it switches back with model.train() afterwards." },
  ],
});

/* ================= Chapter 7: Q, K, V ================= */
segAct("qkvSel", "qkv", false, { compute: false });
ch({
  id: "qkv", num: 7, short: "Q, K, V", title: "Queries, keys, values and heads", group: "The model", kind: "forward",
  thesis: "Three learned linear maps give every position a query (what it is looking for), a key (how it can be matched) and a value (what it hands over). Heads are just a reshape that lets several such matchings run side by side.",
  files: ["model.py"], ops: ["qkv_def", "qkv_proj", "head_split"],
  card: {
    problem: "Produce, per position, the vectors that attention will compare (Q, K) and mix (V).",
    ml: "Three Dense layers applied to every position with shared weights.",
    op: "Q = a W_Qᵀ + b_Q,  K = a W_Kᵀ + b_K,  V = a W_Vᵀ + b_V, then view(B,T,H,Dh).transpose(1,2)",
    shapes: () => `a ${shape(["B", "T", "D"])} → Q, K, V ${shape(["B", "T", "D"])} → ${shape(["B", "H", "T", "Dh"])}`,
    params: () => `3(D² + D) for Q/K/V (+ D² + D for W_O): 216 (+72) micro; ${fmtInt(3 * (16384 + 128))} (+${fmtInt(16512)}) training`,
    changes: "W_Q, W_K, W_V and their biases.",
    fixed: "H and Dh; the head split is a reshape with no parameters.",
    failures: "D not divisible by H; transposing before view (scrambles heads); forgetting .contiguous() before view after a transpose.",
    verify: "Head h of the split equals columns h·Dh…(h+1)·Dh−1 of the unsplit tensor; split → merge returns the original exactly.",
  },
  sections: () => {
    const f = M.fw, which = S.qkv, Wn = `blocks.0.attn.${which}_proj.weight`, bn = `blocks.0.attn.${which}_proj.bias`, out = f[which];
    const [ct, co] = S.qkvCell, Dh = 4;
    INS.a1 = actInspect("LN1(x0)", ["T", "D"], f.ln1.y, { rowLabels: posLabels() });
    INS.W = paramInspect(Wn);
    INS.out = actInspect(which.toUpperCase(), ["T", "D"], out, { rowLabels: posLabels() });
    const colLab = Array.from({ length: 8 }, (_, j) => `h${Math.floor(j / Dh)}·${j % Dh}`);
    return [
      {
        title: "Projection: a · Wᵀ + b", tag: "tensor", prov: "browser",
        ctrl: () => segC("qkvSel", [["q", "Query"], ["k", "Key"], ["v", "Value"]], S.qkv, "projection"),
        body: () => `<div class="tensors">${tensor("a = LN1(x0)", ["T", "D"], "act", heat(f.ln1.y, { kind: "act", rowLabels: posLabels(), rowSel: ct, hl: (r) => r === ct, digits: 2, id: "a1", inspect: INS.a1 }), { inspectKey: "a1", op: "ln1_call" })}
          <span class="op-sym">·</span>${tensor(`W_${which.toUpperCase()}ᵀ (stored as [out, in])`, ["D", "D"], "par", heat(M.P[Wn], { kind: "par", rowLabels: colLab, colLabels: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => "in" + i), rowSel: co, hl: (r) => r === co, digits: 2, id: "W", inspect: INS.W }), { inspectKey: "W", op: "qkv_def" })}
          <span class="op-sym">=</span>${tensor(which.toUpperCase(), ["T", "D"], "act", heat(out, { kind: "act", rowLabels: posLabels(), colLabels: colLab, sel: [ct, co], digits: 2, id: "out", inspect: INS.out, onCell: (r, c) => { S.qkvCell = [r, c]; refresh({ compute: false }); } }), { inspectKey: "out", op: "qkv_proj" })}</div>
          <h3>Element ${which.toUpperCase()}[${ct}, ${co}] term by term <span class="small muted">(select any cell of ${which.toUpperCase()})</span></h3>
          <div class="calc">${which.toUpperCase()}[${ct},${co}] = Σ<sub>i</sub> a[${ct},i] · W[${co},i]  +  b[${co}]
${f.ln1.y[ct].map((a, i) => `   ${fmt(a, 4).padStart(8)} × ${fmt(M.P[Wn][co][i], 4).padStart(8)} = ${fmt(a * M.P[Wn][co][i], 5).padStart(9)}`).join("\n")}
 + b[${co}] = ${fmt(M.P[bn][co], 5)}
 = ${fmt(out[ct][co], 6)}</div>
          <p class="small" style="margin-top:8px"><code>nn.Linear(in, out)</code> stores <code>weight</code> as <b>[out_features, in_features]</b> and computes <code>x @ weight.T + bias</code>. Row o of the stored matrix is the weight vector for output feature o, which is why the dot product above reads row ${co}.</p>`,
      },
      {
        title: "Splitting into heads is a reshape", tag: "tensor", prov: "browser", body: () => {
          const heads = f[which.toUpperCase() === "Q" ? "Q" : which.toUpperCase() === "K" ? "K" : "V"];
          return `<div class="calc">${which} = self.${which}_proj(x)              ${shape(["B", "T", "D"]).replace(/<[^>]+>/g, "")}
${which}.view(B, T, H, Dh)              ${shape(["B", "T", "H", "Dh"]).replace(/<[^>]+>/g, "")}     D = H × Dh  (${S.dims === "train" ? "128 = 4 × 32" : "8 = 2 × 4"})
   .transpose(1, 2)               ${shape(["B", "H", "T", "Dh"]).replace(/<[^>]+>/g, "")}</div>
          <div class="tensors" style="margin-top:12px" data-op="head_split">${heads.map((hm, h) => tensor(`${which.toUpperCase()}<sub>h=${h}</sub> = columns ${h * 4}–${h * 4 + 3}`, ["T", "Dh"], "act", heat(hm, { kind: "act", rowLabels: posLabels(), digits: 2 }))).join("")}</div>
          <p class="small" style="margin-top:8px">No numbers change: head h is literally columns h·Dh … (h+1)·Dh − 1. The transpose moves H next to B so the following matrix products batch over (B, H). Merging reverses it with <code>transpose(1, 2).contiguous().view(B, T, D)</code>; <code>test_shapes_and_head_split_merge</code> asserts the round trip is exact.</p>`;
        },
      },
      {
        title: "What Q, K and V do, in equations", tag: "math", body: () => `<div class="eq">score<sub>h</sub>(i, j) = <span class="v">q<sub>h,i</sub></span> · <span class="v">k<sub>h,j</sub></span> / √Dh            (query i compared with key j)
p<sub>h</sub>(i, ·)     = softmax<sub>j ≤ i</sub>( score<sub>h</sub>(i, j) )
out<sub>h,i</sub>       = Σ<sub>j</sub> p<sub>h</sub>(i, j) · <span class="v">v<sub>h,j</sub></span>                (values are what gets averaged)</div>
        <div class="prose"><p>The query vector of position i "selects" by producing large dot products with some keys. Keys only ever appear inside those scores. Values never affect the weights; they are the content that is mixed. In the micro example at head 0, position ${S.qi} gives its largest weight to key ${M.fw.P[0][S.qi].indexOf(Math.max(...M.fw.P[0][S.qi]))} (chapter 8 shows the numbers).</p>
        <p>Nothing assigns a head a linguistic job. Heads are initialised symmetrically-at-random and specialise only to whatever reduces loss; some trained heads have interpretable patterns, many do not.</p></div>`,
      },
    ];
  },
  quiz: [
    { id: "q-qkv-1", q: "D = 128, H = 4, B = 32, T = 128. What is the shape of q after view(B, T, H, Dh).transpose(1, 2)?", options: ["[32, 128, 4, 32]", "[32, 4, 128, 32]", "[32, 4, 32, 128]", "[4, 32, 128, 32]"], answer: 1, explain: "Dh = D/H = 32; the transpose swaps T and H: [B, H, T, Dh] = [32, 4, 128, 32]." },
  ],
});

/* ================= Chapter 8: Attention ================= */
const STAGES = ["Scores QKᵀ/√Dh", "Causal mask", "Masked scores", "Row-wise softmax", "Probabilities", "Weighted values PV"];
let PLAY = null;
function stopPlay() { if (PLAY) { clearInterval(PLAY); PLAY = null; } }
ACT.stage = (v) => { stopPlay(); S.stage8 = Math.max(0, Math.min(5, v === "next" ? S.stage8 + 1 : v === "prev" ? S.stage8 - 1 : v === "reset" ? 0 : +v)); refresh({ compute: false }); };
ACT.play = () => {
  if (PLAY) { stopPlay(); refresh({ compute: false }); return; }
  if (S.stage8 >= 5) S.stage8 = 0;
  PLAY = setInterval(() => { if (S.stage8 >= 5 || S.ch !== "attn") { stopPlay(); } else S.stage8++; refresh({ compute: false }); }, 1300 / S.speed);
  refresh({ compute: false });
};
segAct("head", "head", true, { compute: false });
segAct("speed", "speed", true, { compute: false });
ACT.mask = (v) => { S.causal = v === "1"; document.querySelectorAll('[data-act="mask"]').forEach((b) => b.setAttribute("aria-pressed", b.dataset.arg === v)); refresh(); };
["trL", "trH"].forEach((k) => segAct(k, k, true, { compute: false }));
ch({
  id: "attn", num: 8, short: "Attention & masking", title: "Attention scores and causal masking", group: "The model", kind: "forward",
  thesis: "Each position scores every key, the mask deletes the future, softmax turns each row into weights that sum to one, and the output is the weighted average of the value vectors.",
  files: ["model.py", "tests/test_llm.py"], ops: ["scores", "mask_buf", "mask_apply", "softmax", "attn_drop", "av", "t_causal"],
  keys: (e) => { if (e.key === " ") { e.preventDefault(); ACT.play(); } else if (e.key === "ArrowRight") ACT.stage("next"); else if (e.key === "ArrowLeft") ACT.stage("prev"); },
  card: {
    problem: "Let each position read information from earlier positions, with weights it computes itself.",
    ml: "A learned, input-dependent weighted average, like a soft nearest-neighbour lookup over the previous positions.",
    op: "softmax(mask(QKᵀ/√Dh)) · V",
    shapes: () => `Q, K, V ${shape(["B", "H", "T", "Dh"])} → scores ${shape(["B", "H", "T", "T"])} → out ${shape(["B", "H", "T", "Dh"])}`,
    params: "None in this step (the projections are chapter 7).",
    changes: "Nothing directly; gradients flow through to W_Q, W_K, W_V.",
    fixed: "The causal mask (a non-persistent buffer) and the 1/√Dh scale.",
    failures: "Softmax over the wrong axis; mask with 0 instead of −∞; no mask (target leakage); fp16 overflow in scores.",
    verify: "Changing tokens after position t leaves logits 0…t unchanged in eval mode; probability rows sum to 1 before dropout; upper triangle is exactly 0.",
  },
  sections: () => {
    const f = M.fw, h = S.head, T = 8, st = S.stage8, qi = S.qi, kj = S.kj;
    const Q = f.Q[h], K = f.K[h], V = f.V[h];
    const Kt = K[0].map((_, c) => K.map((r) => r[c]));
    const L = posLabels();
    const cellPick = (r, c) => { S.qi = r; S.kj = c; refresh({ compute: false }); };
    const masked = (r, c) => S.causal && c > r;
    let main = "";
    const sMat = st === 0 ? f.S[h] : st === 1 ? f.S[h].map((r, i) => r.map((_, j) => (S.causal && j > i ? -Infinity : 0))) : st === 2 ? f.Sm[h] : f.P[h];
    const matKind = st >= 3 ? "act" : "act";
    INS.Smat = actInspect(["scores", "mask", "masked scores", "probs", "probs", "probs"][st] + ` (head ${h})`, ["T", "T"], sMat, { rowLabels: L });
    main += `<div class="steps" role="tablist" aria-label="Attention stages">${STAGES.map((s, i) => `<button class="${i === st ? "cur" : i < st ? "done" : ""}" data-act="stage" data-arg="${i}" role="tab" aria-selected="${i === st}">${i + 1}. ${s}</button>`).join("")}</div>`;
    main += `<div class="stepper" style="margin-bottom:12px"><button class="btn small" data-act="stage" data-arg="reset" aria-label="Reset">⏮</button><button class="btn small" data-act="stage" data-arg="prev" aria-label="Step back">◀</button><button class="btn small primary" data-act="play" aria-label="${PLAY ? "Pause" : "Play"}">${PLAY ? "❚❚ Pause" : "▶ Play"}</button><button class="btn small" data-act="stage" data-arg="next" aria-label="Step forward">▶</button>${segC("speed", [[0.5, "0.5×"], [1, "1×"], [2, "2×"]], S.speed, "speed")}</div>`;
    const sOp = ["scores", "mask_buf", "mask_apply", "softmax", "softmax", "av"][st];
    main += `<div class="tensors">`;
    if (st <= 2) main += `${tensor(`Q<sub>h=${h}</sub>`, ["T", "Dh"], "act", heat(Q, { kind: "act", rowLabels: L, rowSel: qi, hl: (r) => r === qi, digits: 2 }), { op: "head_split" })}<span class="op-sym">·</span>${tensor(`K<sub>h=${h}</sub>ᵀ`, ["Dh", "T"], "act", heat(Kt, { kind: "act", colLabels: L, colSel: kj, hl: (r, c) => c === kj, digits: 2 }), { op: "head_split" })}<span class="op-sym">→</span>`;
    main += tensor(["S = QKᵀ/√Dh", "mask (−∞ above diagonal)", "S masked", "softmax over each row", "P", "P"][st], ["T", "T"], st === 1 ? "in" : "act", heat(st === 1 ? sMat.map((r) => r.map((v) => (v === 0 ? 1 : v))) : sMat, { kind: st === 1 ? "in" : matKind, rowLabels: L, colLabels: L, sel: [qi, kj], masked: st >= 1 ? masked : null, digits: st >= 3 ? 2 : 2, id: "Smat", inspect: INS.Smat, onCell: cellPick, hl: !S.causal && st >= 3 ? (r, c) => c === r + 1 : null, values: st !== 1 }), { op: sOp, inspectKey: "Smat" });
    if (st >= 3) main += `<div><div class="small muted" style="margin-bottom:4px">row sums</div>${heat(f.P[h].map((r) => [r.reduce((a, v) => a + v, 0)]), { kind: "tgt", digits: 3, max: 1.5 })}</div>`;
    if (st === 5) main += `<span class="op-sym">·</span>${tensor(`V<sub>h=${h}</sub>`, ["T", "Dh"], "act", heat(V, { kind: "act", rowLabels: L, digits: 2, hl: (r) => f.P[h][qi][r] > 0.15 }), { op: "head_split" })}<span class="op-sym">=</span>${tensor(`O<sub>h=${h}</sub>`, ["T", "Dh"], "act", heat(f.O[h], { kind: "act", rowLabels: L, rowSel: qi, hl: (r) => r === qi, digits: 2 }), { op: "av" })}`;
    main += `</div>`;
    let detail = "";
    if (st <= 2) {
      const terms = Q[qi].map((q, c) => q * K[kj][c]);
      detail = `<h3>Score for query ${qi} ('${esc(mvis(M.ids[qi]))}') against key ${kj} ('${esc(mvis(M.ids[kj]))}')</h3><div class="calc">S[${qi},${kj}] = ( ${terms.map((_, c) => `q${c}·k${c}`).join(" + ")} ) / √${4}
        = ( ${terms.map((v) => fmt(v, 4)).join(" + ")} ) / 2
        = ${fmt(terms.reduce((a, v) => a + v, 0), 5)} / 2 = ${fmt(f.S[h][qi][kj], 5)}${masked(qi, kj) ? `\n   key ${kj} is in the future of query ${qi} → replaced by −∞ (st. 2–3)` : ""}</div>`;
    } else if (st <= 4) {
      const row = f.Sm[h][qi], m = Math.max(...row.filter(Number.isFinite)), ex = row.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0)), Z = ex.reduce((a, v) => a + v, 0);
      detail = `<h3>Softmax of row ${qi} (over keys j — the last axis)</h3><div class="calc">max = ${fmt(m, 4)}   (subtracted for numerical stability; does not change the result)
${row.map((v, j) => `j=${j}: exp(${fmt(v, 3)} − ${fmt(m, 3)}) = ${fmt(ex[j], 5)}   → p = ${fmt(ex[j] / Z, 5)}`).join("\n")}
Σ exp = ${fmt(Z, 5)},   Σ p = ${fmt(f.P[h][qi].reduce((a, v) => a + v, 0), 6)}</div>`;
    } else {
      detail = `<h3>Output row ${qi} = Σ<sub>j</sub> P[${qi}, j] · V[j]</h3><div class="calc">${f.P[h][qi].map((p, j) => (p > 0 ? `${fmt(p, 3)} × [${V[j].map((v) => fmt(v, 2)).join(", ")}]` : null)).filter(Boolean).join("\n+ ")}
= [${f.O[h][qi].map((v) => fmt(v, 4)).join(", ")}]</div>`;
    }
    const why = [
      "Rows index queries i, columns index keys j. Entry (i, j) is how strongly position i wants information from position j. Select any cell to see its dot product.",
      "A boolean lower-triangular buffer: key j is allowed for query i only if j ≤ i. The diagonal is allowed because position i's own input token is known when predicting token i + 1.",
      "Forbidden scores become −∞, so exp(−∞) = 0 after softmax. Using 0 instead of −∞ would still give future keys weight e⁰ relative to the others.",
      "Softmax runs over the key axis (dim = −1): each query distributes one unit of attention across the keys it may see. Softmax over the query axis would normalise the wrong thing.",
      "Every row sums to 1 before dropout. During training, dropout zeroes some entries and scales the rest by 1/(1−p), so realised rows no longer sum to 1; only the expectation does.",
      "Each output row is a convex combination of value vectors. Heads then concatenate back to width D (chapter 9).",
    ][st];
    return [
      {
        title: "Scaled dot-product attention, step by step", tag: "numerical", prov: "browser",
        ctrl: () => `${segC("head", [[0, "head 0"], [1, "head 1"]], S.head, "head")}${segC("mask", [["1", "causal mask on"], ["0", "mask off (leak demo)"]], S.causal ? "1" : "0", "mask")}`,
        body: () => `${!S.causal ? `<div class="callout bad"><b>Educational setting: the mask is removed</b>This configuration is never used for training. Outlined cells are (i, i+1): the key that holds exactly the token query i must predict.</div>` : ""}${main}<p class="small" style="margin-top:10px">${why}</p>${detail}<p style="margin-top:8px">${S.causal ? recordedCheck(f.P, MICRO.trace.probs, "attention probabilities vs PyTorch:") : ""}</p>`,
      },
      {
        title: "Why the mask matters: target leakage", tag: "numerical", prov: ["browser", "recorded"], body: () => {
          const fm = M.fwMasked, fu = Eng.forward(M.P, CFG, M.ids, M.tg, { causal: false });
          const c = MICRO.curves, avg = (a, i) => a.slice(Math.max(0, i - 25), i + 25).reduce((x, y) => x + y, 0) / Math.min(50, a.length);
          const pts = (a) => a.map((v, i) => [i + 1, v]).filter((_, i) => i % 5 === 0).map(([x], k) => [x, avg(a, x - 1)]);
          return `<div class="grid2"><div><p>With the same weights, removing the mask changes this window's loss from <b>${fmt(fm.loss, 4)}</b> (masked) to <b>${fmt(fu.loss, 4)}</b> (unmasked). The weights were learned under the mask, so suddenly visible future keys look like noise to them.</p>
          <p>Train with the mask removed instead, and the model learns to copy the next input token: the curves show two micro models trained from the same initialisation on the same batches, differing only in <code>causal</code>.</p>
          <div class="callout"><b>Why the low curve is worthless</b>At generation time position t + 1 does not exist yet, so a model that learned to read it has nothing to read. Training loss improved by cheating. <code>test_causal_mask_no_future_leakage</code> guards against this regression.</div></div>
          <div>${lineChart([{ name: "causal mask (valid)", color: "var(--k-act)", points: pts(c.masked) }, { name: "mask removed (leaks targets)", color: "var(--k-grad)", dash: "5 4", points: pts(c.unmasked) }], { xLabel: "micro training step", yLabel: "loss (50-step mean)", h: 220, aria: "Masked vs unmasked training loss" })}</div></div>`;
        },
      },
      {
        title: "Why divide by √Dh", tag: "math", prov: "browser", body: () => {
          const raw = f.S[h].flat().map((v) => v * 2);
          const varOf = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length; };
          return `<div class="eq">if q, k have independent components with mean 0 and variance 1:
Var(q · k) = Σ<sub>c=1..Dh</sub> Var(q<sub>c</sub> k<sub>c</sub>) = Dh        ⇒  Var(q · k / √Dh) = 1</div>
          <p>Without the scale, larger heads produce larger logits, softmax saturates towards one-hot, and its gradient (p(1 − p) terms) vanishes. For this head the variance of raw dot products over all 64 pairs is ${fmt(varOf(raw), 4)} and after scaling ${fmt(varOf(f.S[h].flat()), 4)} (the trained micro weights are not unit-variance, so the ratio is exactly Dh = 4 but the absolute values are not 4 and 1).</p>`;
        },
      },
      {
        title: "A trained model's attention", tag: "tensor", prov: "recorded",
        ctrl: () => `${segC("trL", [0, 1, 2, 3].map((l) => [l, "layer " + l]), S.trL)}${segC("trH", [0, 1, 2, 3].map((x) => [x, "head " + x]), S.trH)}`,
        body: () => {
          const A = TRAINED.attention, mp = A.maps[S.trL][S.trH], chars = Array.from(A.prompt);
          const lab = chars.map((c, i) => vis(c));
          return `<p class="small">The 4-layer model after ${fmtInt(TRAINED.step)} steps, on a validation prompt it never trained on: “${esc(visStr(A.prompt))}”. Rows are queries, columns keys; darker means more weight.</p>
          <div style="overflow-x:auto">${heat(mp, { kind: "act", size: "compact", rowLabels: lab, colLabels: lab, max: 1, masked: (r, c) => c > r })}</div>
          <div class="grid2" style="margin-top:10px"><div><h3 style="margin-top:0">Next-character distribution after the prompt</h3>${bars(A.next.map((n) => n.p), { labels: A.next.map((n) => vis(n.char)), kind: "tgt", h: 120, w: 420, min: 0 })}</div>
          <div class="callout warn"><b>Attention maps are not explanations</b>A weight says how much of a value vector was averaged in, not whether that value mattered for the output: value norms, the output projection, later layers and the residual stream all intervene (Jain & Wallace, 2019). Treat maps as hypotheses to test, e.g. by ablation.</div></div>`;
        },
      },
    ];
  },
  quiz: [
    { id: "q-attn-1", q: "scores has shape [B, H, T, T]. Over which axis must softmax run?", options: ["dim=0 (batch)", "dim=1 (heads)", "dim=-2 (queries)", "dim=-1 (keys)"], answer: 3, explain: "Each query distributes its attention over keys, so each row (fixed b, h, i) must sum to 1: softmax over the last axis." },
    { id: "q-attn-2", q: "Context length doubles from 1,024 to 2,048 with everything else fixed. By what factor does the explicit attention-probability tensor grow?", options: ["2×", "4×", "8×", "unchanged"], answer: 1, explain: "It has B·H·T² entries, so doubling T multiplies it by 4. Activations shaped [B, T, D] only double." },
    { id: "q-attn-3", q: "Explain exactly what leaks if the causal mask is removed during training.", explain: "Target y[t] equals input x[t+1]. Without the mask, query t can attend to key t+1 and copy its token, so the loss can approach zero without learning language. At generation time that future token does not exist, so the learned shortcut fails." },
  ],
});
