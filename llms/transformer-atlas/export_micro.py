"""Record the fully inspectable micro-example used by the visual reference.

Same GPT class, same tokenizer class, same optimizer factory as the real run;
only the sizes differ (B=1, T=8, D=8, H=2, Dh=4, L=1, Dff=32).

Writes web/data/micro.json containing REAL values:
  * the excerpt, its vocabulary and the canonical (input, target) window
  * every parameter (snapshot right before the recorded step)
  * every intermediate activation of the forward pass
  * every parameter gradient from loss.backward()
  * AdamW state before/after and parameter values after optimizer.step()
  * masked vs unmasked training curves (the leakage demonstration)
  * a gradient-accumulation comparison and a finite-difference check
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from config import MICRO, ModelConfig, TrainConfig
from data import load_text
from model import GPT
from tokenizer import CharTokenizer
from train import make_optimizer

SEED = 1337
EXCERPT_CHARS = 250
PRETRAIN_STEPS = 1500


def r(t):
    """tensor -> nested list rounded to float32-meaningful precision"""
    if isinstance(t, torch.Tensor):
        t = t.detach().double().tolist()
    if isinstance(t, list):
        return [r(v) for v in t]
    if isinstance(t, float):
        if math.isinf(t):
            return "-inf" if t < 0 else "inf"
        return float(f"{t:.9g}")
    return t


def windows(ids, T, B, g):
    starts = torch.randint(len(ids) - T, (B,), generator=g)
    x = torch.stack([ids[s:s + T] for s in starts.tolist()])
    y = torch.stack([ids[s + 1:s + T + 1] for s in starts.tolist()])
    return x, y


def pretrain(model, ids, tc, steps, seed):
    opt = make_optimizer(model, tc)
    g = torch.Generator().manual_seed(seed)
    curve = []
    for step in range(steps):
        x, y = windows(ids, model.cfg.block_size, 8, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        curve.append(round(loss.item(), 5))
    return opt, curve


def main():
    text = load_text("data/input.txt")
    excerpt = text[:EXCERPT_CHARS]
    tok = CharTokenizer.build(excerpt)
    ids = torch.tensor(tok.encode(excerpt))
    cfg = ModelConfig(vocab_size=tok.vocab_size, **MICRO)
    tc = TrainConfig(lr=1e-2, weight_decay=0.1, grad_clip=1.0)

    # --- masked model: the valid baseline ---
    torch.manual_seed(SEED)
    model = GPT(cfg)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    opt, curve_masked = pretrain(model, ids, tc, PRETRAIN_STEPS, SEED)

    # --- same init, same batches, mask removed: leakage demo (never a valid config) ---
    leaky = GPT(cfg)
    leaky.load_state_dict(init_state)
    for b in leaky.blocks:
        b.attn.causal = False
    _, curve_leaky = pretrain(leaky, ids, tc, PRETRAIN_STEPS, SEED)

    # --- the recorded step ---
    T = cfg.block_size
    x = ids[None, :T]
    y = ids[None, 1:T + 1]
    model.train()  # dropout = 0, so train/eval give identical numbers here
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    adam_before = {}
    for n, p in model.named_parameters():
        st = opt.state[p]
        adam_before[n] = {"exp_avg": r(st["exp_avg"]), "exp_avg_sq": r(st["exp_avg_sq"]),
                          "step": int(st["step"].item())}

    opt.zero_grad(set_to_none=True)
    trace = {}
    logits, loss = model(x, y, trace=trace)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
    clipped = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    lr = opt.param_groups[0]["lr"]
    opt.step()
    params_after = {n: p.detach().clone() for n, p in model.named_parameters()}
    adam_after = {n: {"exp_avg": r(opt.state[p]["exp_avg"]), "exp_avg_sq": r(opt.state[p]["exp_avg_sq"]),
                      "step": int(opt.state[p]["step"].item())} for n, p in model.named_parameters()}
    with torch.no_grad():
        _, loss_after = model(x, y)

    wd = {}
    for gi, group in enumerate(opt.param_groups):
        for p in group["params"]:
            for n, q in model.named_parameters():
                if q is p:
                    wd[n] = group["weight_decay"]

    # --- unmasked forward on the SAME pre-step weights (illustrates what leaks) ---
    probe = GPT(cfg)
    probe.load_state_dict({n: params_before[n] for n in params_before}, strict=False)
    for b in probe.blocks:
        b.attn.causal = False
    utrace = {}
    with torch.no_grad():
        ulogits, uloss = probe(x, y, trace=utrace)

    # --- gradient accumulation on two windows (real numbers) ---
    ga = GPT(cfg)
    ga.load_state_dict({n: params_before[n] for n in params_before}, strict=False)
    xb = torch.stack([ids[0:T], ids[9:9 + T]])
    yb = torch.stack([ids[1:T + 1], ids[10:10 + T]])
    ga(xb, yb)[1].backward()
    full = ga.lm_head.weight.grad.clone()
    ga.zero_grad()
    for i in range(2):
        (ga(xb[i:i + 1], yb[i:i + 1])[1] / 2).backward()
    acc_scaled = ga.lm_head.weight.grad.clone()
    ga.zero_grad()
    for i in range(2):
        ga(xb[i:i + 1], yb[i:i + 1])[1].backward()
    acc_unscaled = ga.lm_head.weight.grad.clone()
    grad_accum = {"param": "lm_head.weight", "windows": [tok.decode(xb[0].tolist()), tok.decode(xb[1].tolist())],
                  "full": r(full), "accum_scaled": r(acc_scaled), "accum_unscaled": r(acc_unscaled),
                  "max_abs_diff_scaled": (full - acc_scaled).abs().max().item(),
                  "ratio_unscaled": (acc_unscaled.norm() / full.norm()).item()}

    # --- finite-difference check in float64 on the same weights ---
    fd_model = GPT(cfg).double()
    fd_model.load_state_dict({n: params_before[n].double() for n in params_before}, strict=False)
    fd_model(x, y)[1].backward()
    fd = []
    eps = 1e-6
    for name, idx in (("blocks.0.attn.q_proj.weight", (1, 2)), ("tok_emb.weight", (int(x[0, 0]), 3)),
                      ("blocks.0.ffn.fc1.bias", (5,)), ("lm_head.weight", (int(y[0, 0]), 0))):
        p = dict(fd_model.named_parameters())[name]
        old = p.data[idx].item()
        p.data[idx] = old + eps
        lp = fd_model(x, y)[1].item()
        p.data[idx] = old - eps
        lm = fd_model(x, y)[1].item()
        p.data[idx] = old
        fd.append({"param": name, "index": list(idx), "numeric": (lp - lm) / (2 * eps),
                   "autograd": p.grad[idx].item()})

    blk = trace["block0"]
    out = {
        "provenance": "Recorded from export_micro.py (PyTorch %s, CPU, float32)." % torch.__version__,
        "excerpt": excerpt,
        "vocab": tok.itos,
        "config": {**cfg.to_dict(), "head_dim": cfg.head_dim, "d_ff": cfg.d_ff},
        "seed": SEED,
        "pretrain_steps": PRETRAIN_STEPS,
        "optimizer": {"type": "AdamW", "lr": lr, "betas": list(tc.betas), "eps": tc.eps,
                      "weight_decay": wd, "grad_clip": tc.grad_clip},
        "input_ids": x[0].tolist(), "target_ids": y[0].tolist(),
        "input_text": tok.decode(x[0].tolist()), "target_text": tok.decode(y[0].tolist()),
        "params": {n: r(p) for n, p in params_before.items()},
        "shapes": {n: list(p.shape) for n, p in params_before.items()},
        "trace": {
            "x0": r(trace["x0"][0]), "ln1": r(blk["ln1"][0]),
            "scores_raw": r(blk["scores_raw"][0]), "scores_masked": r(blk["scores_masked"][0]),
            "probs": r(blk["probs"][0]), "head_out": r(blk["head_out"][0]), "merged": r(blk["merged"][0]),
            "h1": r(blk["h1"][0]), "ln2": r(blk["ln2"][0]), "ffn_pre": r(blk["ffn_pre"][0]),
            "ffn_act": r(blk["ffn_act"][0]), "h2": r(blk["h2"][0]), "lnf": r(trace["lnf"][0]),
            "logits": r(logits[0]), "loss": loss.item(),
        },
        "grads": {n: r(g) for n, g in grads.items()},
        "grad_norm_total": total_norm,
        "grad_clipped": total_norm > tc.grad_clip,
        "grads_clipped": {n: r(g) for n, g in clipped.items()},
        "adam_before": adam_before, "adam_after": adam_after,
        "params_after": {n: r(p) for n, p in params_after.items()},
        "loss_after_step": loss_after.item(),
        "unmasked": {"probs": r(utrace["block0"]["probs"][0]), "logits": r(ulogits[0]), "loss": uloss.item()},
        "curves": {"masked": curve_masked, "unmasked": curve_leaky},
        "grad_accum": grad_accum,
        "finite_difference": fd,
    }
    Path("web/data").mkdir(parents=True, exist_ok=True)
    Path("web/data/micro.json").write_text(json.dumps(out))
    print(f"V={tok.vocab_size} loss={loss.item():.4f} after={loss_after.item():.4f} "
          f"gnorm={total_norm:.4f} unmasked_loss={uloss.item():.4f} "
          f"final curves masked={curve_masked[-1]} leaky={curve_leaky[-1]}")
    for f in fd:
        print(f)


if __name__ == "__main__":
    main()
