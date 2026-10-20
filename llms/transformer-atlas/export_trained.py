"""Record facts about the trained checkpoint for the visual reference (web/data/trained.json).

Everything here is measured from out/ckpt.pt and out/log.jsonl:
parameter breakdown, checkpoint contents, reload equivalence, exhaustive
validation/test loss, attention maps for one validation prompt, gradient norms
for one batch, generated samples, and one real request to the FastAPI app.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint, load_for_inference, save_checkpoint
from data import load_text, sequential_windows, split_text
from generate import generate

CKPT = Path("out/ckpt.pt")


def rnd(t, d=4):
    return [rnd(x, d) for x in t] if isinstance(t, list) else round(t, d)


@torch.no_grad()
def exhaustive_loss(model, ids, T=128, B=64):
    """Mean CE over non-overlapping windows covering the whole split."""
    xs, ys = zip(*sequential_windows(ids, T, stride=T))
    total, n = 0.0, 0
    for i in range(0, len(xs), B):
        x, y = torch.stack(xs[i:i + B]), torch.stack(ys[i:i + B])
        logits, _ = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
        n += y.numel()
    return total / n, n


def main():
    model, tok, ckpt = load_for_inference(CKPT)
    cfg = model.cfg
    text = load_text("data/input.txt")
    tr_txt, va_txt, te_txt = split_text(text)
    val = torch.tensor(tok.encode(va_txt, strict=False))
    test = torch.tensor(tok.encode(te_txt, strict=False))

    params = [{"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in model.named_parameters()]

    # checkpoint contents
    raw = load_checkpoint(CKPT)
    opt_state = raw["optimizer"]["state"]
    opt_bytes = sum(t.numel() * t.element_size() for s in opt_state.values() for t in s.values()
                    if isinstance(t, torch.Tensor))
    model_bytes = sum(t.numel() * t.element_size() for t in raw["model"].values())
    contents = [
        {"key": "model", "what": f"state_dict, {len(raw['model'])} tensors", "bytes": model_bytes, "infer": True, "resume": True},
        {"key": "model_config", "what": json.dumps(raw["model_config"]), "bytes": None, "infer": True, "resume": True},
        {"key": "tokenizer", "what": f"{len(raw['tokenizer']['chars'])} characters + <unk>", "bytes": None, "infer": True, "resume": True},
        {"key": "optimizer", "what": f"AdamW exp_avg + exp_avg_sq + step for {len(opt_state)} tensors", "bytes": opt_bytes, "infer": False, "resume": True},
        {"key": "step", "what": str(raw["step"]), "bytes": None, "infer": False, "resume": True},
        {"key": "train_config", "what": "lr schedule, batch size, eval settings", "bytes": None, "infer": False, "resume": True},
        {"key": "rng", "what": "Python, NumPy, torch CPU/CUDA, data-sampler generator", "bytes": None, "infer": False, "resume": True},
        {"key": "data", "what": f"sha256 {raw['data']['sha256'][:16]}..., split sizes {raw['data']['split_chars']}", "bytes": None, "infer": False, "resume": True},
    ]

    # reload equivalence: save -> load into a fresh model -> compare logits
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "copy.pt"
        save_checkpoint(p, model=model, optimizer=None, step=raw["step"], tokenizer=tok, train_cfg={})
        m2, _, _ = load_for_inference(p)
        x = val[None, :128]
        with torch.no_grad():
            l1, l2 = model(x)[0], m2(x)[0]
        reload_diff = (l1 - l2).abs().max().item()

    val_loss, val_n = exhaustive_loss(model, val)
    test_loss, test_n = exhaustive_loss(model, test)

    # attention maps for one validation prompt
    start = va_txt.find("\n") + 1
    prompt = va_txt[start:start + 48]
    pid = torch.tensor([tok.encode(prompt)])
    tr = {}
    with torch.no_grad():
        logits, _ = model(pid, trace=tr)
    attn = [[rnd(tr[f"block{l}"]["probs"][0, h].tolist(), 3) for h in range(cfg.n_head)] for l in range(cfg.n_layer)]
    probs = torch.softmax(logits[0, -1], -1)
    top = torch.topk(probs, 10)
    next_dist = [{"char": tok.itos[i], "p": round(p, 4)} for p, i in zip(top.values.tolist(), top.indices.tolist())]

    # one training-mode batch: per-parameter gradient norms
    model.train()
    g = torch.Generator().manual_seed(7)
    ids_tr = torch.tensor(tok.encode(tr_txt[:200000]))
    starts = torch.randint(len(ids_tr) - 129, (32,), generator=g).tolist()
    xb = torch.stack([ids_tr[s:s + 128] for s in starts])
    yb = torch.stack([ids_tr[s + 1:s + 129] for s in starts])
    torch.manual_seed(0)
    _, loss = model(xb, yb)
    loss.backward()
    grad_norms = [{"name": n, "norm": p.grad.norm().item()} for n, p in model.named_parameters()]
    model.zero_grad(set_to_none=True)
    model.eval()

    samples = []
    for prompt_s, temp, seed in (("ROMEO:\n", 0.8, 1), ("JULIET:\n", 0.8, 2), ("\n", 0.5, 3), ("\n", 1.0, 3), ("KING", 0.0, 0)):
        gen = torch.Generator().manual_seed(seed)
        out = generate(model, torch.tensor([tok.encode(prompt_s)]), 300, temp, None, gen)
        samples.append({"prompt": prompt_s, "temperature": temp, "seed": seed, "text": tok.decode(out[0].tolist())})

    # one real request through the FastAPI app
    from fastapi.testclient import TestClient
    os.environ["CKPT"] = str(CKPT)
    import serve
    req = {"prompt": "ROMEO:\n", "max_new_tokens": 120, "temperature": 0.8, "seed": 1}
    with TestClient(serve.app) as c:
        resp = c.post("/generate", json=req).json()
        bad = c.post("/generate", json={"prompt": "Hello 👋"})
        bad_resp = {"status": bad.status_code, "body": bad.json()}

    log = [json.loads(l) for l in open("out/log.jsonl")]
    out = {
        "provenance": "Measured from out/ckpt.pt by export_trained.py",
        "env": {"torch": torch.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
                "device": "cpu", "threads": torch.get_num_threads()},
        "config": {**cfg.to_dict(), "head_dim": cfg.head_dim, "d_ff": cfg.d_ff},
        "train_config": raw["train_config"],
        "step": raw["step"],
        "num_params": model.num_params(),
        "params": params,
        "vocab": tok.itos,
        "data": raw["data"],
        "ckpt_file_bytes": CKPT.stat().st_size,
        "ckpt_contents": contents,
        "reload_max_abs_diff": reload_diff,
        "val_loss_exhaustive": val_loss, "val_targets": val_n,
        "test_loss_exhaustive": test_loss, "test_targets": test_n,
        "attention": {"prompt": prompt, "maps": attn, "next": next_dist},
        "grad_norms": grad_norms, "grad_norm_batch_loss": loss.item(),
        "samples": samples,
        "serve": {"request": req, "response": resp, "bad_request": {"prompt": "Hello 👋"}, "bad_response": bad_resp},
        "log": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in log],
    }
    Path("web/data/trained.json").write_text(json.dumps(out))
    print(f"step={raw['step']} params={model.num_params():,} val={val_loss:.4f} test={test_loss:.4f} "
          f"ppl={math.exp(val_loss):.2f} reload_diff={reload_diff}")


if __name__ == "__main__":
    main()
