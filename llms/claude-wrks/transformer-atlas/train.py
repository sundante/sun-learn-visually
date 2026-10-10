"""Pretraining loop for the char-level GPT on Tiny Shakespeare.

python train.py                       # fresh run (defaults in config.py)
python train.py --resume              # continue from out/ckpt.pt with identical RNG/optimizer state
python train.py --max_steps 200 --device cuda --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint, restore_for_resume, save_checkpoint
from config import ModelConfig, TrainConfig
from data import get_batch, load_text, sha256_of, split_text
from generate import generate
from model import GPT
from tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    # Seeds make runs repeatable on the SAME hardware/software stack. Different GPUs,
    # kernels, thread counts or library versions can still produce tiny numeric drift.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model: GPT, tc: TrainConfig) -> torch.optim.AdamW:
    # Decoupled weight decay on matrices (Linear weights, embeddings); none on biases / LayerNorm.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=tuple(tc.betas), eps=tc.eps)


@torch.no_grad()
def estimate_loss(model, splits: dict, tc: TrainConfig, T: int, device, gen) -> dict:
    model.eval()                                   # dropout OFF for evaluation
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tc.eval_iters)
        for i in range(tc.eval_iters):
            x, y = get_batch(data, tc.batch_size, T, device, gen)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    tc_defaults, mc_defaults = TrainConfig(), ModelConfig()
    for k, v in tc_defaults.to_dict().items():
        if not isinstance(v, list):
            ap.add_argument(f"--{k}", type=type(v), default=v)
    for k in ("block_size", "n_layer", "n_head", "n_embd", "ffn_mult", "dropout"):
        ap.add_argument(f"--{k}", type=type(getattr(mc_defaults, k)), default=getattr(mc_defaults, k))
    ap.add_argument("--tie_weights", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tc = TrainConfig(**{k: getattr(args, k) for k in tc_defaults.to_dict() if hasattr(args, k)})
    out_dir = Path(tc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    set_seed(tc.seed)
    device = pick_device(tc.device)

    # ---- data: split first, then tokenize each split with a train-only vocabulary ----
    text = load_text(tc.data_path)
    train_txt, val_txt, test_txt = split_text(text, tc.split)
    tok = CharTokenizer.build(train_txt)
    splits = {name: torch.tensor(tok.encode(t, strict=False), dtype=torch.long)
              for name, t in (("train", train_txt), ("val", val_txt))}
    data_info = {"path": tc.data_path, "sha256": sha256_of(tc.data_path), "chars": len(text),
                 "split_chars": [len(train_txt), len(val_txt), len(test_txt)],
                 "unk_in_val": int((splits["val"] == 0).sum())}

    mc = ModelConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                     n_head=args.n_head, n_embd=args.n_embd, ffn_mult=args.ffn_mult,
                     dropout=args.dropout, tie_weights=args.tie_weights)
    model = GPT(mc).to(device)
    opt = make_optimizer(model, tc)
    data_gen = torch.Generator().manual_seed(tc.seed)      # dedicated sampler RNG (saved in ckpt)
    eval_gen = torch.Generator()
    start_step = 0
    if args.resume:
        start_step = restore_for_resume(load_checkpoint(ckpt_path, device), model, opt, data_gen)
        print(f"resumed at step {start_step}")
    print(f"device={device}  params={model.num_params():,}  V={mc.vocab_size}  {data_info}")
    tok.save(out_dir / "vocab.json")

    log = open(out_dir / "log.jsonl", "a")
    T, t0 = mc.block_size, time.time()
    for step in range(start_step, tc.max_steps + 1):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % tc.eval_interval == 0 or step == tc.max_steps:
            eval_gen.manual_seed(1234)                       # same eval batches every time
            losses = estimate_loss(model, splits, tc, T, device, eval_gen)
            sample_gen = torch.Generator().manual_seed(0)
            prompt = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
            sample = tok.decode(generate(model, prompt, tc.sample_chars, 0.8, None, sample_gen)[0].tolist())
            rec = {"type": "eval", "step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                   "val_ppl": math.exp(losses["val"]), "sample": sample, "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            save_checkpoint(ckpt_path, model=model, optimizer=opt, step=step, tokenizer=tok,
                            train_cfg=tc.to_dict(), data_gen=data_gen, data_info=data_info,
                            metrics=losses)
        if step == tc.max_steps:
            break

        # ---- one optimizer step (possibly several microbatches) ----
        model.train()
        opt.zero_grad(set_to_none=True)                     # PyTorch ACCUMULATES .grad otherwise
        loss_sum = 0.0
        for _ in range(tc.grad_accum):
            x, y = get_batch(splits["train"], tc.batch_size, T, device, data_gen)
            _, loss = model(x, y)                           # forward + mean CE over B*T targets
            (loss / tc.grad_accum).backward()               # scale so summed grads = mean over all microbatches
            loss_sum += loss.item()
        if tc.grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)  # returns pre-clip norm
        else:
            gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()))
        opt.step()                                          # AdamW update

        if step % tc.log_interval == 0:
            rec = {"type": "train", "step": step, "loss": loss_sum / tc.grad_accum, "lr": lr,
                   "grad_norm": float(gnorm), "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
    log.close()


if __name__ == "__main__":
    main()
