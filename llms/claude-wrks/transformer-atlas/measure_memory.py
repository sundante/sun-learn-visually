"""Measure peak CUDA memory for ONE training step of a given configuration.

python measure_memory.py --batch_size 32 --block_size 128
python measure_memory.py --batch_size 8 --block_size 1024 --n_layer 12 --n_head 12 --n_embd 768 --vocab_size 50257

Reports PyTorch's caching-allocator counters:
  max_memory_allocated  bytes actually held by tensors at the peak
  max_memory_reserved   bytes the allocator reserved from the driver (>= allocated)
Compare with the analytical estimator in the visual reference (chapter 13).
"""
from __future__ import annotations

import argparse

import torch

from config import ModelConfig
from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=66)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: this script measures GPU memory and needs a CUDA device.")

    dev = "cuda"
    cfg = ModelConfig(vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                      n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(dev)
    if args.dtype == "bfloat16":
        model = model.to(torch.bfloat16)   # pure bf16 (no master weights): simplest possible setup
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.block_size), device=dev)
    y = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.block_size), device=dev)

    # warm-up step so AdamW allocates its state (exp_avg, exp_avg_sq) before measuring
    model(x, y)[1].backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()          # params + optimizer state, no grads, no activations
    _, loss = model(x, y)
    after_fwd = torch.cuda.memory_allocated()     # + saved activations
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    gib = 2 ** 30
    print(f"parameters            {model.num_params():,}")
    print(f"resident before step  {base / gib:8.3f} GiB  (weights + AdamW state)")
    print(f"after forward         {after_fwd / gib:8.3f} GiB  (+ activations saved for backward)")
    print(f"peak allocated        {torch.cuda.max_memory_allocated() / gib:8.3f} GiB")
    print(f"peak reserved         {torch.cuda.max_memory_reserved() / gib:8.3f} GiB")


if __name__ == "__main__":
    main()
