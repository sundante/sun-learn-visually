"""Autoregressive generation (no KV cache: the full context is recomputed each step)."""
from __future__ import annotations

import argparse

import torch

from checkpoint import load_for_inference


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """idx: [B, T0] token ids. Returns [B, T0 + max_new_tokens]."""
    model.eval()                                         # disables dropout
    T_max = model.cfg.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -T_max:]                       # crop: learned positions only exist for 0..T_max-1
        logits, _ = model(idx_cond)                      # [B, T, V]
        logits = logits[:, -1, :]                        # only the last position predicts the NEXT token
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)            # greedy
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)                    # [B, V]
            next_id = torch.multinomial(probs, 1, generator=generator)  # [B, 1]
        idx = torch.cat([idx, next_id], dim=1)           # append and repeat: inherently sequential
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--prompt", default="ROMEO:\n")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tok, ckpt = load_for_inference(args.ckpt)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k, g)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
