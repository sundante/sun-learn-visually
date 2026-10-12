"""
01 · Causal Multi-Head Self-Attention from scratch
==================================================
"Attention Is All You Need", Vaswani et al., 2017.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V

  Q (query)  : what each token is looking for
  K (key)    : what each token offers, used to be matched against queries
  V (value)  : the content a token hands over once it is matched
  M (mask)   : -inf above the diagonal so token t never sees tokens > t

Every token attends to every earlier token in ONE matrix multiply. No loop over time.

Run:  python 01_causal_attention.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        # One projection produces Q, K and V together (common in GPT-style code).
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):                    # (B, T, D) -> (B, H, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, use_fused: bool = False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)                 # each (B, T, D)
        q, k, v = map(self.split_heads, (q, k, v))             # each (B, H, T, d_k)

        if use_fused:
            # PyTorch 2.x: one call, dispatches to FlashAttention / memory-efficient kernels.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, H, T, T)
            # Why sqrt(d_k)? Dot products of d_k-dim random vectors have variance ~d_k.
            # Scaling keeps softmax out of its saturated, near-one-hot region.
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))        # hide the future
            weights = scores.softmax(dim=-1)                         # each row sums to 1
            y = weights @ v                                          # (B, H, T, d_k)

        y = y.transpose(1, 2).contiguous().view(B, T, D)             # merge heads
        return self.out(y)


if __name__ == "__main__":
    attn = CausalSelfAttention(d_model=64, n_heads=8)
    x = torch.randn(2, 10, 64)
    manual = attn(x, use_fused=False)
    fused = attn(x, use_fused=True)
    assert torch.allclose(manual, fused, atol=1e-5)
    print("✓ Manual attention matches F.scaled_dot_product_attention", tuple(manual.shape))

    # Causality check: changing a future token must not change earlier outputs.
    x2 = x.clone(); x2[:, -1] += 1.0
    assert torch.allclose(attn(x)[:, :-1], attn(x2)[:, :-1], atol=1e-6)
    print("✓ Causal mask holds: editing token 10 leaves tokens 1–9 untouched")
