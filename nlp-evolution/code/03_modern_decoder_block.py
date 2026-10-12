"""
03 · A modern (Llama-style) decoder block
=========================================
What changed between the 2017 block and a 2023–2025 open LLM block:

  2017 Transformer            ->  Modern decoder (Llama 2/3, Mistral, Qwen)
  ---------------------------     -----------------------------------------
  Post-norm (norm after add)  ->  Pre-norm  (norm before each sub-layer; stabler deep training)
  LayerNorm                   ->  RMSNorm   (Zhang & Sennrich 2019: no mean-centering, cheaper)
  Sinusoidal abs. positions   ->  RoPE      (Su et al. 2021: relative, rotates q and k)
  Multi-Head Attention        ->  GQA       (Ainslie et al. 2023: few K/V heads shared by many Q heads)
  ReLU MLP                    ->  SwiGLU    (Shazeer 2020: gated MLP, better quality per FLOP)
  Recompute everything        ->  KV cache  (store past K/V; each new token costs O(T), not O(T^2))

Run:  python 03_modern_decoder_block.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        # Divide by root-mean-square only. No mean subtraction, no bias.
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x, pos, base=10000.0):
    """x: (B, H, T, d). pos: (T,) absolute positions of these tokens."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    ang = torch.outer(pos.float(), inv)                     # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class GQAttention(nn.Module):
    """Grouped-Query Attention with a KV cache.
    n_heads query heads share n_kv_heads key/value heads (n_kv=1 is MQA, n_kv=n_heads is MHA).
    The KV cache shrinks by n_heads / n_kv_heads, which is what makes long-context serving affordable."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.h, self.kv, self.d_k = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.wo = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, start_pos=0, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)    # (B, H,  T, d_k)
        k = self.wk(x).view(B, T, self.kv, self.d_k).transpose(1, 2)   # (B, KV, T, d_k)
        v = self.wv(x).view(B, T, self.kv, self.d_k).transpose(1, 2)

        pos = torch.arange(start_pos, start_pos + T)
        q, k = rope(q, pos), rope(k, pos)       # positions go into q and k, never v

        if cache is not None:                   # decode: append new K/V to the past
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # Share each K/V head across a group of query heads.
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # Causal only during prefill (T > 1). A single new token may see the whole cache.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # SiLU(x W_gate) acts as a learned, smooth gate on (x W_up).
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_kv_heads=2, d_ff=344):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d_model), GQAttention(d_model, n_heads, n_kv_heads)
        self.norm2, self.mlp = RMSNorm(d_model), SwiGLU(d_model, d_ff)

    def forward(self, x, start_pos=0, cache=None):
        x = x + self.attn(self.norm1(x), start_pos, cache)   # pre-norm + residual
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    block = DecoderBlock().eval()
    x = torch.randn(1, 12, 128)

    with torch.no_grad():
        full = block(x)                                   # all 12 tokens at once

        # Prefill 8 tokens, then decode 4 tokens one at a time from the cache.
        cache = {}
        out = [block(x[:, :8], start_pos=0, cache=cache)]
        for t in range(8, 12):
            out.append(block(x[:, t:t + 1], start_pos=t, cache=cache))
        incremental = torch.cat(out, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)
    print("✓ KV-cached decoding == full forward pass", tuple(full.shape))
    print(f"✓ Cache holds K of shape {tuple(cache['k'].shape)} — 2 KV heads instead of 8 (4× smaller)")
