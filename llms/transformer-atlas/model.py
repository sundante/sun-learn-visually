"""GPT-style decoder-only causal Transformer, written out explicitly.

No nn.Transformer, no nn.MultiheadAttention: Q/K/V projections, head reshaping,
causal masking, softmax and the output projection are all visible below.

The optional `trace` dict is only used by export_micro.py to record intermediate
tensors for the visual reference. It never changes the computation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


def _rec(trace, key, t):
    if trace is not None:
        trace[key] = t.detach().clone()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.n_embd, cfg.n_head
        assert D % H == 0, "D must equal H * Dh"
        self.n_head, self.head_dim = H, D // H
        # nn.Linear stores weight as [out_features, in_features] and computes x @ W.T + b
        self.q_proj = nn.Linear(D, D, bias=cfg.bias)   # W_Q: [D, D]
        self.k_proj = nn.Linear(D, D, bias=cfg.bias)   # W_K: [D, D]
        self.v_proj = nn.Linear(D, D, bias=cfg.bias)   # W_V: [D, D]
        self.out_proj = nn.Linear(D, D, bias=cfg.bias) # W_O: [D, D]
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        T_max = cfg.block_size
        mask = torch.tril(torch.ones(T_max, T_max, dtype=torch.bool))  # True = may attend
        self.register_buffer("causal_mask", mask.view(1, 1, T_max, T_max), persistent=False)
        self.causal = True  # educational switch for the leakage demo; always True in training

    def forward(self, x: torch.Tensor, trace=None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_head, self.head_dim

        q = self.q_proj(x)                                   # [B, T, D]
        k = self.k_proj(x)                                   # [B, T, D]
        v = self.v_proj(x)                                   # [B, T, D]

        q = q.view(B, T, H, Dh).transpose(1, 2)              # [B, H, T, Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)              # [B, H, T, Dh]
        v = v.view(B, T, H, Dh).transpose(1, 2)              # [B, H, T, Dh]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)   # [B, H, T, T]  rows=query i, cols=key j
        _rec(trace, "scores_raw", scores)
        if self.causal:
            allowed = self.causal_mask[:, :, :T, :T]
            scores = scores.masked_fill(~allowed, float("-inf"))  # forbid j > i
        _rec(trace, "scores_masked", scores)

        probs = F.softmax(scores, dim=-1)                    # normalise over keys j; rows sum to 1
        _rec(trace, "probs", probs)
        probs = self.attn_dropout(probs)                     # training only; rows may no longer sum to 1

        y = probs @ v                                        # [B, H, T, Dh]
        _rec(trace, "head_out", y)
        y = y.transpose(1, 2).contiguous().view(B, T, D)     # merge heads -> [B, T, D]
        _rec(trace, "merged", y)
        y = self.resid_dropout(self.out_proj(y))             # [B, T, D]
        return y


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.bias)   # W1: [Dff, D]
        self.act = nn.GELU()                                        # exact (erf) GELU
        self.fc2 = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.bias)   # W2: [D, Dff]
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, trace=None) -> torch.Tensor:
        h = self.fc1(x)                                      # [B, T, Dff]
        _rec(trace, "ffn_pre", h)
        h = self.act(h)                                      # [B, T, Dff]
        _rec(trace, "ffn_act", h)
        return self.dropout(self.fc2(h))                     # [B, T, D]


class Block(nn.Module):
    """Pre-LayerNorm block:  h1 = x + Attn(LN1(x));  h2 = h1 + FFN(LN2(h1))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, trace=None) -> torch.Tensor:
        a = self.ln1(x)
        _rec(trace, "ln1", a)
        h1 = x + self.attn(a, trace)                         # residual add: shapes must match [B, T, D]
        _rec(trace, "h1", h1)
        f = self.ln2(h1)
        _rec(trace, "ln2", f)
        h2 = h1 + self.ffn(f, trace)
        _rec(trace, "h2", h2)
        return h2


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # [V, D]
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # [T_max, D]
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.ln_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # [V, D]
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight          # one tensor, two uses
        self.apply(self._init_weights)
        for name, p in self.named_parameters():                # GPT-2 style residual scaling
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # .parameters() yields a tied tensor once, so this counts UNIQUE parameters
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, trace=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)              # [T]
        tok = self.tok_emb(idx)                               # [B, T, D] row lookup
        x = self.drop(tok + self.pos_emb(pos))                # [T, D] broadcasts over B
        _rec(trace, "x0", x)
        for i, block in enumerate(self.blocks):
            bt = {} if trace is not None else None
            x = block(x, bt)
            if trace is not None:
                trace[f"block{i}"] = bt
        x = self.ln_f(x)                                      # [B, T, D]
        _rec(trace, "lnf", x)
        logits = self.lm_head(x)                              # [B, T, V]
        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))  # expects raw logits
        return logits, loss
