"""Configuration for the decoder-only Transformer.

Notation used everywhere in this project:
    B   batch size (sequences per microbatch)
    T   sequence length of the current input (T <= T_max)
    T_max   maximum context length = block_size (rows of the position table)
    V   vocabulary size
    D   model width (n_embd)
    H   number of attention heads (n_head)
    Dh  head dimension = D // H
    L   number of Transformer blocks (n_layer). The loss is written `loss`, never L.
    Dff feed-forward hidden width = ffn_mult * D
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 66      # V (65 characters + 1 reserved <unk> id)
    block_size: int = 128     # T_max
    n_layer: int = 4          # L
    n_head: int = 4           # H
    n_embd: int = 128         # D
    ffn_mult: int = 4         # Dff = ffn_mult * D
    dropout: float = 0.1
    bias: bool = True         # biases in Linear layers (lm_head never has one)
    tie_weights: bool = False # share tok_emb.weight with lm_head.weight
    ln_eps: float = 1e-5      # LayerNorm epsilon (PyTorch default)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def d_ff(self) -> int:
        return self.ffn_mult * self.n_embd

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd (D={self.n_embd}) must be divisible by n_head (H={self.n_head}): D = H x Dh"
            )
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "ffn_mult"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    data_path: str = "data/input.txt"
    out_dir: str = "out"
    seed: int = 1337
    device: str = "auto"          # auto | cpu | cuda | mps
    batch_size: int = 32          # B (microbatch)
    grad_accum: int = 1           # microbatches per optimizer step
    max_steps: int = 2500         # optimizer updates
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-8
    grad_clip: float = 1.0        # 0 disables clipping
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    sample_chars: int = 300
    split: tuple = (0.9, 0.05, 0.05)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        d["split"] = list(self.split)
        return d


# The fully inspectable configuration used by the visual reference (export_micro.py).
# Same code, same conventions; only sizes differ. Its vocabulary comes from a short excerpt.
MICRO = dict(block_size=8, n_layer=1, n_head=2, n_embd=8, ffn_mult=4, dropout=0.0)
