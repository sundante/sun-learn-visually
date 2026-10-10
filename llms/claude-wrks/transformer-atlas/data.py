"""Corpus loading, splitting, and (input, target) window sampling.

The corpus is split into contiguous character ranges BEFORE any windows are made.
Each split becomes its own 1-D tensor of token ids, so a window can never span
two splits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def load_text(path) -> str:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")                         # fail loudly on invalid UTF-8
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalise line endings only
    return text                                        # case, punctuation, whitespace kept


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_text(text: str, fractions=(0.9, 0.05, 0.05)) -> tuple[str, str, str]:
    """Contiguous split: first 90% train, next 5% validation, last 5% test."""
    assert abs(sum(fractions) - 1.0) < 1e-9
    n = len(text)
    i = int(n * fractions[0])
    j = int(n * (fractions[0] + fractions[1]))
    return text[:i], text[i:j], text[j:]


def get_batch(data: torch.Tensor, B: int, T: int, device, generator: torch.Generator):
    """Sample B random windows of T+1 tokens from ONE split.

    x = tokens[start : start+T]        shape [B, T]
    y = tokens[start+1 : start+T+1]    shape [B, T]   (shifted by one)
    """
    if len(data) < T + 1:
        raise ValueError(f"split has {len(data)} tokens, need at least T+1={T + 1}")
    starts = torch.randint(len(data) - T, (B,), generator=generator)
    x = torch.stack([data[s : s + T] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + T + 1] for s in starts.tolist()])
    return x.to(device), y.to(device)


def sequential_windows(data: torch.Tensor, T: int, stride: int):
    """Deterministic windows (used for exhaustive evaluation). stride=T -> no overlap."""
    for s in range(0, len(data) - T, stride):
        yield data[s : s + T], data[s + 1 : s + T + 1]
