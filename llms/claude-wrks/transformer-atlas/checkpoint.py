"""Checkpoint save/load.

Inference needs:  model state_dict + model config + tokenizer.
Resuming needs:   all of the above + optimizer state (AdamW moments and step counts)
                  + training step + RNG states (Python, NumPy, torch CPU/CUDA, data sampler)
                  + training config + dataset identity.
Saving weights alone is NOT enough to continue training identically.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import GPT
from tokenizer import CharTokenizer

FORMAT_VERSION = 1


def _rng_state(data_gen: torch.Generator | None) -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_generator": data_gen.get_state() if data_gen is not None else None,
    }


def _restore_rng(state: dict, data_gen: torch.Generator | None) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if data_gen is not None and state["data_generator"] is not None:
        data_gen.set_state(state["data_generator"])


def save_checkpoint(path, *, model, optimizer, step, tokenizer, train_cfg: dict,
                    data_gen=None, data_info=None, metrics=None) -> None:
    ckpt = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,                       # number of optimizer updates already applied
        "train_config": train_cfg,
        "rng": _rng_state(data_gen),
        "data": data_info or {},
        "metrics": metrics or {},
        "torch_version": str(torch.__version__),   # plain str: loadable with weights_only=True
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)                   # atomic: never leaves a half-written checkpoint


def load_checkpoint(path, device="cpu") -> dict:
    # weights_only=True: only tensors and plain Python containers are unpickled
    return torch.load(path, map_location=device, weights_only=True)


def load_for_inference(path, device="cpu"):
    ckpt = load_checkpoint(path, device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_dict(ckpt["tokenizer"])
    return model, tok, ckpt


def restore_for_resume(ckpt: dict, model, optimizer, data_gen) -> int:
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    _restore_rng(ckpt["rng"], data_gen)
    return ckpt["step"]
