"""Minimal educational HTTP endpoint (not production serving: no batching, no KV cache).

Run:  CKPT=out/ckpt.pt uvicorn serve:app --port 8000
Try:  curl -s localhost:8000/generate -H 'content-type: application/json' \
        -d '{"prompt": "ROMEO:\\n", "max_new_tokens": 120, "temperature": 0.8, "seed": 1}'
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from checkpoint import load_for_inference
from generate import generate
from tokenizer import UnknownCharacterError

MAX_PROMPT_CHARS = 1000
MAX_NEW_TOKENS = 1000
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tok, ckpt = load_for_inference(os.environ.get("CKPT", "out/ckpt.pt"))  # load ONCE
    STATE.update(model=model, tok=tok, step=ckpt["step"])
    yield
    STATE.clear()


app = FastAPI(title="Tiny Shakespeare char-GPT (educational)", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: int = Field(200, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(0.8, ge=0.0, le=2.0)   # 0 = greedy
    top_k: int | None = Field(None, ge=1)
    seed: int | None = None


class GenerateResponse(BaseModel):
    completion: str
    prompt_tokens: int
    generated_tokens: int
    model_step: int


@app.get("/health")
def health():
    return {"status": "ok", "model_step": STATE.get("step")}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    model, tok = STATE["model"], STATE["tok"]
    try:
        ids = tok.encode(req.prompt, strict=True)
    except UnknownCharacterError as e:
        raise HTTPException(status_code=422, detail={
            "error": "prompt contains characters outside the model vocabulary",
            "unknown_characters": e.chars})
    g = torch.Generator().manual_seed(req.seed) if req.seed is not None else None
    idx = torch.tensor([ids], dtype=torch.long)
    out = generate(model, idx, req.max_new_tokens, req.temperature, req.top_k, g)
    new_ids = out[0, len(ids):].tolist()
    return GenerateResponse(completion=tok.decode(new_ids), prompt_tokens=len(ids),
                            generated_tokens=len(new_ids), model_step=STATE["step"])
