# Transformer Atlas

An interactive technical reference for a decoder-only Transformer language model, built around one
canonical PyTorch implementation that you can run end to end: raw text → split → character tokenizer →
windows → embeddings → Transformer blocks → logits → cross-entropy → backprop → AdamW → evaluation →
checkpoint → reload → generation → HTTP endpoint.

**Open `TransformerAtlas.html` in a browser.** It is a single self-contained file (code, data excerpts,
recorded traces and the in-browser compute engine are embedded). It works offline; only the Geist web
fonts come from Google Fonts, with system-font fallback.

## What is real and what is not

| Label in the page | Meaning |
|---|---|
| Computed in your browser | A JavaScript port of the micro model (forward pass, hand-written backward pass, AdamW). On load it is compared against the PyTorch record; worst difference ≈ 1e-6 (float32 rounding). |
| Recorded from PyTorch | Saved from executing the Python in this folder (`export_micro.py`, `export_trained.py`, `train.py`, pytest). |
| Analytical estimate | Formula-based (GPU memory, FLOPs) with every assumption stated. |
| Illustrative | Teaching sketches and general guidance, not measurements. |

The static page does not train the full model. Training happens in Python.

## Files

| File | Responsibility |
|---|---|
| `config.py` | `ModelConfig`, `TrainConfig`, notation, the micro configuration |
| `tokenizer.py` | Reversible character tokenizer, `<unk>` policy, serialisation |
| `data.py` | UTF-8 loading, contiguous split, window sampling |
| `model.py` | Explicit GPT: embeddings, Pre-LN blocks, causal multi-head attention, GELU FFN, LM head |
| `train.py` | Seeds, device, AdamW groups, warmup + cosine, clipping, accumulation, eval, checkpoints, resume |
| `checkpoint.py` | Save/load for inference and for exact resume (optimizer + RNG states) |
| `generate.py` | Autoregressive sampling (greedy / temperature / top-k) |
| `serve.py` | Minimal FastAPI endpoint (educational) |
| `measure_memory.py` | Peak CUDA memory for one training step |
| `export_micro.py`, `export_trained.py` | Record the traces the page displays |
| `build.py` | Assemble `TransformerAtlas.html` from `web/` |
| `tests/test_llm.py` | 22 verification tests |

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && curl -L -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
shasum -a 256 data/input.txt   # 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed

python -m pytest -q tests                         # verification suite
python train.py                                   # 5,000 steps; writes out/ckpt.pt, out/log.jsonl, out/vocab.json
python train.py --resume                          # continue from out/ckpt.pt
python generate.py --prompt "ROMEO:" --temperature 0.8
CKPT=out/ckpt.pt uvicorn serve:app --port 8000    # then:
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "ROMEO:\n", "max_new_tokens": 120, "temperature": 0.8, "seed": 1}'
python measure_memory.py --batch_size 32 --block_size 128   # needs a CUDA GPU

python export_micro.py && python export_trained.py && python build.py   # rebuild the page
```

Useful training flags: `--device cuda`, `--batch_size`, `--grad_accum`, `--block_size`, `--n_layer`,
`--n_head`, `--n_embd`, `--dropout`, `--tie_weights`, `--lr`, `--max_steps`.

## Dataset

Tiny Shakespeare from `karpathy/char-rnn` (repository README states the MIT license; the text is a subset of
Shakespeare's public-domain works; the source edition is not documented). 1,115,394 characters, 65 distinct,
pure ASCII, no carriage returns. Split contiguously 90 / 5 / 5 % before windowing.

## Scope

The model is small (826,624 parameters, character-level) and learns the surface statistics of the text.
It is not an instruction-following assistant. Deferred topics: subword tokenizers, mixed precision,
distributed training, KV caching, batching, quantization, fine-tuning and preference optimisation.

See `VALIDATION.md` for what was executed and verified.
