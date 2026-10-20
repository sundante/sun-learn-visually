# Validation report

Environment: Python 3.13, torch 2.14.1 (CPU build, 2 threads), no GPU. Dataset sha256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

## Executed

| What | Result |
|---|---|
| `pytest tests` (22 tests) | 22 passed |
| Training run, `python train.py --max_steps 5000 --eval_interval 250` | 826,624 params; final train 1.312, val 1.467 (50-batch estimates) |
| Exhaustive evaluation after training (`export_trained.py`) | val 1.4665 nats/char (perplexity 4.33, 2.12 bits/char); test 1.5993 |
| Checkpoint save → reload, eval mode, same 128 tokens | max abs logit difference 0.0 (bitwise identical) |
| Resume: 6 straight steps vs 3 + resume + 3 | every weight `torch.equal` |
| FastAPI app via TestClient with the real checkpoint | 200 with 120 generated tokens; 422 listing the unknown character for `"Hello 👋"` |
| Micro-example export (`export_micro.py`) | finite differences vs autograd (float64) agree to ≤ 1e-9 on 4 sampled entries |
| Browser engine vs PyTorch record (forward, backward, clipping, AdamW) | worst \|Δ\| 2.7e-6 (float32 rounding) |
| Headless Chromium checks (`tests/check_html.py`) | 11 / 11 passed (see below) |

### Tests in `tests/test_llm.py`
Tokenizer round trip on all splits; unseen-character policy; vocabulary serialisation; one-token shift;
windows never cross the validation boundary (1,280 sampled); tensor shapes and exact head split/merge;
causal masking (changing tokens 5–7 leaves logits 0–4 bit-identical); removing the mask does leak;
attention rows sum to 1 before dropout; explicit attention equals `F.scaled_dot_product_attention`;
finite loss ≈ ln V at init with finite gradients everywhere; `cross_entropy` equals manual NLL and differs
from softmax-first; every parameter changes after one AdamW step; overfit one batch below 0.05;
scaled gradient accumulation equals full batch; finite-difference gradient check (float64, 12 entries);
weight tying parameter count and gradient sum; unused embedding rows get zero gradient; checkpoint round
trip; bit-exact resume; generation shape/validity; HTTP endpoint validation.

### HTML checks
Page boots; all 72 code anchors resolve to source lines; browser engine agrees with PyTorch; all 21
chapters render and 131 controls plus every range, select and checkbox run without errors; no horizontal
overflow at 1440 px or 390 px; code↔visual synchronisation both ways; editing the running example
recomputes dependent results; keyboard navigation and search; `prefers-reduced-motion` honoured; content
readable without animation.

## Notable findings

* `attn.k_proj.bias` has an analytically zero gradient (softmax is invariant to a per-row constant shift).
  Both engines see only rounding noise there, which AdamW's normalisation turns into a visible step, so the
  two engines' updates for that one tensor differ. The page explains this; AdamW agreement is therefore
  checked from PyTorch's own clipped gradients (2.1e-7).
* Test loss (1.60) is higher than validation loss (1.47): both splits are contiguous ranges at the end of
  the file, so the gap most likely reflects differences between the plays in each range (not investigated further).
* 23.2% of non-empty validation lines (395 of 1,703) occur verbatim in training; 98% are speaker tags (e.g. "PETRUCHIO:");
  only 4 of 1,251 lines longer than 10 characters repeat, and none longer than 20.

## Not verified

* No CUDA GPU: GPU memory figures are analytical estimates; `measure_memory.py` was not run on a GPU.
* One training run, one seed; run-to-run variance was not measured.
* Accessibility was checked structurally (labels, keyboard, contrast choices, reduced motion), not with a
  screen reader.
* Web fonts could not load in the build sandbox; the checks ran with system-font fallback.

## Labelling of results in the page
Every panel carries one of: *Computed in your browser*, *Recorded from PyTorch*, *Analytical estimate*,
*Illustrative*. Loss curves, samples, checkpoint contents, attention maps of the trained model and the
HTTP responses are recorded from the executed run; nothing is invented.
