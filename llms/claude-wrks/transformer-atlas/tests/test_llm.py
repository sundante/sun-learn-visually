"""Verification suite.  Run:  python -m pytest -q tests"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_checkpoint, load_for_inference, save_checkpoint  # noqa: E402
from config import ModelConfig, TrainConfig  # noqa: E402
from data import get_batch, load_text, split_text  # noqa: E402
from generate import generate  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import UNK_DECODED, UNK_ID, CharTokenizer, UnknownCharacterError  # noqa: E402
from train import make_optimizer  # noqa: E402

DATA = ROOT / "data" / "input.txt"


@pytest.fixture(scope="module")
def corpus():
    text = load_text(DATA)
    tr, va, te = split_text(text)
    return text, tr, va, te, CharTokenizer.build(tr)


def tiny_cfg(**kw):
    base = dict(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


# ---------------- tokenizer ----------------
def test_tokenizer_roundtrip(corpus):
    _, tr, va, te, tok = corpus
    assert tok.vocab_size == 66 and tok.itos[0] == "<unk>"
    for s in (tr[:5000], va[:5000], te[:5000]):
        assert tok.decode(tok.encode(s)) == s


def test_unseen_characters(corpus):
    tok = corpus[4]
    with pytest.raises(UnknownCharacterError) as e:
        tok.encode("Hello ✓")
    assert e.value.chars == ["✓"]
    ids = tok.encode("Hi✓", strict=False)
    assert ids[-1] == UNK_ID and tok.decode(ids) == "Hi" + UNK_DECODED
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


def test_tokenizer_serialization(tmp_path, corpus):
    tok = corpus[4]
    tok.save(tmp_path / "v.json")
    assert CharTokenizer.load(tmp_path / "v.json").itos == tok.itos


# ---------------- data ----------------
def test_shift_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, 16, 10, "cpu", torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (16, 10)
    assert torch.equal(y, x + 1)                 # with data = arange, target = next token
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_windows_stay_inside_split(corpus):
    _, tr, va, te, tok = corpus
    val = torch.tensor(tok.encode(va))
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        x, y = get_batch(val, 64, 128, "cpu", g)
        for row_x, row_y in zip(x, y):
            window = tok.decode(row_x.tolist()) + tok.decode(row_y[-1:].tolist())
            assert window in va                  # every window is a substring of the val text
    assert len(tr) + len(va) + len(te) == len(corpus[0])


# ---------------- model shapes & attention ----------------
def test_shapes_and_head_split_merge():
    cfg = tiny_cfg()
    m = GPT(cfg).eval()
    idx = torch.randint(0, 20, (3, 8))
    tr = {}
    logits, _ = m(idx, trace=tr)
    B, T, D, H, Dh = 3, 8, 16, 2, 8
    assert logits.shape == (B, T, 20)
    b0 = tr["block0"]
    assert b0["scores_raw"].shape == (B, H, T, T)
    assert b0["head_out"].shape == (B, H, T, Dh)
    assert b0["merged"].shape == (B, T, D)
    q = torch.randn(B, T, D)
    split = q.view(B, T, H, Dh).transpose(1, 2)
    merged = split.transpose(1, 2).contiguous().view(B, T, D)
    assert torch.equal(merged, q)
    assert torch.equal(split[:, 1, :, :], q[:, :, Dh:2 * Dh])   # head 1 owns columns Dh..2Dh-1


def test_causal_mask_no_future_leakage():
    m = GPT(tiny_cfg()).eval()
    a = torch.randint(0, 20, (1, 8))
    b = a.clone()
    b[0, 5:] = (b[0, 5:] + 7) % 20               # change only positions 5..7
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[0, :5], lb[0, :5], atol=0, rtol=0)
    assert not torch.allclose(la[0, 5:], lb[0, 5:])


def test_unmasked_attention_does_leak():
    m = GPT(tiny_cfg()).eval()
    for blk in m.blocks:
        blk.attn.causal = False
    a = torch.randint(0, 20, (1, 8))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % 20
    assert not torch.allclose(m(a)[0][0, 0], m(b)[0][0, 0])   # position 0 now sees position 7


def test_attention_rows_sum_to_one_before_dropout():
    m = GPT(tiny_cfg(dropout=0.5)).train()
    tr = {}
    m(torch.randint(0, 20, (2, 8)), trace=tr)
    p = tr["block0"]["probs"]
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-6)
    assert torch.all(torch.triu(p, diagonal=1) == 0)


def test_matches_pytorch_sdpa():
    cfg = tiny_cfg()
    attn = GPT(cfg).blocks[0].attn.eval()
    x = torch.randn(2, 8, 16)
    ours = attn(x)
    B, T, D, H, Dh = 2, 8, 16, 2, 8
    q = attn.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.out_proj(y.transpose(1, 2).reshape(B, T, D))
    assert torch.allclose(ours, ref, atol=1e-6)


# ---------------- loss, gradients, optimizer ----------------
def test_loss_and_gradients_finite_and_complete():
    m = GPT(tiny_cfg())
    x = torch.randint(0, 20, (4, 8))
    _, loss = m(x, torch.randint(0, 20, (4, 8)))
    assert math.isfinite(loss.item())
    assert abs(loss.item() - math.log(20)) < 0.2   # near-uniform predictions at init
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_cross_entropy_equals_manual_nll():
    logits = torch.randn(6, 11)
    y = torch.randint(0, 11, (6,))
    manual = -(logits.log_softmax(-1)[torch.arange(6), y]).mean()
    assert torch.allclose(F.cross_entropy(logits, y), manual, atol=1e-6)
    double = F.cross_entropy(logits.softmax(-1), y)  # the classic bug: softmax applied twice
    assert not torch.allclose(double, manual)


def test_optimizer_step_changes_parameters():
    m = GPT(tiny_cfg())
    opt = make_optimizer(m, TrainConfig(lr=1e-2))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    _, loss = m(torch.randint(0, 20, (4, 8)), torch.randint(0, 20, (4, 8)))
    loss.backward()
    opt.step()
    changed = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert len(changed) == len(before)


def test_overfit_single_batch():
    torch.manual_seed(0)
    m = GPT(tiny_cfg(n_embd=32))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x, y = torch.randint(0, 20, (2, 8)), torch.randint(0, 20, (2, 8))
    for _ in range(300):
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(0)
    m = GPT(tiny_cfg())
    x, y = torch.randint(0, 20, (4, 8)), torch.randint(0, 20, (4, 8))
    m(x, y)[1].backward()
    full = m.lm_head.weight.grad.clone()
    m.zero_grad()
    for xs, ys in ((x[:2], y[:2]), (x[2:], y[2:])):
        (m(xs, ys)[1] / 2).backward()
    assert torch.allclose(full, m.lm_head.weight.grad, atol=1e-6)


def test_finite_difference_gradient():
    torch.manual_seed(3)
    m = GPT(tiny_cfg()).double()
    x, y = torch.randint(0, 20, (2, 8)), torch.randint(0, 20, (2, 8))
    m(x, y)[1].backward()
    eps = 1e-6
    for name in ("tok_emb.weight", "blocks.0.attn.q_proj.weight", "blocks.1.ffn.fc1.bias", "ln_f.weight"):
        p = dict(m.named_parameters())[name]
        flat = p.data.view(-1)
        for i in (0, flat.numel() // 2, flat.numel() - 1):
            old = flat[i].item()
            flat[i] = old + eps
            lp = m(x, y)[1].item()
            flat[i] = old - eps
            lm = m(x, y)[1].item()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = p.grad.view(-1)[i].item()
            assert abs(num - ana) < 1e-6 + 1e-4 * abs(ana), (name, i, num, ana)


def test_weight_tying_counts_and_gradient_sum():
    untied = GPT(tiny_cfg())
    tied = GPT(tiny_cfg(tie_weights=True))
    assert untied.num_params() - tied.num_params() == 20 * 16
    # copy tied weights into an untied model; tied grad must equal the sum of both paths
    untied.load_state_dict({**tied.state_dict()})
    x, y = torch.randint(0, 20, (2, 8)), torch.randint(0, 20, (2, 8))
    tied(x, y)[1].backward()
    untied(x, y)[1].backward()
    expected = untied.tok_emb.weight.grad + untied.lm_head.weight.grad
    assert torch.allclose(tied.tok_emb.weight.grad, expected, atol=1e-6)


def test_token_ids_receive_no_gradient_but_embeddings_do():
    m = GPT(tiny_cfg())
    x = torch.tensor([[3, 3, 5, 3, 1, 2, 2, 9]])
    _, loss = m(x, torch.randint(0, 20, (1, 8)))
    loss.backward()
    g = m.tok_emb.weight.grad
    assert not x.requires_grad
    unused = [i for i in range(20) if i not in x.tolist()[0]]
    assert torch.all(g[unused] == 0)              # rows never selected get exactly zero gradient
    assert g[3].abs().sum() > 0


# ---------------- persistence & generation ----------------
def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(dropout=0.1))
    tok = CharTokenizer(list("abcdefghijklmnopqrs"))   # 19 chars + <unk> = 20
    opt = make_optimizer(m, TrainConfig())
    save_checkpoint(tmp_path / "c.pt", model=m, optimizer=opt, step=7, tokenizer=tok,
                    train_cfg=TrainConfig().to_dict(), data_gen=torch.Generator().manual_seed(1))
    m2, tok2, ck = load_for_inference(tmp_path / "c.pt")
    m.eval()
    x = torch.randint(0, 20, (2, 8))
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-6, rtol=0)
    assert ck["step"] == 7 and tok2.itos == tok.itos and ck["optimizer"] is not None


def test_resume_is_bit_exact(tmp_path):
    common = [sys.executable, str(ROOT / "train.py"), "--n_layer", "1", "--n_embd", "32",
              "--n_head", "2", "--block_size", "16", "--batch_size", "4", "--eval_interval", "3",
              "--eval_iters", "1", "--sample_chars", "5", "--data_path", str(DATA)]
    run = lambda *a: subprocess.run(common + list(a), check=True, cwd=ROOT, capture_output=True)
    run("--max_steps", "6", "--out_dir", str(tmp_path / "a"))
    run("--max_steps", "3", "--out_dir", str(tmp_path / "b"))
    run("--max_steps", "6", "--out_dir", str(tmp_path / "b"), "--resume")
    a = load_checkpoint(tmp_path / "a" / "ckpt.pt")["model"]
    b = load_checkpoint(tmp_path / "b" / "ckpt.pt")["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_generation_shape_and_validity():
    m = GPT(tiny_cfg())
    out = generate(m, torch.zeros(2, 3, dtype=torch.long), 20, 1.0, 5, torch.Generator().manual_seed(0))
    assert out.shape == (2, 23)
    assert out.min() >= 0 and out.max() < 20
    greedy = generate(m, torch.zeros(1, 3, dtype=torch.long), 5, 0.0)
    assert greedy.shape == (1, 8)


def test_serve_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    m = GPT(tiny_cfg())
    tok = CharTokenizer(list("abcdefghijklmnopqrs"))
    save_checkpoint(tmp_path / "c.pt", model=m, optimizer=None, step=0, tokenizer=tok, train_cfg={})
    os.environ["CKPT"] = str(tmp_path / "c.pt")
    import serve
    with TestClient(serve.app) as c:
        r = c.post("/generate", json={"prompt": "abc", "max_new_tokens": 10, "seed": 1})
        assert r.status_code == 200 and r.json()["generated_tokens"] == 10
        r = c.post("/generate", json={"prompt": "abZ"})
        assert r.status_code == 422 and r.json()["detail"]["unknown_characters"] == ["Z"]
        assert c.post("/generate", json={"prompt": "a", "max_new_tokens": 5000}).status_code == 422
        assert c.post("/generate", json={"prompt": ""}).status_code == 422
