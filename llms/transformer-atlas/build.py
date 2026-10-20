"""Assemble TransformerAtlas.html: one self-contained page with the source code,
corpus facts, recorded traces and test results embedded.

python build.py                       # uses web/data/*.json and runs pytest for the report
python build.py --skip-tests          # reuse web/data/tests.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from data import TINY_SHAKESPEARE_URL, load_text, sha256_of, split_text
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
SOURCES = ["config.py", "tokenizer.py", "data.py", "model.py", "train.py", "checkpoint.py", "generate.py",
           "serve.py", "measure_memory.py", "export_micro.py", "export_trained.py", "build.py",
           "tests/test_llm.py", "requirements.txt", "README.md"]
TEST_DOCS = {
    "test_tokenizer_roundtrip": "decode(encode(s)) == s on train, val and test text; V = 66 with <unk> at 0",
    "test_unseen_characters": "strict encoding raises; non-strict maps to <unk> and decodes to U+FFFD",
    "test_tokenizer_serialization": "vocabulary JSON round trip",
    "test_shift_by_one": "y is x shifted by exactly one token",
    "test_windows_stay_inside_split": "1,280 random validation windows are substrings of the validation text",
    "test_shapes_and_head_split_merge": "tensor shapes; head h = columns h·Dh…; split→merge is exact",
    "test_causal_mask_no_future_leakage": "changing tokens 5–7 leaves logits 0–4 bit-identical",
    "test_unmasked_attention_does_leak": "removing the mask makes position 0 depend on position 7",
    "test_attention_rows_sum_to_one_before_dropout": "rows sum to 1 and the upper triangle is 0, with dropout active",
    "test_matches_pytorch_sdpa": "explicit attention equals F.scaled_dot_product_attention(is_causal=True)",
    "test_loss_and_gradients_finite_and_complete": "finite loss ≈ ln V at init; every parameter gets a finite gradient",
    "test_cross_entropy_equals_manual_nll": "F.cross_entropy == −mean(log_softmax[y]); softmax-first differs",
    "test_optimizer_step_changes_parameters": "one AdamW step changes every parameter tensor",
    "test_overfit_single_batch": "a tiny model drives one batch's loss below 0.05",
    "test_gradient_accumulation_matches_full_batch": "two scaled microbatches reproduce the full-batch gradient",
    "test_finite_difference_gradient": "central differences match autograd in float64 for 12 entries",
    "test_weight_tying_counts_and_gradient_sum": "tying removes V·D parameters; tied grad = sum of both paths",
    "test_token_ids_receive_no_gradient_but_embeddings_do": "unused embedding rows get exactly zero gradient",
    "test_checkpoint_roundtrip": "save → load_for_inference gives equal logits; optimizer state stored",
    "test_resume_is_bit_exact": "6 straight steps == 3 steps + resume + 3 steps, every weight torch.equal",
    "test_generation_shape_and_validity": "output length and id range; greedy works",
    "test_serve_endpoint": "200 on valid prompt; 422 on unknown characters, too many tokens, empty prompt",
}


def corpus_facts() -> dict:
    path = ROOT / "data" / "input.txt"
    raw = path.read_bytes()
    text = load_text(path)
    tr, va, te = split_text(text)
    tok = CharTokenizer.build(tr)
    train_lines = set(l for l in tr.split("\n"))
    def dup_stats(min_len):
        lines = [l for l in va.split("\n") if len(l) > min_len]
        return {"lines": len(lines), "dups": sum(1 for l in lines if l in train_lines)}
    dup = {"any": dup_stats(0), "gt10": dup_stats(10), "gt20": dup_stats(20)}
    n = len(text)
    i, j = len(tr), len(tr) + len(va)
    return {
        "url": TINY_SHAKESPEARE_URL, "sha256": sha256_of(path), "bytes": len(raw), "chars": n,
        "lines": text.count("\n"), "cr": raw.count(b"\r"), "non_ascii": sum(1 for c in text if ord(c) > 127),
        "unique": len(set(text)), "split": [len(tr), len(va), len(te)],
        "boundaries": [{"name": "train | val", "index": i, "before": text[i - 70:i], "after": text[i:i + 70]},
                       {"name": "val | test", "index": j, "before": text[j - 70:j], "after": text[j:j + 70]}],
        "dup_val_lines": dup, "vocab": tok.itos,
        "unk_val": sum(1 for c in va if c not in tok.stoi), "unk_test": sum(1 for c in te if c not in tok.stoi),
        "head": text[:420], "strip_ids": tok.encode(tr[:72]), "bpe_text": tr[:3000],
        "license": "The char-rnn repository (Andrej Karpathy) states the MIT license in its README; it has no separate LICENSE file. "
                   "The text is a subset of William Shakespeare's works, which are in the public domain. "
                   "The edition the excerpt was taken from is not documented.",
    }


def run_tests(skip: bool) -> list:
    out = WEB / "data" / "tests.json"
    if not skip:
        xml = ROOT / "out" / "pytest.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}"], cwd=ROOT)
        res = []
        for tc in ET.parse(xml).getroot().iter("testcase"):
            outcome = "failed" if tc.find("failure") is not None or tc.find("error") is not None else \
                      "skipped" if tc.find("skipped") is not None else "passed"
            res.append({"name": tc.get("name"), "outcome": outcome, "time": float(tc.get("time", 0)),
                        "doc": TEST_DOCS.get(tc.get("name"), "")})
        out.write_text(json.dumps(res))
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    data = {
        "micro": json.loads((WEB / "data" / "micro.json").read_text()),
        "trained": json.loads((WEB / "data" / "trained.json").read_text()),
        "sources": {s: (ROOT / s).read_text(encoding="utf-8") for s in SOURCES},
        "corpus": corpus_facts(),
        "tests": run_tests(args.skip_tests),
    }
    checks = WEB / "data" / "html_checks.json"
    data["htmlChecks"] = json.loads(checks.read_text()) if checks.exists() else None
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\\ufffd")
    tpl = (WEB / "index.template.html").read_text(encoding="utf-8")
    js = "\n".join((WEB / f).read_text(encoding="utf-8") for f in ("engine.js", "app.js", "chapters1.js", "chapters2.js"))
    # Order matters: the embedded sources (build.py itself) contain these markers too,
    # so each one is replaced once, at its first occurrence, before data is inserted after it.
    html = tpl.replace("/*__JS__*/", js.replace("</script", "<\\/script"), 1)
    html = html.replace("/*__CSS__*/", (WEB / "styles.css").read_text(encoding="utf-8"), 1)
    html = html.replace("__DATA__", blob, 1)
    (ROOT / "TransformerAtlas.html").write_text(html, encoding="utf-8")
    print(f"wrote TransformerAtlas.html ({len(html.encode()) / 1024:.0f} KiB); tests: "
          f"{sum(t['outcome'] == 'passed' for t in data['tests'])}/{len(data['tests'])} passed")


if __name__ == "__main__":
    main()
