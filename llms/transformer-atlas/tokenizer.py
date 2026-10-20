"""Reversible character-level tokenizer.

Vocabulary policy
-----------------
* The vocabulary is built from the TRAINING split only, so validation/test text
  cannot influence the model's input space.
* Id 0 is reserved for <unk>. Every observed training character gets an id >= 1,
  assigned in sorted (Unicode code point) order, so the mapping is deterministic.
* Unseen characters: `encode(strict=True)` (the default) raises UnknownCharacterError.
  `encode(strict=False)` maps them to UNK_ID; decoding UNK_ID yields U+FFFD.
  Only strict encoding is guaranteed to round-trip exactly.

Token ids are categorical indices. Id 40 is not "twice" id 20; ids only select
rows of the embedding table.
"""
from __future__ import annotations

import json
from pathlib import Path

UNK_ID = 0
UNK_TOKEN = "<unk>"
UNK_DECODED = "\ufffd"


class UnknownCharacterError(ValueError):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        super().__init__(f"characters not in vocabulary: {self.chars!r}")


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        if UNK_DECODED in chars:
            raise ValueError("U+FFFD is reserved for decoding <unk>")
        self.chars = list(chars)                       # id i+1 -> chars[i]
        self.itos = [UNK_TOKEN] + self.chars
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}

    @classmethod
    def build(cls, train_text: str) -> "CharTokenizer":
        return cls(sorted(set(train_text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, strict: bool = True) -> list[int]:
        unknown = [c for c in text if c not in self.stoi]
        if unknown and strict:
            raise UnknownCharacterError(unknown)
        return [self.stoi.get(c, UNK_ID) for c in text]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if not 0 <= i < self.vocab_size:
                raise ValueError(f"token id {i} outside [0, {self.vocab_size})")
            out.append(UNK_DECODED if i == UNK_ID else self.itos[i])
        return "".join(out)

    def to_dict(self) -> dict:
        return {"type": "char", "unk_id": UNK_ID, "chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        assert d["type"] == "char" and d["unk_id"] == UNK_ID
        return cls(d["chars"])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
