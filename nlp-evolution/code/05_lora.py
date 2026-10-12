"""
05 · LoRA: Low-Rank Adaptation
==============================
Hu et al., 2021. QLoRA (Dettmers et al., 2023) adds a 4-bit frozen base.

Fine-tuning a 7B model updates 7B weights. LoRA freezes W and learns a low-rank update:

    h = x W^T + (alpha / r) * x A^T B^T        A: (r, d_in), B: (d_out, r), r << d

B starts at zero, so training begins from exactly the pre-trained model.
After training, merge: W' = W + (alpha/r) B A, so inference costs nothing extra.

Run:  python 05_lora.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)                 # frozen pre-trained weight
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero init
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


if __name__ == "__main__":
    base = nn.Linear(4096, 4096)                 # one attention projection in a 7B model
    x = torch.randn(3, 4096)
    y0 = base(x).detach()
    lora = LoRALinear(base, r=8)

    assert torch.allclose(lora(x), y0, atol=1e-6)   # identical at init
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
    print(f"✓ Trainable {trainable:,} vs frozen {frozen:,}  ({100 * trainable / frozen:.2f}%)")

    # One training step on a toy target, then check the merged layer is equivalent.
    opt = torch.optim.AdamW([lora.A, lora.B], lr=1e-3)
    loss = (lora(x) - torch.randn_like(y0)).pow(2).mean()
    loss.backward(); opt.step()
    assert base.weight.grad is None                 # the base never received a gradient
    assert torch.allclose(lora(x), lora.merge()(x), atol=1e-4)
    print("✓ Base weight untouched; merged W' gives the same output as base + adapter")
