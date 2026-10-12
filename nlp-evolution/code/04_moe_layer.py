"""
04 · Sparse Mixture-of-Experts (top-k routing)
==============================================
Idea: Jacobs et al. 1991; sparse gating: Shazeer et al. 2017; at LLM scale:
Switch Transformer (2021), Mixtral 8x7B (Dec 2023), DeepSeek-V3 (Dec 2024).

Replace one big MLP with E small expert MLPs. A router picks the top-k experts
per token, so the model has E× the parameters but each token pays for only k.
Mixtral: 8 experts, top-2 -> ~47B total parameters, ~13B active per token.

    g = softmax( TopK( x W_router ) )         y = sum_{i in TopK} g_i * Expert_i(x)

Load-balancing loss (Switch Transformer):  L_aux = E * sum_i f_i * P_i
    f_i = fraction of tokens routed to expert i,  P_i = mean router prob for expert i
It is smallest when tokens spread evenly, which stops a few experts taking everything.

Run:  python 04_moe_layer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate, self.w_up = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))   # SwiGLU expert


class MoELayer(nn.Module):
    def __init__(self, d=64, d_ff=128, n_experts=8, top_k=2):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                                # (N, D), N = B*T tokens
        logits = self.router(x_flat)                             # (N, E)
        probs = logits.softmax(-1)
        top_p, top_i = probs.topk(self.k, dim=-1)                # (N, k)
        top_p = top_p / top_p.sum(-1, keepdim=True)              # renormalise over chosen k

        y = torch.zeros_like(x_flat)
        # Dispatch: each expert processes ONLY the tokens routed to it.
        for e in range(self.E):
            token_idx, slot = (top_i == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y.index_add_(0, token_idx,
                         top_p[token_idx, slot, None] * self.experts[e](x_flat[token_idx]))

        # Load-balancing auxiliary loss.
        f = F.one_hot(top_i, self.E).float().sum(1).mean(0) / self.k   # (E,) share of assignments
        P = probs.mean(0)                                               # (E,)
        aux_loss = self.E * (f * P).sum()
        return y.view(B, T, D), aux_loss, top_i


if __name__ == "__main__":
    moe = MoELayer().eval()
    x = torch.randn(2, 6, 64)
    with torch.no_grad():
        y, aux, routes = moe(x)

        # Reference: the slow, obvious version, token by token.
        x_flat, ref = x.reshape(-1, 64), []
        for n in range(x_flat.shape[0]):
            p = moe.router(x_flat[n]).softmax(-1)
            tp, ti = p.topk(2)
            tp = tp / tp.sum()
            ref.append(sum(tp[j] * moe.experts[ti[j]](x_flat[n]) for j in range(2)))
        ref = torch.stack(ref).view_as(y)

    assert torch.allclose(y, ref, atol=1e-5)
    total = sum(p.numel() for p in moe.experts.parameters())
    active = total * moe.k // moe.E
    print("✓ Batched MoE dispatch matches per-token reference")
    print(f"✓ Expert params: {total:,} total, ~{active:,} active per token (top-{moe.k} of {moe.E})")
    print(f"  aux loss = {aux.item():.3f} (1.0 = perfectly balanced)")
