"""
02 · Rotary Positional Embeddings (RoPE)
========================================
Su et al., "RoFormer", 2021. Used by Llama, Mistral, Qwen, DeepSeek and most open LLMs.

Idea: instead of ADDING a position vector to the token, ROTATE each pair of
dimensions of q and k by an angle proportional to the position m:

    theta_i = base^(-2i/d)        rotate pair i of a vector at position m by angle m * theta_i

Because rotations compose, q_m · k_n depends only on (m - n): attention sees
RELATIVE distance for free, which also helps context-length extension.

Run:  python 02_rope.py
"""
import torch

torch.manual_seed(0)


def rope_cache(seq_len: int, d: int, base: float = 10000.0):
    """Precompute cos/sin tables, shape (seq_len, d/2)."""
    inv_freq = base ** (-torch.arange(0, d, 2).float() / d)   # (d/2,)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)  # (T, d/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (..., T, d). Rotate consecutive pairs (x0,x1), (x2,x3), ..."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2)


if __name__ == "__main__":
    d, T = 64, 32
    cos, sin = rope_cache(T, d)
    q, k = torch.randn(d), torch.randn(d)

    def score(m, n):
        qm = apply_rope(q.expand(T, d), cos, sin)[m]   # q placed at position m
        kn = apply_rope(k.expand(T, d), cos, sin)[n]   # k placed at position n
        return (qm @ kn).item()

    # Same distance (3) at different absolute positions -> same score.
    s1, s2, s3 = score(5, 2), score(20, 17), score(31, 28)
    assert abs(s1 - s2) < 1e-3 and abs(s2 - s3) < 1e-3
    # Rotation preserves vector length, so no information is lost.
    assert torch.allclose(apply_rope(q.expand(T, d), cos, sin).norm(dim=-1), q.norm().expand(T), atol=1e-4)
    print(f"✓ RoPE is relative: score(5,2)={s1:.4f}  score(20,17)={s2:.4f}  score(31,28)={s3:.4f}")
