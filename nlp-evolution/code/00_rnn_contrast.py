"""
00 · One LSTM step (contrast only)
==================================
LSTM: Hochreiter & Schmidhuber, 1997. Shown here only to make one point:
an RNN must walk the sequence ONE token at a time, because h_t depends on h_{t-1}.
That serial loop is exactly what the 2017 Transformer removed.

Run:  python 00_rnn_contrast.py
"""
import torch

torch.manual_seed(0)
d_in, d_h = 8, 16

# One fused weight for the 4 gates: input, forget, cell-candidate, output.
W_x = torch.randn(4 * d_h, d_in) * 0.1   # acts on the current token x_t
W_h = torch.randn(4 * d_h, d_h) * 0.1    # acts on the previous hidden state h_{t-1}
b = torch.zeros(4 * d_h)

def lstm_step(x_t, h_prev, c_prev):
    gates = x_t @ W_x.T + h_prev @ W_h.T + b
    i, f, g, o = gates.chunk(4, dim=-1)
    i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()   # gates live in (0, 1)
    g = g.tanh()                                      # candidate memory
    c = f * c_prev + i * g    # forget part of the old memory, write part of the new
    h = o * c.tanh()          # expose a filtered view of memory
    return h, c

# The serial bottleneck: a Python loop over time steps that cannot be parallelised.
x = torch.randn(1, 5, d_in)                 # (batch, seq, d_in)
h = c = torch.zeros(1, d_h)
for t in range(x.shape[1]):
    h, c = lstm_step(x[:, t], h, c)

# Verify against PyTorch's reference cell (same gate order: i, f, g, o).
ref = torch.nn.LSTMCell(d_in, d_h)
with torch.no_grad():
    ref.weight_ih.copy_(W_x); ref.weight_hh.copy_(W_h)
    ref.bias_ih.copy_(b); ref.bias_hh.zero_()
h_ref = c_ref = torch.zeros(1, d_h)
for t in range(x.shape[1]):
    h_ref, c_ref = ref(x[:, t], (h_ref, c_ref))

assert torch.allclose(h, h_ref, atol=1e-6)
print("✓ LSTM step matches nn.LSTMCell — 5 tokens needed 5 sequential steps")
