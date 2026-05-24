"""
GAM — Gated Associative Memory
===============================
A single-layer recurrent neural module that learns to read and write
an internal associative memory using a delta rule with a learned gate.

Reference: see README.md for full architecture description.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAMLayer(nn.Module):
    """
    Gated Associative Memory layer.

    Maintains a hidden matrix H (d x d) that is updated step by step
    with a gated delta rule.  At each time step:

        read   = H_{t-1} @ q                     (associative recall)
        error  = v - H_{t-1} @ k                  (residual)
        ΔH     = g ⊗ error ⊗ kᵀ                   (outer product)
        H_t    = H_{t-1} + ΔH

    followed by a learned gating, residual, LayerNorm and MLP.

    Parameters per layer:  12d² + 4d  (approx. 12d²).
    Time complexity:        O(T · d²)  — linear in sequence length.
    Memory:                 O(d²)      — constant H matrix.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

        # Projections
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wg = nn.Linear(d, d, bias=True)      # +1 bias for gate

        # Normalisation + MLP
        self.ln1 = nn.LayerNorm(d)
        self.W1  = nn.Linear(d, d * 4)
        self.W2  = nn.Linear(d * 4, d)

        self._reset_parameters()

    def _reset_parameters(self):
        lim = 1.0 / math.sqrt(self.d)
        for w in [self.Wq, self.Wk, self.Wv, self.Wg]:
            nn.init.uniform_(w.weight, -lim, lim)
        nn.init.constant_(self.Wg.bias, -2.0)      # gate starts nearly closed
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, T, d) input sequence.
        Returns:
            Y: (B, T, d) output sequence.
        """
        B, T, d = X.shape
        H = torch.zeros(B, d, d, device=X.device, dtype=X.dtype)
        outs = []

        for t in range(T):
            x = X[:, t, :]                           # (B, d)

            # Projections
            q = self.Wq(x)                           # (B, d)
            k = self.Wk(x)
            v = self.Wv(x)
            g = torch.sigmoid(self.Wg(x))            # (B, d)

            # Associative read
            read = torch.bmm(H, q.unsqueeze(-1)).squeeze(-1)  # (B, d)

            # Delta-rule write
            retrieved = torch.bmm(H, k.unsqueeze(-1)).squeeze(-1)
            error = v - retrieved
            # ΔH = outer(g, error, k)
            delta = g.unsqueeze(-1) * error.unsqueeze(-2) * k.unsqueeze(-1)
            H = H + delta

            # Residual + LayerNorm + MLP
            out = self.ln1(read + x)
            z = F.gelu(self.W1(out))
            z = self.W2(z)
            out = out + z
            outs.append(out.unsqueeze(1))

        return torch.cat(outs, dim=1)
