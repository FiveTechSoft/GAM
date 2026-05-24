"""
GAM — Gated Associative Memory
===============================
A recurrent neural module that learns to read and write an internal
associative memory using a gated delta rule.

Reference: see README.md for full architecture description.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAMLayer(nn.Module):
    """
    Gated Associative Memory layer.

    Maintains a hidden matrix H (d x d) updated step by step with a
    gated delta rule. At each time step:

        read   = H_{t-1} @ q                       (associative recall)
        error  = v - H_{t-1} @ k                   (delta-rule residual)
        ΔH     = g * outer(error, k)               (per-row gated)
        H_t    = H_{t-1} + ΔH

    followed by a residual, LayerNorm and MLP.

    Args:
        d: model dimension.
        detach_every: if >0, detach H from autograd graph every N steps
            (truncated BPTT). 0 disables — full BPTT through T.

    Parameters per layer:  12d² + 4d  (approx. 12d²).
    Time complexity:        O(T · d²) — linear in sequence length.
    Memory:                 O(d²) state per sample (constant in T).
    """

    def __init__(self, d: int, detach_every: int = 0):
        super().__init__()
        self.d = d
        self.detach_every = detach_every

        # Projections (q, k, v read/write; g per-dim write gate)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wg = nn.Linear(d, d, bias=True)

        # Norm + MLP
        self.ln1 = nn.LayerNorm(d)
        self.W1 = nn.Linear(d, d * 4)
        self.W2 = nn.Linear(d * 4, d)

        self._reset_parameters()

    def _reset_parameters(self):
        lim = 1.0 / math.sqrt(self.d)
        for w in (self.Wq, self.Wk, self.Wv, self.Wg):
            nn.init.uniform_(w.weight, -lim, lim)
        nn.init.constant_(self.Wg.bias, -2.0)  # gate ~0.12 at init
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def init_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.d, self.d, device=device, dtype=dtype)

    def forward(
        self,
        X: torch.Tensor,
        H: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        """
        Args:
            X: (B, T, d) input sequence.
            H: optional initial memory (B, d, d). Defaults to zeros.
            return_state: if True, returns (Y, H_final) for streaming use.
        Returns:
            Y: (B, T, d) output sequence, or (Y, H) if return_state=True.
        """
        B, T, d = X.shape
        if H is None:
            H = self.init_state(B, X.device, X.dtype)

        outs = []
        for t in range(T):
            x = X[:, t, :]                                       # (B, d)

            q = self.Wq(x)                                       # (B, d)
            k = self.Wk(x)
            v = self.Wv(x)
            g = torch.sigmoid(self.Wg(x))                        # (B, d)

            # Associative read
            read = torch.bmm(H, q.unsqueeze(-1)).squeeze(-1)     # (B, d)

            # Delta-rule write: ΔH[i,j] = g[i] * (v - H k)[i] * k[j]
            retrieved = torch.bmm(H, k.unsqueeze(-1)).squeeze(-1)
            error = v - retrieved                                # (B, d)
            delta = (g * error).unsqueeze(-1) * k.unsqueeze(-2)  # (B, d, d)
            H = H + delta

            if self.detach_every and (t + 1) % self.detach_every == 0:
                H = H.detach()

            # Residual + LayerNorm + MLP
            out = self.ln1(read + x)
            z = F.gelu(self.W1(out))
            out = out + self.W2(z)
            outs.append(out.unsqueeze(1))

        Y = torch.cat(outs, dim=1)
        if return_state:
            return Y, H
        return Y
