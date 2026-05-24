"""
Shared model components for GAM benchmarks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gam_layer import GAMLayer


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerLayer(nn.Module):
    """Single-head causal self-attention block, matched in parameter
    count to GAMLayer for fair comparison."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.W1 = nn.Linear(d, d * 4)
        self.W2 = nn.Linear(d * 4, d)
        self._reset()

    def _reset(self):
        lim = 1.0 / math.sqrt(self.d)
        for w in (self.Wq, self.Wk, self.Wv, self.Wo):
            nn.init.uniform_(w.weight, -lim, lim)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, X):
        B, T, d = X.shape
        q = self.Wq(X)
        k = self.Wk(X)
        v = self.Wv(X)
        att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=X.device), diagonal=1)
        att = F.softmax(att + mask, dim=-1)
        out = self.Wo(torch.bmm(att, v))
        out = self.ln1(out + X)
        z = F.gelu(self.W1(out))
        return self.ln2(out + self.W2(z))


class SeqModel(nn.Module):
    """Generic stack: embed + PE + N layers + head."""

    def __init__(self, vocab_size, d_model, n_layers, variant):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPE(d_model)
        layer_cls = GAMLayer if variant == "gam" else TransformerLayer
        self.layers = nn.ModuleList([layer_cls(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        e = self.pe(self.embed(x))
        for layer in self.layers:
            e = layer(e)
        return self.head(e)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
