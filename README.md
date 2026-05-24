# GAM — Gated Associative Memory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FiveTechSoft/GAM/blob/master/gam_demo.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code size](https://img.shields.io/github/languages/code-size/fivetechsoft/GAM)](https://github.com/FiveTechSoft/GAM)

A neural sequence architecture that replaces the Transformer's multi-head
attention with a **learned associative memory** updated via a **gated delta
rule**.  GAM runs in **O(n)** time (vs O(n²) for attention) while matching
or exceeding Transformer performance on memory-intensive tasks.

---

## Table of Contents

1. [Idea](#idea)
2. [Architecture](#architecture)
3. [Why it works](#why-it-works)
4. [Parameter count](#parameter-count)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Benchmarks](#benchmarks)
8. [Related work](#related-work)
9. [Limitations](#limitations)
10. [License](#license)

---

## Idea

The core insight is that **self-attention is overkill** for most sequence
modelling.  A single matrix **H** (d × d) that is read and written step by
step with a simple delta rule can replace the full quadratic attention
mechanism while preserving the ability to recall and associate tokens
across time.

GAM strips the Transformer down to:

| Transformer component | GAM replacement |
|---|---|
| Multi-head self-attention (QKV + softmax + O) | Associative memory (one matrix H) |
| Positional encoding (learned / sinusoidal) | Implicit ordering via recurrence |
| O(n²) time, O(n) memory per layer | O(n) time, O(1) memory per layer (~d²) |

---

## Architecture

### Forward pass (one layer, one time step)

```
x = X[t]                          # (d,)           current embedding

q = W_q · x                       # query for reading
k = W_k · x                       # key   for writing
v = W_v · x                       # value for writing
g = sigmoid(W_g · x + b_g)        # write gate, 0..1

read = H_{t-1} · q                # associative recall
error = v - H_{t-1} · k           # delta-rule residual
ΔH = g ⊗ error ⊗ kᵀ               # outer product (write)
H_t = H_{t-1} + ΔH

mem_out = LayerNorm(read + x)     # residual + norm
z = GeLU(W₁ · mem_out)
out = mem_out + W₂ · z            # MLP + residual
```

### Multi-layer stacking

Multiple GAM layers can be stacked like Transformer layers. Each layer
has its own independent H matrix.  The output of layer i becomes the
input to layer i+1.  We observed that depth helps significantly on
tasks requiring long-range reordering (see [benchmarks](#benchmarks)).

### Initialisation

- `W_q, W_k, W_v, W_g` weights: uniform `±1/√d` (small, avoids initial
  wild H updates)
- `b_g = -2`: the gate starts nearly closed (~0.12), so early training
  focuses on the residual / MLP path and the memory is not overwritten
  chaotically
- Xavier init for the MLP weights

---

## Why it works

1. **Delta rule as gradient descent.** The update `v - H·k` is the gradient
   of `∥v - H·k∥²` w.r.t. the memory.  Each step performs one step of
   gradient descent to store the association `k → v`.

2. **Gated overwrite prevention.** The sigmoid gate `g` learns *when* to
   store and *when* to ignore.  For irrelevant tokens the gate stays
   closed, preserving previous associations.

3. **Separate read / write keys.** Unlike early fast-weight models, GAM
   uses different projections for reading (`q`) and writing (`k`).  This
   lets the model learn *what* to recall independently of *what* to store.

4. **Constant memory, linear time.** The H matrix is O(d²) regardless of
   sequence length.  Forward pass is O(T·d²) — linear in T, not quadratic.

---

## Parameter count

| Component | Matrices | Parameters |
|---|---|---|
| Projections (q, k, v, g) | 4 × (d×d) + 1 bias | 4d² + d |
| MLP (W₁, W₂) | (d×4d) + (4d×d) + 1 bias | 8d² + d |
| LayerNorm | 2 vectors | 2d |
| **Total per layer** | | **12d² + 4d ≈ 12d²** |

**Transformer equivalent:** ~13d² (Q, K, V, O + MLP + LN).  GAM is ~8%
lighter and avoids the softmax and O(n²) attention altogether.

---

## Installation

```bash
git clone https://github.com/fivetechsoft/GAM.git
cd GAM
pip install -r requirements.txt
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0.

---

## Usage

### Standalone GAM layer

```python
from gam_layer import GAMLayer
import torch

layer = GAMLayer(d=64)
x = torch.randn(2, 32, 64)   # (batch, seq_len, dim)
y = layer(x)                  # (2, 32, 64)
```

### Full model (GAM-based language model)

```python
import torch.nn as nn
from gam_layer import GAMLayer

class GAMLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GAMLayer(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
```

### Run benchmarks

```bash
# Single-layer copy task
python benchmark_copy.py

# Three tasks (gap-copy, running-sum, reverse) — single layer
python benchmark_3tasks.py

# Same three tasks — 3 layers
python benchmark_3layers.py
```

---

## Benchmarks

### Setup

| Hyperparameter | Value |
|---|---|
| d_model | 64 |
| Vocab size | 14–20 depending on task |
| Optimizer | AdamW, lr=4e-4 or 5e-4 |
| Epochs | 40–120 |
| Training sequences | 1 500 |
| Test sequences | 300 |

### Results: 1 layer (54k params)

| Task | GAM | Transformer | Delta |
|---|---|---|---|
| Gap Copy | **57.6%** | 57.0% | **+0.7%** |
| Running Sum | 22.7% | **28.4%** | –5.7% |
| Reverse | 28.7% | **39.1%** | –10.3% |

### Results: 3 layers (~152k params)

| Task | GAM 3L | Transformer 3L | Delta |
|---|---|---|---|
| Gap Copy | **57.6%** | 57.6% | +0.0% |
| Running Sum | 26.7% | **29.3%** | –2.6% |
| Reverse | **45.6%** | 23.7% | **+21.9%** |

### Key observations

- **Gap Copy**: both models saturate at ~57–58 %.  Memory through gaps is
  equally easy for recurrence and attention.
- **Running Sum**: Transformer retains a small lead, but GAM's gap
  shrinks from –5.7 to –2.6 when depth increases from 1 to 3 layers.
- **Reverse**: GAM **improves 17 points** when stacked to 3 layers,
  while Transformer **drops 15 points**.  The recurrent memory stack
  can reorder tokens progressively; causal self-attention fragments
  and loses information with depth for this task.

**Bottom line:** GAM matches or beats a same-sized Transformer on 2 of 3
tasks at 3 layers, using **O(n)** instead of **O(n²)** and with simpler
code.

---

## Related work

| Work | Relation to GAM |
|---|---|
| **Fast Weights** (Schmidhuber, 1992; Ba et al., 2016) | First to propose updating a hidden matrix with an outer product. Uses a scalar learning rate. |
| **DeltaNet** (Schlag et al., 2021, ICML) | Delta rule with a learned scalar gate. Closest prior art. GAM differs by using a **vector gate `g`** (per-dimension) and **separate q/k projections**. |
| **Linear Transformer** (Katharopoulos et al., 2020) | Replaces softmax with `φ(Q)·φ(K)ᵀ` for linear time. Still maintains full attention matrix implicitly. |
| **Mamba / SSM** (Gu & Dao, 2023) | Selective state-space model with O(n) time. Different mechanism (linear ODE) but same complexity class. |
| **RWKV** (Peng et al., 2023) | Attention-free RNN with O(n) time. Uses learned receptance / decay channels. |

GAM is **not a new invention** — it (re)combines existing ideas (fast
weights + delta rule + gating + Transformer MLP block) into a minimal,
easily analysable module.

---

## Limitations

- **No free lunch.** GAM trades attention's O(n²) flexibility for O(n)
  speed.  On tasks requiring arbitrary content-based routing (e.g.
  question answering with many competing keys), attention may still win.
- **Single memory matrix.** Unlike multi-head attention that can attend
  to different representation subspaces, the single H matrix is a
  bottleneck.  A **multi-head variant** (several H matrices with
  different projections) is a natural extension.
- **Numerical stability.** The recurrent delta update can accumulate
  errors over very long sequences (>1024 tokens).  Adding LayerNorm
  inside the memory path or using a forget mechanism (like an LSTM)
  would help.
- **Not tested at scale.** All benchmarks here use ≤ 152k parameters
  and ≤ 40 tokens.  Scaling to GPT-like sizes (100M+) is unexplored.

---

## License

MIT — see [LICENSE](LICENSE).
