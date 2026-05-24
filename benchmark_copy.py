"""
GAM (Gated Associative Memory) vs Transformer — Copy-Task Comparison
====================================================================
Single-layer, ~equal parameter count, causal language modelling on
synthetic copy sequences.
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32
VOCAB     = 32          # small vocabulary
D_MODEL   = 64          # embedding / hidden dim
TRAIN_LEN = 2000        # sequences
TEST_LEN  = 200
MIN_SEQ   = 8
MAX_SEQ   = 24
BATCH     = 32
LR        = 5e-4
EPOCHS    = 120
SEED      = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}   Vocab: {VOCAB}   d_model: {D_MODEL}")

# ── Data ─────────────────────────────────────────────────────────────
def make_copy_data(n, min_len, max_len):
    """Return (input, target) — input has <BOS> prepended, target has it shifted."""
    xs, ys = [], []
    for _ in range(n):
        L = np.random.randint(min_len, max_len + 1)
        seq = np.random.randint(0, VOCAB, size=L).tolist()
        inp = [VOCAB] + seq                     # <BOS> = VOCAB
        tgt = seq + [VOCAB]                     # predict next token, ends with <BOS>
        xs.append(inp)
        ys.append(tgt)
    return xs, ys

def pad_batch(xs, ys):
    max_len = max(len(s) for s in xs)
    inp = torch.full((len(xs), max_len), VOCAB, dtype=torch.long)
    tgt = torch.full((len(xs), max_len), -100, dtype=torch.long)  # -100 ignored by CE
    for i, (x, y) in enumerate(zip(xs, ys)):
        inp[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        tgt[i, :len(y)] = torch.tensor(y, dtype=torch.long)
    return inp.to(DEVICE), tgt.to(DEVICE)

train_x, train_y = make_copy_data(TRAIN_LEN, MIN_SEQ, MAX_SEQ)
test_x,  test_y  = make_copy_data(TEST_LEN,  MIN_SEQ, MAX_SEQ)

# ── GAM Layer ────────────────────────────────────────────────────────
class GAMLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wg = nn.Linear(d, d, bias=True)
        self.ln1 = nn.LayerNorm(d)
        self.W1 = nn.Linear(d, d * 4)
        self.W2 = nn.Linear(d * 4, d)
        self._reset()

    def _reset(self):
        lim = 1.0 / math.sqrt(D_MODEL)
        for w in [self.Wq, self.Wk, self.Wv, self.Wg]:
            nn.init.uniform_(w.weight, -lim, lim)
        nn.init.constant_(self.Wg.bias, -2.0)   # gate starts nearly closed
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, X):
        B, T, d = X.shape
        H = torch.zeros(B, d, d, device=X.device, dtype=X.dtype)
        outs = []
        for t in range(T):
            x = X[:, t, :]                             # (B, d)
            q = self.Wq(x)                             # (B, d)
            k = self.Wk(x)
            v = self.Wv(x)
            g = torch.sigmoid(self.Wg(x))              # (B, d)

            read = torch.bmm(H, q.unsqueeze(-1)).squeeze(-1)  # (B, d)
            # delta rule
            retrieved = torch.bmm(H, k.unsqueeze(-1)).squeeze(-1)
            error = v - retrieved
            # H += g ⊗ error ⊗ kᵀ   (outer product)
            delta = g.unsqueeze(-1) * error.unsqueeze(-2) * k.unsqueeze(-1)
            H = H + delta

            out = self.ln1(read + x)
            z = F.gelu(self.W1(out))
            z = self.W2(z)
            out = out + z
            outs.append(out.unsqueeze(1))
        return torch.cat(outs, dim=1)

# ── Transformer Layer (single-head, for fair comparison) ────────────
class TransformerLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
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
        lim = 1.0 / math.sqrt(D_MODEL)
        for w in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.uniform_(w.weight, -lim, lim)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, X, mask=None):
        B, T, d = X.shape
        # Causal attention (single head)
        q = self.Wq(X).view(B, T, d)   # (B, T, d)
        k = self.Wk(X).view(B, T, d)
        v = self.Wv(X).view(B, T, d)
        att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)  # (B, T, T)
        if mask is None:
            mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1).to(X.device)
        att = att + mask
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, v)                                # (B, T, d)
        out = self.Wo(out)
        out = self.ln1(out + X)
        z = F.gelu(self.W1(out))
        z = self.W2(z)
        out = self.ln2(out + z)
        return out

# ── Full Models ──────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class GAMModel(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, d)   # +1 for <BOS>
        self.pe = SinusoidalPE(d)
        self.layer = GAMLayer(d)
        self.head  = nn.Linear(d, vocab + 1)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        e = self.embed(x)
        e = self.pe(e)
        e = self.layer(e)
        return self.head(e)

class TransformerModel(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, d)
        self.pe = SinusoidalPE(d)
        self.layer = TransformerLayer(d)
        self.head  = nn.Linear(d, vocab + 1)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        e = self.embed(x)
        e = self.pe(e)
        e = self.layer(e)
        return self.head(e)

# ── Parameter count ──────────────────────────────────────────────────
gam_model   = GAMModel(VOCAB, D_MODEL).to(DEVICE)
trans_model = TransformerModel(VOCAB, D_MODEL).to(DEVICE)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

pg, pt = count_params(gam_model), count_params(trans_model)
print(f"GAM params:         {pg:,}")
print(f"Transformer params: {pt:,}")

# ── Training ─────────────────────────────────────────────────────────
def train_one_epoch(model, opt, data_x, data_y, desc):
    model.train()
    total_loss = 0.0
    n_batches = 0
    indices = list(range(len(data_x)))
    np.random.shuffle(indices)
    for start in range(0, len(indices), BATCH):
        batch_idx = indices[start:start + BATCH]
        bx = [data_x[i] for i in batch_idx]
        by = [data_y[i] for i in batch_idx]
        inp, tgt = pad_batch(bx, by)
        opt.zero_grad()
        logits = model(inp)                         # (B, T, V)
        loss = F.cross_entropy(logits.view(-1, VOCAB + 1),
                               tgt.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches

def evaluate(model, data_x, data_y):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, len(data_x), BATCH):
            bx = data_x[start:start + BATCH]
            by = data_y[start:start + BATCH]
            inp, tgt = pad_batch(bx, by)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, VOCAB + 1),
                                   tgt.view(-1), ignore_index=-100)
            total_loss += loss.item()
            n += 1
    return total_loss / n

# ── Run ──────────────────────────────────────────────────────────────
gam_opt   = torch.optim.AdamW(gam_model.parameters(),   lr=LR)
trans_opt = torch.optim.AdamW(trans_model.parameters(),  lr=LR)

hist = {"gam_train": [], "gam_test": [], "trans_train": [], "trans_test": []}
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    lt = train_one_epoch(gam_model,   gam_opt,   train_x, train_y, "GAM")
    le = evaluate(gam_model,   test_x, test_y)
    hist["gam_train"].append(lt); hist["gam_test"].append(le)

    lt2 = train_one_epoch(trans_model, trans_opt, train_x, train_y, "Transformer")
    le2 = evaluate(trans_model,   test_x, test_y)
    hist["trans_train"].append(lt2); hist["trans_test"].append(le2)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | GAM train {lt:.4f} test {le:.4f} | "
              f"Transformer train {lt2:.4f} test {le2:.4f}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

# ── Accuracy ─────────────────────────────────────────────────────────
def accuracy(model, data_x, data_y):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(data_x), BATCH):
            bx = data_x[start:start + BATCH]
            by = data_y[start:start + BATCH]
            inp, tgt = pad_batch(bx, by)
            logits = model(inp)                     # (B, T, V)
            preds = logits.argmax(dim=-1)
            mask = tgt != -100
            correct += (preds[mask] == tgt[mask]).sum().item()
            total   += mask.sum().item()
    return correct / total

acc_gam   = accuracy(gam_model,   test_x, test_y)
acc_trans = accuracy(trans_model, test_x, test_y)
print(f"\nTest accuracy:  GAM {acc_gam:.4f}   Transformer {acc_trans:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(hist["gam_train"],   label="GAM train")
ax.plot(hist["gam_test"],    label="GAM test")
ax.plot(hist["trans_train"], label="Transformer train")
ax.plot(hist["trans_test"],  label="Transformer test")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.set_title(f"Loss (d={D_MODEL})")

ax = axes[1]
bars = ax.bar(["GAM", "Transformer"], [acc_gam, acc_trans], color=["#4C72B0", "#DD8452"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Test Accuracy")
for bar, val in zip(bars, [acc_gam, acc_trans]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("gam_vs_transformer.png", dpi=150)
print("Plot saved -> gam_vs_transformer.png")
