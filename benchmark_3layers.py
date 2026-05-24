"""
GAM vs Transformer — 3-layer comparison across 3 tasks
=======================================================
Stacking 3 layers for each model.
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL   = 64
N_LAYERS  = 3
TRAIN_LEN = 1500
TEST_LEN  = 300
BATCH     = 32
LR        = 4e-4
EPOCHS    = 40
SEED      = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}  d_model={D_MODEL}  layers={N_LAYERS}  epochs={EPOCHS}")
print("=" * 60)

# ── Components ───────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

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
        nn.init.constant_(self.Wg.bias, -2.0)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
    def forward(self, X):
        B, T, d = X.shape
        H = torch.zeros(B, d, d, device=X.device, dtype=X.dtype)
        outs = []
        for t in range(T):
            x = X[:, t, :]
            q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            g = torch.sigmoid(self.Wg(x))
            read = torch.bmm(H, q.unsqueeze(-1)).squeeze(-1)
            retrieved = torch.bmm(H, k.unsqueeze(-1)).squeeze(-1)
            error = v - retrieved
            delta = g.unsqueeze(-1) * error.unsqueeze(-2) * k.unsqueeze(-1)
            H = H + delta
            out = self.ln1(read + x)
            z = F.gelu(self.W1(out))
            z = self.W2(z)
            out = out + z
            outs.append(out.unsqueeze(1))
        return torch.cat(outs, dim=1)

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
    def forward(self, X):
        B, T, d = X.shape
        q = self.Wq(X).view(B, T, d)
        k = self.Wk(X).view(B, T, d)
        v = self.Wv(X).view(B, T, d)
        att = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1).to(X.device)
        att = att + mask
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, v)
        out = self.Wo(out)
        out = self.ln1(out + X)
        z = F.gelu(self.W1(out))
        z = self.W2(z)
        out = self.ln2(out + z)
        return out

# ── Factory (multi-layer) ───────────────────────────────────────────
def make_model(vocab_size, variant):
    n = N_LAYERS
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, D_MODEL)
            self.pe = SinusoidalPE(D_MODEL)
            layer_cls = GAMLayer if variant == "gam" else TransformerLayer
            self.layers = nn.ModuleList([layer_cls(D_MODEL) for _ in range(n)])
            self.head = nn.Linear(D_MODEL, vocab_size)
            nn.init.normal_(self.embed.weight, std=0.02)
            nn.init.normal_(self.head.weight, std=0.02)
        def forward(self, x):
            e = self.embed(x)
            e = self.pe(e)
            for layer in self.layers:
                e = layer(e)
            return self.head(e)
    return M()

# ── Helpers ─────────────────────────────────────────────────────────
def pad_batch(xs, ys, pad_id):
    max_len = max(len(s) for s in xs)
    inp = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    tgt = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        inp[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        tgt[i, :len(y)] = torch.tensor(y, dtype=torch.long)
    return inp.to(DEVICE), tgt.to(DEVICE)

def run_epoch(model, opt, data_x, data_y, pad_id, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    indices = list(range(len(data_x)))
    if train:
        np.random.shuffle(indices)
    with torch.set_grad_enabled(train):
        for start in range(0, len(indices), BATCH):
            bidx = indices[start:start + BATCH]
            inp, tgt = pad_batch([data_x[i] for i in bidx],
                                 [data_y[i] for i in bidx], pad_id)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tgt.view(-1), ignore_index=-100)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches

def accuracy(model, data_x, data_y, pad_id):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(data_x), BATCH):
            inp, tgt = pad_batch(data_x[start:start + BATCH],
                                 data_y[start:start + BATCH], pad_id)
            preds = model(inp).argmax(dim=-1)
            mask = tgt != -100
            correct += (preds[mask] == tgt[mask]).sum().item()
            total   += mask.sum().item()
    return correct / total

# ═════════════════════════════════════════════════════════════════════
# TASK 1 — GAP COPY
# ═════════════════════════════════════════════════════════════════════
print("\n[ TASK 1: GAP COPY ]")
V1 = 20; BOS1=16; GAP=17; PAD1=18; N_GAP=8
MIN1, MAX1 = 4, 10

def make_gapcopy(n):
    xs, ys = [], []
    for _ in range(n):
        L = np.random.randint(MIN1, MAX1+1)
        seq = np.random.randint(0, 16, size=L).tolist()
        xs.append([BOS1] + [GAP]*N_GAP + seq)
        ys.append([GAP]*N_GAP + seq + [BOS1])
    return xs, ys

gc_train_x, gc_train_y = make_gapcopy(TRAIN_LEN)
gc_test_x,  gc_test_y  = make_gapcopy(TEST_LEN)
gam_gc   = make_model(V1, "gam").to(DEVICE)
trans_gc = make_model(V1, "trans").to(DEVICE)
print(f"  GAM params: {sum(p.numel() for p in gam_gc.parameters()):,}")
print(f"  Transformer params: {sum(p.numel() for p in trans_gc.parameters()):,}")

opt_gam   = torch.optim.AdamW(gam_gc.parameters(),   lr=LR)
opt_trans = torch.optim.AdamW(trans_gc.parameters(),  lr=LR)
for ep in range(1, EPOCHS+1):
    l1 = run_epoch(gam_gc,   opt_gam,   gc_train_x, gc_train_y, PAD1, train=True)
    l2 = run_epoch(trans_gc, opt_trans, gc_train_x, gc_train_y, PAD1, train=True)
    if ep % 10 == 0 or ep == 1:
        print(f"  ep {ep:3d} | GAM {l1:.4f} | Transformer {l2:.4f}")
acc_gam_gc   = accuracy(gam_gc,   gc_test_x, gc_test_y, PAD1)
acc_trans_gc = accuracy(trans_gc, gc_test_x, gc_test_y, PAD1)
print(f"  >> Acc: GAM {acc_gam_gc:.4f}  |  Transformer {acc_trans_gc:.4f}")

# ═════════════════════════════════════════════════════════════════════
# TASK 2 — RUNNING SUM
# ═════════════════════════════════════════════════════════════════════
print("\n[ TASK 2: RUNNING SUM ]")
V2=14; BOS2=10; PLUS=11; PAD2=12; DIGITS=10

def make_sumdata(n):
    xs, ys = [], []
    for _ in range(n):
        L = np.random.randint(4, 10)
        digits = np.random.randint(0, DIGITS, size=L).tolist()
        inp, tgt = [BOS2], []
        s = 0
        for d in digits:
            s = (s + d) % DIGITS
            inp += [d, PLUS]
            tgt += [d, s]
        tgt.append(BOS2)
        xs.append(inp); ys.append(tgt)
    return xs, ys

sum_train_x, sum_train_y = make_sumdata(TRAIN_LEN)
sum_test_x,  sum_test_y  = make_sumdata(TEST_LEN)
gam_sum   = make_model(V2, "gam").to(DEVICE)
trans_sum = make_model(V2, "trans").to(DEVICE)
opt_gam   = torch.optim.AdamW(gam_sum.parameters(),   lr=LR)
opt_trans = torch.optim.AdamW(trans_sum.parameters(),  lr=LR)
for ep in range(1, EPOCHS+1):
    l1 = run_epoch(gam_sum,   opt_gam,   sum_train_x, sum_train_y, PAD2, train=True)
    l2 = run_epoch(trans_sum, opt_trans, sum_train_x, sum_train_y, PAD2, train=True)
    if ep % 10 == 0 or ep == 1:
        print(f"  ep {ep:3d} | GAM {l1:.4f} | Transformer {l2:.4f}")
acc_gam_sum   = accuracy(gam_sum,   sum_test_x, sum_test_y, PAD2)
acc_trans_sum = accuracy(trans_sum, sum_test_x, sum_test_y, PAD2)
print(f"  >> Acc: GAM {acc_gam_sum:.4f}  |  Transformer {acc_trans_sum:.4f}")

# ═════════════════════════════════════════════════════════════════════
# TASK 3 — REVERSE
# ═════════════════════════════════════════════════════════════════════
print("\n[ TASK 3: REVERSE ]")
V3=20; BOS3=16; REV=17; PAD3=18; MIN_R, MAX_R = 3, 8

def make_revdata(n):
    xs, ys = [], []
    for _ in range(n):
        L = np.random.randint(MIN_R, MAX_R+1)
        seq = np.random.randint(0, 16, size=L).tolist()
        rev = list(reversed(seq))
        xs.append([BOS3] + seq + [REV] + rev)
        ys.append(seq + [REV] + rev + [BOS3])
    return xs, ys

rev_train_x, rev_train_y = make_revdata(TRAIN_LEN)
rev_test_x,  rev_test_y  = make_revdata(TEST_LEN)
gam_rev   = make_model(V3, "gam").to(DEVICE)
trans_rev = make_model(V3, "trans").to(DEVICE)
opt_gam   = torch.optim.AdamW(gam_rev.parameters(),   lr=LR)
opt_trans = torch.optim.AdamW(trans_rev.parameters(),  lr=LR)
for ep in range(1, EPOCHS+1):
    l1 = run_epoch(gam_rev,   opt_gam,   rev_train_x, rev_train_y, PAD3, train=True)
    l2 = run_epoch(trans_rev, opt_trans, rev_train_x, rev_train_y, PAD3, train=True)
    if ep % 10 == 0 or ep == 1:
        print(f"  ep {ep:3d} | GAM {l1:.4f} | Transformer {l2:.4f}")
acc_gam_rev   = accuracy(gam_rev,   rev_test_x, rev_test_y, PAD3)
acc_trans_rev = accuracy(trans_rev, rev_test_x, rev_test_y, PAD3)
print(f"  >> Acc: GAM {acc_gam_rev:.4f}  |  Transformer {acc_trans_rev:.4f}")

# ═════════════════════════════════════════════════════════════════════
# FINAL TABLE
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"FINAL RESULTS ({N_LAYERS} layers, test accuracy)")
print("=" * 60)
print(f"{'Task':<20} {'GAM':>12} {'Transformer':>14} {'Delta':>8}")
print("-" * 56)
d1 = acc_gam_gc - acc_trans_gc
d2 = acc_gam_sum - acc_trans_sum
d3 = acc_gam_rev - acc_trans_rev
print(f"{'Gap Copy':<20} {acc_gam_gc:>10.4f}  {acc_trans_gc:>10.4f}  {d1:>+7.4f}")
print(f"{'Running Sum':<20} {acc_gam_sum:>10.4f}  {acc_trans_sum:>10.4f}  {d2:>+7.4f}")
print(f"{'Reverse':<20} {acc_gam_rev:>10.4f}  {acc_trans_rev:>10.4f}  {d3:>+7.4f}")
print("=" * 60)
