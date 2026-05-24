"""
Synthetic tasks used by the GAM benchmarks.
"""
import numpy as np
import torch


# ── Copy ─────────────────────────────────────────────────────────────
def make_copy(n, vocab, min_len, max_len, rng):
    bos = vocab
    xs, ys = [], []
    for _ in range(n):
        L = rng.integers(min_len, max_len + 1)
        seq = rng.integers(0, vocab, size=L).tolist()
        xs.append([bos] + seq)
        ys.append(seq + [bos])
    return xs, ys


# ── Gap Copy ─────────────────────────────────────────────────────────
GAP_CFG = dict(vocab=20, bos=16, gap=17, pad=18, n_gap=8, min_len=4, max_len=10)

def make_gapcopy(n, rng, cfg=GAP_CFG):
    xs, ys = [], []
    for _ in range(n):
        L = rng.integers(cfg["min_len"], cfg["max_len"] + 1)
        seq = rng.integers(0, 16, size=L).tolist()
        xs.append([cfg["bos"]] + [cfg["gap"]] * cfg["n_gap"] + seq)
        ys.append([cfg["gap"]] * cfg["n_gap"] + seq + [cfg["bos"]])
    return xs, ys


# ── Running Sum (mod 10) ─────────────────────────────────────────────
SUM_CFG = dict(vocab=14, bos=10, plus=11, pad=12, digits=10, min_len=4, max_len=10)

def make_sumdata(n, rng, cfg=SUM_CFG):
    xs, ys = [], []
    for _ in range(n):
        L = rng.integers(cfg["min_len"], cfg["max_len"])
        ds = rng.integers(0, cfg["digits"], size=L).tolist()
        inp, tgt = [cfg["bos"]], []
        s = 0
        for d in ds:
            s = (s + d) % cfg["digits"]
            inp += [d, cfg["plus"]]
            tgt += [d, s]
        tgt.append(cfg["bos"])
        xs.append(inp); ys.append(tgt)
    return xs, ys


# ── Reverse ──────────────────────────────────────────────────────────
REV_CFG = dict(vocab=20, bos=16, rev=17, pad=18, min_len=3, max_len=8)

def make_revdata(n, rng, cfg=REV_CFG):
    xs, ys = [], []
    for _ in range(n):
        L = rng.integers(cfg["min_len"], cfg["max_len"] + 1)
        seq = rng.integers(0, 16, size=L).tolist()
        rev = list(reversed(seq))
        xs.append([cfg["bos"]] + seq + [cfg["rev"]] + rev)
        ys.append(seq + [cfg["rev"]] + rev + [cfg["bos"]])
    return xs, ys


# ── Padding util ─────────────────────────────────────────────────────
def pad_batch(xs, ys, pad_id, device):
    max_len = max(len(s) for s in xs)
    inp = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    tgt = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        inp[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        tgt[i, :len(y)] = torch.tensor(y, dtype=torch.long)
    return inp.to(device), tgt.to(device)
