"""
GAM vs Transformer — copy-task comparison.
Single layer, ~equal parameter count, causal LM, multi-seed.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import SeqModel, count_params
from runner import accuracy, run_epoch, summarize
from tasks_data import make_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--min-seq", type=int, default=8)
    p.add_argument("--max-seq", type=int, default=24)
    p.add_argument("--train-n", type=int, default=2000)
    p.add_argument("--test-n", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--plot", type=str, default="gam_vs_transformer.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  d_model: {args.d_model}  vocab: {args.vocab}  seeds: {args.seeds}")

    pad_id = args.vocab  # BOS doubles as pad sentinel in this task
    vocab_size = args.vocab + 1

    results = {}
    hist_for_plot = None

    for variant in ("gam", "trans"):
        accs = []
        last_hist = None
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(seed)
            train_x, train_y = make_copy(args.train_n, args.vocab, args.min_seq, args.max_seq, rng)
            test_x, test_y = make_copy(args.test_n, args.vocab, args.min_seq, args.max_seq, rng)

            model = SeqModel(vocab_size, args.d_model, n_layers=1, variant=variant).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            hist = {"train": [], "test": []}
            t0 = time.time()
            for ep in range(1, args.epochs + 1):
                tr = run_epoch(model, opt, train_x, train_y, pad_id, device, args.batch, train=True)
                te = run_epoch(model, opt, test_x, test_y, pad_id, device, args.batch, train=False)
                hist["train"].append(tr); hist["test"].append(te)
                if ep == 1 or ep % 20 == 0:
                    print(f"  [{variant} seed={seed}] ep {ep:3d}  train {tr:.4f}  test {te:.4f}")
            elapsed = time.time() - t0
            acc = accuracy(model, test_x, test_y, pad_id, device, args.batch)
            accs.append(acc)
            last_hist = hist
            print(f"  [{variant} seed={seed}] acc={acc:.4f}  time={elapsed:.1f}s  params={count_params(model):,}")
        mean, std = summarize(accs)
        results[variant] = (mean, std, accs)
        if variant == "gam":
            hist_for_plot = {"gam_train": last_hist["train"], "gam_test": last_hist["test"]}
        else:
            hist_for_plot.update({"trans_train": last_hist["train"], "trans_test": last_hist["test"]})

    print("\nFINAL (mean ± std over seeds)")
    for v, (m, s, _) in results.items():
        print(f"  {v:>5}: {m:.4f} ± {s:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for k, v in hist_for_plot.items():
        ax.plot(v, label=k)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    ax.set_title(f"Loss curves (d={args.d_model}, last seed)")

    ax = axes[1]
    means = [results["gam"][0], results["trans"][0]]
    stds = [results["gam"][1], results["trans"][1]]
    bars = ax.bar(["GAM", "Transformer"], means, yerr=stds, capsize=6,
                  color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title(f"Test accuracy (n_seeds={len(args.seeds)})")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Plot saved -> {args.plot}")


if __name__ == "__main__":
    main()
