"""
GAM vs Transformer — 3-task benchmark (single layer), multi-seed.
Tasks: Gap Copy | Running Sum | Reverse.
"""
import argparse

import torch

from models import SeqModel, count_params
from runner import summarize, train_and_eval
from tasks_data import GAP_CFG, REV_CFG, SUM_CFG, make_gapcopy, make_revdata, make_sumdata


TASKS = [
    ("Gap Copy",    make_gapcopy, GAP_CFG),
    ("Running Sum", make_sumdata, SUM_CFG),
    ("Reverse",     make_revdata, REV_CFG),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--train-n", type=int, default=1500)
    p.add_argument("--test-n", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  d_model={args.d_model}  n_layers={args.n_layers}  "
          f"epochs={args.epochs}  seeds={args.seeds}")
    print("=" * 72)

    table = []
    for name, make_fn, cfg in TASKS:
        print(f"\n[ {name} ]")
        row = [name]
        for variant in ("gam", "trans"):
            factory = lambda v=variant, vocab=cfg["vocab"]: SeqModel(
                vocab, args.d_model, args.n_layers, v
            ).to(device)
            # Probe param count once
            _probe = factory()
            print(f"  {variant:>5} params: {count_params(_probe):,}")
            del _probe

            accs = train_and_eval(
                factory,
                lambda n, rng, fn=make_fn: fn(n, rng),
                cfg["pad"],
                seeds=args.seeds, epochs=args.epochs, batch=args.batch,
                lr=args.lr, train_n=args.train_n, test_n=args.test_n,
                device=device,
            )
            m, s = summarize(accs)
            print(f"  {variant:>5} acc: {m:.4f} ± {s:.4f}  (seeds: {[f'{a:.3f}' for a in accs]})")
            row += [m, s]
        table.append(row)

    print("\n" + "=" * 72)
    print(f"{'Task':<14} {'GAM mean':>10} {'GAM std':>10} "
          f"{'Trans mean':>12} {'Trans std':>10} {'Delta':>8}")
    print("-" * 72)
    for name, gm, gs, tm, ts in table:
        print(f"{name:<14} {gm:>10.4f} {gs:>10.4f} {tm:>12.4f} {ts:>10.4f} {gm - tm:>+8.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
