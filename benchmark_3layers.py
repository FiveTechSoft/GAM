"""
GAM vs Transformer — 3-layer comparison across 3 tasks, multi-seed.

Convenience wrapper around benchmark_3tasks with n_layers=3, lr=4e-4,
epochs=40 to reproduce the table from the README.
"""
import sys

from benchmark_3tasks import main  # reuse

if __name__ == "__main__":
    # Inject defaults if user did not override
    args = sys.argv[1:]
    def has(flag): return any(a == flag or a.startswith(flag + "=") for a in args)
    if not has("--n-layers"): args += ["--n-layers", "3"]
    if not has("--lr"):       args += ["--lr", "4e-4"]
    if not has("--epochs"):   args += ["--epochs", "40"]
    sys.argv = [sys.argv[0]] + args
    main()
