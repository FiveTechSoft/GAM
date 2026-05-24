# GAM — Roadmap

Ideas for future exploration, ordered roughly by expected impact.

---

## Short term (next)

- [ ] **Multi-head GAM.** Replace the single H matrix with `h` independent
      heads, each with its own projections.  Multi-head attention is one of
      the Transformer's key strengths; GAM should benefit similarly.

- [ ] **LayerNorm inside the memory loop.** Currently H accumulates
      without normalisation.  Adding a small norm (or a forget gate)
      every N steps would improve stability on sequences >512 tokens.

- [ ] **Forget / reset gate.** An additional sigmoid gate that decays
      H before each write (like LSTM's forget gate).  This would let
      the model explicitly discard old associations.

---

## Medium term

- [ ] **Hybrid GAM + local attention.** Use GAM as a global memory layer
      (O(n)) interleaved with a **local sliding-window attention** (also
      O(n)).  This gives the best of both: associative recall across
      arbitrary distances + precise local token mixing.

- [ ] **Causal kernel form.** Derive an equivalent formulation where
      `H` is never materialised, but instead the output is computed via
      an online linear recurrence (like Mamba / RWKV).  This would
      enable fast inference and GPU-friendly training.

- [ ] **Scaling study.** Train GAM at 100M, 1B and 7B parameter scales on
      standard LM benchmarks (Wikitext, C4, The Pile) and compare
      perplexity / throughput against a same-sized Transformer.

- [ ] **Long-context benchmark.** Test GAM on tasks requiring 8k–128k
      context (e.g. GovReport, BookSum, Needle-in-a-Haystack) where
      the O(n²) cost of attention is prohibitive.

---

## Long term

- [ ] **GAM for vision.** Replace the ViT patch-mixing MLP or the
      attention blocks with GAM layers.  Images have a natural spatial
      structure that a recurrent associative memory might exploit
      efficiently.

- [ ] **GAM for RL / POMDP.** The recurrent hidden state H is a natural
      representation for partially observed environments.  Test on
      memory-based RL benchmarks (e.g., PopGym, DMLab).

- [ ] **Hardware-aware kernel.** Write a custom CUDA / Triton kernel
      that fuses the delta-rule loop into a single pass, similar to
      the FlashAttention or Mamba kernels.  This would make GAM
      competitive on wall-clock time at small batch sizes.

- [ ] **Theoretical analysis.** Prove that GAM is a universal
      approximator of sequence-to-sequence functions (or characterise
      its capacity relative to attention).  A formal result would
      strengthen the empirical findings.

- [ ] **arXiv paper.** Write up the architecture, benchmarks, and
      theoretical properties as a scientific paper.

---

## Won't do (but interesting)

- **GAM without recurrence** — a feed-forward version that processes
  all tokens in parallel (like Linear Transformer).  This would lose
  the sequential memory, but might be faster for fixed-length inputs.

- **GAM with complex / quaternion values.**  Complex-valued memories
  can store richer associations, but the engineering overhead is high.

---

*Last updated: 2026-05-24*

Contributions, ideas and PRs welcome.  Open an issue to discuss any
item above.
