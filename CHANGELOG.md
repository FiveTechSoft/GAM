# Changelog

## v1.1.0 — review-fixes

### Fixed
- **Delta-rule outer product** in `GAMLayer`. The previous expression
  `g.unsqueeze(-1) * error.unsqueeze(-2) * k.unsqueeze(-1)` produced
  `ΔH[i,j] = g[i] * error[j] * k[i]`, swapping the roles of `error`
  and `k`. The intended (and documented) form is
  `ΔH[i,j] = g[i] * error[i] * k[j]`, i.e. a per-row gated outer
  product of the delta-rule residual and the write key. The network
  still trained because the learned projections absorbed the
  transpose, but `H @ k` no longer recovered `v` as the delta rule
  prescribes. Re-run all benchmarks after upgrading.

### Added
- `GAMLayer(detach_every=N)` — truncates BPTT through the memory
  matrix every `N` time steps. `0` (default) keeps full BPTT for
  backwards compatibility.
- `GAMLayer.forward(X, H=None, return_state=True)` — accept and
  return the memory matrix for streaming / chunked inference.
- `models.py` with shared `SinusoidalPE`, `TransformerLayer`, and
  `SeqModel`; all benchmarks now import `GAMLayer` from `gam_layer`
  rather than duplicating it.
- `tasks_data.py` and `runner.py` for shared task generators and the
  training loop.
- `--seeds` flag on every benchmark; reported numbers are
  `mean ± std` across seeds.
- `tests/test_gam_layer.py` — pytest suite covering shape, gradient
  flow, parameter count, the delta-rule recovery property,
  multi-layer stacking, `detach_every`, and determinism.
- `.github/workflows/ci.yml` — runs pytest on every push and PR.

### Changed
- Pinned minimum versions in `requirements.txt`.
- README: clarified `O(d²)` memory wording, dropped the placeholder
  Zenodo DOI, and flagged the pre-v1.1 benchmark numbers as
  needing re-generation under the corrected delta rule.

## v1.0.0
- Initial release.
