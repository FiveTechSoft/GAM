"""
Unit tests for GAMLayer.
Run: pytest -q
"""
import math

import pytest
import torch
import torch.nn as nn

from gam_layer import GAMLayer


# ── Shapes ───────────────────────────────────────────────────────────
def test_output_shape():
    layer = GAMLayer(d=16)
    x = torch.randn(2, 5, 16)
    y = layer(x)
    assert y.shape == x.shape


def test_state_return_and_reuse():
    layer = GAMLayer(d=8)
    x = torch.randn(1, 4, 8)
    y1, h1 = layer(x, return_state=True)
    y2, h2 = layer(x, H=h1, return_state=True)
    assert h1.shape == (1, 8, 8)
    assert h2.shape == h1.shape
    # second call differs because state non-zero
    assert not torch.allclose(y1, y2)


# ── Param count matches the documented 12d² + 4d ────────────────────
def test_param_count():
    d = 32
    layer = GAMLayer(d)
    n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    # 4 projections (Wq,Wk,Wv: no bias; Wg: with bias=d)
    proj = 4 * d * d + d
    # MLP: d*4d + 4d + 4d*d + d = 8d² + 5d
    mlp = d * 4 * d + 4 * d + 4 * d * d + d
    # LayerNorm: 2d
    ln = 2 * d
    expected = proj + mlp + ln
    assert n == expected, f"expected {expected}, got {n}"


# ── Gradient flow ───────────────────────────────────────────────────
def test_gradient_flow():
    layer = GAMLayer(d=8)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"


# ── Delta rule sanity: write then read recovers v ───────────────────
def test_delta_rule_recovers_value():
    """
    With a single write where k = q = unit vector, the read after one
    step should approximate the (gated) value.

    We bypass the projections by forcing them to identity, set the gate
    open, and check that read at t=1 reproduces v from t=0.
    """
    torch.manual_seed(0)
    d = 4
    layer = GAMLayer(d)

    # Force projections to identity, gate fully open, kill MLP/LN effect.
    with torch.no_grad():
        nn.init.eye_(layer.Wq.weight)
        nn.init.eye_(layer.Wk.weight)
        nn.init.eye_(layer.Wv.weight)
        nn.init.zeros_(layer.Wg.weight)
        nn.init.constant_(layer.Wg.bias, 50.0)  # sigmoid ≈ 1
        # Cancel the post-residual MLP: W2 → 0
        nn.init.zeros_(layer.W2.weight)
        nn.init.zeros_(layer.W2.bias)

    # Step 0: write unit-vector key/value, then step 1: same key, expect
    # read ≈ that value (modulo LayerNorm rescaling on the output path).
    k_vec = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    x = torch.stack([k_vec, k_vec], dim=1)  # (1, 2, 4)
    _, H = layer(x, return_state=True)

    # H should hold the outer product approximately: error = v - 0 = k
    # delta[i,j] = 1 * k[i] * k[j]  (after gate ≈1), then second step
    # subtracts retrieved, leaving very small residual.
    retrieved = (H @ k_vec.unsqueeze(-1)).squeeze(-1)
    # retrieved ≈ k_vec (since k is in H's column space)
    assert torch.allclose(retrieved, k_vec, atol=1e-4), retrieved


# ── Multi-layer stacking works ──────────────────────────────────────
def test_stacking():
    layers = nn.Sequential(*[GAMLayer(d=16) for _ in range(3)])
    x = torch.randn(2, 7, 16)
    y = layers(x)
    assert y.shape == x.shape
    y.sum().backward()


# ── detach_every truncates BPTT graph ───────────────────────────────
def test_detach_every_reduces_graph():
    """With detach_every=1 the graph through H is severed each step, so
    grads w.r.t. early Wq weights should be smaller (or zero through H)
    than with full BPTT. We check both modes run without error."""
    x = torch.randn(1, 8, 8)
    for de in (0, 1, 4):
        layer = GAMLayer(d=8, detach_every=de)
        y = layer(x)
        y.sum().backward()


# ── Determinism ─────────────────────────────────────────────────────
def test_determinism():
    torch.manual_seed(123)
    layer1 = GAMLayer(d=8)
    x = torch.randn(2, 4, 8)
    y1 = layer1(x)

    torch.manual_seed(123)
    layer2 = GAMLayer(d=8)
    y2 = layer2(x)

    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
