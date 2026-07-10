"""The hand-built primitives: shapes, backward, causality, NaN-safety."""

import math

import pytest

torch = pytest.importorskip("torch")

from nueronce import nn as cnn


def test_shapes_and_backward():
    B, T, D = 2, 10, 16
    x = torch.randn(B, T, D, requires_grad=True)
    for mod in [cnn.LocalAttention(D, 4, window=4),
                cnn.SparseGlobalAttention(D, 4, topk=3),
                cnn.SelectiveSSM(D, d_state=8),
                cnn.GatedMLP(D, 32),
                cnn.MLP(D, 32, D),
                cnn.RMSNorm(D)]:
        y = mod(x)
        assert y.shape == x.shape
        y.sum().backward(retain_graph=True)
    assert x.grad is not None


def test_local_attention_is_causal():
    torch.manual_seed(0)
    D = 16
    attn = cnn.LocalAttention(D, 4, window=100).eval()
    x = torch.randn(1, 12, D)
    with torch.no_grad():
        base = attn(x)
        x2 = x.clone(); x2[0, 8] += 5.0   # perturb a future position
        out = attn(x2)
    assert (base[0, :8] - out[0, :8]).abs().max() < 1e-5  # earlier positions unchanged


def test_selective_ssm_is_causal():
    torch.manual_seed(0)
    D = 16
    ssm = cnn.SelectiveSSM(D, d_state=8).eval()
    x = torch.randn(1, 14, D)
    with torch.no_grad():
        base = ssm(x)
        x2 = x.clone(); x2[0, 10] += 3.0
        out = ssm(x2)
    assert (base[0, :10] - out[0, :10]).abs().max() < 1e-4


def test_masked_softmax_handles_fully_masked_rows():
    scores = torch.randn(1, 1, 3, 4)
    mask = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    mask[0, 0, 0] = False  # row 0 fully masked
    w = cnn.masked_softmax(scores, mask)
    assert torch.isfinite(w).all()
    assert float(w[0, 0, 0].sum()) == 0.0          # masked row -> zeros, not NaN
    assert abs(float(w[0, 0, 1].sum()) - 1.0) < 1e-5
