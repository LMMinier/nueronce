"""The hand-built engine primitives: shapes, backward, causality,
NaN-safety. Mirrors tests/test_nn_primitives.py (which needs real PyTorch)
one-for-one on the from-scratch engine, so this coverage exists even where
PyTorch isn't installed."""

import numpy as np

from nueronce.engine import functional as F
from nueronce.engine import nn as mnn
from nueronce.engine.tensor import Tensor, no_grad


def test_shapes_and_backward():
    B, T, D = 2, 10, 16
    x = Tensor(np.random.randn(B, T, D), requires_grad=True)
    for mod in [mnn.MultiHeadAttention(D, 4, window=4),
                mnn.SparseGlobalAttention(D, 4, topk=3),
                mnn.NUERONCESelectiveSSM(D, d_state=8),
                mnn.GatedMLP(D, 32),
                mnn.MLP(D, 32, D),
                mnn.RMSNorm(D)]:
        y = mod(x)
        assert y.shape == x.shape
        y.sum().backward()
    assert x.grad is not None


def test_local_attention_is_causal():
    np.random.seed(0)
    D = 16
    attn = mnn.MultiHeadAttention(D, 4, window=100)
    x = np.random.randn(1, 12, D)
    with no_grad():
        base = attn(Tensor(x))
        x2 = x.copy(); x2[0, 8] += 5.0   # perturb a future position
        out = attn(Tensor(x2))
    assert np.abs(base.data[0, :8] - out.data[0, :8]).max() < 1e-8  # earlier positions unchanged


def test_selective_ssm_is_causal():
    np.random.seed(0)
    D = 16
    ssm = mnn.NUERONCESelectiveSSM(D, d_state=8)
    x = np.random.randn(1, 14, D)
    with no_grad():
        base = ssm(Tensor(x))
        x2 = x.copy(); x2[0, 10] += 3.0
        out = ssm(Tensor(x2))
    assert np.abs(base.data[0, :10] - out.data[0, :10]).max() < 1e-6


def test_masked_softmax_handles_fully_masked_rows():
    scores = Tensor(np.random.randn(1, 1, 3, 4))
    mask = np.ones((1, 1, 3, 4), dtype=bool)
    mask[0, 0, 0] = False  # row 0 fully masked
    w = F.masked_softmax(scores, mask)
    assert np.isfinite(w.data).all()
    assert float(w.data[0, 0, 0].sum()) == 0.0          # masked row -> zeros, not NaN
    assert abs(float(w.data[0, 0, 1].sum()) - 1.0) < 1e-5
