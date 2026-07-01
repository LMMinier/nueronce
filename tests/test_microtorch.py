"""Verification for the from-scratch autograd engine.

Two independent checks of correctness: finite-difference gradient checks (the
analytic backward matches numeric derivatives), and parity with PyTorch on the
same inputs. Plus optimizer behavior and an end-to-end training assertion.
"""

import numpy as np
import pytest

from cfna.microtorch import Tensor, functional as F, nn, optim
from cfna.microtorch.models import train_overfit


# --------------------------------------------------------------------------- #
# finite-difference gradient checks
# --------------------------------------------------------------------------- #

def _grad_check(fn, x, eps=1e-6):
    """fn: Tensor -> scalar Tensor. Returns max|analytic - numeric| for d/dx."""
    xt = Tensor(x, requires_grad=True)
    fn(xt).backward()
    analytic = xt.grad.copy()
    num = np.zeros_like(x, dtype=np.float64)
    flat = x.astype(np.float64).ravel()
    for i in range(x.size):
        up = flat.copy(); up[i] += eps
        dn = flat.copy(); dn[i] -= eps
        num.ravel()[i] = (fn(Tensor(up.reshape(x.shape))).item()
                          - fn(Tensor(dn.reshape(x.shape))).item()) / (2 * eps)
    return np.abs(num - analytic).max()


@pytest.mark.parametrize("fn", [
    lambda t: (t * t).sum(),
    lambda t: (t * 3.0 + 1.0).sum(),
    lambda t: (t / (t * t + 1.0)).sum(),
    lambda t: (t ** 3).sum(),
    lambda t: t.exp().sum(),
    lambda t: (t * t + 1.0).log().sum(),
    lambda t: t.tanh().sum(),
    lambda t: t.relu().sum(),
    lambda t: F.gelu(t).sum(),
    lambda t: F.silu(t).sum(),
    lambda t: F.sigmoid(t).sum(),
    lambda t: F.softmax(t, axis=-1).sum(),
    lambda t: (t.transpose() @ t).sum(),
    lambda t: t.reshape(-1).sum(),
    lambda t: t.mean(),
    lambda t: t[0].sum(),
])
def test_elementwise_and_reduction_grads(fn):
    x = np.random.randn(4, 4) * 0.7
    assert _grad_check(fn, x) < 1e-4


def test_matmul_and_broadcast_grads():
    a = np.random.randn(2, 3, 4)
    w = np.random.randn(4, 5)
    err = _grad_check(lambda t: (t @ Tensor(w)).sum(), a)
    assert err < 1e-4
    # broadcast add
    b = np.random.randn(5)
    err2 = _grad_check(lambda t: (Tensor(a) @ Tensor(w) + t).sum(), b)
    assert err2 < 1e-4


def test_cross_entropy_grad():
    x = np.random.randn(6, 9)
    tgt = np.array([0, 3, 8, 1, 5, 2])
    assert _grad_check(lambda t: F.cross_entropy(t, tgt), x) < 1e-4


def test_conv1d_causal_grad_and_causality():
    x = np.random.randn(2, 3, 7)
    w = np.random.randn(4, 3, 3)
    assert _grad_check(lambda t: F.conv1d_causal(t, Tensor(w)).sum(), x) < 1e-4
    # causal: output[..., t] must not depend on inputs > t
    xt = Tensor(x)
    base = F.conv1d_causal(xt, Tensor(w)).data
    x2 = x.copy(); x2[:, :, 5] += 3.0
    out = F.conv1d_causal(Tensor(x2), Tensor(w)).data
    assert np.abs(base[:, :, :5] - out[:, :, :5]).max() < 1e-9


# --------------------------------------------------------------------------- #
# parity with PyTorch
# --------------------------------------------------------------------------- #

def test_parity_with_pytorch_linear_softmax_ce():
    torch = pytest.importorskip("torch")
    np.random.seed(0)
    xd = np.random.randn(3, 5)
    wd = np.random.randn(4, 5)
    tgt = np.array([0, 2, 3])

    # microtorch
    x = Tensor(xd, requires_grad=True)
    w = Tensor(wd, requires_grad=True)
    loss = F.cross_entropy(x @ w.transpose(), tgt)
    loss.backward()

    # torch (float64 for clean parity)
    xt = torch.tensor(xd, requires_grad=True, dtype=torch.float64)
    wt = torch.tensor(wd, requires_grad=True, dtype=torch.float64)
    lt = torch.nn.functional.cross_entropy(xt @ wt.t(), torch.tensor(tgt))
    lt.backward()

    assert abs(loss.item() - lt.item()) < 1e-9
    assert np.abs(x.grad - xt.grad.numpy()).max() < 1e-8
    assert np.abs(w.grad - wt.grad.numpy()).max() < 1e-8


# --------------------------------------------------------------------------- #
# optimizers + end-to-end training
# --------------------------------------------------------------------------- #

def test_adamw_minimizes_quadratic():
    p = nn.Parameter(np.array([5.0, -3.0]))
    opt = optim.AdamW([p], lr=0.1)
    for _ in range(300):
        loss = ((p - Tensor([2.0, 1.0])) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    assert np.allclose(p.data, [2.0, 1.0], atol=1e-2)


def test_clip_grad_norm():
    p = nn.Parameter(np.array([3.0, 4.0]))
    p.grad = np.array([3.0, 4.0])  # norm 5
    total = optim.clip_grad_norm_([p], 1.0)
    assert abs(total - 5.0) < 1e-6
    assert abs(np.linalg.norm(p.grad) - 1.0) < 1e-6


def test_microtorch_model_overfits():
    curve = train_overfit(b"the quick brown fox jumps over the lazy dog.", steps=150, lr=5e-3)
    assert curve[-1] < 0.2 * curve[0]
    assert curve[-1] < 1.0
