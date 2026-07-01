"""Functional ops built from the autograd primitives in :mod:`tensor`.

Because these compose ``Tensor`` operations, their gradients are produced by the
engine automatically — there is no hand-written backward here, which is itself a
test that the primitives are complete enough to express the architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .tensor import Tensor, cat

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def sigmoid(x: Tensor) -> Tensor:
    return Tensor(1.0) / ((-x).exp() + 1.0)


def silu(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def gelu(x: Tensor) -> Tensor:
    inner = (x + (x ** 3) * 0.044715) * SQRT_2_OVER_PI
    return x * (inner.tanh() + 1.0) * 0.5


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.data.max(axis=axis, keepdims=True)          # detached shift for stability
    e = (x - Tensor(m)).exp()
    return e / e.sum(axis=axis, keepdims=True)


def masked_softmax(x: Tensor, mask: np.ndarray, axis: int = -1) -> Tensor:
    """Softmax with an additive boolean mask (True = keep). Fully-masked rows
    return zeros (matching the CFNA attention convention)."""
    add = np.where(mask, 0.0, -1e30)
    e = (x + Tensor(add)).exp() * Tensor(np.where(mask, 1.0, 0.0))
    denom = e.sum(axis=axis, keepdims=True)
    return e / (denom + 1e-30)   # fully-masked rows -> ~0 (denom ~0 -> e ~0)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy over a [N, C] logit matrix and [N] integer targets."""
    n = logits.shape[0]
    m = logits.data.max(axis=-1, keepdims=True)
    shifted = logits - Tensor(m)
    logsumexp = shifted.exp().sum(axis=-1, keepdims=True).log()
    logprobs = shifted - logsumexp
    picked = logprobs[np.arange(n), np.asarray(targets)]
    return -picked.mean()


def pad_left(x: Tensor, pad: int, axis: int = -1) -> Tensor:
    """Zero-pad on the left along ``axis`` (used for causal convolution)."""
    if pad == 0:
        return x
    axis = axis % x.ndim
    shape = list(x.shape)
    shape[axis] = pad
    return cat([Tensor(np.zeros(shape)), x], axis=axis)


def conv1d_causal(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                  dilation: int = 1) -> Tensor:
    """Causal 1-D convolution (groups=1).

    x: [B, Cin, T], weight: [Cout, Cin, K] -> [B, Cout, T]. Left-padded so output
    position t depends only on inputs <= t. Built from pad + slice + matmul, so
    autograd handles the backward pass.
    """
    b, cin, t = x.shape
    cout, cin_w, k = weight.shape
    assert cin == cin_w, "in-channel mismatch"
    xpad = pad_left(x, (k - 1) * dilation, axis=2)
    out = None
    for i in range(k):
        xk = xpad[:, :, i * dilation: i * dilation + t]   # [B, Cin, T]
        wk = weight[:, :, i]                              # [Cout, Cin]
        contrib = wk @ xk                                 # broadcast -> [B, Cout, T]
        out = contrib if out is None else out + contrib
    if bias is not None:
        out = out + bias.reshape(1, cout, 1)
    return out


__all__ = ["sigmoid", "silu", "gelu", "softmax", "masked_softmax",
           "cross_entropy", "pad_left", "conv1d_causal"]
