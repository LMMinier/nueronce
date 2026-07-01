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


def softplus(x: Tensor) -> Tensor:
    """log(1 + exp(x)). Composed from existing differentiable ops (exp/+/log),
    so autograd gets the exact sigmoid(x) gradient for free — no stability
    shift is applied since it would have to be a constant split out of the
    max(x,0) term, which is itself part of the function's gradient."""
    return (x.exp() + 1.0).log()


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


def masked_cross_entropy(logits: Tensor, targets: np.ndarray, mask: np.ndarray) -> Tensor:
    """Cross-entropy over a [N, C] logit matrix and [N] targets, restricted to
    the rows where ``mask`` is True. Used for SFT-style losses that should
    only be charged on certain positions (e.g. the response half of a
    dialogue turn, not the user's turn)."""
    idx = np.nonzero(np.asarray(mask).reshape(-1))[0]
    if idx.size == 0:
        return Tensor(0.0)
    return cross_entropy(logits[idx], np.asarray(targets).reshape(-1)[idx])


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


def binary_cross_entropy_with_logits(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean BCE over arbitrary-shaped logits and same-shaped 0/1 targets,
    composed from ``sigmoid``/``log`` so autograd handles the gradient (no
    stability shift, for the same reason as :func:`softplus`: this is a toy-
    scale model, and a numerically-stable rewrite would have to detach part
    of the function's own gradient)."""
    p = sigmoid(logits)
    y = Tensor(np.asarray(targets, dtype=np.float64))
    eps = 1e-12
    loss = -(y * (p + eps).log() + (Tensor(1.0) - y) * (Tensor(1.0) - p + eps).log())
    return loss.mean()


def depthwise_conv1d_causal(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                            dilation: int = 1) -> Tensor:
    """Causal depthwise ("groups=channels") 1-D convolution: each output
    channel is convolved only with its own input channel — no cross-channel
    mixing. x: [B, C, T], weight: [C, 1, K] -> [B, C, T]. Used for the
    selective-SSM's per-channel smoothing conv (matches PyTorch's
    ``F.conv1d(..., groups=C)``, built from elementwise ops instead)."""
    b, c, t = x.shape
    c_w, one, k = weight.shape
    assert c == c_w and one == 1, "depthwise conv expects weight [C, 1, K]"
    xpad = pad_left(x, (k - 1) * dilation, axis=2)
    out = None
    for i in range(k):
        xk = xpad[:, :, i * dilation: i * dilation + t]     # [B, C, T]
        wk = weight[:, 0, i].reshape(1, c, 1)                # [1, C, 1], broadcasts over B, T
        contrib = wk * xk
        out = contrib if out is None else out + contrib
    if bias is not None:
        out = out + bias.reshape(1, c, 1)
    return out


__all__ = ["sigmoid", "silu", "gelu", "softplus", "softmax", "masked_softmax",
           "cross_entropy", "masked_cross_entropy", "binary_cross_entropy_with_logits",
           "pad_left", "conv1d_causal", "depthwise_conv1d_causal"]
