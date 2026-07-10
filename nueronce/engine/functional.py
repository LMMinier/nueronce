"""Functional ops built from the autograd primitives in :mod:`tensor`.

Because these compose ``Tensor`` operations, their gradients are produced by the
engine automatically — there is no hand-written backward here, which is itself a
test that the primitives are complete enough to express the architecture.
"""

from __future__ import annotations

import math
from typing import Optional

from .backend import xp as np

from .tensor import Tensor, cat

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def sigmoid(x: Tensor) -> Tensor:
    """Numerically stable sigmoid.

    The exp-based form can overflow for large negative float32 inputs. Its
    quotient backward then encounters indeterminate ``0 / inf`` expressions.
    The tanh identity is algebraically equivalent and remains finite across the
    practical float32 range used by NUERONCE gates.
    """
    return (x * 0.5).tanh() * 0.5 + 0.5


def silu(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def gelu(x: Tensor) -> Tensor:
    inner = (x + (x ** 3) * 0.044715) * SQRT_2_OVER_PI
    return x * (inner.tanh() + 1.0) * 0.5


def softplus(x: Tensor) -> Tensor:
    return (x.exp() + 1.0).log()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.data.max(axis=axis, keepdims=True)
    e = (x - Tensor(m)).exp()
    return e / e.sum(axis=axis, keepdims=True)


def masked_softmax(x: Tensor, mask: np.ndarray, axis: int = -1) -> Tensor:
    """Softmax with a boolean keep-mask.

    Fully masked rows return zeros. The denominator floor is deliberately
    ``1e-12`` rather than ``1e-30``: in float32 the square of ``1e-30``
    underflows to zero inside division backward and turns otherwise harmless
    zero gradients into ``0 / 0`` NaNs.
    """
    mask = np.asarray(mask, dtype=bool)
    add = np.where(mask, 0.0, -1e30)
    keep = Tensor(np.where(mask, 1.0, 0.0))
    e = (x + Tensor(add)).exp() * keep
    denom = e.sum(axis=axis, keepdims=True)
    return e / (denom + 1e-12)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    n = logits.shape[0]
    m = logits.data.max(axis=-1, keepdims=True)
    shifted = logits - Tensor(m)
    logsumexp = shifted.exp().sum(axis=-1, keepdims=True).log()
    logprobs = shifted - logsumexp
    picked = logprobs[np.arange(n), np.asarray(targets)]
    return -picked.mean()


def masked_cross_entropy(logits: Tensor, targets: np.ndarray, mask: np.ndarray) -> Tensor:
    idx = np.nonzero(np.asarray(mask).reshape(-1))[0]
    if idx.size == 0:
        return Tensor(0.0)
    return cross_entropy(logits[idx], np.asarray(targets).reshape(-1)[idx])


def pad_left(x: Tensor, pad: int, axis: int = -1) -> Tensor:
    if pad == 0:
        return x
    axis = axis % x.ndim
    shape = list(x.shape)
    shape[axis] = pad
    return cat([Tensor(np.zeros(shape)), x], axis=axis)


def conv1d_causal(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                  dilation: int = 1) -> Tensor:
    b, cin, t = x.shape
    cout, cin_w, k = weight.shape
    assert cin == cin_w, "in-channel mismatch"
    xpad = pad_left(x, (k - 1) * dilation, axis=2)
    out = None
    for i in range(k):
        xk = xpad[:, :, i * dilation: i * dilation + t]
        wk = weight[:, :, i]
        contrib = wk @ xk
        out = contrib if out is None else out + contrib
    if bias is not None:
        out = out + bias.reshape(1, cout, 1)
    return out


def binary_cross_entropy_with_logits(logits: Tensor, targets: np.ndarray) -> Tensor:
    p = sigmoid(logits)
    y = Tensor(np.asarray(targets, dtype=np.float64))
    eps = 1e-12
    loss = -(y * (p + eps).log() + (Tensor(1.0) - y) * (Tensor(1.0) - p + eps).log())
    return loss.mean()


def depthwise_conv1d_causal(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                            dilation: int = 1) -> Tensor:
    b, c, t = x.shape
    c_w, one, k = weight.shape
    assert c == c_w and one == 1, "depthwise conv expects weight [C, 1, K]"
    xpad = pad_left(x, (k - 1) * dilation, axis=2)
    out = None
    for i in range(k):
        xk = xpad[:, :, i * dilation: i * dilation + t]
        wk = weight[:, 0, i].reshape(1, c, 1)
        contrib = wk * xk
        out = contrib if out is None else out + contrib
    if bias is not None:
        out = out + bias.reshape(1, c, 1)
    return out


__all__ = ["sigmoid", "silu", "gelu", "softplus", "softmax", "masked_softmax",
           "cross_entropy", "masked_cross_entropy", "binary_cross_entropy_with_logits",
           "pad_left", "conv1d_causal", "depthwise_conv1d_causal"]
