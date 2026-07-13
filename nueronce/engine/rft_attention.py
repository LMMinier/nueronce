"""Experimental phi-grid rotary attention for the Nueronce Engine.

This module applies a norm-preserving rotary transform to Q and K using
frequencies f_k = frac((k + 1) * phi). It does not transform model weights,
does not compress gradients, and adds no trainable parameters.

The classes mirror the engine's existing self-attention modules so a launcher
can install them as an opt-in ablation without changing checkpoint shapes.
"""
from __future__ import annotations

import math
from typing import Optional

from .backend import xp as np
from . import functional as F
from .nn import MultiHeadAttention, SparseGlobalAttention, _causal_window_mask
from .tensor import Tensor, cat, stack

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def phi_rotary(x: Tensor) -> Tensor:
    """Rotate adjacent feature pairs with golden-ratio-spaced frequencies.

    ``x`` is shaped ``[batch, heads, time, head_dim]``. Each adjacent pair is
    rotated independently; an odd final feature is passed through unchanged.
    The operation is position-local and cannot mix future information backward.
    """
    b, h, t, hd = x.shape
    pairs = hd // 2
    if pairs == 0 or t == 0:
        return x

    positions = np.arange(t, dtype=np.float64)
    freqs = (np.arange(1, pairs + 1, dtype=np.float64) * PHI) % 1.0
    phases = 2.0 * math.pi * np.outer(positions, freqs)
    cos = Tensor(np.cos(phases)[None, None, :, :])
    sin = Tensor(np.sin(phases)[None, None, :, :])

    even = x[:, :, :, : 2 * pairs : 2]
    odd = x[:, :, :, 1 : 2 * pairs : 2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    # ``Tensor.stack`` does not normalize negative axes.  The input rank here
    # is four, so axis=4 explicitly appends the pair lane and guarantees the
    # output order [rot_even_0, rot_odd_0, rot_even_1, rot_odd_1, ...].
    rotated = stack([rot_even, rot_odd], axis=4).reshape(b, h, t, 2 * pairs)

    if hd % 2:
        rotated = cat([rotated, x[:, :, :, -1:]], axis=-1)
    return rotated


class PhiRotaryMultiHeadAttention(MultiHeadAttention):
    """Causal self-attention with phi-grid rotary Q/K geometry."""

    def forward(self, x: Tensor, key_padding: Optional[np.ndarray] = None) -> Tensor:
        b, t, d = x.shape
        q = phi_rotary(self._split(self.q(x)))
        k = phi_rotary(self._split(self.k(x)))
        v = self._split(self.v(x))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.hd))
        mask = _causal_window_mask(t, self.window)[None, None]
        if key_padding is not None:
            mask = mask & np.asarray(key_padding)[:, None, None, :]
        attn = F.masked_softmax(scores, mask, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.o(out)


class PhiRotarySparseGlobalAttention(SparseGlobalAttention):
    """Sparse causal global attention with phi-grid rotary Q/K geometry."""

    def forward(self, x: Tensor, importance: Optional[Tensor] = None,
                key_padding: Optional[np.ndarray] = None) -> Tensor:
        b, t, d = x.shape
        q = phi_rotary(self._split(self.q(x)))
        k = phi_rotary(self._split(self.k(x)))
        v = self._split(self.v(x))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.hd))

        mask = np.broadcast_to(
            _causal_window_mask(t, None)[None, None], (b, 1, t, t)
        ).copy()
        if key_padding is not None:
            mask = mask & np.asarray(key_padding)[:, None, None, :]
        if importance is not None:
            imp = importance if isinstance(importance, Tensor) else Tensor(importance)
            scores = scores + imp.reshape(b, 1, 1, t)

        if self.topk < t:
            masked = np.where(mask, scores.data, -1e30)
            kth = np.sort(masked, axis=-1)[..., -self.topk:][..., :1]
            mask = mask & (masked >= kth)

        attn = F.masked_softmax(scores, mask, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.o(out)


def install_phi_rotary_attention() -> None:
    """Install the experimental classes for subsequently constructed models.

    ``HybridBlock`` and decoder layers resolve their attention classes from
    ``nueronce.engine.nueronce_blocks`` when instances are constructed, so this
    opt-in patch preserves all existing checkpoints and the baseline path.
    """
    from . import nueronce_blocks

    nueronce_blocks.MultiHeadAttention = PhiRotaryMultiHeadAttention
    nueronce_blocks.SparseGlobalAttention = PhiRotarySparseGlobalAttention


__all__ = [
    "PHI",
    "phi_rotary",
    "PhiRotaryMultiHeadAttention",
    "PhiRotarySparseGlobalAttention",
    "install_phi_rotary_attention",
]
