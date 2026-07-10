"""Differentiable-structure helpers for dynamic patching — NumPy port of
``nueronce.segment`` for :mod:`nueronce.engine.nueronce_blocks`.

The boundary *decisions* are discrete (non-differentiable) in the original too
(computed under ``@torch.no_grad()``), so this port is a direct, gradient-free
NumPy translation: segment ids, the mean-pool matrix, and the byte→unit causal
cross-attention mask. The pool matrix is later multiplied against a
gradient-carrying ``Tensor`` inside :mod:`nueronce.engine.nueronce_blocks`, exactly
as the constant ``m`` matrix is in the real, PyTorch-backed model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Bytes that tend to mark a boundary (whitespace / punctuation).
_SYNTAX = bytes(b" \t\n\r.,;:!?()[]{}<>\"'`/\\|=+-*&^%$#@~")


def syntax_table() -> np.ndarray:
    t = np.zeros(256, dtype=bool)
    for b in _SYNTAX:
        t[b] = True
    return t


def segment_ids_from_boundaries(
    boundary_prob: np.ndarray,  # [B, T] in [0,1]
    tau: float = 0.5,
    min_patch: int = 3,
    max_patch: int = 24,
    p_max: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy left-to-right segmentation honoring min/max patch length.

    Returns (seg_ids [B,T] int64 in [0, p_max-1], n_units [B] int64).
    """
    b, t = boundary_prob.shape
    seg = np.zeros((b, t), dtype=np.int64)
    cur = np.zeros(b, dtype=np.int64)
    length = np.ones(b, dtype=np.int64)
    ones = np.ones_like(length)
    for i in range(1, t):
        over_min = length >= min_patch
        over_max = length >= max_patch
        cut = ((boundary_prob[:, i] > tau) & over_min) | over_max
        cur = np.minimum(cur + cut.astype(np.int64), p_max - 1)
        seg[:, i] = cur
        length = np.where(cut, ones, length + 1)
    return seg, cur + 1


def pool_matrix(seg_ids: np.ndarray, p_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-pool matrix M [B, p_max, T] and unit_mask [B, p_max] (True=real)."""
    b, t = seg_ids.shape
    onehot = np.zeros((b, t, p_max), dtype=np.float64)
    bi, ti = np.meshgrid(np.arange(b), np.arange(t), indexing="ij")
    onehot[bi, ti, seg_ids] = 1.0
    counts = onehot.sum(axis=1)                            # [B, p_max]
    m = onehot.transpose(0, 2, 1) / np.clip(counts[..., None], 1.0, None)
    return m, counts > 0


def byte_to_unit_mask(seg_ids: np.ndarray, unit_mask: np.ndarray, p_max: int) -> np.ndarray:
    """Causal cross-attention mask: byte t may attend unit j iff j < seg_ids[t].

    Returns [B, T, p_max] boolean (True = attend).
    """
    units = np.arange(p_max)
    allowed = units[None, None, :] < seg_ids[..., None]     # [B,T,p_max]
    return allowed & unit_mask[:, None, :]


def boundary_targets(byte_ids: np.ndarray, syntax: np.ndarray) -> np.ndarray:
    """Self-supervised boundary labels: a boundary starts where a non-syntax
    byte follows a syntax byte (word-onset)."""
    is_syn = syntax[byte_ids]                                # [B,T]
    target = np.zeros_like(is_syn, dtype=np.float64)
    prev_syn = is_syn[:, :-1]
    cur_not_syn = ~is_syn[:, 1:]
    target[:, 1:] = (prev_syn & cur_not_syn).astype(np.float64)
    return target


__all__ = [
    "syntax_table",
    "segment_ids_from_boundaries",
    "pool_matrix",
    "byte_to_unit_mask",
    "boundary_targets",
]
