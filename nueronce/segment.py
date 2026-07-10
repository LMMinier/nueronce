"""Differentiable-structure helpers for dynamic patching in the trainable model.

The boundary *decisions* are discrete (non-differentiable), so they are taken
under ``no_grad``; the boundary *head* is trained via an auxiliary boundary loss.
These helpers turn per-byte boundary probabilities into segment ids, a mean-pool
matrix, and the byte→unit causal cross-attention mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Bytes that tend to mark a boundary (whitespace / punctuation).
_SYNTAX = bytes(b" \t\n\r.,;:!?()[]{}<>\"'`/\\|=+-*&^%$#@~")


def syntax_table(device=None) -> Tensor:
    t = torch.zeros(256, dtype=torch.bool, device=device)
    for b in _SYNTAX:
        t[b] = True
    return t


@torch.no_grad()
def segment_ids_from_boundaries(
    boundary_prob: Tensor,  # [B, T] in [0,1]
    tau: float = 0.5,
    min_patch: int = 3,
    max_patch: int = 24,
    p_max: int = 32,
):
    """Greedy left-to-right segmentation honoring min/max patch length.

    Returns (seg_ids [B,T] long in [0, p_max-1], n_units [B] long).
    """
    b, t = boundary_prob.shape
    device = boundary_prob.device
    seg = torch.zeros(b, t, dtype=torch.long, device=device)
    cur = torch.zeros(b, dtype=torch.long, device=device)
    length = torch.ones(b, dtype=torch.long, device=device)
    ones = torch.ones_like(length)
    for i in range(1, t):
        over_min = length >= min_patch
        over_max = length >= max_patch
        cut = ((boundary_prob[:, i] > tau) & over_min) | over_max
        cur = torch.clamp(cur + cut.long(), max=p_max - 1)
        seg[:, i] = cur
        length = torch.where(cut, ones, length + 1)
    return seg, cur + 1


def pool_matrix(seg_ids: Tensor, p_max: int):
    """Mean-pool matrix M [B, p_max, T] and unit_mask [B, p_max] (True=real)."""
    onehot = F.one_hot(seg_ids, p_max).float()      # [B, T, p_max]
    counts = onehot.sum(dim=1)                       # [B, p_max]
    m = onehot.transpose(1, 2) / counts[..., None].clamp_min(1.0)
    return m, counts > 0


def byte_to_unit_mask(seg_ids: Tensor, unit_mask: Tensor, p_max: int) -> Tensor:
    """Causal cross-attention mask: byte t may attend unit j iff j < seg_ids[t].

    Returns [B, T, p_max] boolean (True = attend). Bytes in segment 0 attend to
    no units (fully masked rows are handled NaN-safely downstream).
    """
    units = torch.arange(p_max, device=seg_ids.device)
    allowed = units[None, None, :] < seg_ids[..., None]          # [B,T,p_max]
    return allowed & unit_mask[:, None, :]


def boundary_targets(byte_ids: Tensor, syntax: Tensor) -> Tensor:
    """Self-supervised boundary labels: a boundary starts where a non-syntax byte
    follows a syntax byte (word-onset), giving the boundary head a real signal.
    """
    is_syn = syntax[byte_ids]                                    # [B,T]
    target = torch.zeros_like(is_syn, dtype=torch.float)
    prev_syn = is_syn[:, :-1]
    cur_not_syn = ~is_syn[:, 1:]
    target[:, 1:] = (prev_syn & cur_not_syn).float()
    return target


__all__ = [
    "syntax_table",
    "segment_ids_from_boundaries",
    "pool_matrix",
    "byte_to_unit_mask",
    "boundary_targets",
]
