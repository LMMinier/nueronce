"""Entropy-driven adaptive byte patching (BLT-style) for NUERONCE.

STATUS: additive and opt-in. This module is **not** imported by
``nueronce.model``/``nueronce.segment`` and does **not** touch ``ModelConfig``,
``NUERONCEModel``, or any preset. Activating entropy patching changes the
boundary-head training target, which changes the model and breaks checkpoint
resume, so it is deliberately kept out of the default path until the in-flight
``base_35m`` run establishes a clean syntactic-patching baseline. The exact
post-run integration diff and the A/B protocol live in
``docs/ADAPTIVE_PATCHING_SPEC.md``.

Idea (Byte Latent Transformer, Meta 2024): allocate compute where information
is dense. A cheap byte-level predictor yields a per-position next-byte entropy
H_t; a new unit boundary is placed where the model is *surprised* (high or
rising entropy) and predictable runs are glued into long patches. This is the
informational replacement for the fixed syntactic target in
:func:`nueronce.segment.boundary_targets` ("boundary = word onset after a
space/punctuation byte").

Index convention matches the rest of the model: ``logits[:, t]`` predicts byte
``t+1``, and ``target[:, i] == 1`` means "byte ``i`` starts a new unit" — the
same semantics ``nueronce.segment.boundary_targets`` and
``segment_ids_from_boundaries`` already use, so this drops in without touching
the greedy scan.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .nn import Embedding, Linear, RMSNorm

LN_256 = math.log(256.0)  # max possible next-byte entropy in nats


class _CausalConv1d(nn.Module):
    """Left-padded depth-preserving causal conv (hand-parameterized, matching
    the style of nueronce.blocks so nothing stock is introduced)."""

    def __init__(self, c_in: int, c_out: int, kernel: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c_out, c_in, kernel) * (1.0 / (c_in * kernel) ** 0.5))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.pad = kernel - 1

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, T]
        return F.conv1d(F.pad(x, (self.pad, 0)), self.weight, self.bias)


class ByteEntropyHead(nn.Module):
    """A small, cheap causal byte LM whose only job is to produce a per-position
    next-byte distribution (and hence an entropy). This is the BLT "entropy
    model": deliberately tiny relative to the main model, trained with plain
    next-byte cross-entropy. It never shares weights with the main model, so it
    can be trained once and frozen, or trained jointly — the spec covers both.
    """

    def __init__(self, dim: int = 64, kernel: int = 5, layers: int = 2):
        super().__init__()
        self.embed = Embedding(256, dim)
        self.convs = nn.ModuleList([_CausalConv1d(dim, dim, kernel) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.head = Linear(dim, 256)

    def forward(self, byte_ids: Tensor) -> Tensor:
        x = self.embed(byte_ids).transpose(1, 2)          # [B, dim, T]
        for conv in self.convs:
            x = x + F.gelu(conv(x))
        x = self.norm(x.transpose(1, 2))                  # [B, T, dim]
        return self.head(x)                               # [B, T, 256]

    def lm_loss(self, byte_ids: Tensor) -> Tensor:
        logits = self.forward(byte_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1))


def predictive_entropy(logits: Tensor) -> Tensor:
    """Per-position Shannon entropy (nats) of the next-byte distribution.

    ``logits``: [B, T, 256], where ``logits[:, t]`` predicts byte ``t+1``.
    Returns H [B, T] with ``H[:, t]`` = entropy of the distribution that
    predicts byte ``t+1``. Computed in fp32 for stability under AMP.
    """
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(dim=-1)                    # [B, T]


def entropy_boundary_targets(
    logits: Tensor,
    *,
    mode: str = "global",
    global_theta: float = 2.5,
    relative_theta: float = 0.2,
) -> Tensor:
    """Informational boundary targets, drop-in for segment.boundary_targets.

    A boundary at byte ``i`` means "byte ``i`` starts a new unit". Byte ``i``'s
    surprise is the entropy of the distribution that predicted it, i.e.
    ``H[:, i-1]`` (``logits[:, i-1]`` predicts byte ``i``).

    - ``mode="global"``   : boundary where surprise exceeds ``global_theta``.
    - ``mode="relative"`` : boundary where surprise *rises* by more than
      ``relative_theta`` vs. the previous byte (BLT "approximate monotonic";
      catches the *onset* of surprise, which aligns with word/morpheme starts).

    Thresholds are in nats (max entropy = ln 256 ≈ 5.545). ``target[:, 0]`` is
    always 0 by convention (byte 0 opens unit 0; the greedy scan never cuts at
    position 0).
    """
    if mode not in {"global", "relative"}:
        raise ValueError(f"unknown mode: {mode!r}")
    H = predictive_entropy(logits)                        # [B, T]
    b, t = H.shape
    target = torch.zeros_like(H)
    if t < 2:
        return target
    surprise = H[:, :-1]                                  # surprise[:, i-1] -> byte i, for i in 1..T-1
    if mode == "global":
        fires = surprise > global_theta                  # [B, T-1]
        target[:, 1:] = fires.float()
    else:  # relative
        if t < 3:
            return target
        rise = surprise[:, 1:] - surprise[:, :-1]         # H[i-1]-H[i-2] -> byte i, for i in 2..T-1
        target[:, 2:] = (rise > relative_theta).float()
    return target


@torch.no_grad()
def patch_stats(seg_ids: Tensor, byte_mask: Optional[Tensor] = None) -> dict:
    """Compute the efficiency numbers the thesis is measured on.

    ``seg_ids``: [B, T] long unit-id per byte (from
    ``segment_ids_from_boundaries``). ``byte_mask``: optional [B, T] bool of
    real (non-pad) bytes. Returns mean bytes-per-unit and units-per-byte across
    the batch — higher bytes-per-unit on predictable text = more compute saved.
    """
    b, t = seg_ids.shape
    if byte_mask is None:
        byte_mask = torch.ones_like(seg_ids, dtype=torch.bool)
    n_bytes = byte_mask.sum(dim=1).clamp_min(1)           # [B]
    # units used per example = distinct seg ids over valid bytes = max+1
    seg_valid = seg_ids.masked_fill(~byte_mask, -1)
    n_units = (seg_valid.max(dim=1).values + 1).clamp_min(1)  # [B]
    bytes_per_unit = (n_bytes.float() / n_units.float())
    units_per_byte = (n_units.float() / n_bytes.float())
    return {
        "bytes_per_unit": float(bytes_per_unit.mean()),
        "units_per_byte": float(units_per_byte.mean()),
        "mean_units": float(n_units.float().mean()),
        "mean_bytes": float(n_bytes.float().mean()),
    }


__all__ = [
    "ByteEntropyHead",
    "predictive_entropy",
    "entropy_boundary_targets",
    "patch_stats",
    "LN_256",
]
