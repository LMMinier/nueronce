"""Multiscale perception and dynamic information-unit formation.

This is a core departure from standard LLM stacks: fixed tokenization is replaced
with *dynamic information units* derived from raw bytes, local predictability,
entropy, semantic shift, and syntax boundaries (cf. the Byte Latent Transformer).

- :class:`ByteCharPerception` is the learned local encoder (CNN-style). Its
  ``forward`` needs a neural backend.
- :func:`dynamic_patching` is fully implemented: given per-byte local features
  and boundary logits, it segments a byte stream into variable-length patches.
- :func:`encode_information_units` builds per-patch representations; the exact
  (lexical) and pooled-local parts are real, the contextual encoder is injected.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PerceptionConfig
from .ops import SYNTAX_BYTES, hash_ngram_features, l2, mean_pool, norm, sigmoid


class ByteCharPerception:
    """Byte embedding + stacked causal (dilated) 1D convolutions + a boundary head.

    forward(input_bytes: long[B, T]) -> (local_feats[B, T, d_local],
                                         boundary_logits[B, T])

    Thin adapter over the real torch operator in :mod:`cfna.blocks`. torch is
    imported lazily so the pure-logic functions in this module (``dynamic_patching``,
    ``encode_information_units``) stay importable without a backend.
    """

    def __init__(self, cfg: Optional[PerceptionConfig] = None):
        self.cfg = cfg or PerceptionConfig()
        from .blocks import BytePerceptionEncoder

        self.module = BytePerceptionEncoder(self.cfg.byte_embed_dim, self.cfg.d_local)

    def forward(self, input_bytes):
        return self.module(input_bytes)

    def __call__(self, input_bytes):
        return self.module(input_bytes)


def dynamic_patching(
    bytes_: Sequence[int],
    local_feats,
    boundary_logits,
    cfg: Optional[PerceptionConfig] = None,
) -> List[Tuple[int, int]]:
    """Segment a byte stream into variable-length patches.

    The boundary decision combines a learned boundary logit, a local
    semantic-shift estimate (L2 delta of adjacent local features), and a syntax
    spike (punctuation / brackets / whitespace). A boundary is cut when the score
    exceeds ``boundary_tau`` and the patch has reached ``min_patch``, or when the
    patch reaches ``max_patch``.
    """
    cfg = cfg or PerceptionConfig()
    feats = np.asarray(local_feats, dtype=np.float64)
    logits = np.asarray(boundary_logits, dtype=np.float64)
    T = len(bytes_)
    if T == 0:
        return []

    patches: List[Tuple[int, int]] = []
    start = 0
    for t in range(1, T):
        semantic_shift = l2(feats[t] - feats[t - 1]) if feats.ndim == 2 else 0.0
        syntax_spike = 1.0 if bytes_[t] in SYNTAX_BYTES else 0.0
        learned = float(sigmoid(logits[t])) if logits.size else 0.0
        score = (
            cfg.w_learned * learned
            + cfg.w_shift * norm(semantic_shift)
            + cfg.w_syntax * syntax_spike
        )
        patch_len = t - start
        if (score > cfg.boundary_tau and patch_len >= cfg.min_patch) or patch_len >= cfg.max_patch:
            patches.append((start, t))
            start = t
    if start < T:
        patches.append((start, T))
    return patches


# Contextual encoder maps a patch's local-feature slice -> a fixed contextual vec.
PatchContextEncoder = Callable[[np.ndarray], np.ndarray]


def encode_information_units(
    bytes_: Sequence[int],
    patch_spans: List[Tuple[int, int]],
    local_feats,
    patch_context_encoder: Optional[PatchContextEncoder] = None,
) -> List[Dict]:
    """Build per-patch unit dicts: exact (sparse lexical), local (pooled), and
    contextual representations.

    ``exact`` and ``local`` are fully computed. ``contextual`` uses the injected
    encoder when present; otherwise it falls back to a mean-pool so the pipeline
    stays runnable end-to-end for testing.
    """
    feats = np.asarray(local_feats, dtype=np.float64)
    raw = bytes(int(b) & 0xFF for b in bytes_)
    units: List[Dict] = []
    for s, e in patch_spans:
        patch_bytes = raw[s:e]
        local_repr = mean_pool(feats[s:e], axis=0) if feats.size else np.zeros(0)
        if patch_context_encoder is not None:
            contextual_repr = patch_context_encoder(feats[s:e])
        else:
            contextual_repr = local_repr  # runnable fallback
        units.append(
            {
                "span": (s, e),
                "exact": hash_ngram_features(patch_bytes),
                "local": local_repr,
                "contextual": contextual_repr,
            }
        )
    return units


__all__ = [
    "ByteCharPerception",
    "dynamic_patching",
    "PatchContextEncoder",
    "encode_information_units",
]
