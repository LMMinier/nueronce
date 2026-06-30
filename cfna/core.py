"""Hybrid cognitive fabric: the shared core, reused recurrently.

Real implementation. Each block merges a selective state-space path, local
attention, sparse global attention, and (optional) retrieval injection via a
*per-position* adaptive router, then a gated FFN with residuals. A few physical
blocks are reused over mode-dependent logical depth to cut memory pressure.

The actual torch modules live in :mod:`cfna.blocks` (built from the hand-rolled
primitives in :mod:`cfna.nn`); this module exposes the design-named classes and
the mode→depth control flow on top of them.
"""

from __future__ import annotations

from typing import List, Optional

from .config import CoreConfig


class HybridBlock:
    """Design-named wrapper around :class:`cfna.blocks.HybridBlock`."""

    def __init__(self, cfg: Optional[CoreConfig] = None):
        self.cfg = cfg or CoreConfig()
        from .blocks import HybridBlock as _HybridBlock

        self.module = _HybridBlock(
            self.cfg.d_model,
            n_heads=self.cfg.n_local_heads,
            local_window=self.cfg.local_window,
            sparse_topk=self.cfg.sparse_global_topk,
            ffn_mult=self.cfg.ffn_mult,
        )

    def forward(self, x, key_padding=None, retrieval_ctx=None, retrieval_mask=None, importance=None):
        return self.module(x, key_padding, retrieval_ctx, retrieval_mask, importance)


def convergence_detected(x, step: int, mode: str) -> bool:
    """Hook for early-exit of the logical recurrence. Default: never early-exit."""
    return False


class CFNACore:
    """Drives a few physical blocks over mode-dependent logical depth.

    Wraps :class:`cfna.blocks.HybridCoreStack`; ``run`` keeps the design's
    block-reuse loop and accepts a string ``mode`` (FAST/DELIBERATE/RESEARCH).
    """

    def __init__(self, cfg: Optional[CoreConfig] = None, physical_blocks: Optional[int] = None):
        self.cfg = cfg or CoreConfig()
        from .blocks import HybridCoreStack

        self.stack = HybridCoreStack(
            self.cfg.d_model,
            physical_blocks=physical_blocks or self.cfg.physical_blocks,
            n_heads=self.cfg.n_local_heads,
            local_window=self.cfg.local_window,
            sparse_topk=self.cfg.sparse_global_topk,
            ffn_mult=self.cfg.ffn_mult,
        )

    def depth_for(self, mode: str) -> int:
        return self.cfg.logical_depth[mode]

    def run(self, x, mode: str = "DELIBERATE", key_padding=None, retrieval_ctx=None,
            retrieval_mask=None, importance=None):
        return self.stack(
            x, self.depth_for(mode), key_padding=key_padding,
            retrieval_ctx=retrieval_ctx, retrieval_mask=retrieval_mask, importance=importance,
        )

    @property
    def blocks(self) -> List:
        return list(self.stack.blocks)


__all__ = ["HybridBlock", "convergence_detected", "CFNACore"]
