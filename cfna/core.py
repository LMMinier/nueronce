"""Hybrid cognitive fabric: the shared core that is reused recurrently.

Each :class:`HybridBlock` merges a selective state-space / recurrent path, local
attention, sparse global attention, and a retrieval-injection path via an
adaptive (softmax-weighted) router, then a gated FFN with residuals. A small
number of *physical* blocks are reused over a larger number of *logical* steps
to cut memory pressure — one of the main goals of the 350M prototype.

These are learned modules; their forward passes need a neural backend. The
mode→depth control flow in :class:`CFNACore.run` is real.
"""

from __future__ import annotations

from typing import List, Optional

from ._backend import needs_backend
from .config import CoreConfig
from .memory import TypedState


class HybridBlock:
    """Shared physical block reused recurrently.

    forward(x[B, N, d_model], state, retrieval_ctx, importance_mask)
        -> (x[B, N, d_model], state)
    """

    def __init__(self, cfg: Optional[CoreConfig] = None):
        self.cfg = cfg or CoreConfig()

    def forward(self, x, state: TypedState, retrieval_ctx, importance_mask):
        raise needs_backend(
            "HybridBlock.forward",
            "Paths: selective-SSM, local attention, sparse-global attention "
            "(on important positions), retrieval cross-integration; merged by a "
            "softmax router, then gated FFN + residuals.",
        )


def convergence_detected(x, step: int, mode: str) -> bool:
    """Hook for early-exit of the logical recurrence. Default: never early-exit.

    A real implementation compares successive states (e.g. small relative delta)
    and may be mode-dependent. Kept conservative so depth == configured depth.
    """
    return False


class CFNACore:
    """Drives a few physical blocks over mode-dependent logical depth."""

    def __init__(self, blocks: List[HybridBlock], cfg: Optional[CoreConfig] = None):
        if not blocks:
            raise ValueError("CFNACore requires at least one HybridBlock")
        self.blocks = blocks
        self.cfg = cfg or CoreConfig()

    def depth_for(self, mode: str) -> int:
        return self.cfg.logical_depth[mode]

    def run(self, x, state: TypedState, retrieval_ctx, importance_mask, mode: str):
        depth = self.depth_for(mode)
        for t in range(depth):
            block = self.blocks[t % len(self.blocks)]
            x, state = block.forward(x, state, retrieval_ctx, importance_mask)
            if convergence_detected(x, t, mode):
                break
        return x, state


__all__ = ["HybridBlock", "convergence_detected", "CFNACore"]
