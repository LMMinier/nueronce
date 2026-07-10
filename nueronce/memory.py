"""Typed multi-timescale recurrent memory + evidence-gated consolidation.

The memory substrate uses *typed channels* (semantic, structural, goal, evidence,
uncertainty, authority, procedure) with separate read / write / retention gates,
authority-controlled writes, and distinct timescales. Conceptually descended from
LSTM's protected cell state and gating, with efficiency motivation from selective
state-space models.

Per channel k:
    f_t^k = sigma(F_k([x_t, h_{t-1}, g_t, u_t]))            # forget
    w_t^k = sigma(W_k([x_t, h_{t-1}, m_t, e_t]))            # write
    r_t^k = sigma(R_k([x_t, c_t^k, g_t]))                   # read
    Δc_t^k = T_k([x_t, h_{t-1}, m_t, e_t])                  # candidate
    c_t^k = λ_k ⊙ f_t^k ⊙ c_{t-1}^k + a_t^k ⊙ w_t^k ⊙ Δc_t^k
    h_t^k = r_t^k ⊙ φ_k(c_t^k)
where a_t^k is an authority permission mask and λ_k is a retention timescale.

The cell step needs a backend (the gate MLPs). Consolidation scoring is pure and
fully implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import MemoryConfig
from .types import CHANNELS, MemoryRecord, Tensor


@dataclass
class TypedState:
    """Fused typed-memory state. ``cell``/``hidden`` are [B, K*Dk] tensors; the
    per-channel views are recoverable by reshaping to [B, K, Dk]."""

    cell: Tensor
    hidden: Tensor


class TypedRecurrentMemoryCell:
    """Real typed multi-timescale gated memory (delegates to the torch operator).

    Provides both a single-step API (``init_state`` / ``step``) realizing the
    design's per-channel forget/write/read/retention/authority equations, and a
    sequence API (``forward``) that returns a causal per-unit read-out summary.
    """

    def __init__(self, cfg: Optional[MemoryConfig] = None, d_model: Optional[int] = None):
        self.cfg = cfg or MemoryConfig()
        from .config import CoreConfig
        from .blocks import TypedRecurrentMemory

        self.d_model = d_model or CoreConfig().d_model
        self.module = TypedRecurrentMemory(self.d_model, channel_dim=self.cfg.channel_dim)

    def init_state(self, batch: int = 1) -> TypedState:
        import torch

        z = torch.zeros(batch, self.module.state_dim)
        return TypedState(cell=z.clone(), hidden=z.clone())

    def step(self, x_t, prev: TypedState, authority_mask=None) -> TypedState:
        """One fused gated update. ``x_t``: [B, d_model]; ``authority_mask``:
        optional [B, K] permission in [0,1]."""
        import torch

        m = self.module
        z = torch.cat([x_t, prev.hidden], dim=-1)
        f = torch.sigmoid(m.forget(z))
        w = torch.sigmoid(m.write(z))
        r = torch.sigmoid(m.read(z))
        dc = torch.tanh(m.cand(z))
        a = (authority_mask.repeat_interleave(m.dk, dim=-1)
             if authority_mask is not None else torch.ones_like(f))
        c = m.retention * f * prev.cell + a * w * dc
        h = r * torch.tanh(c)
        return TypedState(cell=c, hidden=h)

    def forward(self, units, authority=None):
        """Sequence read-out summary [B, P, d_model] (causal)."""
        return self.module(units, authority)


# --------------------------------------------------------------------------- #
# Evidence-gated consolidation (pure logic — fully implemented)
# --------------------------------------------------------------------------- #

def consolidation_score(
    corroboration: float,
    source_diversity: float,
    authority: float,
    temporal_stability: float,
    contradiction: float,
) -> float:
    """Weighted consolidation score (matches the design doc weights)."""
    return (
        0.25 * corroboration
        + 0.20 * source_diversity
        + 0.20 * authority
        + 0.20 * temporal_stability
        - 0.15 * contradiction
    )


# Routing thresholds for what a scored cluster becomes.
PROMOTE_THRESHOLD = 0.82
REVIEW_THRESHOLD = 0.60
MAX_CONTRADICTION_FOR_PROMOTE = 0.2


def consolidation_decision(score: float, contradiction: float) -> str:
    """Map a consolidation score + contradiction strength to an action.

    Returns one of: ``"semantic"``, ``"review_queue"``, ``"episodic_only"``.
    """
    if score >= PROMOTE_THRESHOLD and contradiction < MAX_CONTRADICTION_FOR_PROMOTE:
        return "semantic"
    if score >= REVIEW_THRESHOLD:
        return "review_queue"
    return "episodic_only"


@dataclass
class ConsolidationHooks:
    """Cluster-analysis callables the consolidator depends on."""

    cluster_equivalent_claims: object
    independent_support: object
    measure_source_diversity: object
    contradiction_strength: object
    authority_aggregate: object
    temporal_stability_score: object
    store_semantic_memory: object
    store_review_queue: object
    keep_episodic_only: object


class MemoryConsolidator:
    """Promotes corroborated, low-contradiction episodic clusters to semantic
    memory; routes borderline clusters to a review queue; keeps the rest episodic.
    """

    def __init__(self, hooks: Optional[ConsolidationHooks] = None):
        self.hooks = hooks

    def consolidate(self, episodic_candidates: List[MemoryRecord]) -> List[str]:
        h = self.hooks
        if h is None:
            raise NotImplementedError(
                "MemoryConsolidator needs ConsolidationHooks (clustering + cluster "
                "metric functions). The scoring/decision logic lives in "
                "consolidation_score / consolidation_decision."
            )
        decisions: List[str] = []
        for cluster in h.cluster_equivalent_claims(episodic_candidates):
            score = consolidation_score(
                corroboration=h.independent_support(cluster),
                source_diversity=h.measure_source_diversity(cluster),
                authority=h.authority_aggregate(cluster),
                temporal_stability=h.temporal_stability_score(cluster),
                contradiction=h.contradiction_strength(cluster),
            )
            decision = consolidation_decision(score, h.contradiction_strength(cluster))
            if decision == "semantic":
                h.store_semantic_memory(cluster)
            elif decision == "review_queue":
                h.store_review_queue(cluster)
            else:
                h.keep_episodic_only(cluster)
            decisions.append(decision)
        return decisions


__all__ = [
    "TypedState",
    "TypedRecurrentMemoryCell",
    "consolidation_score",
    "consolidation_decision",
    "PROMOTE_THRESHOLD",
    "REVIEW_THRESHOLD",
    "MAX_CONTRADICTION_FOR_PROMOTE",
    "ConsolidationHooks",
    "MemoryConsolidator",
]
