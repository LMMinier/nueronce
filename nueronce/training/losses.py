"""Multi-objective WPGCP loss.

The notes reject single-objective next-token pretraining in favor of a weighted
multi-objective loss covering language, byte reconstruction, semantic equivalence,
entailment, contradiction, retrieval, grounding, world prediction, temporal order,
routing, memory decisions, planning, calibration, and revision (H7).

The aggregation (weighted sum over whichever terms are present) is implemented and
testable with plain numbers. The individual term functions are backend objectives
and are injected as a registry of callables.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

# Canonical loss-term order; mirrors curriculum.phase_dependent_weights keys.
LOSS_TERMS = (
    "lang", "byte", "sem", "ent", "contra", "retr", "ground",
    "world", "temp", "route", "mem", "plan", "cal", "rev",
)


def aggregate_losses(losses: Dict[str, float], lambdas: Dict[str, float]) -> float:
    """Weighted sum over present, non-None terms.

    total = sum(lambdas[k] * losses[k] for present k)
    """
    total = 0.0
    for k, v in losses.items():
        if v is None:
            continue
        total += float(lambdas.get(k, 0.0)) * float(v)
    return total


class LossRegistry:
    """Holds the per-term loss callables. Each maps (model_out, targets) -> scalar.

    A term is skipped when its callable is missing or its target is absent, so the
    same registry works across curriculum phases that activate different terms.
    """

    def __init__(self, terms: Optional[Dict[str, Callable]] = None):
        self.terms: Dict[str, Callable] = dict(terms or {})

    def register(self, name: str, fn: Callable) -> None:
        self.terms[name] = fn

    def compute(self, model_out: dict, targets: dict, lambdas: Dict[str, float]):
        losses: Dict[str, float] = {}
        for name, fn in self.terms.items():
            target = targets.get(name)
            if target is None:
                continue
            losses[name] = fn(model_out, target)
        total = aggregate_losses(losses, lambdas)
        return total, losses


__all__ = ["LOSS_TERMS", "aggregate_losses", "LossRegistry"]
