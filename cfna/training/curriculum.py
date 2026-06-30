"""Curriculum scheduler and phase-dependent loss weighting (WPGCP).

The curriculum moves through phases: perception/patching → bidirectional semantic
reconstruction → retrieval/provenance/evidence → world/tool/planning → mixed
steady state. Domain mixture weights can be reweighted DoReMi-style from proxy
stats. The phase boundaries and the default loss-weight schedules are real; the
mixture optimizer is an injectable hook.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

# Phase step boundaries (training steps). Tunable defaults.
PHASE_BOUNDARIES = [
    (50_000, 0),    # perception / patching
    (200_000, 1),   # bidirectional semantic reconstruction
    (350_000, 2),   # retrieval / provenance / evidence
    (500_000, 3),   # world / tool / planning
]
STEADY_STATE_PHASE = 4


def sample_phase(step: int) -> int:
    for boundary, phase in PHASE_BOUNDARIES:
        if step < boundary:
            return phase
    return STEADY_STATE_PHASE


# Per-phase loss weights. Keys correspond to cfna.training.losses terms.
_ALL_TERMS = (
    "lang", "byte", "sem", "ent", "contra", "retr", "ground",
    "world", "temp", "route", "mem", "plan", "cal", "rev",
)


def phase_dependent_weights(phase: int) -> Dict[str, float]:
    """Return loss weights emphasizing the current curriculum phase's objectives."""
    base = {k: 0.1 for k in _ALL_TERMS}
    emphasis = {
        0: {"byte": 1.0, "route": 0.4},
        1: {"lang": 1.0, "sem": 0.7, "ent": 0.5},
        2: {"retr": 1.0, "ground": 1.0, "contra": 0.8, "temp": 0.6},
        3: {"world": 1.0, "plan": 0.8, "mem": 0.6, "route": 0.6},
        4: {k: 0.5 for k in _ALL_TERMS},  # balanced steady state
    }
    base.update(emphasis.get(phase, {}))
    return base


class CurriculumScheduler:
    def __init__(self, mixture_optimizer: Optional[Callable[[dict, dict], dict]] = None):
        self.phase = 0
        self.domain_weights: Dict[str, float] = {
            "books": 0.25,
            "papers": 0.20,
            "code": 0.20,
            "qa": 0.12,
            "tools": 0.10,
            "evidence": 0.08,
            "general": 0.05,
        }
        self._mixture_optimizer = mixture_optimizer

    def update_domain_weights(self, proxy_stats: dict) -> Dict[str, float]:
        if self._mixture_optimizer is None:
            return self.domain_weights
        self.domain_weights = self._mixture_optimizer(proxy_stats, self.domain_weights)
        return self.domain_weights

    def sample_phase(self, step: int) -> int:
        self.phase = sample_phase(step)
        return self.phase


__all__ = [
    "PHASE_BOUNDARIES",
    "STEADY_STATE_PHASE",
    "sample_phase",
    "phase_dependent_weights",
    "CurriculumScheduler",
]
