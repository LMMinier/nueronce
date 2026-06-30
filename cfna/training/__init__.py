"""CFNA training pipelines.

Two phases, keeping the source notes' architecturally meaningful names:

- **WPGCP** — Web-Scale Provenance-Grounded Curriculum Pretraining
  (multi-objective, multi-phase curriculum; see :mod:`cfna.training.curriculum`,
  :mod:`cfna.training.episodes`, :mod:`cfna.training.losses`).
- **VGRFT** — Verifier-Guided Residual Fine-Tuning (structured instruction
  tuning, tool grounding, verifier training, residual correction experts; see
  :mod:`cfna.training.vgrft`).
"""

from __future__ import annotations

from . import curriculum, episodes, losses, vgrft

__all__ = ["curriculum", "episodes", "losses", "vgrft"]
