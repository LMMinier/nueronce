"""CFNA training pipelines.

Two phases, keeping the source notes' architecturally meaningful names:

- **WPGCP** — Web-Scale Provenance-Grounded Curriculum Pretraining
  (multi-objective, multi-phase curriculum; see :mod:`cfna.training.curriculum`,
  :mod:`cfna.training.episodes`, :mod:`cfna.training.losses`).
- **VGRFT** — Verifier-Guided Residual Fine-Tuning (structured instruction
  tuning, tool grounding, verifier training, residual correction experts; see
  :mod:`cfna.training.vgrft`). Stage 1 (SFT) has two interchangeable backends:
  :mod:`cfna.training.sft` (PyTorch, fine-tunes ``CFNAModel``) and
  :mod:`cfna.microtorch.models` (from-scratch, NumPy-only). Both share the
  turn data/encoding in :mod:`cfna.training.dialogue_data`.
"""

from __future__ import annotations

import importlib

from . import curriculum, dialogue_data, episodes, losses, vgrft

__all__ = ["curriculum", "dialogue_data", "episodes", "losses", "vgrft", "sft"]


def __getattr__(name):
    # cfna.training.sft imports torch (it drives real backprop), so it stays
    # lazy — the same reason cfna/__init__.py defers CFNAModel.
    if name == "sft":
        return importlib.import_module(".sft", __name__)
    raise AttributeError(f"module 'cfna.training' has no attribute {name!r}")
