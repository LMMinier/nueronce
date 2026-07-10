"""NUERONCE training pipelines.

Two phases, keeping the source notes' architecturally meaningful names:

- **WPGCP** — Web-Scale Provenance-Grounded Curriculum Pretraining
  (multi-objective, multi-phase curriculum; see :mod:`nueronce.training.curriculum`,
  :mod:`nueronce.training.episodes`, :mod:`nueronce.training.losses`).
- **VGRFT** — Verifier-Guided Residual Fine-Tuning (structured instruction
  tuning, tool grounding, verifier training, residual correction experts; see
  :mod:`nueronce.training.vgrft`). Stage 1 (SFT) has two interchangeable backends:
  :mod:`nueronce.training.sft` (PyTorch, fine-tunes ``NUERONCEModel``) and
  :mod:`nueronce.engine.models` (from-scratch, NumPy-only). Both share the
  turn data/encoding in :mod:`nueronce.training.dialogue_data`.
"""

from __future__ import annotations

import importlib

from . import curriculum, dialogue_data, episodes, losses, vgrft

__all__ = ["curriculum", "dialogue_data", "episodes", "losses", "vgrft", "sft"]


def __getattr__(name):
    # nueronce.training.sft imports torch (it drives real backprop), so it stays
    # lazy — the same reason nueronce/__init__.py defers NUERONCEModel.
    if name == "sft":
        return importlib.import_module(".sft", __name__)
    raise AttributeError(f"module 'nueronce.training' has no attribute {name!r}")
