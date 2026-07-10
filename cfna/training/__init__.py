"""CFNA training pipelines.

- **WPGCP** — provenance-grounded curriculum pretraining.
- **VGRFT** — verifier-guided residual fine-tuning.
- **CPU streaming** — bounded-window, block-local learning intended to test
  CFNA's no-VRAM / low-memory training thesis.
"""
from __future__ import annotations

import importlib

from . import curriculum, dialogue_data, episodes, losses, vgrft

__all__ = [
    "curriculum", "dialogue_data", "episodes", "losses", "vgrft",
    "sft", "cpu_streaming",
]


def __getattr__(name):
    # These modules import torch and stay lazy so metadata/data tooling remains
    # usable without importing the training backend.
    if name in {"sft", "cpu_streaming"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'cfna.training' has no attribute {name!r}")
