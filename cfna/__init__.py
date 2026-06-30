"""CFNA — a hybrid foundational model architecture.

CFNA reorganizes the standard ``next_token = MODEL(previous_tokens)`` recipe into
a pipeline with explicit division of labor:

    perceive → represent meaning → determine intent → retrieve evidence →
    reason → plan → communicate → verify → revise

This package is the engineer-facing scaffold for that architecture. The data
model and pure-logic paths (provenance gating, dynamic patching, retrieval score
fusion, consolidation scoring, LoRA) are implemented and tested; the learned
neural components expose typed interfaces that raise ``BackendNotConfigured``
until a PyTorch/JAX backend is wired in.

See ``docs/CFNA_design.md`` for the full design and ``docs/architecture.md`` for
the module map.
"""

from __future__ import annotations

from . import config, ops, types
from ._backend import BackendNotConfigured

__version__ = "0.1.0"

__all__ = [
    "config",
    "ops",
    "types",
    "BackendNotConfigured",
    "__version__",
]
