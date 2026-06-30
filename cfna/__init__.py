"""CFNA — a hybrid foundational model architecture.

CFNA reorganizes the standard ``next_token = MODEL(previous_tokens)`` recipe into
a pipeline with explicit division of labor:

    perceive → represent meaning → determine intent → retrieve evidence →
    reason → plan → communicate → verify → revise

This package is a runnable implementation of that architecture. Every operator is
hand-built from primitives (PyTorch is used only as a tensor/autograd substrate —
no stock transformer, attention, or state-space modules). The full pipeline
trains end-to-end and is verified causal; see ``cfna.model.CFNAModel``,
``cfna.pipeline.respond``, and the demos under ``scripts/``.

See ``docs/CFNA_design.md`` for the full design and ``docs/architecture.md`` for
the module map.
"""

from __future__ import annotations

from . import config, ops, types
from ._backend import BackendNotConfigured

__version__ = "0.2.0"


def __getattr__(name):
    # Lazy access to the torch-backed pieces so importing cfna stays light.
    if name == "CFNAModel":
        from .model import CFNAModel

        return CFNAModel
    if name == "respond":
        from .pipeline import respond

        return respond
    raise AttributeError(f"module 'cfna' has no attribute {name!r}")


__all__ = [
    "config",
    "ops",
    "types",
    "BackendNotConfigured",
    "CFNAModel",
    "respond",
    "__version__",
]
