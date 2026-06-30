"""Neural-backend boundary.

CFNA is framework-agnostic at the data-model level. The learned components
(CNN perception, MLP heads, selective state-space mixers, attention) need a real
tensor backend (PyTorch / JAX). Rather than hard-wire one into the scaffold, the
learned methods raise :class:`BackendNotConfigured` until a backend is injected.

This keeps the whole package importable and the pure-logic paths testable with
nothing installed but numpy, while making the "this needs real weights" boundary
explicit and greppable.
"""

from __future__ import annotations


class BackendNotConfigured(NotImplementedError):
    """Raised by learned components that need a real tensor backend."""


def needs_backend(component: str, detail: str = "") -> "BackendNotConfigured":
    msg = (
        f"{component} requires a neural backend (PyTorch/JAX) and trained "
        f"weights. This scaffold defines the interface and the pure-logic "
        f"control flow; wire a backend to make it runnable."
    )
    if detail:
        msg += f" {detail}"
    return BackendNotConfigured(msg)


__all__ = ["BackendNotConfigured", "needs_backend"]
