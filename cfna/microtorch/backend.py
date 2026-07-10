"""Array backend used by :mod:`cfna.microtorch`.

MicroTorch is intentionally NumPy-first so a clean checkout works on ordinary
CPUs without PyTorch, CUDA, or CuPy. Keeping the backend behind this tiny
module lets experimental accelerators be added later without changing tensor,
NN, functional, or optimizer code.
"""

from __future__ import annotations

from typing import Any

import numpy as xp


def to_cpu(value: Any) -> Any:
    """Return *value* as a CPU/NumPy object.

    NumPy arrays are already resident on the CPU, so this is normally an
    identity operation. The defensive ``get`` branch also makes the helper
    compatible with CuPy-like arrays if an experimental backend supplies one.
    """

    get = getattr(value, "get", None)
    if callable(get):
        return get()
    return value


__all__ = ["xp", "to_cpu"]
