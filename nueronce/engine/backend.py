"""Array and dtype policy for the Nueronce Engine runtime.

Training can select float32 so model storage is practical at large scale.
Tests that need finite-difference precision retain float64 by default and may
also use ``dtype_policy`` or ``MICROTORCH_DTYPE`` explicitly.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as xp


@dataclass(frozen=True)
class DTypePolicy:
    param: Any = xp.float64
    activation: Any = xp.float64
    grad: Any = xp.float64
    optimizer: Any = xp.float32


def _env_dtype():
    name = os.environ.get("MICROTORCH_DTYPE", "float64").lower()
    if name not in {"float32", "float64"}:
        raise ValueError("MICROTORCH_DTYPE must be float32 or float64")
    return getattr(xp, name)


_POLICY = DTypePolicy(param=_env_dtype(), activation=_env_dtype(),
                      grad=_env_dtype(), optimizer=xp.float32)


def get_dtype_policy() -> DTypePolicy:
    return _POLICY


def set_dtype_policy(policy: DTypePolicy) -> None:
    global _POLICY
    _POLICY = policy


@contextmanager
def dtype_policy(policy: DTypePolicy):
    previous = get_dtype_policy()
    set_dtype_policy(policy)
    try:
        yield
    finally:
        set_dtype_policy(previous)


def to_cpu(value: Any) -> Any:
    get = getattr(value, "get", None)
    return get() if callable(get) else value


__all__ = ["xp", "to_cpu", "DTypePolicy", "get_dtype_policy",
           "set_dtype_policy", "dtype_policy"]
