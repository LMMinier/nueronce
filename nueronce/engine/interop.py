"""Checkpoint interoperability between PyTorch NUERONCE and Nueronce Engine.

The two implementations intentionally share architecture and tensor shapes,
but their module containers expose a handful of attention names differently.
This module keeps that translation explicit and rejects partial conversions.
"""
from __future__ import annotations

from typing import Iterator, Mapping

from .backend import xp as np
from .nn import Module, Parameter


def named_parameters(module: Module) -> Iterator[tuple[str, Parameter]]:
    """Yield stable dotted parameter names for an engine module tree."""
    seen: set[int] = set()

    def walk(value, prefix: str):
        for key, child in value.__dict__.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(child, Parameter):
                if id(child) not in seen:
                    seen.add(id(child))
                    yield name, child
            elif isinstance(child, Module):
                yield from walk(child, name)
            elif isinstance(child, (list, tuple)):
                for index, item in enumerate(child):
                    item_name = f"{name}.{index}"
                    if isinstance(item, Parameter):
                        if id(item) not in seen:
                            seen.add(id(item))
                            yield item_name, item
                    elif isinstance(item, Module):
                        yield from walk(item, item_name)

    yield from walk(module, "")


def torch_to_engine_name(name: str) -> str:
    """Translate the PyTorch attention-wrapper naming convention."""
    for projection in ("q", "k", "v", "o"):
        name = name.replace(f".proj.{projection}.", f".{projection}.")
    return name


def load_torch_state_dict(model: Module, state_dict: Mapping[str, object]) -> dict:
    """Load a PyTorch-style state dict into an equivalent engine model.

    Values only need ``shape`` and array conversion support, so callers may
    pass actual torch tensors or NumPy arrays. Every source tensor must load;
    silent partial conversion is forbidden.
    """
    targets = dict(named_parameters(model))
    loaded: list[str] = []
    missing: list[str] = []
    mismatched: list[tuple[str, tuple, tuple]] = []

    for source_name, value in state_dict.items():
        array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
        if source_name == "memory.retention":
            target = getattr(getattr(model, "memory", None), "retention", None)
            if target is None:
                missing.append(source_name)
                continue
        else:
            target = targets.get(torch_to_engine_name(source_name))
            if target is None:
                missing.append(source_name)
                continue
        if tuple(array.shape) != tuple(target.data.shape):
            mismatched.append((source_name, tuple(array.shape), tuple(target.data.shape)))
            continue
        target.data[...] = array.astype(target.data.dtype, copy=False)
        loaded.append(source_name)

    if missing or mismatched or len(loaded) != len(state_dict):
        raise ValueError(
            f"incomplete PyTorch→engine conversion: loaded={len(loaded)}/{len(state_dict)}, "
            f"missing={missing[:8]}, mismatched={mismatched[:8]}"
        )
    return {"loaded": len(loaded), "source_tensors": len(state_dict)}


__all__ = ["named_parameters", "torch_to_engine_name", "load_torch_state_dict"]
