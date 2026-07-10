"""Activation recomputation for the Nueronce Engine autograd engine.

``checkpoint`` executes a single-output stage without recording its internal
forward graph. During backward it recreates just that stage, seeds it with the
incoming output gradient, accumulates parameter gradients, and passes input
gradients back into the outer graph. This separates logical model depth from
physical graph residency without custom derivatives for every NUERONCE block.
"""
from __future__ import annotations

from typing import Callable, Iterable

from .tensor import Tensor, no_grad


def checkpoint(fn: Callable[..., Tensor], *inputs: Tensor,
               parameters: Iterable[Tensor] = (), name: str = "stage") -> Tensor:
    inputs = tuple(inputs)
    params = tuple(parameters)
    if not inputs:
        raise ValueError("checkpoint requires at least one Tensor input")

    with no_grad():
        value = fn(*inputs)
    if not isinstance(value, Tensor):
        raise TypeError("checkpointed function must return one Tensor")

    tracked = tuple(x for x in (*inputs, *params) if x.requires_grad)
    out = Tensor(value.data.copy(), requires_grad=bool(tracked),
                 _children=tracked, _op=f"checkpoint:{name}")

    def _backward():
        if out.grad is None:
            return
        replay_inputs = tuple(
            Tensor(x.data.copy(), requires_grad=x.requires_grad) for x in inputs
        )
        replay = fn(*replay_inputs)
        if not isinstance(replay, Tensor):
            raise TypeError("checkpointed function changed return type during replay")
        replay.backward(out.grad)
        for original, recreated in zip(inputs, replay_inputs):
            if original.requires_grad and recreated.grad is not None:
                original._accum(recreated.grad)

    out._backward = _backward
    return out


__all__ = ["checkpoint"]
