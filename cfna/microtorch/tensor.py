"""Core tensor with reverse-mode automatic differentiation.

Every operation records how to push gradients to its inputs (``_backward``); a
topological sort from the output runs them in reverse. Gradients are
broadcasting-correct: a value that was broadcast in the forward pass has its
gradient summed back down to its original shape (:func:`_unbroadcast`).

Values are float64 so finite-difference gradient checks are clean.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional, Tuple, Union

from .backend import to_cpu, xp as np

ArrayLike = Union["Tensor", np.ndarray, float, int, list]

# Global switch so inference / eval can skip building the graph.
_GRAD_ENABLED = True


@contextmanager
def no_grad():
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` back down to ``shape`` (reverse of NumPy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data: ArrayLike, requires_grad: bool = False,
                 _children: Iterable["Tensor"] = (), _op: str = ""):
        self.data = data.data if isinstance(data, Tensor) else np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad and _GRAD_ENABLED
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev = tuple(_children) if _GRAD_ENABLED else ()
        self._op = _op

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return self.transpose()

    def _accum(self, g: np.ndarray):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    def _make(self, data, children, op) -> "Tensor":
        req = _GRAD_ENABLED and any(c.requires_grad for c in children)
        return Tensor(data, requires_grad=req, _children=children if req else (), _op=op)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return to_cpu(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------ #
    # elementwise
    # ------------------------------------------------------------------ #
    def _bin(self, other, fwd, back_self, back_other, op):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._make(fwd(self.data, other.data), (self, other), op)

        def _backward():
            g = out.grad
            if self.requires_grad:
                self._accum(_unbroadcast(back_self(g, self.data, other.data), self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(back_other(g, self.data, other.data), other.shape))

        out._backward = _backward
        return out

    def __add__(self, other):
        return self._bin(other, lambda a, b: a + b, lambda g, a, b: g, lambda g, a, b: g, "+")

    def __sub__(self, other):
        return self._bin(other, lambda a, b: a - b, lambda g, a, b: g, lambda g, a, b: -g, "-")

    def __mul__(self, other):
        return self._bin(other, lambda a, b: a * b, lambda g, a, b: g * b, lambda g, a, b: g * a, "*")

    def __truediv__(self, other):
        return self._bin(other, lambda a, b: a / b,
                         lambda g, a, b: g / b, lambda g, a, b: -g * a / (b * b), "/")

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (Tensor(other) if not isinstance(other, Tensor) else other).__sub__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return (Tensor(other) if not isinstance(other, Tensor) else other).__truediv__(self)

    def __neg__(self):
        out = self._make(-self.data, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self._accum(-out.grad)

        out._backward = _backward
        return out

    def __pow__(self, p: float):
        assert isinstance(p, (int, float))
        out = self._make(self.data ** p, (self,), f"**{p}")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * p * self.data ** (p - 1))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # unary math
    # ------------------------------------------------------------------ #
    def _unary(self, fwd, back, op):
        y = fwd(self.data)
        out = self._make(y, (self,), op)

        def _backward():
            if self.requires_grad:
                self._accum(back(out.grad, self.data, y))

        out._backward = _backward
        return out

    def exp(self):
        return self._unary(np.exp, lambda g, x, y: g * y, "exp")

    def log(self):
        return self._unary(np.log, lambda g, x, y: g / x, "log")

    def tanh(self):
        return self._unary(np.tanh, lambda g, x, y: g * (1 - y * y), "tanh")

    def relu(self):
        return self._unary(lambda x: np.maximum(0, x), lambda g, x, y: g * (x > 0), "relu")

    def sqrt(self):
        return self ** 0.5

    # ------------------------------------------------------------------ #
    # matmul
    # ------------------------------------------------------------------ #
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._make(self.data @ other.data, (self, other), "@")

        def _backward():
            g = out.grad
            if self.requires_grad:
                self._accum(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # reductions / shape
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims=False):
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                ax = (axis,) if isinstance(axis, int) else tuple(axis)
                g = np.expand_dims(g, ax)
            self._accum(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in ((axis,) if isinstance(axis, int) else axis)])
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        if axes and len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = self._make(np.transpose(self.data, axes), (self,), "transpose")

        def _backward():
            if not self.requires_grad:
                return
            if axes is None:
                self._accum(np.transpose(out.grad))
            else:
                inv = np.argsort(axes)
                self._accum(np.transpose(out.grad, inv))

        out._backward = _backward
        return out

    def swapaxes(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    def __getitem__(self, idx):
        out = self._make(self.data[idx], (self,), "getitem")

        def _backward():
            if not self.requires_grad:
                return
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self._accum(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # backprop
    # ------------------------------------------------------------------ #
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = None


def tensor(data, requires_grad=False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def cat(tensors, axis=0) -> Tensor:
    tensors = list(tensors)
    data = np.concatenate([t.data for t in tensors], axis=axis)
    req = _GRAD_ENABLED and any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=req, _children=tensors if req else (), _op="cat")
    sizes = [t.shape[axis] for t in tensors]

    def _backward():
        g = out.grad
        offset = 0
        for t, s in zip(tensors, sizes):
            if t.requires_grad:
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(offset, offset + s)
                t._accum(g[tuple(sl)])
            offset += s

    out._backward = _backward
    return out


def stack(tensors, axis=0) -> Tensor:
    return cat([t.reshape(t.shape[:axis] + (1,) + t.shape[axis:]) for t in tensors], axis=axis)


__all__ = ["Tensor", "tensor", "cat", "stack", "no_grad", "_unbroadcast"]
