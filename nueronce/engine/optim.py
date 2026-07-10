"""Memory-conscious optimizers for the Nueronce Engine engine."""
from __future__ import annotations

import math
from typing import Iterable, List

from .backend import get_dtype_policy, xp as np
from .tensor import Tensor


class NonFiniteGradientError(FloatingPointError):
    """Raised before any parameter or optimizer state is mutated."""


def validate_finite_gradients(params: Iterable[Tensor]) -> int:
    """Validate all present gradients and return their count.

    The pass happens before an optimizer changes parameters or moments, making
    a rejected step safe to retry after lowering the learning rate or restoring
    a checkpoint.
    """
    count = 0
    for index, p in enumerate(params):
        if p.grad is None:
            continue
        count += 1
        finite = np.isfinite(p.grad)
        if not bool(finite.all()):
            nan_count = int(np.isnan(p.grad).sum())
            inf_count = int(np.isinf(p.grad).sum())
            raise NonFiniteGradientError(
                f"parameter {index} shape={p.shape} has nonfinite gradient: "
                f"nan={nan_count} inf={inf_count}"
            )
    return count


class Optimizer:
    def __init__(self, params: Iterable[Tensor]):
        self.params: List[Tensor] = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        del state


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(params)
        self.lr, self.momentum = lr, momentum
        self.vel = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        validate_finite_gradients(self.params)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.vel[i] *= self.momentum
            self.vel[i] += p.grad
            p.data -= self.lr * self.vel[i]


class AdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd = eps, weight_decay
        dtype = get_dtype_policy().optimizer
        self.m = [np.zeros(p.shape, dtype=dtype) for p in self.params]
        self.v = [np.zeros(p.shape, dtype=dtype) for p in self.params]
        self.t = 0

    def step(self):
        validate_finite_gradients(self.params)
        next_t = self.t + 1
        c1, c2 = 1 - self.b1 ** next_t, 1 - self.b2 ** next_t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.wd:
                p.data *= 1.0 - self.lr * self.wd
            g = p.grad
            self.m[i] *= self.b1
            self.m[i] += (1 - self.b1) * g
            self.v[i] *= self.b2
            self.v[i] += (1 - self.b2) * (g * g)
            p.data -= self.lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)
        self.t = next_t

    def state_dict(self):
        return {"name": "adamw", "t": self.t, "lr": self.lr,
                "m": [x.copy() for x in self.m], "v": [x.copy() for x in self.v]}

    def load_state_dict(self, state):
        self.t, self.lr = int(state["t"]), float(state["lr"])
        self.m = [x.copy() for x in state["m"]]
        self.v = [x.copy() for x in state["v"]]


class StreamFactor(Optimizer):
    """Factorized, tiled adaptive optimizer for constrained CPU/RAM training."""
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 momentum: bool = True, tile_rows: int = 256,
                 clip_threshold: float = 1.0):
        super().__init__(params)
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd = eps, weight_decay
        self.use_momentum = momentum
        self.tile_rows = max(1, int(tile_rows))
        self.clip_threshold = float(clip_threshold)
        self.t = 0
        dtype = get_dtype_policy().optimizer
        self.m = [np.zeros(p.shape, dtype=dtype) if momentum else None for p in self.params]
        self.v = []
        for p in self.params:
            if p.ndim >= 2:
                rows, cols = p.shape[0], int(np.prod(p.shape[1:]))
                self.v.append((np.zeros(rows, dtype=dtype), np.zeros(cols, dtype=dtype)))
            else:
                self.v.append(np.zeros(p.shape, dtype=dtype))

    def _matrix_step(self, i, p, g, c1, c2):
        shape = p.shape
        w2, g2 = p.data.reshape(shape[0], -1), g.reshape(shape[0], -1)
        vr, vc = self.v[i]
        vr *= self.b2
        vr += (1 - self.b2) * np.mean(g2 * g2, axis=1)
        vc *= self.b2
        vc += (1 - self.b2) * np.mean(g2 * g2, axis=0)
        vr_hat, vc_hat = vr / c2, vc / c2
        normalizer = max(float(vr_hat.mean()), self.eps)
        m2 = self.m[i].reshape(w2.shape) if self.use_momentum else None
        for start in range(0, w2.shape[0], self.tile_rows):
            stop = min(start + self.tile_rows, w2.shape[0])
            gt = g2[start:stop]
            if m2 is not None:
                mt = m2[start:stop]
                mt *= self.b1
                mt += (1 - self.b1) * gt
                numerator = mt / c1
            else:
                numerator = gt
            denom = np.sqrt((vr_hat[start:stop, None] * vc_hat[None, :]) / normalizer) + self.eps
            update = numerator / denom
            rms = float(np.sqrt(np.mean(update * update)))
            if rms > self.clip_threshold:
                update *= self.clip_threshold / (rms + self.eps)
            w2[start:stop] -= self.lr * update

    def step(self):
        validate_finite_gradients(self.params)
        next_t = self.t + 1
        c1 = max(1 - self.b1 ** next_t, self.eps)
        c2 = max(1 - self.b2 ** next_t, self.eps)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.wd:
                p.data *= 1.0 - self.lr * self.wd
            g = p.grad.astype(get_dtype_policy().optimizer, copy=False)
            if p.ndim >= 2:
                self._matrix_step(i, p, g, c1, c2)
            else:
                v = self.v[i]
                v *= self.b2
                v += (1 - self.b2) * (g * g)
                if self.use_momentum:
                    m = self.m[i]
                    m *= self.b1
                    m += (1 - self.b1) * g
                    numerator = m / c1
                else:
                    numerator = g
                update = numerator / (np.sqrt(v / c2) + self.eps)
                rms = float(np.sqrt(np.mean(update * update))) if update.size else 0.0
                if rms > self.clip_threshold:
                    update *= self.clip_threshold / (rms + self.eps)
                p.data -= self.lr * update
        self.t = next_t

    def state_dict(self):
        packed_v = []
        for item in self.v:
            packed_v.append(tuple(x.copy() for x in item) if isinstance(item, tuple) else item.copy())
        return {"name": "streamfactor", "t": self.t, "lr": self.lr,
                "momentum": self.use_momentum, "tile_rows": self.tile_rows,
                "clip_threshold": self.clip_threshold,
                "m": [None if x is None else x.copy() for x in self.m], "v": packed_v}

    def load_state_dict(self, state):
        self.t, self.lr = int(state["t"]), float(state["lr"])
        self.m = [None if x is None else x.copy() for x in state["m"]]
        self.v = [tuple(x.copy() for x in item) if isinstance(item, tuple) else item.copy()
                  for item in state["v"]]


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> float:
    params = [p for p in params if p.grad is not None]
    validate_finite_gradients(params)
    total = math.sqrt(sum(float(np.sum(p.grad ** 2)) for p in params))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad *= scale
    return total


__all__ = ["Optimizer", "SGD", "AdamW", "StreamFactor", "NonFiniteGradientError",
           "validate_finite_gradients", "clip_grad_norm_"]
