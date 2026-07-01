"""Optimizers for the microtorch engine: SGD and AdamW, plus grad-norm clipping."""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor]):
        self.params: List[Tensor] = [p for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(params)
        self.lr, self.momentum = lr, momentum
        self.vel = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.vel[i] = self.momentum * self.vel[i] + p.grad
            p.data -= self.lr * self.vel[i]


class AdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.wd:
                p.data -= self.lr * self.wd * p.data     # decoupled weight decay
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> float:
    params = [p for p in params if p.grad is not None]
    total = math.sqrt(sum(float(np.sum(p.grad ** 2)) for p in params))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad *= scale
    return total


__all__ = ["Optimizer", "SGD", "AdamW", "clip_grad_norm_"]
