"""Addressable recurrent execution on the from-scratch MicroTorch engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .backend import xp as np
from . import functional as F
from .nn import Linear, MLP, Module, RMSNorm
from .tensor import Tensor, cat


@dataclass
class ExecutionTrace:
    states: List[Tensor]
    read_weights: List[Tensor]
    halt_probabilities: List[Tensor]


class AddressableExecutionRegister(Module):
    def __init__(self, dim: int):
        self.dim = dim
        self.query = Linear(dim, dim, bias=False)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=False)
        self.candidate = MLP(2 * dim, 2 * dim, dim)
        self.write_gate = Linear(2 * dim, dim)
        self.halt = Linear(2 * dim, 1)
        self.norm = RMSNorm(dim)

    def step(self, state: Tensor, keys: Tensor, values: Tensor,
             memory_mask: Optional[np.ndarray] = None):
        squeeze_query = state.ndim == 2
        q = self.query(self.norm(state))
        if squeeze_query:
            q = q.reshape(q.shape[0], 1, q.shape[1])
        scores = (q @ self.key(keys).transpose(0, 2, 1)) * (1.0 / math.sqrt(self.dim))
        weights = F.masked_softmax(scores, memory_mask, axis=-1)
        read = weights @ self.value(values)
        if squeeze_query:
            read = read[:, 0]
            weights = weights[:, 0]
        joined = cat([state, read], axis=-1)
        candidate = self.candidate(joined)
        gate = F.sigmoid(self.write_gate(joined))
        next_state = self.norm(state + gate * (candidate - state))
        halt_probability = F.sigmoid(self.halt(joined)).reshape(state.shape[:-1])
        return next_state, weights, halt_probability

    def forward(self, state: Tensor, keys: Tensor, values: Tensor, steps: int,
                memory_mask: Optional[np.ndarray] = None, halt_threshold: float = 0.0,
                min_steps: int = 1):
        states, reads, halts = [], [], []
        for index in range(steps):
            state, read, halt = self.step(state, keys, values, memory_mask)
            states.append(state); reads.append(read); halts.append(halt)
            if (halt_threshold > 0 and index + 1 >= min_steps
                    and bool(np.all(halt.data >= halt_threshold))):
                break
        return state, ExecutionTrace(states, reads, halts)


__all__ = ["AddressableExecutionRegister", "ExecutionTrace"]
