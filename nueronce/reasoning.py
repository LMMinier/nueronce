"""Addressable recurrent execution for algorithmic latent reasoning.

Unlike generic recurrent depth, this module has an explicit working register and
performs one content-addressed memory read followed by one shared state
transition per step. Increasing ``steps`` therefore extends an algorithm rather
than merely applying more unrelated feature transformations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from .nn import Linear, MLP, RMSNorm


@dataclass
class ExecutionTrace:
    states: List[Tensor]
    read_weights: List[Tensor]
    halt_probabilities: List[Tensor]


class AddressableExecutionRegister(nn.Module):
    """A small differentiable machine with shared transition weights.

    ``keys`` and ``values`` form immutable evidence memory. ``state`` is the
    mutable register. Each recurrence reads the value at the address represented
    by the current state and writes the result back into that register.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.query = Linear(dim, dim, bias=False)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=False)
        self.candidate = MLP(2 * dim, 2 * dim, dim)
        self.write_gate = Linear(2 * dim, dim)
        self.halt = Linear(2 * dim, 1)
        self.norm = RMSNorm(dim)

    def step(self, state: Tensor, keys: Tensor, values: Tensor,
             memory_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        squeeze_query = state.ndim == 2
        q = self.query(self.norm(state))
        if squeeze_query:
            q = q[:, None, :]
        scores = torch.matmul(q, self.key(keys).transpose(-1, -2)) / math.sqrt(self.dim)
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        read = torch.matmul(weights, self.value(values))
        if squeeze_query:
            read = read.squeeze(1)
            weights = weights.squeeze(1)
        joined = torch.cat([state, read], dim=-1)
        candidate = self.candidate(joined)
        gate = torch.sigmoid(self.write_gate(joined))
        next_state = self.norm(state + gate * (candidate - state))
        halt_probability = torch.sigmoid(self.halt(joined)).squeeze(-1)
        return next_state, weights, halt_probability

    def forward(self, state: Tensor, keys: Tensor, values: Tensor, steps: int,
                memory_mask: Optional[Tensor] = None, halt_threshold: float = 0.0,
                min_steps: int = 1) -> Tuple[Tensor, ExecutionTrace]:
        states, reads, halts = [], [], []
        for index in range(steps):
            state, read, halt = self.step(state, keys, values, memory_mask)
            states.append(state)
            reads.append(read)
            halts.append(halt)
            if (halt_threshold > 0 and index + 1 >= min_steps
                    and bool(torch.all(halt >= halt_threshold))):
                break
        return state, ExecutionTrace(states, reads, halts)


__all__ = ["AddressableExecutionRegister", "ExecutionTrace"]
