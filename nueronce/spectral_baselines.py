"""Matched sparse baselines for the NUERONCE RFT experiment."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .nn import GatedMLP
from .rft import RFTGatedMLP, _golden_sparse_indices


def orthonormal_dct_basis(dim: int, *, dtype=torch.float32, device=None) -> Tensor:
    if dim < 1:
        raise ValueError("dim must be positive")
    n = torch.arange(dim, dtype=dtype, device=device)[:, None]
    k = torch.arange(dim, dtype=dtype, device=device)[None, :]
    basis = torch.cos(math.pi / dim * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1.0 / dim)
    if dim > 1:
        basis[:, 1:] *= math.sqrt(2.0 / dim)
    return basis


class SparseLinear(nn.Module):
    """Direct-coordinate sparse layer with two scalars per active edge.

    The auxiliary scalar multiplies a cyclically shifted source channel. This
    matches the two-real-scalar budget of each complex RFT coefficient.
    """

    def __init__(self, d_in: int, d_out: int, *, fan_in: int, bias: bool = True):
        super().__init__()
        self.d_in, self.d_out = int(d_in), int(d_out)
        rows, cols = _golden_sparse_indices(d_in, d_out, fan_in)
        self.register_buffer("row_index", rows)
        self.register_buffer("col_index", cols)
        self.weight_primary = nn.Parameter(torch.randn(len(rows)) / math.sqrt(fan_in))
        self.weight_aux = nn.Parameter(torch.zeros(len(rows)))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    @property
    def active_connections(self) -> int:
        return int(self.row_index.numel())

    @property
    def density(self) -> float:
        return self.active_connections / (self.d_in * self.d_out)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.d_in)
        primary = flat[:, self.col_index] * self.weight_primary
        shifted = flat[:, (self.col_index + 1) % self.d_in] * self.weight_aux
        output = flat.new_zeros(flat.shape[0], self.d_out)
        output.index_add_(1, self.row_index, primary + shifted)
        output = output.reshape(*shape, self.d_out)
        return output if self.bias is None else output + self.bias


class DCTSparseLinear(nn.Module):
    """Sparse layer between fixed orthonormal DCT coordinates."""

    def __init__(self, d_in: int, d_out: int, *, fan_in: int, bias: bool = True):
        super().__init__()
        self.d_in, self.d_out = int(d_in), int(d_out)
        self.register_buffer("in_basis", orthonormal_dct_basis(d_in))
        self.register_buffer("out_basis", orthonormal_dct_basis(d_out))
        rows, cols = _golden_sparse_indices(d_in, d_out, fan_in)
        self.register_buffer("row_index", rows)
        self.register_buffer("col_index", cols)
        self.weight_primary = nn.Parameter(torch.randn(len(rows)) / math.sqrt(fan_in))
        self.weight_aux = nn.Parameter(torch.zeros(len(rows)))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    @property
    def active_connections(self) -> int:
        return int(self.row_index.numel())

    @property
    def density(self) -> float:
        return self.active_connections / (self.d_in * self.d_out)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        coefficients = (x @ self.in_basis).reshape(-1, self.d_in)
        primary = coefficients[:, self.col_index] * self.weight_primary
        auxiliary = coefficients[:, (self.col_index + 1) % self.d_in] * self.weight_aux
        mixed = coefficients.new_zeros(coefficients.shape[0], self.d_out)
        mixed.index_add_(1, self.row_index, primary + auxiliary)
        output = (mixed @ self.out_basis.T).reshape(*shape, self.d_out)
        return output if self.bias is None else output + self.bias


class SparseGatedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, *, fan_in: int):
        super().__init__()
        self.up = SparseLinear(dim, hidden, fan_in=fan_in, bias=False)
        self.gate = SparseLinear(dim, hidden, fan_in=fan_in, bias=False)
        self.down = SparseLinear(hidden, dim, fan_in=fan_in, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DCTSparseGatedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, *, fan_in: int):
        super().__init__()
        self.up = DCTSparseLinear(dim, hidden, fan_in=fan_in, bias=False)
        self.gate = DCTSparseLinear(dim, hidden, fan_in=fan_in, bias=False)
        self.down = DCTSparseLinear(hidden, dim, fan_in=fan_in, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _dense_budget_hidden(dim: int, sparse_hidden: int, fan_in: int) -> int:
    """Approximate dense FFN width matching sparse FFN trainable scalars."""
    sparse_scalars = 2 * fan_in * (2 * sparse_hidden + dim)
    return max(1, int(round(sparse_scalars / (3 * dim))))


def replace_core_ffn(model, variant: str, *, fan_in: int, ffn_mult: int = 3,
                     rft_block_size: int = 64):
    dim = int(model.cfg.d_model)
    hidden = int(ffn_mult * dim)
    dense_budget_hidden = _dense_budget_hidden(dim, hidden, fan_in)
    factories: dict[str, Callable[[], nn.Module]] = {
        "dense_budget": lambda: GatedMLP(dim, dense_budget_hidden),
        "ordinary_sparse": lambda: SparseGatedMLP(dim, hidden, fan_in=fan_in),
        "dct_sparse": lambda: DCTSparseGatedMLP(dim, hidden, fan_in=fan_in),
        "rft_sparse": lambda: RFTGatedMLP(
            dim, hidden, fan_in=fan_in, block_size=rft_block_size
        ),
    }
    if variant == "dense":
        return model
    if variant not in factories:
        raise ValueError(f"unknown variant {variant!r}")
    for block in model.core.blocks:
        block.ffn = factories[variant]()
    model.matched_ffn_metadata = {
        "variant": variant,
        "full_hidden": hidden,
        "dense_budget_hidden": dense_budget_hidden,
        "fan_in": fan_in,
    }
    return model


__all__ = [
    "orthonormal_dct_basis",
    "SparseLinear",
    "DCTSparseLinear",
    "SparseGatedMLP",
    "DCTSparseGatedMLP",
    "replace_core_ffn",
]
