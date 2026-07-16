"""Matched sparse baselines for the NUERONCE RFT experiment.

The operators in this file share the same active complex-weight budget and COO
connectivity as :class:`nueronce.rft.RFTSparseLinear` where applicable. They
exist to distinguish an RFT-specific effect from sparsity alone.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rft import RFTGatedMLP, _golden_sparse_indices


def orthonormal_dct_basis(
    dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tensor:
    """Return an orthonormal DCT-II basis matrix."""
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
    """Direct-coordinate sparse linear layer with fixed COO connectivity."""

    def __init__(self, d_in: int, d_out: int, *, fan_in: int, bias: bool = True):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        rows, cols = _golden_sparse_indices(d_in, d_out, fan_in)
        self.register_buffer("row_index", rows)
        self.register_buffer("col_index", cols)
        self.weight = nn.Parameter(torch.randn(len(rows)) / math.sqrt(fan_in))
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
        selected = flat[:, self.col_index] * self.weight
        output = flat.new_zeros(flat.shape[0], self.d_out)
        output.index_add_(1, self.row_index, selected)
        output = output.reshape(*shape, self.d_out)
        return output if self.bias is None else output + self.bias


class DCTSparseLinear(nn.Module):
    """Sparse linear layer operating between fixed orthonormal DCT coordinates."""

    def __init__(self, d_in: int, d_out: int, *, fan_in: int, bias: bool = True):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.register_buffer("in_basis", orthonormal_dct_basis(d_in))
        self.register_buffer("out_basis", orthonormal_dct_basis(d_out))
        rows, cols = _golden_sparse_indices(d_in, d_out, fan_in)
        self.register_buffer("row_index", rows)
        self.register_buffer("col_index", cols)
        # Two trainable scalars per connection, matching RFT real+imag storage.
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
        # The auxiliary scalar enters as an independent quadrature-like channel
        # through a one-position cyclic shift. This keeps parameter budgets equal
        # without pretending the real DCT has a complex phase.
        primary = coefficients[:, self.col_index] * self.weight_primary
        shifted_cols = (self.col_index + 1) % self.d_in
        auxiliary = coefficients[:, shifted_cols] * self.weight_aux
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


def replace_core_ffn(
    model,
    variant: str,
    *,
    fan_in: int,
    ffn_mult: int = 3,
    rft_block_size: int = 64,
):
    """Replace only hybrid-core FFNs for a controlled matched experiment."""
    dim = int(model.cfg.d_model)
    hidden = int(ffn_mult * dim)
    factories: dict[str, Callable[[], nn.Module]] = {
        "ordinary_sparse": lambda: SparseGatedMLP(dim, hidden, fan_in=fan_in),
        "dct_sparse": lambda: DCTSparseGatedMLP(dim, hidden, fan_in=fan_in),
        "rft_sparse": lambda: RFTGatedMLP(
            dim,
            hidden,
            fan_in=fan_in,
            block_size=rft_block_size,
        ),
    }
    if variant == "dense":
        return model
    if variant not in factories:
        raise ValueError(f"unknown variant {variant!r}")
    for block in model.core.blocks:
        block.ffn = factories[variant]()
    return model


__all__ = [
    "orthonormal_dct_basis",
    "SparseLinear",
    "DCTSparseLinear",
    "SparseGatedMLP",
    "DCTSparseGatedMLP",
    "replace_core_ffn",
]
