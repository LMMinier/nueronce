"""Canonical RFT neural operators for NUERONCE.

The canonical basis is built from a golden-ratio-spaced complex dictionary

    U = Phi (Phi^H Phi)^(-1/2)

and stored as a non-trainable unitary buffer. RFTSparseLinear trains only the
active spectral connections; inactive dense weights and optimizer state do not
exist.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def canonical_rft_basis(
    dim: int,
    *,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return the dim x dim canonical unitary RFT basis."""
    if dim < 1:
        raise ValueError("dim must be positive")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    n = torch.arange(dim, dtype=real_dtype, device=device)[:, None]
    k = torch.arange(1, dim + 1, dtype=real_dtype, device=device)[None, :]
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    frequencies = torch.remainder(k * phi, 1.0)
    raw = torch.exp(2j * math.pi * n * frequencies) / math.sqrt(dim)
    raw = raw.to(dtype)

    gram = raw.conj().transpose(-1, -2) @ raw
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    floor = torch.finfo(real_dtype).eps * dim
    inv_sqrt = eigenvectors @ torch.diag_embed(
        eigenvalues.clamp_min(floor).rsqrt().to(dtype)
    ) @ eigenvectors.conj().transpose(-1, -2)
    return raw @ inv_sqrt


def _golden_sparse_indices(d_in: int, d_out: int, fan_in: int) -> tuple[Tensor, Tensor]:
    """Deterministic phi-stride connectivity with exactly fan_in inputs per output."""
    if not 1 <= fan_in <= d_in:
        raise ValueError("fan_in must be in [1, d_in]")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    stride = max(1, int(round(d_in / phi)))
    rows, cols = [], []
    for row in range(d_out):
        chosen = []
        candidate = (row * stride) % d_in
        while len(chosen) < fan_in:
            if candidate not in chosen:
                chosen.append(candidate)
            candidate = (candidate + stride) % d_in
            if len(chosen) < fan_in and candidate in chosen:
                candidate = (candidate + 1) % d_in
        rows.extend([row] * fan_in)
        cols.extend(chosen)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


class CanonicalRFT(nn.Module):
    """Exact blockwise differentiable analysis/synthesis with a fixed RFT basis."""

    def __init__(
        self,
        dim: int,
        block_size: Optional[int] = None,
        complex_dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.dim = int(dim)
        self.block_size = int(block_size or dim)
        if self.dim % self.block_size != 0:
            raise ValueError("dim must be divisible by block_size")
        self.blocks = self.dim // self.block_size
        self.complex_dtype = complex_dtype
        self.register_buffer(
            "basis", canonical_rft_basis(self.block_size, dtype=complex_dtype)
        )

    def analysis(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        shape = x.shape
        blocked = x.to(self.basis.dtype).reshape(
            *shape[:-1], self.blocks, self.block_size
        )
        return (blocked @ self.basis.conj()).reshape(*shape)

    def synthesis(self, coefficients: Tensor) -> Tensor:
        if coefficients.shape[-1] != self.dim:
            raise ValueError(
                f"expected last dimension {self.dim}, got {coefficients.shape[-1]}"
            )
        shape = coefficients.shape
        blocked = coefficients.reshape(*shape[:-1], self.blocks, self.block_size)
        return (blocked @ self.basis.transpose(-1, -2)).reshape(*shape)

    def forward(self, x: Tensor) -> Tensor:
        return self.synthesis(self.analysis(x)).real.to(x.dtype)


class RFTSparseLinear(nn.Module):
    """Sparse linear operator whose trainable weights live in RFT coordinates.

    y = Re[ U_out S U_in^H x ] + bias

    S is represented only by active COO entries. This avoids allocating dense
    inactive weights, gradients, or optimizer moments.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        fan_in: int,
        bias: bool = True,
        complex_dtype: torch.dtype = torch.complex64,
        block_size: int = 64,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.fan_in = int(fan_in)
        in_block = min(block_size, d_in)
        out_block = min(block_size, d_out)
        if d_in % in_block != 0 or d_out % out_block != 0:
            raise ValueError("d_in and d_out must be divisible by their RFT block sizes")
        self.in_rft = CanonicalRFT(
            d_in, block_size=in_block, complex_dtype=complex_dtype
        )
        self.out_rft = CanonicalRFT(
            d_out, block_size=out_block, complex_dtype=complex_dtype
        )

        rows, cols = _golden_sparse_indices(d_in, d_out, fan_in)
        self.register_buffer("row_index", rows)
        self.register_buffer("col_index", cols)

        scale = 1.0 / math.sqrt(fan_in)
        self.weight_real = nn.Parameter(torch.randn(len(rows)) * scale)
        self.weight_imag = nn.Parameter(torch.zeros(len(rows)))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    @property
    def active_connections(self) -> int:
        return int(self.row_index.numel())

    @property
    def dense_connections(self) -> int:
        return self.d_in * self.d_out

    @property
    def spectral_density(self) -> float:
        return self.active_connections / self.dense_connections

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape[:-1]
        coefficients = self.in_rft.analysis(x).reshape(-1, self.d_in)
        values = torch.complex(self.weight_real, self.weight_imag).to(
            coefficients.dtype
        )

        selected = coefficients[:, self.col_index] * values
        mixed = coefficients.new_zeros(coefficients.shape[0], self.d_out)
        mixed.index_add_(1, self.row_index, selected)

        output = self.out_rft.synthesis(mixed).real
        output = output.reshape(*original_shape, self.d_out).to(x.dtype)
        return output if self.bias is None else output + self.bias.to(output.dtype)


class RFTGatedMLP(nn.Module):
    """SwiGLU-style feed-forward network using sparse RFT projections."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        *,
        fan_in: int,
        block_size: int = 64,
    ):
        super().__init__()
        if fan_in > min(dim, hidden):
            raise ValueError("fan_in exceeds an RFT projection input dimension")
        self.up = RFTSparseLinear(
            dim, hidden, fan_in=fan_in, bias=False, block_size=block_size
        )
        self.gate = RFTSparseLinear(
            dim, hidden, fan_in=fan_in, bias=False, block_size=block_size
        )
        self.down = RFTSparseLinear(
            hidden, dim, fan_in=fan_in, bias=False, block_size=block_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "canonical_rft_basis",
    "CanonicalRFT",
    "RFTSparseLinear",
    "RFTGatedMLP",
]
