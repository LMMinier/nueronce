"""From-scratch neural primitives for NUERONCE.

PyTorch is used **only** as a tensor / autograd / optimizer substrate. None of
the stock transformer machinery is used: no ``nn.Transformer``,
``nn.TransformerEncoderLayer``, ``nn.MultiheadAttention``, ``nn.Linear``,
``nn.LayerNorm``, ``nn.Embedding``, or ``scaled_dot_product_attention``, and no
external state-space (Mamba) package. Every parametric layer below is built from
``nn.Parameter`` + raw tensor ops so the architecture is genuinely hand-rolled.

``nn.Module`` is used purely as a parameter container, and ``torch.nn.functional``
is used only for elementwise/conv math primitives (``gelu``, ``silu``,
``conv1d``), never for attention or transformer blocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn



# --------------------------------------------------------------------------- #
# Parametric primitives (hand-built)
# --------------------------------------------------------------------------- #

class Linear(nn.Module):
    """y = x W^T (+ b). Hand-rolled so no stock nn.Linear is involved."""

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.t()
        return y if self.bias is None else y + self.bias


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, dim) * 0.02)

    def forward(self, idx: Tensor) -> Tensor:
        return self.weight[idx]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Norm statistics in fp32 regardless of autocast dtype: squaring fp16
        # activations overflows past |x| ~ 16 under AMP, so the reduction must
        # not run in half precision (the standard LLaMA-style RMSNorm pattern).
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).to(x.dtype) * self.gain.to(x.dtype)


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.fc1 = Linear(d_in, hidden)
        self.fc2 = Linear(hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GatedMLP(nn.Module):
    """SwiGLU-style gated feedforward, hand-built."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = Linear(dim, hidden, bias=False)
        self.gate = Linear(dim, hidden, bias=False)
        self.down = Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# --------------------------------------------------------------------------- #
# Attention, built from matmul + softmax (no fused / stock attention)
# --------------------------------------------------------------------------- #

def masked_softmax(scores: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Softmax over the last dim with an additive boolean mask (True = keep).

    Rows that are fully masked out return all-zero weights instead of NaN.

    Dtype-aware constants so this stays NaN-free under AMP/fp16: a fixed -1e30
    fill overflows to -inf in half precision, and for a fully-masked row
    ``-inf - (-inf)`` is NaN; a fixed 1e-20 denominator clamp underflows to 0.
    ``finfo.min``/``finfo.tiny`` are finite in every dtype, so the fully-masked
    row degrades to exact zeros in fp16 the same way it does in fp32.
    """
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores)
    if mask is not None:
        weights = weights * mask
    denom = weights.sum(dim=-1, keepdim=True)
    return weights / denom.clamp_min(torch.finfo(weights.dtype).tiny)


class MultiHeadProjection(nn.Module):
    """Q/K/V/O projections shared by the local and sparse-global attentions."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0, "dim must divide n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(dim, dim, bias=False)
        self.v = Linear(dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)

    def split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,hd]

    def merge(self, x: Tensor) -> Tensor:
        b, h, t, hd = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * hd)


def causal_window_mask(t: int, window: Optional[int], device) -> Tensor:
    """[T, T] boolean mask: position i may attend to j if j<=i and (i-j)<window."""
    idx = torch.arange(t, device=device)
    causal = idx[:, None] >= idx[None, :]
    if window is not None:
        causal = causal & ((idx[:, None] - idx[None, :]) < window)
    return causal


class LocalAttention(nn.Module):
    """Causal self-attention restricted to a sliding local window."""

    def __init__(self, dim: int, n_heads: int, window: int):
        super().__init__()
        self.proj = MultiHeadProjection(dim, n_heads)
        self.window = window

    def forward(self, x: Tensor, key_padding: Optional[Tensor] = None) -> Tensor:
        b, t, _ = x.shape
        q = self.proj.split(self.proj.q(x))
        k = self.proj.split(self.proj.k(x))
        v = self.proj.split(self.proj.v(x))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.proj.head_dim)
        mask = causal_window_mask(t, self.window, x.device)[None, None]  # [1,1,T,T]
        if key_padding is not None:  # [B,T] True=valid
            mask = mask & key_padding[:, None, None, :]
        attn = masked_softmax(scores, mask)
        return self.proj.o(self.proj.merge(attn @ v))


class SparseGlobalAttention(nn.Module):
    """Global attention where each query keeps only its top-k causal keys.

    This is the "sparse global routing budget" from the design: exact content-
    addressed access, but only to the k most relevant earlier positions.
    """

    def __init__(self, dim: int, n_heads: int, topk: int):
        super().__init__()
        self.proj = MultiHeadProjection(dim, n_heads)
        self.topk = topk

    def forward(
        self,
        x: Tensor,
        importance: Optional[Tensor] = None,
        key_padding: Optional[Tensor] = None,
    ) -> Tensor:
        b, t, _ = x.shape
        q = self.proj.split(self.proj.q(x))
        k = self.proj.split(self.proj.k(x))
        v = self.proj.split(self.proj.v(x))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.proj.head_dim)

        mask = causal_window_mask(t, None, x.device)[None, None].expand(b, 1, t, t)
        if key_padding is not None:
            mask = mask & key_padding[:, None, None, :]
        if importance is not None:  # bias toward important keys before top-k
            scores = scores + importance[:, None, None, :]

        # keep top-k keys per query (within the causal mask); finfo.min instead
        # of a fixed -1e30 so the fill stays finite under AMP/fp16
        if self.topk < t:
            masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            kth = masked_scores.topk(self.topk, dim=-1).values[..., -1:]
            mask = mask & (masked_scores >= kth)

        attn = masked_softmax(scores, mask)
        return self.proj.o(self.proj.merge(attn @ v))


class CrossAttention(nn.Module):
    """Generic cross-attention from a query stream to a context stream.

    Used for (a) byte→unit decoding and (b) retrieval injection. An explicit
    boolean ``mask`` [B, Tq, Tc] controls which context items each query sees.
    """

    def __init__(self, dim: int, n_heads: int, ctx_dim: Optional[int] = None):
        super().__init__()
        ctx_dim = ctx_dim or dim
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(ctx_dim, dim, bias=False)
        self.v = Linear(ctx_dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, ctx: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self._split(self.q(x))
        k = self._split(self.k(ctx))
        v = self._split(self.v(ctx))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask[:, None]  # [B,1,Tq,Tc]
        attn = masked_softmax(scores, mask)
        out = attn @ v
        b, h, t, hd = out.shape
        out = out.transpose(1, 2).contiguous().view(b, t, h * hd)
        return self.o(out)


# --------------------------------------------------------------------------- #
# Selective state-space mixer (Mamba-style math, hand-implemented)
# --------------------------------------------------------------------------- #

class SelectiveSSM(nn.Module):
    """Input-dependent (selective) state-space recurrence.

    Implements the selective scan from the design's recurrent path:
        Ā = exp(Δ ⊙ A),  B̄ = Δ ⊙ B
        h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t
        y_t = (C · h_t) + D ⊙ x_t
    with Δ, B, C produced from the input (selectivity). A is a learned negative
    decay. The scan is an explicit sequential loop (clear over fast); fine for the
    short unit sequences NUERONCE operates on.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_k: int = 4):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.conv_k = conv_k

        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False)
        # depthwise causal conv over channels (F.conv1d is a math primitive)
        self.conv_weight = nn.Parameter(torch.randn(self.d_inner, 1, conv_k) * 0.05)
        self.conv_bias = nn.Parameter(torch.zeros(self.d_inner))
        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = Linear(self.dt_rank, self.d_inner)
        # A stored as log for stability; A = -exp(A_log) is strictly negative.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor, key_padding: Optional[Tensor] = None) -> Tensor:
        b, t, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)  # [B,T,d_inner] each

        # causal depthwise conv: pad left by conv_k-1
        xc = x_in.transpose(1, 2)  # [B,d_inner,T]
        xc = F.pad(xc, (self.conv_k - 1, 0))
        xc = F.conv1d(xc, self.conv_weight, self.conv_bias, groups=self.d_inner)
        x_in = F.silu(xc.transpose(1, 2))  # [B,T,d_inner]

        dbc = self.x_proj(x_in)
        delta, B_in, C_in = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))            # [B,T,d_inner]
        A = -torch.exp(self.A_log)                          # [d_inner,d_state]

        if key_padding is not None:
            delta = delta * key_padding[..., None]

        # selective scan
        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for i in range(t):
            d_i = delta[:, i]                               # [B,d_inner]
            A_bar = torch.exp(d_i[..., None] * A)           # [B,d_inner,d_state]
            B_bar = d_i[..., None] * B_in[:, i][:, None, :]  # [B,d_inner,d_state]
            h = A_bar * h + B_bar * x_in[:, i][..., None]
            y_i = (h * C_in[:, i][:, None, :]).sum(dim=-1)  # [B,d_inner]
            ys.append(y_i)
        y = torch.stack(ys, dim=1) + self.D * x_in          # [B,T,d_inner]
        y = y * F.silu(z)
        return self.out_proj(y)


__all__ = [
    "Linear",
    "Embedding",
    "RMSNorm",
    "MLP",
    "GatedMLP",
    "masked_softmax",
    "LocalAttention",
    "SparseGlobalAttention",
    "CrossAttention",
    "SelectiveSSM",
    "causal_window_mask",
]
