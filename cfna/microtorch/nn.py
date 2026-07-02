"""Neural building blocks on the microtorch engine.

Mirrors the CFNA primitives (cfna/nn.py) but on the from-scratch autograd engine,
proving the engine *serves the architecture*: Linear, Embedding, RMSNorm,
MLP/GatedMLP, hand-written multi-head attention (with masking, for the
local/sparse optimizations), and a selective state-space scan.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional

import numpy as np

from . import functional as F
from .tensor import Tensor, cat, stack


class Parameter(Tensor):
    """A Tensor that is optimized (requires_grad=True)."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def parameters(self) -> Iterator[Parameter]:
        seen = set()
        for _, v in self.__dict__.items():
            if isinstance(v, Parameter) and id(v) not in seen:
                seen.add(id(v)); yield v
            elif isinstance(v, Module):
                yield from v.parameters()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        yield from item.parameters()
                    elif isinstance(item, Parameter) and id(item) not in seen:
                        seen.add(id(item)); yield item

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _kaiming(d_in: int, d_out: int) -> np.ndarray:
    return np.random.randn(d_out, d_in) / math.sqrt(d_in)


class Linear(Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        self.weight = Parameter(_kaiming(d_in, d_out))
        self.bias = Parameter(np.zeros(d_out)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.transpose()
        return y if self.bias is None else y + self.bias


class Embedding(Module):
    def __init__(self, num: int, dim: int):
        self.weight = Parameter(np.random.randn(num, dim) * 0.02)

    def forward(self, idx: np.ndarray) -> Tensor:
        return self.weight[np.asarray(idx)]


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.gain = Parameter(np.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        ms = (x * x).mean(axis=-1, keepdims=True)
        inv = (ms + self.eps) ** -0.5
        return x * inv * self.gain


class MLP(Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        self.fc1 = Linear(d_in, hidden)
        self.fc2 = Linear(hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GatedMLP(Module):
    def __init__(self, dim: int, hidden: int):
        self.up = Linear(dim, hidden, bias=False)
        self.gate = Linear(dim, hidden, bias=False)
        self.down = Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _causal_window_mask(t: int, window: Optional[int]) -> np.ndarray:
    idx = np.arange(t)
    mask = idx[:, None] >= idx[None, :]
    if window is not None:
        mask = mask & ((idx[:, None] - idx[None, :]) < window)
    return mask


class MultiHeadAttention(Module):
    """Causal multi-head self-attention with an optional window (local) mask —
    the same content-addressed mixing CFNA uses, hand-built on the engine."""

    def __init__(self, dim: int, n_heads: int, window: Optional[int] = None):
        assert dim % n_heads == 0
        self.h, self.hd, self.window = n_heads, dim // n_heads, window
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(dim, dim, bias=False)
        self.v = Linear(dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        return x.reshape(b, t, self.h, self.hd).transpose(0, 2, 1, 3)  # [B,H,T,hd]

    def forward(self, x: Tensor, key_padding: Optional[np.ndarray] = None) -> Tensor:
        b, t, d = x.shape
        q, k, v = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.hd))  # [B,H,T,T]
        mask = _causal_window_mask(t, self.window)[None, None]
        if key_padding is not None:                       # [B,T] True=valid
            mask = mask & np.asarray(key_padding)[:, None, None, :]
        attn = F.masked_softmax(scores, mask, axis=-1)
        out = attn @ v                                   # [B,H,T,hd]
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.o(out)


class SparseGlobalAttention(Module):
    """Global attention where each query keeps only its top-k causal keys —
    exact content-addressed access, but to a bounded routing budget, matching
    ``cfna.nn.SparseGlobalAttention``."""

    def __init__(self, dim: int, n_heads: int, topk: int):
        assert dim % n_heads == 0
        self.h, self.hd, self.topk = n_heads, dim // n_heads, topk
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(dim, dim, bias=False)
        self.v = Linear(dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        return x.reshape(b, t, self.h, self.hd).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, importance: Optional[Tensor] = None,
                key_padding: Optional[np.ndarray] = None) -> Tensor:
        b, t, d = x.shape
        q, k, v = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.hd))  # [B,H,T,T]

        mask = np.broadcast_to(_causal_window_mask(t, None)[None, None], (b, 1, t, t)).copy()
        if key_padding is not None:
            mask = mask & np.asarray(key_padding)[:, None, None, :]
        if importance is not None:  # bias toward important keys before top-k
            imp = importance if isinstance(importance, Tensor) else Tensor(importance)
            scores = scores + imp.reshape(b, 1, 1, t)

        # Top-k key selection is a discrete, data-dependent decision (like the
        # boundary/segmentation decisions elsewhere): computed on plain arrays,
        # not backpropped through. Gradient still flows through the *scores*
        # feeding the eventual softmax over the selected keys.
        if self.topk < t:
            masked = np.where(mask, scores.data, -1e30)
            kth = np.sort(masked, axis=-1)[..., -self.topk:][..., :1]
            mask = mask & (masked >= kth)

        attn = F.masked_softmax(scores, mask, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.o(out)


class CrossAttention(Module):
    """Generic cross-attention from a query stream to a separate context
    stream, matching ``cfna.nn.CrossAttention`` (byte->unit decoding,
    retrieval injection)."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: Optional[int] = None):
        ctx_dim = ctx_dim or dim
        assert dim % n_heads == 0
        self.h, self.hd = n_heads, dim // n_heads
        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(ctx_dim, dim, bias=False)
        self.v = Linear(ctx_dim, dim, bias=False)
        self.o = Linear(dim, dim, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        return x.reshape(b, t, self.h, self.hd).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, ctx: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        b, t, d = x.shape
        q, k, v = self._split(self.q(x)), self._split(self.k(ctx)), self._split(self.v(ctx))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.hd))  # [B,H,Tq,Tc]
        m = mask[:, None] if mask is not None else None    # [B,1,Tq,Tc]
        attn = F.masked_softmax(scores, m, axis=-1) if m is not None else F.softmax(scores, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.o(out)


class SelectiveSSM(Module):
    """Input-dependent state-space recurrence (Mamba-style selective scan),
    hand-built on the engine — the architecture's linear-time recurrent path.

        h_t = exp(Δ_t ⊙ A) ⊙ h_{t-1} + (Δ_t ⊙ B_t) ⊙ x_t ;  y_t = Σ_n C_t ⊙ h_t
    """

    def __init__(self, d_model: int, d_state: int = 8):
        self.d, self.n = d_model, d_state
        self.in_proj = Linear(d_model, d_model, bias=False)
        self.x_proj = Linear(d_model, 2 * d_state + 1, bias=False)  # -> B, C, Δ
        self.A_log = Parameter(np.log(np.arange(1, d_state + 1).astype(float))[None, :]
                               .repeat(d_model, axis=0))
        self.out_proj = Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        xin = self.in_proj(x)
        params = self.x_proj(x)                            # [B,T,2n+1]
        B = params[:, :, : self.n]                         # [B,T,n]
        C = params[:, :, self.n: 2 * self.n]               # [B,T,n]
        dt = F.softmax(params[:, :, 2 * self.n:], axis=-1) # positive-ish gate [B,T,1]
        A = -(self.A_log.exp())                            # [d,n], negative
        h = Tensor(np.zeros((b, d, self.n)))
        ys = []
        for i in range(t):
            dti = dt[:, i]                                  # [B,1]
            xi = xin[:, i]                                  # [B,d]
            Bi = B[:, i].reshape(b, 1, self.n)             # [B,1,n]
            Ci = C[:, i].reshape(b, 1, self.n)             # [B,1,n]
            decay = (dti.reshape(b, 1, 1) * A.reshape(1, d, self.n)).exp()  # [B,d,n]
            h = decay * h + (dti.reshape(b, 1, 1) * Bi) * xi.reshape(b, d, 1)
            yi = (h * Ci).sum(axis=-1)                      # [B,d]
            ys.append(yi)
        y = stack(ys, axis=1)                              # [B,T,d]
        return self.out_proj(y)


class CFNASelectiveSSM(Module):
    """Faithful port of ``cfna.nn.SelectiveSSM``: input-dependent (selective)
    state-space recurrence with a depthwise causal conv + rank-projected Δ +
    softplus, matching the real production model's math exactly (rather than
    the simplified ``SelectiveSSM`` above, which predates this port and is
    kept for ``MicroByteLM``/``scripts/microtorch_demo.py`` backward compat).

        Ā = exp(Δ ⊙ A),  B̄ = Δ ⊙ B
        h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t
        y_t = (C · h_t) + D ⊙ x_t
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_k: int = 4):
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.conv_k = conv_k

        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv_weight = Parameter(np.random.randn(self.d_inner, 1, conv_k) * 0.05)
        self.conv_bias = Parameter(np.zeros(self.d_inner))
        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = Linear(self.dt_rank, self.d_inner)
        self.A_log = Parameter(np.log(np.arange(1, d_state + 1).astype(float))
                               [None, :].repeat(self.d_inner, axis=0))
        self.D = Parameter(np.ones(self.d_inner))
        self.out_proj = Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor, key_padding: Optional[np.ndarray] = None) -> Tensor:
        b, t, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz[:, :, : self.d_inner], xz[:, :, self.d_inner:]

        xc = x_in.transpose(0, 2, 1)                        # [B,d_inner,T]
        xc = F.depthwise_conv1d_causal(xc, self.conv_weight, self.conv_bias)
        x_in = F.silu(xc.transpose(0, 2, 1))                # [B,T,d_inner]

        dbc = self.x_proj(x_in)
        delta_in = dbc[:, :, : self.dt_rank]
        B_in = dbc[:, :, self.dt_rank: self.dt_rank + self.d_state]
        C_in = dbc[:, :, self.dt_rank + self.d_state:]
        delta = F.softplus(self.dt_proj(delta_in))          # [B,T,d_inner]
        A = -(self.A_log.exp())                             # [d_inner,d_state]

        if key_padding is not None:
            delta = delta * Tensor(np.asarray(key_padding)[..., None])

        h = Tensor(np.zeros((b, self.d_inner, self.d_state)))
        ys = []
        for i in range(t):
            d_i = delta[:, i]                                # [B,d_inner]
            A_bar = (d_i.reshape(b, self.d_inner, 1) * A.reshape(1, self.d_inner, self.d_state)).exp()
            B_bar = d_i.reshape(b, self.d_inner, 1) * B_in[:, i].reshape(b, 1, self.d_state)
            h = A_bar * h + B_bar * x_in[:, i].reshape(b, self.d_inner, 1)
            y_i = (h * C_in[:, i].reshape(b, 1, self.d_state)).sum(axis=-1)  # [B,d_inner]
            ys.append(y_i)
        y = stack(ys, axis=1) + self.D * x_in                # [B,T,d_inner]
        y = y * F.silu(z)
        return self.out_proj(y)


__all__ = ["Parameter", "Module", "Linear", "Embedding", "RMSNorm", "MLP",
           "GatedMLP", "MultiHeadAttention", "SparseGlobalAttention", "CrossAttention",
           "SelectiveSSM", "CFNASelectiveSSM"]
