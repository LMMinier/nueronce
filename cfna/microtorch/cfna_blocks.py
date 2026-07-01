"""CFNA operator modules on the from-scratch microtorch engine — a faithful
port of ``cfna.blocks`` (PyTorch) so the real production architecture (not
just a smaller demo model) can run on nothing but NumPy.

Mirrors ``cfna.blocks`` module-for-module:

- :class:`BytePerceptionEncoder` — causal byte CNN + learned boundary head.
- :class:`TypedRecurrentMemory` — the typed multi-timescale gated memory cell.
- :class:`HybridBlock` / :class:`HybridCoreStack` — the hybrid cognitive
  fabric (selective SSM + local attention + sparse global attention +
  retrieval injection, merged by an adaptive router), reused recurrently.
- :class:`UnitEmbedder` — pooled-unit → d_model projection.
- :class:`ByteDecoder` — byte-level causal decoder cross-attending to the
  completed-unit context (+ optional retrieved context).

Everything here is causal, same as the PyTorch version.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import functional as F
from .nn import (
    CFNASelectiveSSM,
    CrossAttention,
    Embedding,
    GatedMLP,
    Linear,
    MLP,
    Module,
    MultiHeadAttention,
    Parameter,
    RMSNorm,
    SparseGlobalAttention,
)
from .tensor import Tensor, cat, stack
from ..types import CHANNELS, K_CHANNELS


# --------------------------------------------------------------------------- #
# Perception
# --------------------------------------------------------------------------- #

class _CausalConv1d(Module):
    """Depth-preserving causal 1D conv (left-padded, groups=1)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int = 1):
        self.weight = Parameter(np.random.randn(c_out, c_in, kernel) * (1.0 / (c_in * kernel) ** 0.5))
        self.bias = Parameter(np.zeros(c_out))
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, T]
        return F.conv1d_causal(x, self.weight, self.bias, dilation=self.dilation)


class BytePerceptionEncoder(Module):
    """Raw bytes -> local features + boundary logits (causal)."""

    def __init__(self, byte_embed_dim: int = 64, d_local: int = 96):
        self.embed = Embedding(256, byte_embed_dim)
        self.conv3 = _CausalConv1d(byte_embed_dim, d_local, kernel=3)
        self.conv7 = _CausalConv1d(d_local, d_local, kernel=7)
        self.dilated = _CausalConv1d(d_local, d_local, kernel=3, dilation=4)
        self.norm = RMSNorm(d_local)
        self.boundary_head = MLP(d_local, 128, 1)

    def forward(self, byte_ids: np.ndarray):
        x = self.embed(byte_ids).transpose(0, 2, 1)          # [B, emb, T]
        x = F.gelu(self.conv3(x))
        x = x + F.gelu(self.conv7(x))
        x = x + F.gelu(self.dilated(x))
        feats = self.norm(x.transpose(0, 2, 1))              # [B, T, d_local]
        boundary_logits = self.boundary_head(feats).reshape(feats.shape[0], feats.shape[1])
        return feats, boundary_logits


# --------------------------------------------------------------------------- #
# Typed multi-timescale recurrent memory
# --------------------------------------------------------------------------- #

class TypedRecurrentMemory(Module):
    """Typed gated recurrence over a unit sequence (see ``cfna.blocks`` for the
    per-channel forget/write/read/retention/authority equations)."""

    def __init__(self, d_model: int, channel_dim: int = 24):
        self.K = K_CHANNELS
        self.dk = channel_dim
        self.state_dim = self.K * channel_dim
        in_dim = d_model + self.state_dim
        self.forget = Linear(in_dim, self.state_dim)
        self.write = Linear(in_dim, self.state_dim)
        self.read = Linear(in_dim, self.state_dim)
        self.cand = Linear(in_dim, self.state_dim)
        self.readout = Linear(self.state_dim, d_model)
        lam = np.array([0.98, 0.97, 0.95, 0.99, 0.90, 0.999, 0.98])
        self.retention = Tensor(np.repeat(lam, channel_dim))  # fixed, not trained

    def forward(self, units: Tensor, authority: Optional[np.ndarray] = None) -> Tensor:
        b, p, _ = units.shape
        c = Tensor(np.zeros((b, self.state_dim)))
        h = Tensor(np.zeros((b, self.state_dim)))
        if authority is None:
            a = Tensor(np.ones((b, self.state_dim)))
        else:
            a = Tensor(np.repeat(np.asarray(authority), self.dk, axis=-1))
        outs = []
        for i in range(p):
            x = units[:, i]
            z = cat([x, h], axis=-1)
            f = F.sigmoid(self.forget(z))
            w = F.sigmoid(self.write(z))
            r = F.sigmoid(self.read(z))
            dc = self.cand(z).tanh()
            c = self.retention * f * c + a * w * dc
            h = r * c.tanh()
            outs.append(self.readout(h))
        return stack(outs, axis=1)  # [B, P, d_model], causal


# --------------------------------------------------------------------------- #
# Hybrid cognitive fabric
# --------------------------------------------------------------------------- #

class HybridBlock(Module):
    """SSM + local attn + sparse global attn (+ optional retrieval) via a router."""

    def __init__(self, d_model: int, n_heads: int = 4, local_window: int = 32,
                 sparse_topk: int = 16, d_state: int = 16, ffn_mult: int = 3):
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ssm = CFNASelectiveSSM(d_model, d_state=d_state)
        self.local = MultiHeadAttention(d_model, n_heads, window=local_window)
        self.sparse = SparseGlobalAttention(d_model, n_heads, topk=sparse_topk)
        self.retrieval = CrossAttention(d_model, n_heads)
        self.ffn = GatedMLP(d_model, ffn_mult * d_model)
        self.router = MLP(4 * d_model, d_model, 4)

    def forward(self, x: Tensor, key_padding: Optional[np.ndarray] = None,
                retrieval_ctx: Optional[Tensor] = None,
                retrieval_mask: Optional[np.ndarray] = None,
                importance: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        y_ssm = self.ssm(h, key_padding)
        y_local = self.local(h, key_padding)
        y_global = self.sparse(h, importance, key_padding)
        if retrieval_ctx is not None:
            y_ret = self.retrieval(h, retrieval_ctx, retrieval_mask)
        else:
            y_ret = Tensor(np.zeros(h.shape))

        # Per-position routing: each position's mix depends only on its own
        # (causal) path outputs, so no future position can influence it.
        route_in = cat([y_ssm, y_local, y_global, y_ret], axis=-1)  # [B,P,4*d]
        alpha = F.softmax(self.router(route_in), axis=-1)            # [B,P,4]
        merged = (alpha[:, :, 0:1] * y_ssm + alpha[:, :, 1:2] * y_local
                  + alpha[:, :, 2:3] * y_global + alpha[:, :, 3:4] * y_ret)
        x = x + merged
        x = x + self.ffn(self.norm2(x))
        return x


class HybridCoreStack(Module):
    """A few physical blocks reused over mode-dependent logical depth."""

    def __init__(self, d_model: int, physical_blocks: int = 2, **block_kw):
        self.blocks: List[HybridBlock] = [HybridBlock(d_model, **block_kw) for _ in range(physical_blocks)]

    def forward(self, x: Tensor, logical_depth: int, key_padding: Optional[np.ndarray] = None,
                retrieval_ctx: Optional[Tensor] = None, retrieval_mask: Optional[np.ndarray] = None,
                importance: Optional[Tensor] = None) -> Tensor:
        for t in range(logical_depth):
            block = self.blocks[t % len(self.blocks)]
            x = block(x, key_padding, retrieval_ctx, retrieval_mask, importance)
        return x


# --------------------------------------------------------------------------- #
# Unit embedding + byte decoder
# --------------------------------------------------------------------------- #

class UnitEmbedder(Module):
    def __init__(self, d_local: int, d_model: int):
        self.proj = Linear(d_local, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, pooled_local: Tensor) -> Tensor:
        return self.norm(self.proj(pooled_local))


class _DecoderLayer(Module):
    def __init__(self, d_model: int, n_heads: int, window: int):
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, window=window)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)         # byte -> completed units (causal)
        self.norm_r = RMSNorm(d_model)
        self.ret_attn = CrossAttention(d_model, n_heads)           # byte -> retrieved context
        self.norm3 = RMSNorm(d_model)
        self.ffn = GatedMLP(d_model, 3 * d_model)

    def forward(self, x, unit_ctx, cross_mask, key_padding, ret_ctx=None, ret_mask=None):
        x = x + self.self_attn(self.norm1(x), key_padding)
        x = x + self.cross_attn(self.norm2(x), unit_ctx, cross_mask)
        if ret_ctx is not None:
            x = x + self.ret_attn(self.norm_r(x), ret_ctx, ret_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class ByteDecoder(Module):
    """Predict the next byte from causal byte history + completed-unit context
    + (optionally) retrieved external context."""

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4, window: int = 32):
        self.embed = Embedding(256, d_model)
        self.layers: List[_DecoderLayer] = [_DecoderLayer(d_model, n_heads, window) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)
        self.head = Linear(d_model, 256)

    def forward(self, byte_ids: np.ndarray, unit_ctx: Tensor, cross_mask: np.ndarray,
                key_padding: Optional[np.ndarray] = None,
                ret_ctx: Optional[Tensor] = None, ret_mask: Optional[np.ndarray] = None) -> Tensor:
        x = self.embed(byte_ids)
        for layer in self.layers:
            x = layer(x, unit_ctx, cross_mask, key_padding, ret_ctx, ret_mask)
        return self.head(self.norm(x))


__all__ = [
    "BytePerceptionEncoder",
    "TypedRecurrentMemory",
    "HybridBlock",
    "HybridCoreStack",
    "UnitEmbedder",
    "ByteDecoder",
]
