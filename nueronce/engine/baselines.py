"""Matched, from-scratch baselines for honest comparison against NueronceModel
— the engine port of ``nueronce.baselines`` (which needs PyTorch).

Built from the same hand-rolled engine primitives (:mod:`nueronce.engine.nn`)
— no stock transformer/RNN modules here either — so the comparison reflects
the *architecture*, not a library advantage. Each baseline exposes the same
eval interface NueronceModel does: ``forward``, ``lm_loss``, ``num_params``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import functional as F
from .nn import Embedding, GatedMLP, Linear, Module, MultiHeadAttention, Parameter, RMSNorm, NUERONCESelectiveSSM
from .tensor import Tensor


@dataclass
class BaselineConfig:
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    max_len: int = 256
    d_state: int = 16


class _Block(Module):
    def __init__(self, cfg: BaselineConfig, kind: str):
        self.norm1 = RMSNorm(cfg.d_model)
        if kind == "transformer":
            self.mix = MultiHeadAttention(cfg.d_model, cfg.n_heads, window=cfg.max_len)  # full causal
        else:  # ssm
            self.mix = NUERONCESelectiveSSM(cfg.d_model, d_state=cfg.d_state)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = GatedMLP(cfg.d_model, 3 * cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _ByteLM(Module):
    """Shared decoder-only byte LM skeleton; ``kind`` selects the mixer."""

    def __init__(self, cfg: BaselineConfig, kind: str):
        self.cfg = cfg
        self.kind = kind
        self.embed = Embedding(256, cfg.d_model)
        self.pos = Parameter(np.random.randn(1, cfg.max_len, cfg.d_model) * 0.02)
        self.blocks = [_Block(cfg, kind) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model)
        self.head = Linear(cfg.d_model, 256)

    def forward(self, byte_ids: np.ndarray) -> Tensor:
        t = byte_ids.shape[1]
        x = self.embed(byte_ids) + self.pos[:, :t]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def lm_loss(self, byte_ids: np.ndarray) -> Tensor:
        logits = self.forward(byte_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, 256), np.asarray(byte_ids)[:, 1:].reshape(-1))

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())


class ByteTransformerLM(_ByteLM):
    """Decoder-only causal-attention byte LM (the canonical baseline)."""

    def __init__(self, cfg: Optional[BaselineConfig] = None):
        super().__init__(cfg or BaselineConfig(), kind="transformer")


class ByteSSMLM(_ByteLM):
    """Pure selective-state-space byte LM (the 'is hybrid better than SSM?' baseline)."""

    def __init__(self, cfg: Optional[BaselineConfig] = None):
        super().__init__(cfg or BaselineConfig(), kind="ssm")


__all__ = ["BaselineConfig", "ByteTransformerLM", "ByteSSMLM"]
