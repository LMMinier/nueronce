"""A tiny byte language model built entirely on the microtorch engine.

Demonstrates end-to-end training on the from-scratch autograd: embedding →
(attention + selective-SSM) block → RMSNorm → byte head, trained with AdamW to
overfit a short string. Small on purpose — the engine is for correctness and
clarity, not speed.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import functional as F
from .nn import Embedding, GatedMLP, Linear, Module, MultiHeadAttention, RMSNorm, SelectiveSSM
from .optim import AdamW, clip_grad_norm_
from .tensor import Tensor, no_grad


class MicroByteLM(Module):
    """Hybrid (attention + SSM) byte LM — a miniature of the CFNA core."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, window: int = 8, d_state: int = 8):
        self.embed = Embedding(256, d_model)
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, window=window)
        self.norm2 = RMSNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.norm3 = RMSNorm(d_model)
        self.ffn = GatedMLP(d_model, 2 * d_model)
        self.norm_f = RMSNorm(d_model)
        self.head = Linear(d_model, 256)

    def forward(self, ids: np.ndarray) -> Tensor:
        x = self.embed(ids)                    # [B,T,d]
        x = x + self.attn(self.norm1(x))
        x = x + self.ssm(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return self.head(self.norm_f(x))       # [B,T,256]

    def loss(self, ids: np.ndarray) -> Tensor:
        logits = self.forward(ids[:, :-1])     # predict next byte
        b, t, _ = logits.shape
        flat = logits.reshape(b * t, 256)
        targets = np.asarray(ids[:, 1:]).reshape(b * t)
        return F.cross_entropy(flat, targets)

    def generate(self, prompt: bytes, max_new: int = 40, greedy: bool = True) -> bytes:
        ids = list(prompt)
        with no_grad():
            for _ in range(max_new):
                logits = self.forward(np.array([ids]))
                nxt = logits.data[0, -1]
                ids.append(int(nxt.argmax()) if greedy else
                           int(np.random.choice(256, p=_softmax_np(nxt))))
        return bytes(ids)


def _softmax_np(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def train_overfit(text: bytes, steps: int = 300, lr: float = 5e-3, seed: int = 0) -> List[float]:
    """Overfit a short byte string; returns the loss curve (should fall sharply)."""
    np.random.seed(seed)
    model = MicroByteLM()
    opt = AdamW(list(model.parameters()), lr=lr, weight_decay=0.0)
    ids = np.array([list(text)])
    curve = []
    for _ in range(steps):
        loss = model.loss(ids)
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append(loss.item())
    return curve


__all__ = ["MicroByteLM", "train_overfit"]
