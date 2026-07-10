"""A tiny byte language model built entirely on the Nueronce Engine.

Demonstrates end-to-end training on the from-scratch autograd: embedding →
(attention + selective-SSM) block → RMSNorm → byte head, trained with AdamW to
overfit a short string. Small on purpose — the engine is for correctness and
clarity, not speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import functional as F
from .nn import Embedding, GatedMLP, Linear, Module, MultiHeadAttention, RMSNorm, SelectiveSSM
from .optim import AdamW, clip_grad_norm_
from .tensor import Tensor, no_grad


class MicroByteLM(Module):
    """Hybrid (attention + SSM) byte LM — a miniature of the NUERONCE core."""

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

    def masked_loss(self, ids: np.ndarray, target_mask: np.ndarray) -> Tensor:
        """Next-byte CE restricted to positions flagged True in ``target_mask``
        (e.g. SFT: only the assistant's response bytes, not the user's turn).
        Same shift convention as :meth:`loss`."""
        logits = self.forward(ids[:, :-1])
        b, t, c = logits.shape
        flat = logits.reshape(b * t, c)
        targets = np.asarray(ids[:, 1:]).reshape(b * t)
        mask = np.asarray(target_mask)[:, 1:].reshape(b * t)
        return F.masked_cross_entropy(flat, targets, mask)

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


def train_dialogue_sft(model, examples: Sequence, *, steps: int = 300,
                        batch_size: int = 8, val_examples: Optional[Sequence] = None,
                        lr: float = 5e-3, seed: int = 0, log_every: int = 25) -> List[Dict[str, float]]:
    """Masked SFT training loop, entirely on the from-scratch engine
    engine (NumPy only, no PyTorch). Fine-tunes on (prompt, response) turns
    from ``nueronce.training.dialogue_data``, masking the loss to response bytes
    only — the same idea as ``nueronce.training.sft.train_sft``, minus the torch
    dependency. Duck-typed on ``.masked_loss(ids, mask)`` / ``.parameters()`` /
    ``.zero_grad()``, so it runs equally over the smaller :class:`MicroByteLM`
    demo model or the full, faithfully-ported
    :class:`~nueronce.engine.nueronce_model.NueronceModel`."""
    from ..training.dialogue_data import make_sft_batch

    rng = np.random.default_rng(seed)
    opt = AdamW(list(model.parameters()), lr=lr, weight_decay=0.0)
    val_batch = make_sft_batch(val_examples) if val_examples else None
    history: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(examples), size=min(batch_size, len(examples)))
        batch = make_sft_batch([examples[i] for i in idx])
        loss = model.masked_loss(batch["byte_ids"], batch["target_mask"])
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "train_loss": loss.item()}
            if val_batch is not None:
                with no_grad():
                    rec["val_loss"] = model.masked_loss(val_batch["byte_ids"], val_batch["target_mask"]).item()
            history.append(rec)
    return history


@dataclass
class MicroSFTBackend:
    """``VGRFTTrainer`` backend for stage 1 (supervised instruction tuning)
    running entirely on the from-scratch Nueronce Engine — useful wherever
    PyTorch isn't installed. ``model`` can be the small ``MicroByteLM`` demo
    or the full ``NueronceModel`` (the real, ported production architecture);
    both satisfy the same duck-typed interface. See
    ``nueronce.training.sft.TorchSFTBackend`` for the PyTorch/``NUERONCEModel``
    counterpart."""

    model: object
    lr: float = 5e-3

    def train(self, dataset: Sequence, **kwargs) -> List[Dict[str, float]]:
        kwargs.setdefault("lr", self.lr)
        return train_dialogue_sft(self.model, dataset, **kwargs)


__all__ = ["MicroByteLM", "train_overfit", "train_dialogue_sft", "MicroSFTBackend"]
