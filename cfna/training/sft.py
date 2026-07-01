"""Byte-level supervised instruction tuning (VGRFT stage 1, made real) for the
real, PyTorch-backed :class:`~cfna.model.CFNAModel`.

``cfna.chat`` is honest that an untuned checkpoint gives "English-shaped
continuations, not an instruct assistant" — every checkpoint so far is trained
with pure next-byte cross-entropy on monologic prose (books, speeches), which
teaches what English looks like, not what to say back when spoken to.

This module supplies the missing training signal: it takes the small,
hand-written (prompt, response) dialogue set from
:mod:`cfna.training.dialogue_data` (the same ``User: `` / ``Assistant: ``
turn layout ``cfna.chat.Conversation`` uses at inference) and fine-tunes an
already-pretrained ``CFNAModel`` with the loss masked to the response bytes
only (so the model isn't asked to predict the user's turn, just to continue
it appropriately). Small on purpose: this is an SFT *pass* meant to run on
top of an existing byte-LM checkpoint, not a from-scratch pretraining corpus.

See :mod:`cfna.microtorch.models` for the same idea running on the from-scratch
(NumPy-only) autograd engine instead of PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..model import CFNAModel
from .dialogue_data import (
    BOT_TAG,
    SFT_DATASET,
    SFTExample,
    USER_TAG,
    encode_example,
    held_out_split,
    make_sft_batch as _make_sft_batch_np,
)


def make_sft_batch(examples: Sequence[SFTExample], **kwargs) -> Dict[str, torch.Tensor]:
    """``dialogue_data.make_sft_batch`` as torch tensors, ready for CFNAModel."""
    np_batch = _make_sft_batch_np(examples, **kwargs)
    return {
        "byte_ids": torch.from_numpy(np_batch["byte_ids"]),
        "target_mask": torch.from_numpy(np_batch["target_mask"]),
    }


def sft_step(model: CFNAModel, opt: torch.optim.Optimizer, batch: Dict[str, torch.Tensor]) -> float:
    model.train()
    logits, _ = model(batch["byte_ids"])
    loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


@torch.no_grad()
def sft_eval(model: CFNAModel, batch: Dict[str, torch.Tensor]) -> float:
    model.eval()
    logits, _ = model(batch["byte_ids"])
    return float(model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"]).item())


def train_sft(model: CFNAModel, opt: torch.optim.Optimizer, examples: Sequence[SFTExample],
              *, steps: int = 300, batch_size: int = 8,
              val_examples: Optional[Sequence[SFTExample]] = None,
              system: str = "", user_tag: str = USER_TAG, bot_tag: str = BOT_TAG,
              seed: int = 0, log_every: int = 25) -> List[Dict[str, float]]:
    """Fine-tune ``model`` in place on (prompt, response) turns, masking the
    loss to response bytes only. Mirrors ``cfna.retrieval_train.train_retrieval``:
    a real, small, runnable training loop, not a stub."""
    rng = np.random.default_rng(seed)
    val_batch = (make_sft_batch(val_examples, system=system, user_tag=user_tag, bot_tag=bot_tag)
                 if val_examples else None)
    history: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(examples), size=min(batch_size, len(examples)))
        batch = make_sft_batch([examples[i] for i in idx], system=system,
                                user_tag=user_tag, bot_tag=bot_tag)
        loss = sft_step(model, opt, batch)
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "train_loss": loss}
            if val_batch is not None:
                rec["val_loss"] = sft_eval(model, val_batch)
            history.append(rec)
    return history


@dataclass
class TorchSFTBackend:
    """``VGRFTTrainer`` backend for stage 1 (supervised instruction tuning):
    a ``CFNAModel`` + optimizer pair, trained with PyTorch autograd. Stages 2-4
    (tool grounding, verifier training, residual experts) need tool traces /
    verifier ground truth that don't exist yet, so this backend does not
    attempt them."""

    model: CFNAModel
    optimizer: torch.optim.Optimizer

    def train(self, dataset: Sequence[SFTExample], **kwargs) -> List[Dict[str, float]]:
        return train_sft(self.model, self.optimizer, dataset, **kwargs)


__all__ = [
    "SFTExample", "SFT_DATASET", "USER_TAG", "BOT_TAG",
    "encode_example", "make_sft_batch", "held_out_split",
    "sft_step", "sft_eval", "train_sft", "TorchSFTBackend",
]
