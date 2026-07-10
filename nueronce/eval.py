"""Held-out evaluation harness: train models on one region, measure bits/byte on
a disjoint held-out region.

This exists because reporting *training-corpus* loss as model quality is
meaningless (the reviewer's point #7). Bits/byte on never-seen bytes is the
honest signal. ``compare`` trains a set of matched models and reports both train
and held-out bits/byte so memorization vs generalization is visible.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch
from torch import Tensor

LN2 = math.log(2.0)


@torch.no_grad()
def bits_per_byte(model, batches: List[Tensor]) -> float:
    model.eval()
    losses = [model.lm_loss(b).item() for b in batches]
    return float(sum(losses) / max(1, len(losses)) / LN2)


def train_model(model, train_batches: List[Tensor], lr: float = 3e-3,
                clip: float = 1.0) -> List[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve = []
    model.train()
    for batch in train_batches:
        loss = model.lm_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        curve.append(loss.item() / LN2)
    return curve


def compare(model_factories: Dict[str, Callable], train_batches: List[Tensor],
            val_batches: List[Tensor], lr: float = 3e-3) -> Dict[str, dict]:
    """Train each model on train_batches, report train+held-out bits/byte.

    ``model_factories``: name -> zero-arg callable returning a fresh model that
    exposes ``lm_loss`` and ``num_params``. Same batches/optimizer/steps for all.
    """
    results: Dict[str, dict] = {}
    for name, make in model_factories.items():
        torch.manual_seed(0)  # identical init RNG stream per model for fairness
        model = make()
        curve = train_model(model, train_batches, lr=lr)
        results[name] = {
            "params": model.num_params(),
            "final_train_bpb": curve[-1] if curve else float("nan"),
            "heldout_bpb": bits_per_byte(model, val_batches),
        }
    return results


__all__ = ["bits_per_byte", "train_model", "compare", "LN2"]
