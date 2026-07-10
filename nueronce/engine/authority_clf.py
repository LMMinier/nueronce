"""Source/authority classifier on the from-scratch Nueronce Engine.

API-identical mirror of the canonical torch implementation in
:mod:`nueronce.authority_clf` (same features via :mod:`nueronce.authority_features`,
same 2-layer MLP shape, same AdamW recipe, same temperature-scaled
calibration + abstention), so the provenance/authority benchmarks and tests
run on machines without PyTorch. ``nueronce.authority_clf`` aliases to this class
automatically when torch is missing.

Differences from the torch backend, stated honestly:
- Weight init follows engine's Linear (normal / sqrt(fan_in)) rather than
  torch's kaiming-uniform, so trained weights are not bit-identical across
  backends; the evals assert thresholds, not exact values.
- Temperature is fit by deterministic golden-section search on log T
  (minimizing held-out NLL) instead of LBFGS — same objective, same scalar.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..authority_data import AUTHORITY_CLASSES, Doc, UNTRUSTED
from ..authority_features import CLASS_TO_IDX, FEAT_DIM, featurize
from . import functional as F
from .nn import Linear, Module
from .optim import AdamW
from .tensor import Tensor, no_grad


class _Net(Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
        self.l1 = Linear(in_dim, hidden)
        self.l2 = Linear(hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(x).relu())


def _nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    logprob = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logprob[np.arange(len(y)), y].mean())


class AuthorityClassifier:
    def __init__(self, seed: int = 0):
        np.random.seed(seed)
        self.net = _Net(FEAT_DIM, len(AUTHORITY_CLASSES))
        self.temperature = 1.0

    # --- training ---------------------------------------------------------- #
    def fit(self, docs: List[Doc], steps: int = 400, lr: float = 5e-3, batch: int = 128,
            seed: int = 0) -> None:
        X = np.stack([featurize(d.features()) for d in docs]).astype(np.float64)
        y = np.array([CLASS_TO_IDX[d.label_authority] for d in docs], dtype=np.int64)
        opt = AdamW(list(self.net.parameters()), lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            idx = rng.integers(0, len(docs), size=min(batch, len(docs)))
            logits = self.net(Tensor(X[idx]))
            loss = F.cross_entropy(logits, y[idx])
            self.net.zero_grad()
            loss.backward()
            opt.step()

    def calibrate(self, docs: List[Doc]) -> float:
        """Temperature scaling: fit a scalar T minimizing NLL on held-out docs
        via golden-section search over log T in [-2, 2] (deterministic)."""
        with no_grad():
            logits = self.net(Tensor(
                np.stack([featurize(d.features()) for d in docs]).astype(np.float64))).data
        y = np.array([CLASS_TO_IDX[d.label_authority] for d in docs], dtype=np.int64)

        gr = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = -2.0, 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc, fd = _nll(logits, y, math.exp(c)), _nll(logits, y, math.exp(d))
        for _ in range(60):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = _nll(logits, y, math.exp(c))
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = _nll(logits, y, math.exp(d))
        self.temperature = float(math.exp((lo + hi) / 2.0))
        return self.temperature

    # --- inference --------------------------------------------------------- #
    def predict_proba(self, features: Dict) -> np.ndarray:
        with no_grad():
            x = Tensor(featurize(features).astype(np.float64)[None, :])
            logits = self.net(x).data[0] / self.temperature
        z = logits - logits.max()
        e = np.exp(z)
        return (e / e.sum()).astype(np.float64)

    def predict(self, features: Dict, abstain_below: float = 0.0
                ) -> Tuple[Optional[str], float]:
        p = self.predict_proba(features)
        idx = int(p.argmax())
        conf = float(p[idx])
        if conf < abstain_below:
            return None, conf
        return AUTHORITY_CLASSES[idx], conf

    def predict_trusted(self, features: Dict) -> bool:
        label, _ = self.predict(features)
        return label is not None and label not in UNTRUSTED


__all__ = ["AuthorityClassifier"]
