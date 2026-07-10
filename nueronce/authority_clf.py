"""A small, from-scratch source/authority classifier (Phase 3, first learned module).

Features are deterministic (hashed character n-grams of the raw text + provenance
metadata: channel, signature, domain cue, document type) — shared with the
engine backend via :mod:`nueronce.authority_features`. The model is a tiny MLP
trained with plain AdamW over CPU tensors — consistent with the repo rule that
torch is only a tensor/autograd substrate, no prebuilt NLP stack. Includes
temperature-scaled calibration and confidence-based abstention.

When PyTorch is not installed, ``AuthorityClassifier`` aliases to the
API-identical from-scratch implementation in
:mod:`nueronce.engine.authority_clf`, so every provenance/authority benchmark
runs on NumPy alone.

Subsystem label: **REAL / TRAINABLE**.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .authority_data import AUTHORITY_CLASSES, Doc, UNTRUSTED
from .authority_features import (  # re-exported for backward compatibility
    CLASS_TO_IDX as _CLASS_TO_IDX,
    FEAT_DIM,
    TEXT_DIM,
    featurize,
)

try:
    import torch
    _HAVE_TORCH = True
except ModuleNotFoundError:
    _HAVE_TORCH = False


if _HAVE_TORCH:

    class _Net(torch.nn.Module):
        def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
            super().__init__()
            self.l1 = torch.nn.Linear(in_dim, hidden)
            self.l2 = torch.nn.Linear(hidden, n_classes)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    class AuthorityClassifier:
        def __init__(self, seed: int = 0):
            torch.manual_seed(seed)
            self.net = _Net(FEAT_DIM, len(AUTHORITY_CLASSES))
            self.temperature = 1.0

        # --- training ------------------------------------------------------ #
        def fit(self, docs: List[Doc], steps: int = 400, lr: float = 5e-3, batch: int = 128,
                seed: int = 0) -> None:
            X = torch.from_numpy(np.stack([featurize(d.features()) for d in docs]))
            y = torch.tensor([_CLASS_TO_IDX[d.label_authority] for d in docs], dtype=torch.long)
            opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
            lossf = torch.nn.CrossEntropyLoss()
            rng = np.random.default_rng(seed)
            self.net.train()
            for _ in range(steps):
                idx = rng.integers(0, len(docs), size=min(batch, len(docs)))
                bi = torch.from_numpy(idx)
                opt.zero_grad()
                loss = lossf(self.net(X[bi]), y[bi])
                loss.backward()
                opt.step()

        def calibrate(self, docs: List[Doc]) -> float:
            """Temperature scaling: fit a scalar T minimizing NLL on held-out docs."""
            with torch.no_grad():
                logits = self.net(torch.from_numpy(
                    np.stack([featurize(d.features()) for d in docs])))
            y = torch.tensor([_CLASS_TO_IDX[d.label_authority] for d in docs], dtype=torch.long)
            logT = torch.zeros(1, requires_grad=True)
            opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
            lossf = torch.nn.CrossEntropyLoss()

            def closure():
                opt.zero_grad()
                loss = lossf(logits / logT.exp(), y)
                loss.backward()
                return loss
            opt.step(closure)
            self.temperature = float(logT.exp().item())
            return self.temperature

        # --- inference ------------------------------------------------------ #
        @torch.no_grad()
        def predict_proba(self, features: Dict) -> np.ndarray:
            self.net.eval()
            x = torch.from_numpy(featurize(features)).unsqueeze(0)
            logits = self.net(x) / self.temperature
            return torch.softmax(logits, dim=-1)[0].numpy()

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

else:
    from .engine.authority_clf import AuthorityClassifier  # noqa: F401


__all__ = ["AuthorityClassifier", "featurize", "FEAT_DIM", "TEXT_DIM"]
