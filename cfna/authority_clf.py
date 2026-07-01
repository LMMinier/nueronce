"""A small, from-scratch source/authority classifier (Phase 3, first learned module).

Features are deterministic (hashed character n-grams of the raw text + provenance
metadata: channel, signature, domain cue, document type). The model is a tiny MLP
trained with plain SGD/AdamW over CPU tensors — consistent with the repo rule that
torch is only a tensor/autograd substrate, no prebuilt NLP stack. Includes
temperature-scaled calibration and confidence-based abstention.

Subsystem label: **REAL / TRAINABLE**.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .authority_data import AUTHORITY_CLASSES, CHANNELS, Doc, UNTRUSTED

_CLASS_TO_IDX = {c: i for i, c in enumerate(AUTHORITY_CLASSES)}
_CHAN_TO_IDX = {c: i for i, c in enumerate(CHANNELS)}
TEXT_DIM = 2048
_CAT_DIM = len(CHANNELS) + 1 + 3 + 32   # channel + signed + domain-cue + doc_type hash
FEAT_DIM = TEXT_DIM + _CAT_DIM


def _stable_hash(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = (h * 16777619 + ord(ch)) & 0xFFFFFFFF
    return h


def _text_vec(text: str) -> np.ndarray:
    v = np.zeros(TEXT_DIM, dtype=np.float32)
    t = f"^{text.lower()}$"
    for n in (3, 4, 5):
        for i in range(len(t) - n + 1):
            v[_stable_hash(t[i:i + n]) % TEXT_DIM] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _domain_cue(domain: str) -> np.ndarray:
    d = domain.lower()
    return np.array([float(d.endswith(".gov") or "courts" in d),
                     float(d.endswith(".com")),
                     float("example" in d or d.endswith(".example"))], dtype=np.float32)


def featurize(features: Dict) -> np.ndarray:
    text = features.get("text", "")
    vec = np.zeros(FEAT_DIM, dtype=np.float32)
    vec[:TEXT_DIM] = _text_vec(text)
    off = TEXT_DIM
    ch = features.get("channel", "unknown")
    vec[off + _CHAN_TO_IDX.get(ch, _CHAN_TO_IDX["unknown"])] = 1.0
    off += len(CHANNELS)
    vec[off] = 1.0 if features.get("signed") else 0.0
    off += 1
    vec[off:off + 3] = _domain_cue(features.get("domain", ""))
    off += 3
    vec[off + _stable_hash(features.get("doc_type", "")) % 32] = 1.0
    return vec


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

    # --- training ---------------------------------------------------------- #
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

    # --- inference --------------------------------------------------------- #
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


__all__ = ["AuthorityClassifier", "featurize", "FEAT_DIM", "TEXT_DIM"]
