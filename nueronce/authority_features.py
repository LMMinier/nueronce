"""Deterministic feature extraction for the source/authority classifier.

Pure NumPy — extracted from ``authority_clf`` so both classifier backends (the
canonical torch MLP in :mod:`nueronce.authority_clf` and the from-scratch
engine mirror in :mod:`nueronce.engine.authority_clf`) consume the *exact
same* features: hashed character n-grams of the raw text + provenance
metadata (channel, signature flag, domain cue, hashed document type).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .authority_data import AUTHORITY_CLASSES, CHANNELS

CLASS_TO_IDX = {c: i for i, c in enumerate(AUTHORITY_CLASSES)}
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


__all__ = ["featurize", "FEAT_DIM", "TEXT_DIM", "CLASS_TO_IDX",
           "_stable_hash", "_text_vec", "_domain_cue"]
