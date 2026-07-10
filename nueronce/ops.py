"""Pure numeric / hashing utilities used across NUERONCE subsystems.

Everything here is fully implemented and backend-light: it depends only on the
standard library plus numpy. These are the building blocks the design doc refers
to with names like ``cosine``, ``sparse_dot``, ``mean_pool``, ``l2``,
``hash_ngram_features``, ``sha256_bytes`` and ``now_iso``.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, Iterable, Sequence

import numpy as np

# Bytes that tend to mark a syntactic boundary (punctuation / brackets / quotes
# / whitespace / newline). Used by the dynamic patcher.
SYNTAX_BYTES = frozenset(
    b" \t\n\r.,;:!?()[]{}<>\"'`/\\|=+-*&^%$#@~"
)

SPARSE_HASH_BITS = 20
SPARSE_HASH_SIZE = 1 << SPARSE_HASH_BITS  # 2**20 hashed lexical features


# --------------------------------------------------------------------------- #
# Hashing / identity
# --------------------------------------------------------------------------- #

def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of raw bytes (content-addressed identity for sources)."""
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Elementwise math
# --------------------------------------------------------------------------- #

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def softmax(x, axis: int = -1):
    a = np.asarray(x, dtype=np.float64)
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    a = np.asarray(x, dtype=np.float64)
    return 0.5 * a * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a ** 3)))


def l2(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def norm(x) -> float:
    """Scalar magnitude helper; tolerant of scalars and vectors."""
    a = np.asarray(x, dtype=np.float64)
    return float(np.linalg.norm(a)) if a.ndim else float(abs(a))


def mean_pool(x, axis: int = 0):
    return np.mean(np.asarray(x, dtype=np.float64), axis=axis)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def cosine(a, b) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Dot product of two sparse {feature_id: weight} maps."""
    if not a or not b:
        return 0.0
    # iterate over the smaller map
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b.get(k, 0.0) for k, w in a.items()))


# --------------------------------------------------------------------------- #
# Sparse lexical features
# --------------------------------------------------------------------------- #

def _hash_feature(token: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little") % SPARSE_HASH_SIZE


def hash_ngram_features(
    data: bytes,
    ngram_sizes: Sequence[int] = (3, 4, 5),
) -> Dict[int, float]:
    """Hashed character n-gram features for exact-identity / lexical matching.

    Returns a sparse {feature_id: count} map over ``SPARSE_HASH_SIZE`` buckets.
    This is the cheap, dependency-free stand-in for a learned sparse retriever
    (SPLADE-style) referenced in the design doc.
    """
    feats: Dict[int, float] = {}
    if not data:
        return feats
    for n in ngram_sizes:
        if len(data) < n:
            continue
        for i in range(len(data) - n + 1):
            fid = _hash_feature(data[i : i + n])
            feats[fid] = feats.get(fid, 0.0) + 1.0
    return feats


def l2_normalize_sparse(feats: Dict[int, float]) -> Dict[int, float]:
    mag = math.sqrt(sum(v * v for v in feats.values()))
    if mag == 0.0:
        return dict(feats)
    return {k: v / mag for k, v in feats.items()}


def merge_sparse(maps: Iterable[Dict[int, float]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for m in maps:
        for k, v in m.items():
            out[k] = out.get(k, 0.0) + v
    return out


__all__ = [
    "SYNTAX_BYTES",
    "SPARSE_HASH_BITS",
    "SPARSE_HASH_SIZE",
    "sha256_bytes",
    "now_iso",
    "sigmoid",
    "softmax",
    "gelu",
    "l2",
    "norm",
    "mean_pool",
    "cosine",
    "sparse_dot",
    "hash_ngram_features",
    "l2_normalize_sparse",
    "merge_sparse",
]
