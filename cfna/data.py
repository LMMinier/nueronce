"""Tiny byte-level corpus + batching for the CFNA training demo.

The corpus is small and structured on purpose: the point of the demo is to show
the hand-built architecture *learns* (loss falls far below the uniform-byte
baseline and the model can continue the text), not to train a general model.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

# A small, self-contained corpus drawn from the project's own design vocabulary.
CORPUS = (
    "CFNA separates understanding, thinking, remembering, and speaking. "
    "Perception forms dynamic information units from raw bytes. "
    "Typed memory keeps semantic, evidence, and authority channels apart. "
    "The hybrid core mixes a selective state space path with local attention. "
    "A planner decides what to say before the renderer decides how to word it. "
    "An independent verifier checks claims against evidence and revises. "
    "Retrieval is dense, sparse, and late interaction together. "
    "Provenance and authority are first class, not an afterthought. "
)


def corpus_bytes(repeat: int = 8) -> bytes:
    return (CORPUS * repeat).encode("utf-8")


def make_batches(data: bytes, seq_len: int, batch_size: int, n_batches: int,
                 seed: int = 0, device=None) -> List[Tensor]:
    """Random contiguous windows of length ``seq_len`` as byte-id tensors."""
    g = torch.Generator().manual_seed(seed)
    buf = torch.tensor(list(data), dtype=torch.long)
    hi = len(buf) - seq_len - 1
    if hi <= 0:
        raise ValueError("corpus too short for seq_len; increase repeat or shorten seq_len")
    batches = []
    for _ in range(n_batches):
        starts = torch.randint(0, hi, (batch_size,), generator=g)
        rows = torch.stack([buf[s : s + seq_len] for s in starts])
        batches.append(rows.to(device) if device else rows)
    return batches


UNIFORM_BYTE_BPB = 8.0  # log2(256): the no-skill baseline in bits/byte


__all__ = ["CORPUS", "corpus_bytes", "make_batches", "UNIFORM_BYTE_BPB"]
