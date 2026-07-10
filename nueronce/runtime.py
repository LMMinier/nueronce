"""Compact, low-VRAM runtime and the top-level NUERONCE_Respond orchestration.

The notes ask for a practical build path: avoid giant KV caches, reuse shared
recurrent blocks, use small task adapters (LoRA), and back memory with RAM/SSD.

- :class:`LoRAAdapter` is a real numpy implementation of low-rank adaptation of a
  frozen projection W: W' = W + (alpha/r) * B @ A.
- :class:`SSDBackedMemoryStore` is a thin, real key/value + index facade.
- :func:`nueronce_respond` documents the end-to-end control flow (perceive → patch →
  bundle → retrieve → core → workspace → plan → tools → render → verify →
  consolidate). The neural stages raise BackendNotConfigured until wired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .config import AdapterConfig


class LoRAAdapter:
    """Low-rank adapter for a frozen base projection W in R^{d_out x d_in}.

    Learns A in R^{r x d_in}, B in R^{d_out x r}; effective weight is
    W' = W + scale * (B @ A), with scale = alpha / r.
    """

    def __init__(self, d_in: int, d_out: int, cfg: Optional[AdapterConfig] = None, seed: int = 0):
        cfg = cfg or AdapterConfig()
        self.rank = cfg.lora_rank
        self.scale = cfg.lora_alpha / cfg.lora_rank
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((self.rank, d_in)).astype(np.float64) * 0.01
        self.B = np.zeros((d_out, self.rank), dtype=np.float64)  # zero-init: starts as identity delta

    def delta_weight(self) -> np.ndarray:
        return self.scale * (self.B @ self.A)

    def apply(self, x, W) -> np.ndarray:
        """y = x @ W^T + scale * (x @ A^T) @ B^T."""
        x = np.asarray(x, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        base = x @ W.T
        lora = (x @ self.A.T) @ self.B.T * self.scale
        return base + lora


class SSDBackedMemoryStore:
    """Keeps the memory corpus (and its indexes) off the training VRAM path."""

    def __init__(self, dense_index, sparse_index, doc_kv_store: Dict[str, Any]):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.doc_kv_store = doc_kv_store

    def read(self, ids: List[str]) -> List[Any]:
        return [self.doc_kv_store[i] for i in ids]

    def write(self, key: str, record: Any) -> None:
        self.doc_kv_store[key] = record


# The full streaming runtime (CompactRuntime) and nueronce_respond orchestration are
# documented in docs/architecture.md and the design doc's NUERONCE_Respond pseudocode.
# They compose the subsystem objects above; wiring them requires a neural backend
# for perception/embeddings/core/workspace/rendering.

__all__ = ["LoRAAdapter", "SSDBackedMemoryStore"]
