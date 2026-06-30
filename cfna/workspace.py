"""Global workspace: the latent reasoning substrate — real implementation.

Slots hold hypotheses, supporting/contradicting evidence, unknowns, constraint
monitors, numerical checks, and calibration/uncertainty. Initialization seeds slot
latents from the task + retrieval + aggregated core state (with a per-role
embedding); ``iterate`` runs cross-slot message passing (slot self-attention) plus
a gated update; ``extract_reasoning_result`` reads confidence/selection heads.

Built from the hand-rolled primitives in :mod:`cfna.nn`. Heads are randomly
initialized (trained jointly in a full run) but fully runnable today.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .config import WorkspaceConfig
from .types import MemoryRecord, TaskState, WorkspaceSlot

DEFAULT_ROLES = [
    "problem_repr", "user_goal", "known_facts", "unknowns",
    "hypothesis_a", "hypothesis_b", "counterexample", "constraint_monitor",
    "numerical_check", "uncertainty", "response_structure", "tool_need",
]


class GlobalWorkspace:
    def __init__(self, cfg: Optional[WorkspaceConfig] = None, d_model: int = 128):
        import torch
        from torch import nn

        from .nn import MLP, RMSNorm, CrossAttention

        self.cfg = cfg or WorkspaceConfig()
        self.d_model = d_model
        self._t = torch
        n = self.cfg.n_slots
        self.slots: List[WorkspaceSlot] = [
            WorkspaceSlot(slot_id=i, role="free", latent_state=np.zeros((d_model,), np.float32))
            for i in range(n)
        ]
        # real parameters (frozen here; jointly trained in a full run)
        self.role_embed = nn.Parameter(torch.randn(n, d_model) * 0.02, requires_grad=False)
        self.message = CrossAttention(d_model, n_heads=4)   # slot<-slot message passing
        self.update = MLP(d_model, 2 * d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.confidence_head = MLP(d_model, d_model, 1)
        for mod in (self.message, self.update, self.norm, self.confidence_head):
            for p in mod.parameters():
                p.requires_grad_(False)
        self._H = None  # [1, n, d] latent tensor

    # ------------------------------------------------------------------ #
    def initialize(self, task: TaskState, retrieval_items: List[MemoryRecord], core_state: Any) -> None:
        torch = self._t
        n = self.cfg.n_slots
        for i, role in enumerate(DEFAULT_ROLES):
            if i >= n:
                break
            self.slots[i].role = role
            self.slots[i].status = "active"
        seed = _to_vec(core_state, self.d_model)
        ret = _retrieval_summary(retrieval_items, self.d_model)
        base = torch.tensor(seed + ret)[None, None, :]      # [1,1,d]
        self._H = base + self.role_embed[None]              # [1,n,d]
        self._sync_slots()

    def iterate(self, verifier_requests: Optional[list] = None) -> None:
        torch = self._t
        if self._H is None:
            raise RuntimeError("call initialize() before iterate()")
        with torch.no_grad():
            h = self.norm(self._H)
            msg = self.message(h, h)                        # cross-slot message passing
            self._H = self._H + msg
            self._H = self._H + self.update(self.norm(self._H))
        self._sync_slots()

    def extract_reasoning_result(self) -> dict:
        torch = self._t
        with torch.no_grad():
            conf = torch.sigmoid(self.confidence_head(self._H)).squeeze(-1).squeeze(0)  # [n]
        conf_np = conf.numpy()
        hyp_idx = [i for i, s in enumerate(self.slots) if s.role.startswith("hypothesis")]
        best = max(hyp_idx, key=lambda i: conf_np[i]) if hyp_idx else int(conf_np.argmax())
        unknown_idx = [i for i, s in enumerate(self.slots) if s.role == "unknowns"]
        return {
            "best_hypothesis": self.slots[best].role,
            "best_slot_id": int(best),
            "alternatives": [self.slots[i].role for i in hyp_idx if i != best],
            "unknowns": [self.slots[i].role for i in unknown_idx],
            "required_tools": [],
            "confidence": float(conf_np[best]),
            "slot_confidence": conf_np.tolist(),
        }

    # ------------------------------------------------------------------ #
    def _sync_slots(self) -> None:
        h = self._H[0].detach().numpy()
        for i, s in enumerate(self.slots):
            s.latent_state = h[i]
            s.confidence = float(1.0 / (1.0 + np.exp(-h[i].mean())))


def _to_vec(x: Any, dim: int) -> np.ndarray:
    if x is None:
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(getattr(x, "numpy", lambda: x)(), dtype=np.float32).ravel() \
        if hasattr(x, "numpy") else np.asarray(x, dtype=np.float32).ravel()
    if arr.shape[0] >= dim:
        return arr[:dim]
    return np.pad(arr, (0, dim - arr.shape[0]))


def _retrieval_summary(items: List[MemoryRecord], dim: int) -> np.ndarray:
    if not items:
        return np.zeros(dim, dtype=np.float32)
    vecs = []
    for it in items:
        emb = getattr(it, "embeddings", {}) or {}
        ds = emb.get("dense_semantic")
        if ds is not None:
            vecs.append(_to_vec(ds, dim))
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


__all__ = ["GlobalWorkspace", "DEFAULT_ROLES"]
