"""Global workspace: the latent reasoning substrate.

Slots hold hypotheses, supporting/contradicting evidence, unknowns, candidate
actions, constraint monitors, numerical checks, and calibration/uncertainty. The
workspace is initialized from the task + retrieval + aggregated core state, then
iterated (cross-slot message passing, hypothesis generation, consequence testing,
pruning) before a reasoning result is extracted.

Slot seeding and the per-iteration neural operations need a backend; the slot
bookkeeping and role assignment are real.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ._backend import needs_backend
from .config import WorkspaceConfig
from .types import MemoryRecord, TaskState, WorkspaceSlot

DEFAULT_ROLES = [
    "problem_repr",
    "user_goal",
    "known_facts",
    "unknowns",
    "hypothesis_a",
    "hypothesis_b",
    "counterexample",
    "constraint_monitor",
    "numerical_check",
    "uncertainty",
    "response_structure",
    "tool_need",
]


class GlobalWorkspace:
    def __init__(self, cfg: Optional[WorkspaceConfig] = None, d_model: int = 1536):
        self.cfg = cfg or WorkspaceConfig()
        self.d_model = d_model
        self.slots: List[WorkspaceSlot] = [
            WorkspaceSlot(
                slot_id=i,
                role="free",
                latent_state=np.zeros((d_model,), dtype=np.float32),
            )
            for i in range(self.cfg.n_slots)
        ]

    def initialize(
        self,
        task: TaskState,
        retrieval_items: List[MemoryRecord],
        core_state: Any,
    ) -> None:
        # Role assignment is deterministic; latent seeding needs a backend.
        for i, role in enumerate(DEFAULT_ROLES):
            if i >= len(self.slots):
                break
            self.slots[i].role = role
            self.slots[i].status = "active"
        self._seed_latents(task, retrieval_items, core_state)

    def _seed_latents(self, task, retrieval_items, core_state) -> None:
        raise needs_backend(
            "GlobalWorkspace._seed_latents",
            "seed_slot(role, task, retrieval_items, core_state) -> latent vector.",
        )

    def iterate(self, verifier_requests: Optional[list] = None) -> None:
        raise needs_backend(
            "GlobalWorkspace.iterate",
            "cross-slot message passing -> generate hypotheses -> test "
            "consequences -> prune unsupported -> apply verifier requests.",
        )

    def extract_reasoning_result(self) -> dict:
        raise needs_backend(
            "GlobalWorkspace.extract_reasoning_result",
            "Select best hypothesis, list alternatives/unknowns, infer tool "
            "needs, aggregate confidence from the slots.",
        )


__all__ = ["GlobalWorkspace", "DEFAULT_ROLES"]
