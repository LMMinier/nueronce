"""Decomposed CPU-first training runtime for Nueronce Engine NUERONCE.

The runtime separates model structure, execution planning, tensor residency,
optimizer state and evaluation. Heavy single-output NUERONCE stages are now
activation checkpoints: their forward internals are discarded and rebuilt
locally when the output gradient reaches that stage. Optimizer state remains
block-paged so logical depth no longer implies whole-model graph residency.
"""
from __future__ import annotations

import gc
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .tensor import Tensor


class Residency(str, Enum):
    KEEP = "keep"
    RELEASE = "release"
    RECOMPUTE = "recompute"
    MEMMAP = "memmap"
    REVERSIBLE = "reversible"


@dataclass
class TrainableBlock:
    name: str
    module: object
    checkpoint_policy: Residency = Residency.RECOMPUTE
    update_every: int = 1

    def parameters(self) -> List[Tensor]:
        return list(self.module.parameters())

    @property
    def parameter_count(self) -> int:
        return sum(p.data.size for p in self.parameters())


@dataclass
class ExecutionPlan:
    blocks: List[TrainableBlock]
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_nueronce(cls, model) -> "ExecutionPlan":
        names = [
            "perception", "unit_embed", "memory", "core", "decoder",
            "ret_byte_embed", "ret_proj", "boundary_proj",
        ]
        checkpointed = {"unit_embed", "memory", "core", "decoder", "ret_proj", "boundary_proj"}
        blocks = []
        for name in names:
            module = getattr(model, name, None)
            if module is not None:
                policy = Residency.RECOMPUTE if name in checkpointed else Residency.KEEP
                blocks.append(TrainableBlock(name, module, policy))
        return cls(blocks, {
            "runtime": "nueronce-decomposed-v2",
            "tape": "activation-recompute/local-stage-backward",
        })

    def validate(self, model) -> None:
        planned = {id(p) for b in self.blocks for p in b.parameters()}
        actual = {id(p) for p in model.parameters()}
        missing = actual - planned
        duplicate_count = sum(len(b.parameters()) for b in self.blocks) - len(planned)
        if missing or duplicate_count:
            raise ValueError(
                f"invalid execution plan: missing={len(missing)} duplicates={duplicate_count}"
            )


class ActivationRecomputeTape:
    """Seeds the outer graph; checkpoint nodes perform each local replay."""

    mode = "activation-recompute/local-stage-backward"

    def backward(self, loss: Tensor) -> None:
        loss.backward()

    def clear(self) -> None:
        gc.collect()


# Backward-compatible import name for callers created during v1.
CompatibilityTape = ActivationRecomputeTape


class BlockStateManager:
    """Disk-backed optimizer state with only one block resident at a time."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, block_name: str) -> Path:
        return self.root / f"{block_name}.pkl"

    def load(self, block_name: str):
        path = self._path(block_name)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def save(self, block_name: str, state) -> None:
        path = self._path(block_name)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def manifest(self, plan: ExecutionPlan) -> None:
        payload = {
            "version": 2,
            "metadata": plan.metadata,
            "blocks": [
                {
                    "name": b.name,
                    "params": b.parameter_count,
                    "policy": b.checkpoint_policy.value,
                    "update_every": b.update_every,
                }
                for b in plan.blocks
            ],
        }
        (self.root / "manifest.json").write_text(json.dumps(payload, indent=2))


class BlockStreamFactor:
    """Adafactor-style block optimizer whose state is externally pageable."""

    def __init__(
        self,
        lr=1e-3,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        clip_threshold=1.0,
        tile_rows=128,
    ):
        self.lr, self.beta2, self.eps = lr, beta2, eps
        self.weight_decay, self.clip_threshold = weight_decay, clip_threshold
        self.tile_rows = tile_rows

    def init_state(self, params: Iterable[Tensor]):
        state = {"step": 0, "v": []}
        for p in params:
            if p.data.ndim >= 2:
                rows = p.data.shape[0]
                cols = int(np.prod(p.data.shape[1:]))
                state["v"].append(
                    (np.zeros(rows, np.float32), np.zeros(cols, np.float32))
                )
            else:
                state["v"].append(np.zeros_like(p.data, dtype=np.float32))
        return state

    def step(self, params: List[Tensor], state):
        state["step"] += 1
        correction = max(1.0 - self.beta2 ** state["step"], self.eps)
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            if self.weight_decay:
                p.data *= 1.0 - self.lr * self.weight_decay
            g = np.asarray(p.grad, dtype=np.float32)
            if p.data.ndim >= 2:
                w = p.data.reshape(p.data.shape[0], -1)
                gg = g.reshape(w.shape)
                vr, vc = state["v"][i]
                vr *= self.beta2
                vr += (1 - self.beta2) * np.mean(gg * gg, axis=1)
                vc *= self.beta2
                vc += (1 - self.beta2) * np.mean(gg * gg, axis=0)
                vrh, vch = vr / correction, vc / correction
                normalizer = max(float(vrh.mean()), self.eps)
                for start in range(0, w.shape[0], self.tile_rows):
                    stop = min(start + self.tile_rows, w.shape[0])
                    update = gg[start:stop] / (
                        np.sqrt(vrh[start:stop, None] * vch[None, :] / normalizer)
                        + self.eps
                    )
                    rms = float(np.sqrt(np.mean(update * update)))
                    if rms > self.clip_threshold:
                        update *= self.clip_threshold / (rms + self.eps)
                    w[start:stop] -= self.lr * update
            else:
                v = state["v"][i]
                v *= self.beta2
                v += (1 - self.beta2) * (g * g)
                update = g / (np.sqrt(v / correction) + self.eps)
                rms = float(np.sqrt(np.mean(update * update))) if update.size else 0.0
                if rms > self.clip_threshold:
                    update *= self.clip_threshold / (rms + self.eps)
                p.data -= self.lr * update
            p.grad = None
        return state


class IndependentEvaluator:
    def __init__(self, fn: Callable):
        self.fn = fn

    def run(self, model) -> Dict[str, float]:
        return dict(self.fn(model))


class DecomposedTrainer:
    def __init__(
        self,
        model,
        plan: ExecutionPlan,
        state_manager: BlockStateManager,
        optimizer: BlockStreamFactor,
        tape: Optional[ActivationRecomputeTape] = None,
    ):
        self.model, self.plan = model, plan
        self.state_manager, self.optimizer = state_manager, optimizer
        self.tape = tape or ActivationRecomputeTape()
        self.step_index = 0
        plan.validate(model)
        state_manager.manifest(plan)

    def train_step(self, loss_fn: Callable[[], Tensor]) -> Dict[str, float]:
        for p in self.model.parameters():
            p.grad = None
        loss = loss_fn()
        value = float(loss.item())
        self.tape.backward(loss)
        updated = 0
        for block in reversed(self.plan.blocks):
            if self.step_index % block.update_every:
                continue
            params = block.parameters()
            state = self.state_manager.load(block.name)
            if state is None:
                state = self.optimizer.init_state(params)
            state = self.optimizer.step(params, state)
            self.state_manager.save(block.name, state)
            updated += block.parameter_count
            del state
            gc.collect()
        self.tape.clear()
        self.step_index += 1
        return {
            "loss": value,
            "updated_parameters": float(updated),
            "step": float(self.step_index),
            "tape_mode": self.tape.mode,
        }


__all__ = [
    "Residency", "TrainableBlock", "ExecutionPlan", "ActivationRecomputeTape",
    "CompatibilityTape", "BlockStateManager", "BlockStreamFactor",
    "IndependentEvaluator", "DecomposedTrainer",
]
