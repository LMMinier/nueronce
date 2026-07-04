"""CPU-first bounded-memory training for CFNA.

This is an experimental alternative to end-to-end full-model backpropagation.
It trains one subsystem at a time with local predictive objectives over short
streaming byte windows. Frozen stages run under ``torch.no_grad()``, so their
activations are not retained for backward. Only the active subsystem and its
small local head receive gradients.

The design targets low peak RAM, not maximum throughput:

    bytes -> short window -> local objective -> update active block -> discard

It does not claim equivalence to global backprop. Its purpose is to make the
original CFNA thesis testable: useful learning with bounded activation memory,
CPU execution, sparse/blockwise updates, and no VRAM requirement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model import CFNAModel
from ..segment import boundary_targets, byte_to_unit_mask, pool_matrix, segment_ids_from_boundaries

STAGES = ("perception", "units", "core", "decoder")


@dataclass
class CPUStreamingConfig:
    chunk_size: int = 128
    batch_size: int = 1
    steps_per_stage: int = 32
    learning_rates: Mapping[str, float] = field(default_factory=lambda: {
        "perception": 3e-3, "units": 2e-3, "core": 1e-3, "decoder": 2e-3,
    })
    boundary_weight: float = 0.2
    grad_clip: float = 1.0
    min_bytes: int = 8
    seed: int = 7


class LocalPredictionHeads(nn.Module):
    """Small disposable/portable heads that provide local credit assignment."""
    def __init__(self, model: CFNAModel):
        super().__init__()
        c = model.cfg
        self.perception = nn.Linear(c.d_local, 256)
        self.units = nn.Linear(c.d_model, 256)
        self.core = nn.Linear(c.d_model, 256)


class CPUStreamingTrainer:
    """Block-coordinate trainer whose peak activation graph is one short window.

    Each call updates exactly one subsystem:
      * perception: next-byte + boundary prediction from local byte features
      * units: next-byte distribution per learned patch
      * core: next-byte distribution after memory/routed core processing
      * decoder: normal next-byte loss with the upstream stack detached

    SGD is intentionally used by default: unlike AdamW it keeps no two extra
    optimizer tensors per parameter. Optimizers are rebuilt on stage changes so
    only active parameters are referenced.
    """
    def __init__(self, model: CFNAModel, cfg: CPUStreamingConfig | None = None):
        self.cfg = cfg or CPUStreamingConfig()
        if self.cfg.chunk_size < self.cfg.min_bytes:
            raise ValueError("chunk_size must be >= min_bytes")
        torch.manual_seed(self.cfg.seed)
        self.model = model.cpu()
        self.heads = LocalPredictionHeads(model).cpu()
        self.step_index = 0
        self.stage_index = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self._activate(STAGES[0])

    @property
    def stage(self) -> str:
        return STAGES[self.stage_index]

    def _all_modules(self) -> Sequence[nn.Module]:
        return (self.model, self.heads)

    def _activate(self, stage: str) -> None:
        for module in self._all_modules():
            for p in module.parameters():
                p.requires_grad_(False)
        groups = {
            "perception": (self.model.perception, self.heads.perception),
            "units": (self.model.unit_embed, self.model.boundary_proj,
                      self.model.memory, self.heads.units),
            "core": (self.model.core, self.heads.core),
            "decoder": (self.model.decoder,),
        }[stage]
        params: List[nn.Parameter] = []
        for module in groups:
            module.train()
            for p in module.parameters():
                p.requires_grad_(True)
                params.append(p)
        self.optimizer = torch.optim.SGD(params, lr=float(self.cfg.learning_rates[stage]))

    def _advance_stage_if_needed(self) -> None:
        if self.step_index and self.step_index % self.cfg.steps_per_stage == 0:
            self.stage_index = (self.stage_index + 1) % len(STAGES)
            self._activate(self.stage)

    def _tensorize(self, chunks: Sequence[bytes | bytearray | str]) -> Tensor:
        rows: List[List[int]] = []
        for chunk in chunks[: self.cfg.batch_size]:
            b = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
            b = b[: self.cfg.chunk_size]
            if len(b) < self.cfg.min_bytes:
                continue
            rows.append(list(b))
        if not rows:
            raise ValueError("batch contains no chunk long enough to train")
        width = min(len(r) for r in rows)
        return torch.tensor([r[:width] for r in rows], dtype=torch.long)

    def _segment(self, byte_ids: Tensor, feats: Tensor, boundary_logits: Tensor):
        c = self.model.cfg
        seg_ids, _ = segment_ids_from_boundaries(
            torch.sigmoid(boundary_logits.detach()), tau=c.tau,
            min_patch=c.min_patch, max_patch=c.max_patch, p_max=c.p_max,
        )
        matrix, unit_mask = pool_matrix(seg_ids, c.p_max)
        pooled = matrix @ feats
        return seg_ids, matrix, unit_mask, pooled

    @staticmethod
    def _unit_targets(byte_ids: Tensor, matrix: Tensor) -> Tensor:
        shifted = torch.cat([byte_ids[:, 1:], byte_ids[:, -1:]], dim=1)
        one_hot = F.one_hot(shifted, num_classes=256).float()
        return torch.argmax(matrix @ one_hot, dim=-1)

    def _perception_loss(self, ids: Tensor):
        feats, boundary_logits = self.model.perception(ids)
        logits = self.heads.perception(feats[:, :-1])
        lm = F.cross_entropy(logits.reshape(-1, 256), ids[:, 1:].reshape(-1))
        bnd = F.binary_cross_entropy_with_logits(
            boundary_logits, boundary_targets(ids, self.model._syntax)
        )
        return lm + self.cfg.boundary_weight * bnd, {"local_lm": lm, "boundary": bnd}

    def _units_loss(self, ids: Tensor):
        with torch.no_grad():
            feats, boundary_logits = self.model.perception(ids)
            _, matrix, unit_mask, pooled = self._segment(ids, feats, boundary_logits)
            targets = self._unit_targets(ids, matrix)
            boundary_prob = torch.sigmoid(boundary_logits)
            unit_boundary = matrix @ boundary_prob[..., None]
        units = self.model.unit_embed(pooled)
        if self.model.cfg.trainable_segmentation:
            units = units + self.model.boundary_proj(unit_boundary)
        units = units + self.model.memory(units)
        logits = self.heads.units(units)
        return F.cross_entropy(logits[unit_mask], targets[unit_mask]), {}

    def _core_loss(self, ids: Tensor):
        with torch.no_grad():
            feats, boundary_logits = self.model.perception(ids)
            _, matrix, unit_mask, pooled = self._segment(ids, feats, boundary_logits)
            targets = self._unit_targets(ids, matrix)
            units = self.model.unit_embed(pooled)
            if self.model.cfg.trainable_segmentation:
                prob = torch.sigmoid(boundary_logits)
                units = units + self.model.boundary_proj(matrix @ prob[..., None])
            units = units + self.model.memory(units)
        core = self.model.core(units, self.model.cfg.logical_depth, key_padding=unit_mask)
        logits = self.heads.core(core)
        return F.cross_entropy(logits[unit_mask], targets[unit_mask]), {}

    def _decoder_loss(self, ids: Tensor):
        with torch.no_grad():
            g, cross_mask, _, _, _ = self.model.encode_units(ids)
        logits = self.model.decoder(ids, g, cross_mask)
        return F.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)), {}

    def train_step(self, chunks: Sequence[bytes | bytearray | str]) -> Dict[str, float]:
        self._advance_stage_if_needed()
        ids = self._tensorize(chunks)
        loss_fn = {
            "perception": self._perception_loss,
            "units": self._units_loss,
            "core": self._core_loss,
            "decoder": self._decoder_loss,
        }[self.stage]
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        loss, extras = loss_fn(ids)
        loss.backward()
        active = [p for group in self.optimizer.param_groups for p in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(active, self.cfg.grad_clip)
        self.optimizer.step()
        self.step_index += 1
        result = {"step": float(self.step_index), "stage": self.stage,
                  "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                  "bytes": float(ids.numel())}
        result.update({k: float(v.detach()) for k, v in extras.items()})
        return result

    def train_stream(self, chunks: Iterable[bytes], max_steps: int) -> Iterator[Dict[str, float]]:
        batch: List[bytes] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) < self.cfg.batch_size:
                continue
            yield self.train_step(batch)
            batch.clear()
            if self.step_index >= max_steps:
                return

    def memory_contract(self) -> Dict[str, int | str]:
        active = [p for p in self.model.parameters() if p.requires_grad]
        active += [p for p in self.heads.parameters() if p.requires_grad]
        active_params = sum(p.numel() for p in active)
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "device": "cpu", "stage": self.stage,
            "chunk_size": self.cfg.chunk_size,
            "active_parameters": active_params,
            "total_model_parameters": total_params,
            "estimated_gradient_bytes_fp32": active_params * 4,
            "optimizer_state_bytes": 0,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "heads": self.heads.state_dict(),
                    "trainer_config": asdict(self.cfg), "step": self.step_index,
                    "stage_index": self.stage_index}, path)


def stream_file_chunks(paths: Sequence[str | Path], chunk_size: int,
                       overlap: int = 1) -> Iterator[bytes]:
    """Read files incrementally; never load the corpus into memory."""
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    stride = chunk_size - overlap
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("rb") as handle:
            carry = b""
            while True:
                block = handle.read(stride)
                if not block:
                    break
                chunk = carry + block
                if len(chunk) >= 2:
                    yield chunk[:chunk_size]
                carry = chunk[-overlap:] if overlap else b""


__all__ = ["CPUStreamingConfig", "CPUStreamingTrainer", "LocalPredictionHeads",
           "STAGES", "stream_file_chunks"]
