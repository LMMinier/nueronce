"""NueronceModel — the real, production NUERONCE architecture running entirely on
the from-scratch Nueronce Engine (NumPy only, no PyTorch anywhere in the
call graph).

This is a faithful port of ``nueronce.model.NUERONCEModel`` — same subsystems, same
composition, same causal shift conventions:

    bytes -> causal byte perception (CNN) + boundary head -> dynamic patching
    -> unit embedding -> typed recurrent memory -> hybrid core (SSM + local
    attention + sparse global attention + retrieval) -> byte decoder ->
    next-byte logits

not a smaller stand-in like ``nueronce.engine.models.MicroByteLM`` (which
predates this port and stays around for the simpler engine demo /
``scripts/nueronce_engine_demo.py``). Training objective and interface match
``NUERONCEModel``: next-byte cross-entropy + auxiliary boundary loss, plus
``masked_token_loss``/``lm_loss`` for retrieval and SFT-style training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from . import functional as F
from . import segment
from .checkpoint import checkpoint
from .nueronce_blocks import (
    ByteDecoder,
    BytePerceptionEncoder,
    HybridCoreStack,
    TypedRecurrentMemory,
    UnitEmbedder,
)
from .nn import Embedding, Linear, Module
from .tensor import Tensor, cat, no_grad

LN2 = 0.6931471805599453


@dataclass
class NueronceConfig:
    """Mirrors ``nueronce.model.ModelConfig`` field-for-field."""

    byte_embed_dim: int = 48
    d_local: int = 96
    d_model: int = 128
    p_max: int = 32
    physical_blocks: int = 2
    logical_depth: int = 4
    n_heads: int = 4
    unit_window: int = 24
    decoder_window: int = 32
    decoder_layers: int = 2
    d_state: int = 16
    channel_dim: int = 24
    ret_byte_dim: int = 32
    tau: float = 0.5
    min_patch: int = 3
    max_patch: int = 24
    boundary_loss_weight: float = 0.3
    trainable_segmentation: bool = True
    activation_checkpointing: bool = False


def _softmax_np(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


class NueronceModel(Module):
    def __init__(self, cfg: Optional[NueronceConfig] = None):
        self.cfg = cfg or NueronceConfig()
        c = self.cfg
        self.perception = BytePerceptionEncoder(c.byte_embed_dim, c.d_local)
        self.unit_embed = UnitEmbedder(c.d_local, c.d_model)
        self.memory = TypedRecurrentMemory(c.d_model, channel_dim=c.channel_dim)
        self.core = HybridCoreStack(
            c.d_model, physical_blocks=c.physical_blocks, n_heads=c.n_heads,
            local_window=c.unit_window, sparse_topk=max(4, c.p_max // 2), d_state=c.d_state,
        )
        self.decoder = ByteDecoder(c.d_model, n_layers=c.decoder_layers,
                                   n_heads=c.n_heads, window=c.decoder_window)
        self.ret_byte_embed = Embedding(256, c.ret_byte_dim)
        self.ret_proj = Linear(c.d_local + c.ret_byte_dim, c.d_model)
        self.boundary_proj = Linear(1, c.d_model)
        self._syntax = segment.syntax_table()

    def encode_retrieval(self, neighbor_ids: np.ndarray, neighbor_mask: Optional[np.ndarray] = None):
        b, m, lc = neighbor_ids.shape
        flat = neighbor_ids.reshape(b * m, lc)
        feats, _ = self.perception(flat)
        zeros_col = Tensor(np.zeros((feats.shape[0], 1, feats.shape[2])))
        shifted = cat([zeros_col, feats[:, :-1]], axis=1)
        bemb = self.ret_byte_embed(flat)
        joined = cat([shifted, bemb], axis=-1)
        ctx = (checkpoint(self.ret_proj, joined, parameters=self.ret_proj.parameters(), name="ret_proj")
               if self.cfg.activation_checkpointing else self.ret_proj(joined))
        ctx = ctx.reshape(b, m * lc, -1)
        if neighbor_mask is None:
            ctx_mask = np.ones((b, m * lc), dtype=bool)
        else:
            ctx_mask = np.asarray(neighbor_mask).reshape(b, m * lc).astype(bool)
        return ctx, ctx_mask

    def encode_units(self, byte_ids: np.ndarray, ret_ctx: Optional[Tensor] = None,
                     ret_mask: Optional[np.ndarray] = None):
        c = self.cfg
        feats, boundary_logits = self.perception(byte_ids)
        prob_hard = 1.0 / (1.0 + np.exp(-boundary_logits.data))
        seg_ids, _ = segment.segment_ids_from_boundaries(
            prob_hard, tau=c.tau, min_patch=c.min_patch, max_patch=c.max_patch, p_max=c.p_max
        )
        m, unit_mask = segment.pool_matrix(seg_ids, c.p_max)
        pooled_local = Tensor(m) @ feats
        units = (checkpoint(self.unit_embed, pooled_local,
                            parameters=self.unit_embed.parameters(), name="unit_embed")
                 if c.activation_checkpointing else self.unit_embed(pooled_local))
        if c.trainable_segmentation:
            prob = F.sigmoid(boundary_logits)
            unit_boundary = Tensor(m) @ prob.reshape(prob.shape[0], prob.shape[1], 1)
            boundary_delta = (checkpoint(self.boundary_proj, unit_boundary,
                                         parameters=self.boundary_proj.parameters(), name="boundary_proj")
                              if c.activation_checkpointing else self.boundary_proj(unit_boundary))
            units = units + boundary_delta
        memory_delta = (checkpoint(self.memory, units,
                                   parameters=self.memory.parameters(), name="memory")
                        if c.activation_checkpointing else self.memory(units))
        units = units + memory_delta
        core_ret_mask = None
        if ret_ctx is not None:
            core_ret_mask = np.broadcast_to(ret_mask[:, None, :],
                                            (ret_mask.shape[0], c.p_max, ret_mask.shape[-1]))
        if c.activation_checkpointing:
            if ret_ctx is None:
                g = checkpoint(
                    lambda x: self.core(x, c.logical_depth, key_padding=unit_mask,
                                        retrieval_ctx=None, retrieval_mask=None),
                    units, parameters=self.core.parameters(), name="core",
                )
            else:
                g = checkpoint(
                    lambda x, r: self.core(x, c.logical_depth, key_padding=unit_mask,
                                           retrieval_ctx=r, retrieval_mask=core_ret_mask),
                    units, ret_ctx, parameters=self.core.parameters(), name="core",
                )
        else:
            g = self.core(units, c.logical_depth, key_padding=unit_mask,
                          retrieval_ctx=ret_ctx, retrieval_mask=core_ret_mask)
        cross_mask = segment.byte_to_unit_mask(seg_ids, unit_mask, c.p_max)
        return g, cross_mask, boundary_logits, seg_ids, unit_mask

    def forward(self, byte_ids: np.ndarray, neighbor_ids: Optional[np.ndarray] = None,
                neighbor_mask: Optional[np.ndarray] = None) -> Tuple[Tensor, Tensor]:
        byte_ids = np.asarray(byte_ids)
        ret_ctx = ret_mask = None
        if neighbor_ids is not None:
            ret_ctx, ret_mask = self.encode_retrieval(np.asarray(neighbor_ids), neighbor_mask)
        g, cross_mask, boundary_logits, _, _ = self.encode_units(byte_ids, ret_ctx, ret_mask)
        t = byte_ids.shape[1]
        dec_ret_mask = (np.broadcast_to(ret_mask[:, None, :],
                                        (ret_mask.shape[0], t, ret_mask.shape[-1]))
                        if ret_ctx is not None else None)
        if self.cfg.activation_checkpointing:
            if ret_ctx is None:
                logits = checkpoint(
                    lambda x: self.decoder(byte_ids, x, cross_mask, ret_ctx=None, ret_mask=None),
                    g, parameters=self.decoder.parameters(), name="decoder",
                )
            else:
                logits = checkpoint(
                    lambda x, r: self.decoder(byte_ids, x, cross_mask,
                                              ret_ctx=r, ret_mask=dec_ret_mask),
                    g, ret_ctx, parameters=self.decoder.parameters(), name="decoder",
                )
        else:
            logits = self.decoder(byte_ids, g, cross_mask,
                                  ret_ctx=ret_ctx, ret_mask=dec_ret_mask)
        return logits, boundary_logits

    def loss(self, byte_ids: np.ndarray) -> Tuple[Tensor, Dict[str, float]]:
        byte_ids = np.asarray(byte_ids)
        logits, boundary_logits = self.forward(byte_ids)
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1))
        b_target = segment.boundary_targets(byte_ids, self._syntax)
        bnd = F.binary_cross_entropy_with_logits(boundary_logits, b_target)
        total = lm + bnd * self.cfg.boundary_loss_weight
        stats = {"loss": total.item(), "lm": lm.item(), "boundary": bnd.item(), "bpb": lm.item() / LN2}
        return total, stats

    def lm_loss(self, byte_ids: np.ndarray) -> Tensor:
        byte_ids = np.asarray(byte_ids)
        logits, _ = self.forward(byte_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1))

    def masked_token_loss(self, logits: Tensor, byte_ids: np.ndarray, target_mask: np.ndarray) -> Tensor:
        byte_ids, target_mask = np.asarray(byte_ids), np.asarray(target_mask)
        pred = logits[:, :-1].reshape(-1, 256)
        tgt = byte_ids[:, 1:].reshape(-1)
        sel = target_mask[:, 1:].reshape(-1)
        return F.masked_cross_entropy(pred, tgt, sel)

    def masked_loss(self, byte_ids: np.ndarray, target_mask: np.ndarray) -> Tensor:
        logits, _ = self.forward(byte_ids)
        return self.masked_token_loss(logits, byte_ids, target_mask)

    def generate(self, prompt: bytes, max_new: int = 64, temperature: float = 0.8,
                 greedy: bool = False, max_ctx: int = 256,
                 stop_bytes: bytes = b"", min_new: int = 0) -> bytes:
        ids = list(prompt) or [32]
        new_count = 0
        with no_grad():
            for _ in range(max_new):
                ctx = np.array([ids[-max_ctx:]])
                logits, _ = self.forward(ctx)
                nxt = logits.data[0, -1]
                if greedy:
                    idx = int(nxt.argmax())
                else:
                    probs = _softmax_np(nxt / max(1e-5, temperature))
                    idx = int(np.random.choice(256, p=probs))
                ids.append(idx)
                new_count += 1
                if stop_bytes and new_count >= min_new and idx in stop_bytes:
                    break
        return bytes(ids)

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())


def large_config() -> NueronceConfig:
    return NueronceConfig(
        byte_embed_dim=128, d_local=512, d_model=1024, p_max=64, physical_blocks=6,
        logical_depth=12, n_heads=8, unit_window=256, decoder_window=256,
        decoder_layers=6, d_state=16, channel_dim=64, ret_byte_dim=64,
        min_patch=4, max_patch=128, activation_checkpointing=True,
    )


def preset_configs() -> dict:
    presets = {
        "chat_11m": dict(byte_embed_dim=64, d_local=128, d_model=256, p_max=48,
                          physical_blocks=3, logical_depth=4, n_heads=8, unit_window=48,
                          decoder_window=64, decoder_layers=3, d_state=16, channel_dim=24,
                          ret_byte_dim=32, min_patch=3, max_patch=24, boundary_loss_weight=0.2),
        "base_35m": dict(byte_embed_dim=96, d_local=208, d_model=416, p_max=56,
                          physical_blocks=4, logical_depth=6, n_heads=8, unit_window=64,
                          decoder_window=96, decoder_layers=3, d_state=16, channel_dim=32,
                          ret_byte_dim=48, min_patch=3, max_patch=32, boundary_loss_weight=0.2),
        "base_90m": dict(byte_embed_dim=112, d_local=288, d_model=608, p_max=64,
                          physical_blocks=5, logical_depth=8, n_heads=8, unit_window=80,
                          decoder_window=112, decoder_layers=4, d_state=16, channel_dim=40,
                          ret_byte_dim=56, min_patch=3, max_patch=40, boundary_loss_weight=0.2),
    }
    return {name: NueronceConfig(**kw) for name, kw in presets.items()}


__all__ = ["NueronceConfig", "NueronceModel", "large_config", "preset_configs"]
