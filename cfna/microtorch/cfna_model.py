"""MicroCFNAModel — the real, production CFNA architecture running entirely on
the from-scratch microtorch engine (NumPy only, no PyTorch anywhere in the
call graph).

This is a faithful port of ``cfna.model.CFNAModel`` — same subsystems, same
composition, same causal shift conventions:

    bytes -> causal byte perception (CNN) + boundary head -> dynamic patching
    -> unit embedding -> typed recurrent memory -> hybrid core (SSM + local
    attention + sparse global attention + retrieval) -> byte decoder ->
    next-byte logits

not a smaller stand-in like ``cfna.microtorch.models.MicroByteLM`` (which
predates this port and stays around for the simpler engine demo /
``scripts/microtorch_demo.py``). Training objective and interface match
``CFNAModel``: next-byte cross-entropy + auxiliary boundary loss, plus
``masked_token_loss``/``lm_loss`` for retrieval and SFT-style training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import functional as F
from . import segment
from .cfna_blocks import (
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
class MicroModelConfig:
    """Mirrors ``cfna.model.ModelConfig`` field-for-field."""

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


def _softmax_np(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


class MicroCFNAModel(Module):
    def __init__(self, cfg: Optional[MicroModelConfig] = None):
        self.cfg = cfg or MicroModelConfig()
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

    # ------------------------------------------------------------------ #
    def encode_retrieval(self, neighbor_ids: np.ndarray, neighbor_mask: Optional[np.ndarray] = None):
        """Encode retrieved neighbor byte chunks into a context the core and
        the decoder can cross-attend. Byte-level (not pooled) so exact
        identifiers/values stay copyable, same induction-style copy setup as
        ``CFNAModel.encode_retrieval``."""
        b, m, lc = neighbor_ids.shape
        flat = neighbor_ids.reshape(b * m, lc)
        feats, _ = self.perception(flat)                              # [B*M, Lc, d_local]
        zeros_col = Tensor(np.zeros((feats.shape[0], 1, feats.shape[2])))
        shifted = cat([zeros_col, feats[:, :-1]], axis=1)              # left context, excludes byte p
        bemb = self.ret_byte_embed(flat)                               # [B*M, Lc, ret_byte_dim]
        ctx = self.ret_proj(cat([shifted, bemb], axis=-1)).reshape(b, m * lc, -1)
        if neighbor_mask is None:
            ctx_mask = np.ones((b, m * lc), dtype=bool)
        else:
            ctx_mask = np.asarray(neighbor_mask).reshape(b, m * lc).astype(bool)
        return ctx, ctx_mask

    def encode_units(self, byte_ids: np.ndarray, ret_ctx: Optional[Tensor] = None,
                     ret_mask: Optional[np.ndarray] = None):
        """Run perception -> patching -> units -> memory -> core."""
        c = self.cfg
        feats, boundary_logits = self.perception(byte_ids)
        # Discrete segmentation structure from detached (hard) probabilities.
        prob_hard = 1.0 / (1.0 + np.exp(-boundary_logits.data))
        seg_ids, _ = segment.segment_ids_from_boundaries(
            prob_hard, tau=c.tau, min_patch=c.min_patch, max_patch=c.max_patch, p_max=c.p_max
        )
        m, unit_mask = segment.pool_matrix(seg_ids, c.p_max)           # [B,P,T], [B,P]
        pooled_local = Tensor(m) @ feats                               # [B,P,d_local]
        units = self.unit_embed(pooled_local)
        if c.trainable_segmentation:
            # A *differentiable* per-unit boundary feature carries the LM
            # gradient back to the boundary head (m itself is detached/fixed).
            prob = F.sigmoid(boundary_logits)                           # [B,T], grad-enabled
            unit_boundary = Tensor(m) @ prob.reshape(prob.shape[0], prob.shape[1], 1)  # [B,P,1]
            units = units + self.boundary_proj(unit_boundary)
        units = units + self.memory(units)
        core_ret_mask = None
        if ret_ctx is not None:
            core_ret_mask = np.broadcast_to(ret_mask[:, None, :], (ret_mask.shape[0], c.p_max, ret_mask.shape[-1]))
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
        dec_ret_mask = (np.broadcast_to(ret_mask[:, None, :], (ret_mask.shape[0], t, ret_mask.shape[-1]))
                        if ret_ctx is not None else None)
        logits = self.decoder(byte_ids, g, cross_mask, ret_ctx=ret_ctx, ret_mask=dec_ret_mask)
        return logits, boundary_logits

    # ------------------------------------------------------------------ #
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
        """Next-byte cross-entropy only. bits/byte = lm_loss / ln 2."""
        byte_ids = np.asarray(byte_ids)
        logits, _ = self.forward(byte_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1))

    def masked_token_loss(self, logits: Tensor, byte_ids: np.ndarray, target_mask: np.ndarray) -> Tensor:
        """Cross-entropy only at positions flagged in ``target_mask`` (True at
        the *target* byte). logits[t] predicts byte[t+1], so target position v
        uses logits[v-1] — same convention as ``CFNAModel.masked_token_loss``."""
        byte_ids, target_mask = np.asarray(byte_ids), np.asarray(target_mask)
        pred = logits[:, :-1].reshape(-1, 256)
        tgt = byte_ids[:, 1:].reshape(-1)
        sel = target_mask[:, 1:].reshape(-1)
        return F.masked_cross_entropy(pred, tgt, sel)

    def masked_loss(self, byte_ids: np.ndarray, target_mask: np.ndarray) -> Tensor:
        """``forward`` + ``masked_token_loss`` in one call — the one-shot
        interface :mod:`cfna.microtorch.models`' ``train_dialogue_sft``/
        ``MicroSFTBackend`` expect, so the *real* ported architecture is a
        drop-in alternative to the smaller ``MicroByteLM`` for VGRFT stage 1."""
        logits, _ = self.forward(byte_ids)
        return self.masked_token_loss(logits, byte_ids, target_mask)

    # ------------------------------------------------------------------ #
    def generate(self, prompt: bytes, max_new: int = 64, temperature: float = 0.8,
                 greedy: bool = False, max_ctx: int = 256) -> bytes:
        ids = list(prompt) or [32]
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
        return bytes(ids)

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())


__all__ = ["MicroModelConfig", "MicroCFNAModel"]
