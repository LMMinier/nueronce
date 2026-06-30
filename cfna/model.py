"""CFNAModel — the end-to-end, trainable realization of the architecture.

A two-level byte language model that exercises the full CFNA pipeline:

    bytes
      → causal byte perception (CNN) + learned boundary head
      → dynamic patching into variable-length information units
      → unit embedding
      → typed multi-timescale recurrent memory (added to units)
      → hybrid cognitive fabric (SSM + local/sparse-global attention), reused
        over logical depth
      → byte decoder that cross-attends to *completed* units (causal)
      → next-byte logits

Training objective = next-byte cross-entropy + an auxiliary boundary loss, so the
patcher is genuinely learned. Everything is causal, so this is a valid
autoregressive model whose loss decreasing demonstrates the wiring is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import (
    BytePerceptionEncoder,
    ByteDecoder,
    HybridCoreStack,
    TypedRecurrentMemory,
    UnitEmbedder,
)
from .nn import Linear
from .segment import (
    boundary_targets,
    byte_to_unit_mask,
    pool_matrix,
    segment_ids_from_boundaries,
    syntax_table,
)


@dataclass
class ModelConfig:
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


class CFNAModel(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
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
        from .nn import Embedding
        self.ret_byte_embed = Embedding(256, c.ret_byte_dim)
        self.ret_proj = Linear(c.d_local + c.ret_byte_dim, c.d_model)  # retrieved -> model dim
        self.register_buffer("_syntax", syntax_table(), persistent=False)

    # ------------------------------------------------------------------ #
    def encode_retrieval(self, neighbor_ids: Tensor, neighbor_mask: Optional[Tensor] = None):
        """Encode retrieved neighbor byte chunks into a context the core and the
        decoder can cross-attend.

        neighbor_ids:  [B, M, Lc]  (M neighbors per example, Lc bytes each)
        neighbor_mask: [B, M, Lc]  (True = real byte)  -> returns
            ctx [B, M*Lc, d_model], ctx_mask [B, M*Lc]
        Byte-level (not pooled) so exact identifiers/values stay copyable —
        retrieval as exact, content-addressed access.
        """
        b, m, lc = neighbor_ids.shape
        flat = neighbor_ids.reshape(b * m, lc)
        feats, _ = self.perception(flat)                             # causal [B*M, Lc, d_local]
        # Decouple key vs value: position p exposes its LEFT context (feats[p-1],
        # excluding byte p) for matching, and an embedding of byte p as the
        # copyable content. This makes retrieval an induction-style copy.
        shifted = torch.zeros_like(feats)
        shifted[:, 1:] = feats[:, :-1]
        bemb = self.ret_byte_embed(flat)                            # [B*M, Lc, ret_byte_dim]
        ctx = self.ret_proj(torch.cat([shifted, bemb], dim=-1)).reshape(b, m * lc, -1)
        if neighbor_mask is None:
            ctx_mask = ctx.new_ones(b, m * lc, dtype=torch.bool)
        else:
            ctx_mask = neighbor_mask.reshape(b, m * lc).bool()
        return ctx, ctx_mask

    def encode_units(self, byte_ids: Tensor, ret_ctx: Optional[Tensor] = None,
                     ret_mask: Optional[Tensor] = None):
        """Run perception → patching → units → memory → core. Returns the
        contextual unit states + structures needed by the decoder."""
        c = self.cfg
        feats, boundary_logits = self.perception(byte_ids)
        prob = torch.sigmoid(boundary_logits.detach())
        seg_ids, _ = segment_ids_from_boundaries(
            prob, tau=c.tau, min_patch=c.min_patch, max_patch=c.max_patch, p_max=c.p_max
        )
        m, unit_mask = pool_matrix(seg_ids, c.p_max)        # [B,P,T], [B,P]
        pooled_local = m @ feats                            # [B,P,d_local]
        units = self.unit_embed(pooled_local)
        units = units + self.memory(units)                 # typed-memory conditioning
        core_ret_mask = None
        if ret_ctx is not None:                            # units may attend all retrieved ctx
            core_ret_mask = ret_mask[:, None, :].expand(-1, c.p_max, -1)
        g = self.core(units, c.logical_depth, key_padding=unit_mask,
                      retrieval_ctx=ret_ctx, retrieval_mask=core_ret_mask)
        cross_mask = byte_to_unit_mask(seg_ids, unit_mask, c.p_max)
        return g, cross_mask, boundary_logits, seg_ids, unit_mask

    def forward(self, byte_ids: Tensor, neighbor_ids: Optional[Tensor] = None,
                neighbor_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        ret_ctx = ret_mask = None
        if neighbor_ids is not None:
            ret_ctx, ret_mask = self.encode_retrieval(neighbor_ids, neighbor_mask)
        g, cross_mask, boundary_logits, _, _ = self.encode_units(byte_ids, ret_ctx, ret_mask)
        t = byte_ids.shape[1]
        dec_ret_mask = ret_mask[:, None, :].expand(-1, t, -1) if ret_ctx is not None else None
        logits = self.decoder(byte_ids, g, cross_mask, ret_ctx=ret_ctx, ret_mask=dec_ret_mask)
        return logits, boundary_logits

    # ------------------------------------------------------------------ #
    def loss(self, byte_ids: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        logits, boundary_logits = self.forward(byte_ids)
        lm = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1)
        )
        b_target = boundary_targets(byte_ids, self._syntax)
        bnd = F.binary_cross_entropy_with_logits(boundary_logits, b_target)
        total = lm + self.cfg.boundary_loss_weight * bnd
        stats = {"loss": total.detach().item(), "lm": lm.detach().item(),
                 "boundary": bnd.detach().item(),
                 "bpb": lm.detach().item() / 0.6931471805599453}
        return total, stats

    def masked_token_loss(self, logits: Tensor, byte_ids: Tensor, target_mask: Tensor) -> Tensor:
        """Cross-entropy only at positions flagged in ``target_mask`` (True at the
        *target* byte). logits[t] predicts byte[t+1], so target position v uses
        logits[v-1]. Used to isolate the retrieval-dependent tokens."""
        pred = logits[:, :-1].reshape(-1, 256)
        tgt = byte_ids[:, 1:].reshape(-1)
        sel = target_mask[:, 1:].reshape(-1).float()
        ce = F.cross_entropy(pred, tgt, reduction="none")
        return (ce * sel).sum() / sel.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt: bytes, max_new: int = 64, temperature: float = 0.8,
                 greedy: bool = False, max_ctx: int = 256) -> bytes:
        self.eval()
        device = next(self.parameters()).device
        ids = list(prompt) or [32]
        for _ in range(max_new):
            ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
            logits, _ = self.forward(ctx)
            nxt = logits[0, -1]
            if greedy:
                idx = int(nxt.argmax())
            else:
                probs = torch.softmax(nxt / max(1e-5, temperature), dim=-1)
                idx = int(torch.multinomial(probs, 1))
            ids.append(idx)
        return bytes(ids)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["ModelConfig", "CFNAModel"]
