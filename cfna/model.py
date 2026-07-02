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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _prompt_to_bytes(prompt: bytes | bytearray | str | Sequence[int]) -> bytes:
    if isinstance(prompt, bytes):
        return prompt
    if isinstance(prompt, bytearray):
        return bytes(prompt)
    if isinstance(prompt, str):
        return prompt.encode("utf-8")
    return bytes(int(x) & 0xFF for x in prompt)


def _retrieval_context_to_tensors(
    retrieval_context: object,
    *,
    device: torch.device,
    max_neighbors: int = 4,
    max_len: int = 160,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if retrieval_context is None:
        return None, None
    if isinstance(retrieval_context, (str, bytes, bytearray)):
        items: Iterable[object] = [retrieval_context]
    else:
        items = retrieval_context if isinstance(retrieval_context, Iterable) else [retrieval_context]
    chunks: List[bytes] = []
    for item in items:
        if len(chunks) >= max_neighbors:
            break
        content = getattr(item, "content", None) or getattr(item, "raw_text", None) or item
        b = _prompt_to_bytes(content)[:max_len]
        if b:
            chunks.append(b)
    if not chunks:
        return None, None
    width = max(len(c) for c in chunks)
    ids = torch.zeros((1, len(chunks), width), dtype=torch.long, device=device)
    mask = torch.zeros((1, len(chunks), width), dtype=torch.bool, device=device)
    for i, chunk in enumerate(chunks):
        vals = torch.tensor(list(chunk), dtype=torch.long, device=device)
        ids[0, i, : len(chunk)] = vals
        mask[0, i, : len(chunk)] = True
    return ids, mask


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
    trainable_segmentation: bool = True   # let LM loss reach the boundary head


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
        # Differentiable boundary feature: the mean boundary probability inside a
        # unit is projected and added to that unit's representation, so the
        # next-byte loss flows gradient into the boundary head (the discrete
        # segmentation structure stays straight-through / detached).
        self.boundary_proj = Linear(1, c.d_model)
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
        # Discrete segmentation structure from a detached (hard) decision ...
        prob_hard = torch.sigmoid(boundary_logits.detach())
        seg_ids, _ = segment_ids_from_boundaries(
            prob_hard, tau=c.tau, min_patch=c.min_patch, max_patch=c.max_patch, p_max=c.p_max
        )
        m, unit_mask = pool_matrix(seg_ids, c.p_max)        # [B,P,T], [B,P]
        pooled_local = m @ feats                            # [B,P,d_local]
        units = self.unit_embed(pooled_local)
        if c.trainable_segmentation:
            # ... but a *differentiable* per-unit boundary feature carries the LM
            # gradient back to the boundary head (m is detached structure).
            prob = torch.sigmoid(boundary_logits)           # [B,T] (grad-enabled)
            unit_boundary = m.detach() @ prob[..., None]     # [B,P,1] mean prob per unit
            units = units + self.boundary_proj(unit_boundary)
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

    def lm_loss(self, byte_ids: Tensor) -> Tensor:
        """Next-byte cross-entropy only (no auxiliary terms). Common eval interface
        shared with the baselines; bits/byte = lm_loss / ln 2."""
        logits, _ = self.forward(byte_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1))

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
    def generate(
        self,
        prompt: bytes | bytearray | str | Sequence[int],
        *,
        neighbor_ids: Optional[Tensor] = None,
        neighbor_mask: Optional[Tensor] = None,
        retrieval_context: object = None,
        max_new: int = 256,
        max_ctx: Optional[int] = None,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[Sequence[bytes | str]] = None,
        greedy: bool = False,
        continuation_only: bool = True,
        return_scores: bool = False,
    ):
        """Autoregressively continue ``prompt`` with optional retrieval context.

        By default this returns only newly generated bytes. Set
        ``continuation_only=False`` for the historical prompt+continuation
        behavior. Retrieval tensors are forwarded through every decoding step.
        """
        self.eval()
        device = next(self.parameters()).device
        prompt_bytes = _prompt_to_bytes(prompt)
        ids = list(prompt_bytes) or [32]
        generated: List[int] = []
        max_ctx = int(max_ctx or 256)
        max_new = max(0, int(max_new))
        temperature = float(temperature)
        repetition_penalty = max(1e-6, float(repetition_penalty))

        if neighbor_ids is None and retrieval_context is not None:
            neighbor_ids, neighbor_mask = _retrieval_context_to_tensors(
                retrieval_context, device=device
            )
        elif neighbor_ids is not None:
            neighbor_ids = neighbor_ids.to(device=device, dtype=torch.long)
            neighbor_mask = (neighbor_mask.to(device=device, dtype=torch.bool)
                             if neighbor_mask is not None else None)

        stops = []
        for seq in stop_sequences or ():
            b = seq.encode("utf-8") if isinstance(seq, str) else bytes(seq)
            if b:
                stops.append(b)

        logprobs: List[float] = []
        entropies: List[float] = []
        stopped_len = 0
        stopped_sequence: Optional[bytes] = None

        for _ in range(max_new):
            ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
            logits, _ = self.forward(ctx, neighbor_ids=neighbor_ids, neighbor_mask=neighbor_mask)
            nxt = torch.nan_to_num(logits[0, -1].float(), nan=-1e9, posinf=1e9, neginf=-1e9)

            if repetition_penalty != 1.0:
                for token in set(generated or ids):
                    if nxt[token] < 0:
                        nxt[token] *= repetition_penalty
                    else:
                        nxt[token] /= repetition_penalty

            if greedy:
                probs = torch.softmax(torch.nan_to_num(nxt, nan=-1e9), dim=-1)
                idx = int(torch.argmax(nxt).item())
            else:
                scaled = nxt / max(temperature, 1e-6)
                if top_k is not None:
                    k = max(1, min(int(top_k), scaled.numel()))
                    keep = torch.topk(scaled, k).indices
                    masked = torch.full_like(scaled, -float("inf"))
                    masked[keep] = scaled[keep]
                    scaled = masked
                probs = torch.softmax(torch.nan_to_num(scaled, nan=-float("inf")), dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                if top_p is not None:
                    p = float(top_p)
                    if not (0.0 < p <= 1.0):
                        raise ValueError("top_p must be in (0, 1]")
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumulative <= p
                    keep[0] = True
                    filtered = torch.zeros_like(probs)
                    filtered[sorted_idx[keep]] = sorted_probs[keep]
                    probs = filtered
                total = probs.sum()
                if not torch.isfinite(total) or float(total.item()) <= 0.0:
                    idx = int(torch.argmax(nxt).item())
                    probs = torch.softmax(torch.nan_to_num(nxt, nan=-1e9), dim=-1)
                else:
                    probs = probs / total
                    idx = int(torch.multinomial(probs, 1).item())

            prob = torch.clamp(probs[idx], min=1e-12)
            logprobs.append(float(torch.log(prob).item()))
            entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum()
            entropies.append(float(entropy.item()))

            ids.append(idx)
            generated.append(idx)
            current = bytes(generated)
            hit = next((s for s in stops if current.endswith(s)), None)
            if hit is not None:
                stopped_len = len(hit)
                stopped_sequence = hit
                break

        if stopped_len:
            generated = generated[:-stopped_len]
        out = bytes(generated) if continuation_only else bytes(ids if prompt_bytes else ids[1:])
        if not return_scores:
            return out
        return {
            "bytes": out,
            "logprobs": logprobs[: len(generated)],
            "avg_logprob": (sum(logprobs[: len(generated)]) / len(generated)) if generated else 0.0,
            "entropy": entropies[: len(generated)],
            "avg_entropy": (sum(entropies[: len(generated)]) / len(generated)) if generated else 0.0,
            "stopped": stopped_sequence is not None,
            "stop_sequence": stopped_sequence.decode("utf-8", errors="replace") if stopped_sequence else None,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def large_config() -> ModelConfig:
    """A ~337M-parameter configuration realizing the design's ~350M budget.

    This makes the 350M target a *constructable, counted* model rather than the
    bookkeeping-only ``cfna.config.CFNAConfig``. It is not trained here (that needs
    real data and compute); ``test_scaling`` verifies it builds at the right scale.
    """
    return ModelConfig(
        byte_embed_dim=128, d_local=512, d_model=1024, p_max=64, physical_blocks=6,
        logical_depth=12, n_heads=8, unit_window=256, decoder_window=256,
        decoder_layers=6, d_state=16, channel_dim=64, ret_byte_dim=64,
        min_patch=4, max_patch=128,
    )


__all__ = ["ModelConfig", "CFNAModel", "large_config"]
