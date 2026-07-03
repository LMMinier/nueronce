"""State-cached incremental generation for the PyTorch ``CFNAModel`` — the
torch twin of :mod:`cfna.microtorch.incremental`, exploiting the same three
structural facts (everything causal; the decoder masks each byte away from
its own in-progress unit, so the unit stack is constant between patch
completions; the decoder has no absolute positions, so its stacked local
window bounds the bytes needed for exact last-position logits).

The neural core is untouched: this module only *calls* the model's existing
submodules on smaller tensors. Equivalence is a tested claim
(``tests/test_incremental_torch.py``, torch-gated): greedy output must be
byte-identical to ``CFNAModel.generate`` and per-step logits must match to
float tolerance. The NumPy-backend twin passes the same contract and measured
**17.0x** wall-clock speedup at a 256-byte prompt (scripts/bench_incremental.py,
this repository); the torch ratio must be re-measured on target hardware
before any speed claim is made for it.

Retrieval-conditioned generation is not supported; callers with retrieval
context should use the dense path (``cfna.chat`` falls back automatically).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .model import CFNAModel
from .segment import byte_to_unit_mask

_PERCEPTION_SPAN = 17  # conv3 (2) + conv7 (6) + dilated conv3 d=4 (8) + self


class IncrementalGenerator:
    """Incremental drop-in for ``CFNAModel.generate`` (no retrieval)."""

    def __init__(self, model: CFNAModel, margin: int = 8):
        self.model = model
        c = model.cfg
        self.device = next(model.parameters()).device
        self._per_span = _PERCEPTION_SPAN + margin
        self._dec_span = c.decoder_layers * c.decoder_window + margin

    # -------------------------------------------------------------- state
    @torch.no_grad()
    def prime(self, ids: List[int]) -> None:
        c = self.model.cfg
        self.ids = list(ids)
        t = torch.tensor([self.ids], dtype=torch.long, device=self.device)
        feats, blog = self.model.perception(t)
        f = feats[0]                                   # [T, d_local]
        prob = torch.sigmoid(blog[0])                  # [T]
        self.seg: List[int] = [0]
        self._cur, self._len = 0, 1
        self._done_feats: List[torch.Tensor] = []
        self._done_prob: List[float] = []
        self._acc_f, self._acc_p, self._acc_n = f[0].clone(), float(prob[0]), 1
        for i in range(1, len(self.ids)):
            self._advance_segmentation(f[i], float(prob[i]))
        self._rebuild_units()

    def _advance_segmentation(self, feat_i: torch.Tensor, prob_i: float) -> None:
        """Exact mirror of ``segment_ids_from_boundaries``'s per-step rule."""
        c = self.model.cfg
        cut = ((prob_i > c.tau) and self._len >= c.min_patch) or self._len >= c.max_patch
        if cut:
            if self._cur < c.p_max - 1:
                self._done_feats.append(self._acc_f / self._acc_n)
                self._done_prob.append(self._acc_p / self._acc_n)
                self._acc_f, self._acc_p, self._acc_n = feat_i.clone(), prob_i, 1
                self._cur += 1
                self._units_dirty = True
            else:  # p_max cap: byte merges into the last unit
                self._acc_f += feat_i
                self._acc_p += prob_i
                self._acc_n += 1
            self._len = 1
        else:
            self._acc_f += feat_i
            self._acc_p += prob_i
            self._acc_n += 1
            self._len += 1
        self.seg.append(self._cur)

    @torch.no_grad()
    def _rebuild_units(self) -> None:
        c = self.model.cfg
        pc = len(self._done_feats)
        pooled = torch.zeros(1, c.p_max, c.d_local, device=self.device)
        mean_p = torch.zeros(1, c.p_max, 1, device=self.device)
        if pc:
            pooled[0, :pc] = torch.stack(self._done_feats)
            mean_p[0, :pc, 0] = torch.tensor(self._done_prob, device=self.device)
        unit_mask = torch.zeros(1, c.p_max, dtype=torch.bool, device=self.device)
        unit_mask[0, :pc] = True
        units = self.model.unit_embed(pooled)
        if c.trainable_segmentation:
            units = units + self.model.boundary_proj(mean_p)
        units = units + self.model.memory(units)
        self._g = self.model.core(units, c.logical_depth, key_padding=unit_mask)
        self._unit_mask = unit_mask
        self._units_dirty = False

    # -------------------------------------------------------------- steps
    @torch.no_grad()
    def _last_logits(self) -> torch.Tensor:
        c = self.model.cfg
        if self._units_dirty:
            self._rebuild_units()
        span = min(len(self.ids), self._dec_span)
        win = torch.tensor([self.ids[-span:]], dtype=torch.long, device=self.device)
        seg_win = torch.tensor([self.seg[-span:]], dtype=torch.long, device=self.device)
        cross = byte_to_unit_mask(seg_win, self._unit_mask, c.p_max)
        logits = self.model.decoder(win, self._g, cross)
        return logits[0, -1]

    @torch.no_grad()
    def _append(self, byte_id: int) -> None:
        self.ids.append(byte_id)
        span = min(len(self.ids), self._per_span)
        t = torch.tensor([self.ids[-span:]], dtype=torch.long, device=self.device)
        feats, blog = self.model.perception(t)
        self._advance_segmentation(feats[0, -1], float(torch.sigmoid(blog[0, -1])))

    # ----------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, prompt, *, max_new: int = 256, max_ctx: Optional[int] = None,
                 temperature: float = 0.7, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, greedy: bool = False,
                 stop_sequences: Optional[Sequence[bytes]] = None,
                 continuation_only: bool = True) -> bytes:
        """Mirrors the dense ``CFNAModel.generate`` contract for the supported
        argument subset (no retrieval, no repetition penalty)."""
        self.model.eval()
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        ids = list(bytes(prompt)) or [32]
        stops = [s.encode("utf-8") if isinstance(s, str) else bytes(s)
                 for s in (stop_sequences or [])]
        max_ctx = max_ctx or self.model.cfg.p_max * self.model.cfg.max_patch
        out = list(ids)
        new: List[int] = []
        self.prime(out[-max_ctx:])
        for _ in range(max_new):
            logits = self._last_logits()
            if greedy:
                idx = int(logits.argmax())
            else:
                logits = logits / max(1e-5, temperature)
                if top_k:
                    kth = torch.topk(logits, top_k).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p:
                    sorted_l, order = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_l, dim=-1).cumsum(-1)
                    cut = probs > top_p
                    cut[..., 1:] = cut[..., :-1].clone()
                    cut[..., 0] = False
                    logits = logits.masked_fill(
                        torch.zeros_like(logits, dtype=torch.bool).scatter(0, order, cut), float("-inf"))
                idx = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            out.append(idx)
            new.append(idx)
            tail = bytes(new)
            if any(s and tail.endswith(s) for s in stops):
                break
            if len(self.ids) >= max_ctx:
                self.prime(out[-max_ctx:])   # dense path slides here; re-prime
            else:
                self._append(idx)
        return bytes(new) if continuation_only else bytes(out)


__all__ = ["IncrementalGenerator"]
