"""State-cached (KV/state-style) incremental generation for NueronceModel.

``NueronceModel.generate`` recomputes the full forward pass — perception,
segmentation, unit stack, and decoder over the whole context — for every new
byte. This module exploits three structural facts of the architecture to make
generation incremental **without changing a single weight or forward
definition** (the verified neural core is untouched; this is inference
plumbing around it):

1. Every stage is causal, so appending a byte never changes any cached
   quantity for earlier positions.
2. The decoder masks each byte away from its *own* (in-progress) unit
   (``byte_to_unit_mask`` allows only units strictly before the byte's
   segment), so the expensive unit stack — memory + hybrid core — only needs
   recomputing when a patch *completes* (every ``min_patch``..``max_patch``
   bytes), not every byte. Between cuts, the completed-unit context ``g`` is
   provably constant.
3. The decoder has no absolute positions: stacked causal local attention with
   window ``w`` over ``L`` layers reads at most ``L*(w-1)+1`` bytes back, so
   running the decoder on the last ``L*w + margin`` bytes yields the exact
   same last-position logits as running it on the whole context.

Equivalence is a *tested claim*, not an assumption: under greedy decoding the
incremental path must produce byte-identical output to the dense path
(``tests/test_incremental_generation.py``), and per-step logits must agree to
float tolerance. When the context would exceed ``max_ctx`` (where the dense
path starts sliding its window, which is inherently non-incremental), this
engine re-primes from the slid window so equivalence is preserved at dense
cost for those steps.

Retrieval-conditioned generation is not supported here; callers with
retrieval context should use the dense path.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .nueronce_model import NueronceModel
from .segment import byte_to_unit_mask
from .tensor import Tensor, no_grad

# Perception receptive field: conv3 (2 back) + conv7 (6 back) + dilated conv3
# with dilation 4 (8 back) = 16 bytes back + the byte itself.
_PERCEPTION_SPAN = 17


def _softmax_np(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


class IncrementalGenerator:
    """Incremental drop-in for ``NueronceModel.generate`` (no retrieval)."""

    def __init__(self, model: NueronceModel, margin: int = 8):
        self.model = model
        c = model.cfg
        self._per_span = _PERCEPTION_SPAN + margin
        # Safe over-cover of the decoder's stacked receptive field.
        self._dec_span = c.decoder_layers * c.decoder_window + margin

    # -------------------------------------------------------------- state
    def prime(self, ids: List[int]) -> None:
        """Build all caches from a full pass over ``ids`` (dense cost, once)."""
        c = self.model.cfg
        self.ids: List[int] = list(ids)
        feats, blog = self.model.perception(np.array([self.ids]))
        f = feats.data[0]                                   # [T, d_local]
        prob = 1.0 / (1.0 + np.exp(-blog.data[0]))          # [T]
        # Replay the exact greedy segmentation to build per-unit accumulators.
        self.seg: List[int] = [0]
        self._cur, self._len = 0, 1
        self._done_feats: List[np.ndarray] = []             # completed-unit mean feats
        self._done_prob: List[float] = []                   # completed-unit mean boundary prob
        self._acc_f, self._acc_p, self._acc_n = f[0].copy(), float(prob[0]), 1
        for i in range(1, len(self.ids)):
            self._advance_segmentation(f[i], float(prob[i]))
        self._rebuild_units()

    def _advance_segmentation(self, feat_i: np.ndarray, prob_i: float) -> None:
        """Exact mirror of ``segment_ids_from_boundaries``'s per-step rule."""
        c = self.model.cfg
        cut = ((prob_i > c.tau) and self._len >= c.min_patch) or self._len >= c.max_patch
        if cut:
            if self._cur < c.p_max - 1:
                # current unit completes; the new byte starts the next unit
                self._done_feats.append(self._acc_f / self._acc_n)
                self._done_prob.append(self._acc_p / self._acc_n)
                self._acc_f, self._acc_p, self._acc_n = feat_i.copy(), prob_i, 1
                self._cur += 1
                self._units_dirty = True
            else:
                # p_max cap: seg id stays at p_max-1, byte merges into it
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

    def _rebuild_units(self) -> None:
        """Recompute the unit stack over *completed* units only (padded to
        p_max exactly like the dense path pads, so causal core/memory outputs
        for real rows match the dense computation bit-for-bit)."""
        c = self.model.cfg
        pc = len(self._done_feats)
        pooled = np.zeros((1, c.p_max, c.d_local))
        mean_p = np.zeros((1, c.p_max, 1))
        if pc:
            pooled[0, :pc] = np.stack(self._done_feats)
            mean_p[0, :pc, 0] = self._done_prob
        unit_mask = np.zeros((1, c.p_max), dtype=bool)
        unit_mask[0, :pc] = True
        units = self.model.unit_embed(Tensor(pooled))
        if c.trainable_segmentation:
            units = units + self.model.boundary_proj(Tensor(mean_p))
        units = units + self.model.memory(units)
        self._g = self.model.core(units, c.logical_depth, key_padding=unit_mask)
        self._unit_mask = unit_mask
        self._units_dirty = False

    # -------------------------------------------------------------- steps
    def _last_logits(self) -> np.ndarray:
        c = self.model.cfg
        if self._units_dirty:
            self._rebuild_units()
        span = min(len(self.ids), self._dec_span)
        win = np.array([self.ids[-span:]])
        seg_win = np.array([self.seg[-span:]])
        cross = byte_to_unit_mask(seg_win, self._unit_mask, c.p_max)
        logits = self.model.decoder(win, self._g, cross)
        return logits.data[0, -1]

    def _append(self, byte_id: int) -> None:
        self.ids.append(byte_id)
        span = min(len(self.ids), self._per_span)
        feats, blog = self.model.perception(np.array([self.ids[-span:]]))
        prob_i = float(1.0 / (1.0 + np.exp(-blog.data[0, -1])))
        self._advance_segmentation(feats.data[0, -1], prob_i)

    # ----------------------------------------------------------- generate
    def generate(self, prompt: bytes, max_new: int = 64, temperature: float = 0.8,
                 greedy: bool = False, max_ctx: int = 256,
                 stop_bytes: bytes = b"", min_new: int = 0,
                 rng: Optional[np.random.Generator] = None) -> bytes:
        """API- and output-compatible with ``NueronceModel.generate``.
        Byte-identical to the dense path under ``greedy=True`` (tested)."""
        ids = list(prompt) or [32]
        out: List[int] = list(ids)
        new_count = 0
        with no_grad():
            self.prime(out[-max_ctx:])
            for _ in range(max_new):
                nxt = self._last_logits()
                if greedy:
                    idx = int(nxt.argmax())
                elif rng is not None:
                    idx = int(rng.choice(256, p=_softmax_np(nxt / max(1e-5, temperature))))
                else:
                    idx = int(np.random.choice(256, p=_softmax_np(nxt / max(1e-5, temperature))))
                out.append(idx)
                new_count += 1
                if stop_bytes and new_count >= min_new and idx in stop_bytes:
                    break
                if len(self.ids) >= max_ctx:
                    # Dense path slides its window here; sliding invalidates
                    # append-only caches, so re-prime for exactness.
                    self.prime(out[-max_ctx:])
                else:
                    self._append(idx)
        return bytes(out)


__all__ = ["IncrementalGenerator"]
