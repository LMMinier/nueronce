"""Cognitive embedding bundle compiler — real implementation.

One information unit maps to several *typed* representations (semantic, lexical,
structural, hierarchical, temporal, provenance, belief, authority) rather than a
single vector. The heads are hand-built MLPs (from :mod:`cfna.nn`); the bundle's
``sparse_lexical`` is the unit's exact hashed-n-gram map (pass-through), and
``uncertainty`` is ``1 - max(belief)``.

Heads are randomly initialized (this stage is trained jointly in a full run); the
module is fully runnable today and produces deterministic typed embeddings used by
the relation routers and the hybrid retriever.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .config import EmbeddingConfig
from .types import CognitiveEmbeddingBundle, K_CHANNELS


class CognitiveEmbeddingCompiler:
    def __init__(self, cfg: Optional[EmbeddingConfig] = None, d_local: int = 96):
        import torch
        from torch import nn

        from .nn import MLP

        self.cfg = cfg or EmbeddingConfig()
        self.d_local = d_local
        c = self.cfg
        # fused input = [local, contextual, section_feats(4), source_feats(4)]
        fused_dim = 2 * d_local + 8
        self._t = torch
        self.heads = nn.ModuleDict({
            "semantic": MLP(fused_dim, 256, c.d_sem),
            "structural": MLP(fused_dim, 128, c.d_struct),
            "hierarchical": MLP(8, 64, c.d_hier),
            "temporal": MLP(4, 64, c.d_time),
            "provenance": MLP(4, 64, c.d_prov),
            "belief": MLP(c.d_sem, 128, c.belief_classes),
            "authority": MLP(c.d_prov, 64, K_CHANNELS),
        })
        for p in self.heads.parameters():
            p.requires_grad_(False)

    def compile(
        self,
        unit: Dict[str, Any],
        parser_meta: Optional[Dict[str, Any]] = None,
        source_meta: Optional[Any] = None,
    ) -> CognitiveEmbeddingBundle:
        torch = self._t
        parser_meta = parser_meta or {}
        local = np.asarray(unit.get("local"), dtype=np.float32).ravel()
        contextual = np.asarray(unit.get("contextual", local), dtype=np.float32).ravel()
        local = _fit(local, self.d_local)
        contextual = _fit(contextual, self.d_local)

        section_path = parser_meta.get("section_path", [])
        section_feats = np.array([
            len(section_path), parser_meta.get("position", 0.0),
            float(unit.get("span", (0, 0))[0]), float(unit.get("span", (0, 1))[1]),
        ], dtype=np.float32)
        q = float(getattr(source_meta, "quality_score", 0.5)) if source_meta is not None else 0.5
        pii = float(getattr(source_meta, "pii_risk", 0.0)) if source_meta is not None else 0.0
        source_feats = np.array([q, pii, 1.0, 0.0], dtype=np.float32)

        fused = torch.tensor(np.concatenate([local, contextual, section_feats, source_feats]))
        with torch.no_grad():
            dense_semantic = self.heads["semantic"](fused)
            structural = self.heads["structural"](fused)
            hier_in = torch.tensor(np.concatenate([section_feats, source_feats]))
            hierarchical = self.heads["hierarchical"](hier_in)
            temporal = self.heads["temporal"](torch.tensor(_temporal_feats(source_meta)))
            prov_in = torch.tensor(source_feats)
            provenance = self.heads["provenance"](prov_in)
            belief = torch.softmax(self.heads["belief"](dense_semantic), dim=-1)
            authority = torch.sigmoid(self.heads["authority"](provenance))

        belief_np = belief.numpy()
        return CognitiveEmbeddingBundle(
            dense_semantic=dense_semantic.numpy(),
            sparse_lexical=unit.get("exact", {}),
            structural_vec=structural.numpy(),
            hierarchical_vec=hierarchical.numpy(),
            temporal_vec=temporal.numpy(),
            provenance_vec=provenance.numpy(),
            belief_vec=belief_np,
            uncertainty=float(1.0 - belief_np.max()),
            authority_mask=authority.numpy(),
            source_id=getattr(source_meta, "source_id", parser_meta.get("source_id", "?")),
            unit_id=parser_meta.get("unit_id", "?"),
        )


def _fit(v: np.ndarray, dim: int) -> np.ndarray:
    if v.shape[0] == dim:
        return v
    if v.shape[0] > dim:
        return v[:dim]
    return np.pad(v, (0, dim - v.shape[0]))


def _temporal_feats(source_meta) -> np.ndarray:
    date = getattr(source_meta, "publication_date", None) if source_meta is not None else None
    if not date:
        return np.zeros(4, dtype=np.float32)
    parts = str(date).split("-")
    try:
        y = float(parts[0]); mo = float(parts[1]) if len(parts) > 1 else 0.0
    except ValueError:
        return np.zeros(4, dtype=np.float32)
    return np.array([(y - 2000.0) / 50.0, mo / 12.0, 1.0, 0.0], dtype=np.float32)


__all__ = ["CognitiveEmbeddingCompiler"]
