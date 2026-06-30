"""Cognitive embedding bundle compiler.

One information unit should not map to a single vector; it maps to semantic,
lexical, structural, hierarchical, temporal, provenance, belief, and authority
representations (cf. SBERT / SimCSE for semantic geometry, SPLADE for sparse
lexical, ColBERTv2 for late interaction, Poincaré embeddings for hierarchy).

The MLP heads need a neural backend; the assembly of the bundle (including the
real sparse-lexical pass-through and the uncertainty derived from belief) is laid
out here so the contract is explicit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._backend import needs_backend
from .config import EmbeddingConfig
from .types import CognitiveEmbeddingBundle


class CognitiveEmbeddingCompiler:
    """Converts dynamic information units + parser metadata into a typed bundle.

    Heads (semantic / structural / hierarchical / temporal / provenance / belief /
    authority) are learned projections requiring a backend. ``compile`` documents
    the exact fusion and which inputs feed each head.
    """

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        self.cfg = cfg or EmbeddingConfig()

    def compile(
        self,
        unit: Dict[str, Any],
        parser_meta: Dict[str, Any],
        source_meta: Any,
    ) -> CognitiveEmbeddingBundle:
        raise needs_backend(
            "CognitiveEmbeddingCompiler.compile",
            "Fuses [dense_from_exact(unit.exact), unit.local, unit.contextual, "
            "section_position_features, source_type_features] then runs the typed "
            "heads. sparse_lexical passes through unit['exact']; uncertainty = "
            "1 - max(belief_vec).",
        )


__all__ = ["CognitiveEmbeddingCompiler"]
