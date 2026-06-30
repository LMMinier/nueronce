"""Relation-specific routers.

CFNA treats semantic similarity, lexical identity, structural dependency,
support/contradiction, temporal relation, goal relevance, and authority as
*different relations* — not one attention score. Each gets its own scoring
function. The geometric scores (semantic, lexical, structural, temporal,
authority gate) are fully implemented; the learned evidence-relation classifier
needs a backend.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ._backend import needs_backend
from .ops import cosine, sparse_dot
from .types import CHANNELS, CognitiveEmbeddingBundle


def temporal_compatibility(a, b) -> float:
    """Default temporal compatibility in [0, 1]: cosine mapped from [-1, 1].

    Replace with a validity-window comparison when temporal vectors encode
    explicit [start, end] features.
    """
    return 0.5 * (cosine(a, b) + 1.0)


class RelationRouters:
    def __init__(self, evidence_classifier: Optional[Callable] = None):
        # evidence_classifier: [3*d_sem] -> [3] logits (support/contradict/unrelated)
        self._evidence_classifier = evidence_classifier

    def semantic_score(self, q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle) -> float:
        return cosine(q.dense_semantic, c.dense_semantic)

    def lexical_score(self, q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle) -> float:
        return sparse_dot(q.sparse_lexical, c.sparse_lexical)

    def structural_score(self, q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle) -> float:
        return cosine(q.structural_vec, c.structural_vec) + 0.5 * cosine(
            q.hierarchical_vec, c.hierarchical_vec
        )

    def temporal_score(self, q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle) -> float:
        return temporal_compatibility(q.temporal_vec, c.temporal_vec)

    def evidence_relation(
        self, q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle
    ) -> Dict[str, float]:
        if self._evidence_classifier is None:
            raise needs_backend(
                "RelationRouters.evidence_relation",
                "Provide a trained classifier mapping "
                "[q_sem, c_sem, |q_sem - c_sem|] -> 3 logits.",
            )
        qs = np.asarray(q.dense_semantic, dtype=np.float64).ravel()
        cs = np.asarray(c.dense_semantic, dtype=np.float64).ravel()
        x = np.concatenate([qs, cs, np.abs(qs - cs)])
        from .ops import softmax

        p = softmax(self._evidence_classifier(x))
        return {"support": float(p[0]), "contradict": float(p[1]), "unrelated": float(p[2])}

    def authority_allowed(self, q: CognitiveEmbeddingBundle, destination_channel: str) -> bool:
        idx = CHANNELS.index(destination_channel)
        return bool(np.asarray(q.authority_mask).ravel()[idx] > 0.5)


__all__ = ["RelationRouters", "temporal_compatibility"]
