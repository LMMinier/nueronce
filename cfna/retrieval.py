"""Hybrid retrieval: dense ANN + sparse inverted index + late-interaction rerank.

Follows the design doc's split-signal retriever (SBERT/SimCSE dense, SPLADE
sparse, ColBERTv2 late interaction) with provenance, temporal, and contradiction
terms folded into the combined score.

:func:`combine_scores` (the score fusion) is fully implemented and tested. The
index searches and the late-interaction op are injected protocols, since they
depend on a vector store / inverted index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from .config import RetrievalConfig
from .ops import cosine, sparse_dot
from .types import CognitiveEmbeddingBundle


@dataclass
class Candidate:
    bundle: CognitiveEmbeddingBundle
    source: Any  # SourceRecord-like, must expose .quality_score


def combine_scores(
    dense_score: float,
    sparse_score: float,
    late_score: float,
    temporal_validity: float,
    provenance_quality: float,
    contradiction_penalty: float,
    cfg: Optional[RetrievalConfig] = None,
) -> float:
    """Weighted fusion of the retrieval signals (matches RetrievalConfig weights)."""
    cfg = cfg or RetrievalConfig()
    return (
        cfg.w_dense * dense_score
        + cfg.w_sparse * _normalize_sparse(sparse_score)
        + cfg.w_late * _normalize_late(late_score)
        + cfg.w_temporal * temporal_validity
        + cfg.w_provenance * provenance_quality
        - cfg.w_contradiction * contradiction_penalty
    )


def _normalize_sparse(x: float) -> float:
    # squashes an unbounded sparse dot product into [0, 1)
    return x / (1.0 + x) if x > 0 else 0.0


def _normalize_late(x: float) -> float:
    return x / (1.0 + x) if x > 0 else 0.0


class DenseIndex(Protocol):
    def search(self, vec, topk: int) -> List[Candidate]: ...


class SparseIndex(Protocol):
    def search(self, sparse_vec: Dict[int, float], topk: int) -> List[Candidate]: ...


# late_interaction(query_bundle, cand_bundle) -> float
LateInteraction = Callable[[CognitiveEmbeddingBundle, CognitiveEmbeddingBundle], float]
# predict_contradiction_penalty(query_bundle, cand_bundle) -> float in [0, 1]
ContradictionPenalty = Callable[[CognitiveEmbeddingBundle, CognitiveEmbeddingBundle], float]


class HybridRetriever:
    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        late_interaction: LateInteraction,
        contradiction_penalty: ContradictionPenalty,
        temporal_compatibility: Callable,
        cfg: Optional[RetrievalConfig] = None,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.late_interaction = late_interaction
        self.contradiction_penalty = contradiction_penalty
        self.temporal_compatibility = temporal_compatibility
        self.cfg = cfg or RetrievalConfig()

    def retrieve(self, query: CognitiveEmbeddingBundle) -> List[Candidate]:
        cfg = self.cfg
        dense_hits = self.dense_index.search(query.dense_semantic, cfg.k_recall)
        sparse_hits = self.sparse_index.search(query.sparse_lexical, cfg.k_recall)
        merged = self._merge_unique(dense_hits, sparse_hits)

        scored = []
        for cand in merged:
            combined = combine_scores(
                dense_score=cosine(query.dense_semantic, cand.bundle.dense_semantic),
                sparse_score=sparse_dot(query.sparse_lexical, cand.bundle.sparse_lexical),
                late_score=self.late_interaction(query, cand.bundle),
                temporal_validity=self.temporal_compatibility(
                    query.temporal_vec, cand.bundle.temporal_vec
                ),
                provenance_quality=cand.source.quality_score,
                contradiction_penalty=self.contradiction_penalty(query, cand.bundle),
                cfg=cfg,
            )
            scored.append((combined, cand))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[: cfg.k_rerank]]

    @staticmethod
    def _merge_unique(*hit_lists: List[Candidate]) -> List[Candidate]:
        seen: Dict[str, Candidate] = {}
        for hits in hit_lists:
            for cand in hits:
                seen.setdefault(cand.bundle.unit_id, cand)
        return list(seen.values())


__all__ = [
    "Candidate",
    "combine_scores",
    "DenseIndex",
    "SparseIndex",
    "LateInteraction",
    "ContradictionPenalty",
    "HybridRetriever",
]
