"""Episode generator for WPGCP.

An :class:`Episode` packages inputs, targets, and per-objective loss weights for a
given curriculum phase. The phase-dispatch structure is implemented; the actual
data transforms (masking spans, corrupting text, choosing claims, retrieving
candidate units, selecting tool traces) are injected hooks since they depend on
the concrete corpus representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..types import KnowledgeUnit


@dataclass
class Episode:
    episode_type: str
    inputs: Dict[str, Any]
    targets: Dict[str, Any]
    weights: Dict[str, float] = field(default_factory=dict)


class EpisodeGenerator:
    """Dispatches to a per-phase episode builder.

    Provide ``hooks`` (an object exposing the data-transform callables named in
    the design doc) to enable episode construction.
    """

    def __init__(self, hooks: Optional[Any] = None):
        self.hooks = hooks

    def make_episode(self, unit_batch: List[KnowledgeUnit], phase: int) -> Episode:
        builders = {
            0: self._perception,
            1: self._bidirectional,
            2: self._retrieval_evidence,
            3: self._world_tool,
        }
        return builders.get(phase, self._mixed)(unit_batch)

    def _require_hooks(self, name: str):
        if self.hooks is None:
            raise NotImplementedError(
                f"EpisodeGenerator.{name} needs data-transform hooks "
                f"(masking/corruption/claim-selection/retrieval/tool-trace)."
            )
        return self.hooks

    def _perception(self, units):
        h = self._require_hooks("_perception")
        x = h.bytes_from_units(units)
        return Episode(
            episode_type="perception_reconstruct",
            inputs={"bytes": h.mask_random_spans(x)},
            targets={
                "raw_bytes": x,
                "boundaries": h.true_boundaries(x),
                "doc_structure": h.structure_labels(units),
            },
            weights={"byte": 1.0, "boundary": 0.6, "structure": 0.4},
        )

    def _bidirectional(self, units):
        h = self._require_hooks("_bidirectional")
        excerpt = h.concatenate_units(units)
        return Episode(
            episode_type="semantic_reconstruct",
            inputs={"text": h.corrupt_spans(excerpt)},
            targets={
                "restored_text": excerpt,
                "entities": h.extract_entities(excerpt),
                "relations": h.extract_relations(excerpt),
            },
            weights={"reconstruct": 1.0, "semantic": 0.7},
        )

    def _retrieval_evidence(self, units):
        h = self._require_hooks("_retrieval_evidence")
        claim = h.choose_claim(units)
        candidates = h.retrieve_candidate_units_for_claim(claim)
        return Episode(
            episode_type="claim_evidence_relation",
            inputs={"claim": claim.text, "candidates": [c.text for c in candidates]},
            targets={
                "relations": [h.label_relation(claim, c) for c in candidates],
                "best_sources": h.top_supporting_sources(claim, candidates),
            },
            weights={"evidence": 1.0, "retrieval": 0.8, "contradiction": 0.8, "provenance": 0.5},
        )

    def _world_tool(self, units):
        h = self._require_hooks("_world_tool")
        proc = h.choose_procedure_or_tool_trace(units)
        return Episode(
            episode_type="world_tool_predict",
            inputs={"trace": proc["prefix"]},
            targets={"next_step": proc["next_step"], "tool_obs": proc["obs"], "success": proc["success"]},
            weights={"causal": 0.8, "tool": 1.0, "planning": 0.6},
        )

    def _mixed(self, units):
        h = self._require_hooks("_mixed")
        return h.make_mixed_episode(units)


__all__ = ["Episode", "EpisodeGenerator"]
