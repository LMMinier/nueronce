"""End-to-end CFNA inference: perceive → retrieve → reason → plan → render → verify.

This composes the *real* components — the trained :class:`cfna.model.CFNAModel`
(perception/core/renderer), the hybrid retriever over an in-memory corpus, the
latent workspace, the planner, and the independent verifier — into a single
``respond`` call. It is the runnable realization of the design's ``CFNA_Respond``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from . import impl
from .embeddings import CognitiveEmbeddingCompiler
from .model import CFNAModel
from .ops import now_iso
from .planning import Planner
from .retrieval import HybridRetriever
from .types import MemoryRecord, TaskState, VerificationReport
from .verification import IndependentVerifier, verify_and_revise
from .workspace import GlobalWorkspace


@dataclass
class ModelRenderer:
    """Adapter that makes the byte model satisfy the renderer interface used by
    ``verify_and_revise`` (``render(semantic_draft, style) -> str``)."""

    model: CFNAModel
    max_new: int = 96

    def render(self, semantic_draft: dict, style: dict) -> str:
        # Render once, then persist into the draft so verifier-driven revisions
        # (which edit draft['text']) survive across revision rounds.
        if "text" not in semantic_draft:
            prompt = semantic_draft.get("prompt", "")
            out = self.model.generate(prompt.encode("utf-8"), max_new=self.max_new, greedy=True)
            semantic_draft["text"] = out.decode("utf-8", errors="replace")
        return semantic_draft["text"]


def _candidate_to_memory(text: str, doc_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=doc_id, memory_type="semantic", content=text,
        source_ids=[doc_id], embeddings={"dense_semantic": impl.embed_text(text)},
        structured_repr={}, authority_level="verified_secondary_source",
        creation_time=now_iso(), last_verified_time=now_iso(), confidence=0.8,
        review_status="verified", consolidation_status="semantic",
    )


def respond(
    model: CFNAModel,
    query: str,
    corpus_texts: List[str],
    mode: str = "DELIBERATE",
    max_rounds: int = 2,
) -> Tuple[str, VerificationReport, dict]:
    """Run the full pipeline and return (answer, verification_report, trace)."""
    d_model = model.cfg.d_model

    # 1. retrieve evidence (real hybrid retriever over real embeddings)
    candidates = impl.build_corpus_candidates(corpus_texts, dim=768)
    retriever = HybridRetriever(
        dense_index=impl.InMemoryDenseIndex(candidates),
        sparse_index=impl.InMemorySparseIndex(candidates),
        late_interaction=impl.late_interaction,
        contradiction_penalty=impl.contradiction_penalty,
        temporal_compatibility=impl.temporal_compat,
    )
    qbundle = impl.build_corpus_candidates([query], dim=768)[0].bundle
    hits = retriever.retrieve(qbundle)
    id_to_text = {f"doc{i}": t for i, t in enumerate(corpus_texts)}
    evidence = [_candidate_to_memory(id_to_text[h.bundle.source_id], h.bundle.source_id)
                for h in hits if h.bundle.source_id in id_to_text]

    # 2. perceive the query through the real core to seed reasoning
    with torch.no_grad():
        ids = torch.tensor([list(query.encode("utf-8"))[: 256] or [32]], dtype=torch.long)
        g, _, _, _, unit_mask = model.encode_units(ids)
        core_state = g[0][unit_mask[0]].mean(0) if unit_mask[0].any() else g[0].mean(0)

    # 3. workspace reasoning
    task = TaskState(
        literal_request=query, inferred_goal=query, expected_output="answer",
        required_precision=0.7, stakes=0.5, ambiguity=0.3,
        completion_criteria=[],
    )
    ws = GlobalWorkspace(d_model=d_model)
    ws.initialize(task, evidence, core_state)
    for _ in range(model.cfg.logical_depth):
        ws.iterate()
    reasoning = ws.extract_reasoning_result()

    # 4. plan (real heuristic hooks)
    planner = Planner(hooks=impl.default_planner_hooks())
    plan = planner.build_plan(task, reasoning, evidence_map={h.bundle.source_id: 1.0 for h in hits})
    plan["completion_checklist"] = []

    # 5. render + 6. verify→revise
    renderer = ModelRenderer(model)
    verifier = IndependentVerifier(impl.default_verifier_hooks())
    draft = {"prompt": query}
    text, report = verify_and_revise(
        plan=plan, semantic_draft=draft, style={"register": "neutral"},
        evidence_items=evidence, tool_obs=[], renderer=renderer, verifier=verifier,
        revise_semantic_draft=impl.revise_draft, max_rounds=max_rounds,
    )

    trace = {
        "retrieved": [h.bundle.source_id for h in hits],
        "reasoning": reasoning,
        "plan_sections": plan["section_order"],
        "verification": {
            "passes": report.passes,
            "supported_fraction": report.supported_claim_fraction,
            "n_failures": len(report.failures),
        },
    }
    return text, report, trace


__all__ = ["respond", "ModelRenderer"]
