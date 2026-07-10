"""End-to-end NUERONCE inference with evidence-conditioned generation.

The pipeline connects retrieval, reasoning, planning, model generation, and
verification. The renderer receives both textual evidence in the canonical
prompt and retrieval tensors through ``NUERONCEModel.generate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from . import impl
from .evidence import gate_hits as _gate_hits, source_to_memory as _candidate_to_memory
from .model import NUERONCEModel
from .planning import Planner
from .prompting import (
    STOP_SEQUENCES,
    evidence_text,
    extract_assistant_continuation,
    format_inference_prompt,
    format_revision_prompt,
    plan_text,
)
from .retrieval import HybridRetriever
from .types import MemoryRecord, SourceRecord, TaskState, VerificationReport
from .verification import IndependentVerifier
from .workspace import GlobalWorkspace


SYSTEM_MESSAGE = (
    "You are NUERONCE. Answer only from trusted evidence and the response plan. "
    "Do not invent unsupported facts. If evidence is missing or restricted, qualify the answer."
)


def evidence_to_retrieval_tensors(
    evidence_items: List[MemoryRecord],
    *,
    max_neighbors: int = 4,
    max_len: int = 160,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert approved evidence records into NUERONCEModel retrieval tensors."""
    chunks = [ev.content.encode("utf-8")[:max_len] for ev in evidence_items[:max_neighbors] if ev.content]
    if not chunks:
        return None, None
    dev = device or torch.device("cpu")
    width = max(len(c) for c in chunks)
    ids = torch.zeros((1, len(chunks), width), dtype=torch.long, device=dev)
    mask = torch.zeros((1, len(chunks), width), dtype=torch.bool, device=dev)
    for i, chunk in enumerate(chunks):
        vals = torch.tensor(list(chunk), dtype=torch.long, device=dev)
        ids[0, i, : len(chunk)] = vals
        mask[0, i, : len(chunk)] = True
    return ids, mask


def verification_feedback(report: VerificationReport) -> dict:
    """Structured verifier feedback suitable for a one-shot revision prompt."""
    return {
        "unsupported_claims": [
            f.target_claim for f in report.failures if f.category == "unsupported_claim" and f.target_claim
        ],
        "contradictions": [
            f.target_claim for f in report.failures if f.category == "contradiction" and f.target_claim
        ],
        "missing_evidence": [
            f.instruction for f in report.failures if f.category in ("unsupported_claim", "incomplete")
        ],
        "format_failures": [
            f.instruction for f in report.failures
            if f.category not in ("unsupported_claim", "contradiction", "incomplete")
        ],
        "passed": report.passes,
    }


@dataclass
class ModelRenderer:
    """Evidence-conditioned renderer backed by ``NUERONCEModel.generate``."""

    model: NUERONCEModel
    max_new: int = 96
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0

    def render(self, semantic_draft: dict, style: dict) -> str:
        del style
        prompt = self._prompt(semantic_draft)
        out = self.model.generate(
            prompt.encode("utf-8"),
            neighbor_ids=semantic_draft.get("neighbor_ids"),
            neighbor_mask=semantic_draft.get("neighbor_mask"),
            retrieval_context=semantic_draft.get("evidence", []),
            max_new=self.max_new,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            stop_sequences=STOP_SEQUENCES,
            greedy=(self.temperature <= 0),
            continuation_only=True,
        )
        return extract_assistant_continuation(out)

    def render_revision(self, semantic_draft: dict, first_draft: str, feedback: dict) -> str:
        prompt = format_revision_prompt(
            system_message=semantic_draft.get("system_message", SYSTEM_MESSAGE),
            user_request=semantic_draft.get("query", ""),
            trusted_evidence=evidence_text(semantic_draft.get("evidence", [])),
            response_plan=self._plan_text(semantic_draft),
            first_draft=first_draft,
            verifier_feedback=feedback,
        )
        out = self.model.generate(
            prompt.encode("utf-8"),
            neighbor_ids=semantic_draft.get("neighbor_ids"),
            neighbor_mask=semantic_draft.get("neighbor_mask"),
            retrieval_context=semantic_draft.get("evidence", []),
            max_new=self.max_new,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            stop_sequences=STOP_SEQUENCES,
            greedy=(self.temperature <= 0),
            continuation_only=True,
        )
        return extract_assistant_continuation(out)

    def _prompt(self, semantic_draft: dict) -> str:
        return format_inference_prompt(
            system_message=semantic_draft.get("system_message", SYSTEM_MESSAGE),
            user_request=semantic_draft.get("query", semantic_draft.get("prompt", "")),
            trusted_evidence=evidence_text(semantic_draft.get("evidence", [])),
            response_plan=self._plan_text(semantic_draft),
        )

    @staticmethod
    def _plan_text(semantic_draft: dict) -> str:
        tool_outputs = semantic_draft.get("tool_outputs") or []
        parts = [
            "Reasoning summary:",
            plan_text(semantic_draft.get("reasoning", {})),
            "Response plan:",
            plan_text(semantic_draft.get("plan", {})),
        ]
        if tool_outputs:
            parts += ["Tool outputs:", evidence_text(tool_outputs)]
        return "\n".join(p for p in parts if p)


def respond(
    model: NUERONCEModel,
    query: str,
    corpus_texts: List[str],
    mode: str = "DELIBERATE",
    max_rounds: int = 2,
    source_records: Optional[List[SourceRecord]] = None,
    tool_outputs: Optional[List[MemoryRecord]] = None,
    max_new: int = 96,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> Tuple[str, VerificationReport, dict]:
    """Run retrieval, reasoning, planning, evidence-conditioned generation, and verification."""
    del mode
    d_model = model.cfg.d_model

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
    id_to_source = {f"doc{i}": rec for i, rec in enumerate(source_records or [])}
    evidence, rejected = _gate_hits(hits, id_to_text, id_to_source)

    with torch.no_grad():
        ids = torch.tensor([list(query.encode("utf-8"))[:256] or [32]], dtype=torch.long)
        ids = ids.to(next(model.parameters()).device)
        g, _, _, _, unit_mask = model.encode_units(ids)
        core_state = g[0][unit_mask[0]].mean(0) if unit_mask[0].any() else g[0].mean(0)

    task = TaskState(
        literal_request=query,
        inferred_goal=query,
        expected_output="answer",
        required_precision=0.7,
        stakes=0.5,
        ambiguity=0.3,
        completion_criteria=[],
    )
    ws = GlobalWorkspace(d_model=d_model)
    ws.initialize(task, evidence, core_state)
    for _ in range(model.cfg.logical_depth):
        ws.iterate()
    reasoning = ws.extract_reasoning_result()

    planner = Planner(hooks=impl.default_planner_hooks())
    plan = planner.build_plan(task, reasoning, evidence_map={h.bundle.source_id: 1.0 for h in hits})
    plan["completion_checklist"] = []

    device = next(model.parameters()).device
    neighbor_ids, neighbor_mask = evidence_to_retrieval_tensors(evidence, device=device)
    draft = {
        "system_message": SYSTEM_MESSAGE,
        "query": query,
        "evidence": evidence,
        "reasoning": reasoning,
        "plan": plan,
        "tool_outputs": tool_outputs or [],
        "neighbor_ids": neighbor_ids,
        "neighbor_mask": neighbor_mask,
    }
    renderer = ModelRenderer(
        model,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    verifier = IndependentVerifier(impl.default_verifier_hooks())

    first_draft = renderer.render(draft, {"register": "neutral"})
    first_report = verifier.verify(first_draft, plan, evidence, tool_outputs or [])
    text, report = first_draft, first_report
    revision = None
    second_report = None
    if not first_report.passes and max_rounds > 1:
        revision = renderer.render_revision(draft, first_draft, verification_feedback(first_report))
        second_report = verifier.verify(revision, plan, evidence, tool_outputs or [])
        text, report = revision, second_report

    trace = {
        "retrieved": [h.bundle.source_id for h in hits],
        "selected_evidence": [ev.memory_id for ev in evidence],
        "rejected_by_provenance": rejected,
        "provenance": {
            ev.memory_id: {
                "authenticity_status": ev.authenticity_status,
                "issuer_id": ev.issuer_id,
                "key_id": ev.key_id,
                "failure_reason": ev.provenance_failure_reason,
            }
            for ev in evidence
        },
        "reasoning": reasoning,
        "plan": plan,
        "plan_sections": plan["section_order"],
        "first_draft": first_draft,
        "revision": revision,
        "retrieval_tensors": {
            "neighbor_ids_shape": list(neighbor_ids.shape) if neighbor_ids is not None else None,
            "neighbor_mask_shape": list(neighbor_mask.shape) if neighbor_mask is not None else None,
        },
        "verification": {
            "passes": report.passes,
            "supported_fraction": report.supported_claim_fraction,
            "n_failures": len(report.failures),
            "provenance_statuses": report.provenance_statuses,
            "rejected_evidence": report.rejected_evidence,
            "first_feedback": verification_feedback(first_report),
            "revision_feedback": verification_feedback(second_report) if second_report is not None else None,
        },
    }
    return text, report, trace


__all__ = [
    "respond",
    "ModelRenderer",
    "evidence_to_retrieval_tensors",
    "verification_feedback",
    "SYSTEM_MESSAGE",
]
