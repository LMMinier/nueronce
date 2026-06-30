"""Planner and the two-stage renderer.

CFNA separates *what to say* (a structured plan with evidence map, uncertainty
map, required qualifications, discourse order, and an explicit list of prohibited
unsupported claims) from *how to word it* (a semantic draft, then a constrained
causal renderer). Plan-before-wording is a falsifiable design claim (H4).

The plan/draft assembly is structural and depends on injected analysis hooks; the
final autoregressive renderer needs a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ._backend import needs_backend
from .types import TaskState


@dataclass
class PlannerHooks:
    extract_supporting_claims: Callable[[dict], List[str]]
    build_uncertainty_map: Callable[[dict], dict]
    derive_caveats: Callable[[dict, dict], List[str]]
    discourse_order: Callable[[str, dict], List[str]]
    unsupported_claims: Callable[[dict, dict], List[str]]


class Planner:
    def __init__(self, hooks: Optional[PlannerHooks] = None):
        self.hooks = hooks

    def build_plan(self, task: TaskState, reasoning: dict, evidence_map: dict) -> dict:
        h = self.hooks
        if h is None:
            raise NotImplementedError(
                "Planner needs PlannerHooks (claim extraction, uncertainty map, "
                "caveat derivation, discourse ordering, unsupported-claim detection)."
            )
        return {
            "user_goal": task.inferred_goal,
            "central_answer": reasoning["best_hypothesis"],
            "supporting_claims": h.extract_supporting_claims(reasoning),
            "evidence_map": evidence_map,
            "uncertainty_map": h.build_uncertainty_map(reasoning),
            "required_qualifications": h.derive_caveats(reasoning, evidence_map),
            "section_order": h.discourse_order(task.expected_output, reasoning),
            "prohibited_claims": h.unsupported_claims(reasoning, evidence_map),
            "tool_actions": reasoning.get("required_tools", []),
            "completion_checklist": task.completion_criteria,
        }


class SemanticRenderer:
    """Builds a semantic draft (sections -> paragraph specs) from a plan."""

    def __init__(self, build_paragraph_specs: Optional[Callable] = None):
        self._build_paragraph_specs = build_paragraph_specs

    def generate_semantic_draft(self, plan: dict) -> dict:
        if self._build_paragraph_specs is None:
            raise needs_backend(
                "SemanticRenderer.generate_semantic_draft",
                "build_paragraph_specs(section, plan) -> ordered paragraph intents.",
            )
        doc: Dict[str, Any] = {"sections": []}
        for sec in plan["section_order"]:
            specs = self._build_paragraph_specs(sec, plan)
            doc["sections"].append({"section": sec, "paragraphs": specs})
        return doc


class CausalLanguageRenderer:
    """Constrained autoregressive renderer: paragraph intent -> surface text."""

    def render(self, semantic_draft: dict, style: dict) -> str:
        raise needs_backend(
            "CausalLanguageRenderer.render",
            "Per paragraph: para_to_span_intent -> render_span_autoregressive(intent, style).",
        )


__all__ = ["PlannerHooks", "Planner", "SemanticRenderer", "CausalLanguageRenderer"]
