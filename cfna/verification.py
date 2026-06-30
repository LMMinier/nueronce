"""Independent verifier ensemble and the verify-and-revise loop.

Truthfulness is checked by a verifier that is *not just another pass through the
same generator* (design claim H5). The ensemble checks claim/evidence alignment,
contradictions, plan completeness, tool-result fidelity, and confidence
calibration, emitting actionable :class:`VerificationFailure` instructions.

The individual checkers are model/logic functions injected as hooks (they depend
on claim extraction, entailment, etc.). The ensemble aggregation and the
verify→revise control loop are fully implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .config import VerifierConfig
from .types import (
    MemoryRecord,
    VerificationFailure,
    VerificationReport,
)


@dataclass
class VerifierHooks:
    extract_claims: Callable[[str], List[str]]
    match_claim_to_evidence: Callable[[str, List[MemoryRecord]], float]
    detect_text_evidence_contradictions: Callable[[str, List[MemoryRecord]], List[str]]
    checklist_item_satisfied: Callable[[str, str], bool]
    output_respects_tool_result: Callable[[str, MemoryRecord], bool]
    calibration_gap: Callable[[str, List[MemoryRecord]], float]
    measure_support_fraction: Callable[[List[str], List[MemoryRecord]], float]


class IndependentVerifier:
    def __init__(self, hooks: VerifierHooks, cfg: Optional[VerifierConfig] = None):
        self.h = hooks
        self.cfg = cfg or VerifierConfig()

    def verify(
        self,
        candidate_text: str,
        plan: dict,
        evidence_items: List[MemoryRecord],
        tool_obs: List[MemoryRecord],
    ) -> VerificationReport:
        h, cfg = self.h, self.cfg
        failures: List[VerificationFailure] = []
        claims = h.extract_claims(candidate_text)

        # 1. claim-evidence alignment
        for claim in claims:
            support = h.match_claim_to_evidence(claim, evidence_items)
            if support < cfg.unsupported_claim_threshold:
                failures.append(
                    VerificationFailure(
                        category="unsupported_claim",
                        target_claim=claim,
                        severity=1.0,
                        instruction=f"remove, cite, or weaken claim: {claim}",
                    )
                )

        # 2. contradiction scan
        contradictions = h.detect_text_evidence_contradictions(candidate_text, evidence_items)
        for c in contradictions:
            failures.append(
                VerificationFailure("contradiction", c, 1.0, f"resolve contradiction: {c}")
            )

        # 3. plan completeness
        for item in plan.get("completion_checklist", []):
            if not h.checklist_item_satisfied(candidate_text, item):
                failures.append(
                    VerificationFailure(
                        "incomplete", None, 0.8, f"cover missing requirement: {item}"
                    )
                )

        # 4. tool verification
        for obs in tool_obs:
            if obs.authority_level == "tool_observation" and not h.output_respects_tool_result(
                candidate_text, obs
            ):
                failures.append(
                    VerificationFailure("tool_misread", None, 1.0, "align with observed tool output")
                )

        # 5. confidence calibration
        cal_err = h.calibration_gap(candidate_text, evidence_items)
        if cal_err > cfg.calibration_tolerance:
            failures.append(
                VerificationFailure("miscalibration", None, 0.7, "adjust confidence wording")
            )

        blocking = [f for f in failures if f.severity >= cfg.blocking_severity]
        return VerificationReport(
            passes=len(blocking) == 0,
            failures=failures,
            supported_claim_fraction=h.measure_support_fraction(claims, evidence_items),
            contradiction_fraction=len(contradictions) / max(1, len(claims)),
            calibration_error=cal_err,
        )


def verify_and_revise(
    plan: dict,
    semantic_draft: dict,
    style: dict,
    evidence_items: List[MemoryRecord],
    tool_obs: List[MemoryRecord],
    renderer,
    verifier: IndependentVerifier,
    revise_semantic_draft: Callable[[dict, List[VerificationFailure]], dict],
    max_rounds: int = 3,
):
    """Render → verify → (revise)* loop. Returns (final_text, report).

    ``renderer`` exposes ``render(semantic_draft, style) -> str``;
    ``revise_semantic_draft(draft, failures) -> draft`` applies fixes.
    """
    current = semantic_draft
    report = None
    for _ in range(max_rounds):
        text = renderer.render(current, style)
        report = verifier.verify(text, plan, evidence_items, tool_obs)
        if report.passes:
            return text, report
        current = revise_semantic_draft(current, report.failures)
    final_text = renderer.render(current, style)
    return final_text, report


__all__ = ["VerifierHooks", "IndependentVerifier", "verify_and_revise"]
