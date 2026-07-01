"""A small, fully-symbolic cognitive loop over the frozen :mod:`cfna.contract`.

Purpose (the corrected CFNA thesis): show that splitting a task across explicit
stages — perceive, intent, retrieve, *resolve by authority + supersession*, plan,
draft, verify, revise — produces correct behavior that a single continuation
network does **not** get for free, and that *removing any stage* breaks it in a
specific, predictable way.

This module is deliberately decoder-free: the "answer" is decided by typed
reasoning over provenance, not by sampling bytes. That is the point — it isolates
the architecture from language-model fluency so the ablation result is about the
*cognitive structure*, not about how well the byte model was trained.

Subsystem label: **REAL / HEURISTIC**.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .contract import (
    Claim,
    CognitiveTrace,
    Draft,
    EvidenceItem,
    EvidenceSet,
    IntentState,
    MemoryQuery,
    PerceptionState,
    Plan,
    ReasoningState,
    Revision,
    SemanticState,
    content_hash,
)
from .types import AuthorityLevel, VerificationFailure, VerificationReport

# Authority levels that may never silently overwrite a verified belief.
UNTRUSTED: Tuple[AuthorityLevel, ...] = (
    "unverified_external_content",
    "generated_hypothesis",
)

# The cognitive stages that can be individually disabled to prove necessity.
ABLATION_FLAGS = (
    "no_retrieval",
    "no_authority",
    "no_supersession",
    "no_planning",
    "no_verification",
)


# --------------------------------------------------------------------------- #
# Scenario description (a controlled falsification task)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    query: str
    entity: str
    attribute: str
    working_memory: Tuple[EvidenceItem, ...]   # facts already known (user turns)
    external: Tuple[EvidenceItem, ...]         # facts reachable via retrieval
    expected_value: Optional[str]              # None => correct answer is "uncertain"
    requires_citation: bool
    targets: Tuple[str, ...]                    # modules this scenario is built to stress


def fact(entity: str, attribute: str, value: str, source_id: str,
         authority: AuthorityLevel, timestamp: str, score: float = 1.0) -> EvidenceItem:
    """Construct a provenance-carrying evidence item for a scenario."""
    return EvidenceItem(
        value=value, source_id=source_id, authority=authority, timestamp=timestamp,
        content_hash=content_hash(f"{entity}|{attribute}|{value}|{source_id}"),
        score=score, trusted=True, claim_key=(entity.lower(), attribute.lower()),
    )


# --------------------------------------------------------------------------- #
# Stage 5: resolve competing evidence by authority + supersession
# --------------------------------------------------------------------------- #

def resolve(query: MemoryQuery, items: List[EvidenceItem],
            ablations: Dict[str, bool]) -> ReasoningState:
    """Pick the believed value from competing evidence, honoring ablation flags.

    FULL behavior: drop untrusted sources, then prefer the most authoritative
    trusted item, breaking ties toward the most recent (temporal supersession).
    """
    no_authority = ablations.get("no_authority", False)
    no_supersession = ablations.get("no_supersession", False)

    key = (query.entity or "").lower(), (query.attribute or "").lower()
    relevant = [i for i in items if i.claim_key == key]

    rejected: List[EvidenceItem] = []
    pool: List[EvidenceItem] = []
    for it in relevant:
        # Authority representation = the gate that keeps untrusted content from
        # overwriting beliefs. Removing it lets a poisoned document win.
        if not no_authority and it.authority in UNTRUSTED:
            rejected.append(replace(it, trusted=False))
            continue
        pool.append(it)

    if not pool:
        return ReasoningState(
            resolved_value=None, winning_evidence=None,
            rejected_untrusted=tuple(rejected),
            rationale="no trusted evidence for the queried fact",
            confidence=0.0,
        )

    if no_authority:
        # No notion of trust: latest-asserted wins, poison included.
        winner = max(pool, key=lambda i: i.timestamp)
        rationale = "no_authority: chose most recent assertion regardless of source"
    elif no_supersession:
        # No temporal update: the first belief sticks; corrections are ignored.
        winner = min(pool, key=lambda i: i.timestamp)
        rationale = "no_supersession: kept earliest belief, ignored later corrections"
    else:
        # Most authoritative, then most recent.
        winner = sorted(pool, key=lambda i: (i.rank, _neg_iso(i.timestamp)))[0]
        rationale = (f"chose {winner.authority} from {winner.source_id} "
                     f"(authority rank {winner.rank}, ts {winner.timestamp})")

    superseded = tuple(i for i in pool if i is not winner and i.value != winner.value)
    return ReasoningState(
        resolved_value=winner.value, winning_evidence=winner,
        superseded=superseded, rejected_untrusted=tuple(rejected),
        rationale=rationale, confidence=1.0 - 0.1 * len(superseded),
    )


def _neg_iso(ts: str) -> str:
    # Sort ISO timestamps descending by inverting each character's code point.
    return "".join(chr(0x10FFFF - ord(c)) for c in ts)


# --------------------------------------------------------------------------- #
# The cognitive loop
# --------------------------------------------------------------------------- #

def run(scenario: Scenario, ablations: Optional[Dict[str, bool]] = None) -> CognitiveTrace:
    """Run perceive -> intent -> retrieve -> resolve -> plan -> draft -> verify
    -> revise over a scenario, logging every typed intermediate state."""
    ablations = {k: bool(ablations.get(k, False)) for k in ABLATION_FLAGS} if ablations else \
        {k: False for k in ABLATION_FLAGS}
    trace = CognitiveTrace(ablations=ablations)

    # 1. perceive
    trace.perception = PerceptionState.perceive(scenario.query)

    # 2. represent meaning (the query references one entity/attribute)
    trace.semantics = SemanticState(
        surface=scenario.query, entities=(scenario.entity,),
        concepts=(scenario.attribute,),
    )

    # 3. intent
    trace.intent = IntentState(
        task_type="answer_fact", target_entity=scenario.entity,
        target_attribute=scenario.attribute,
        requires_citation=scenario.requires_citation,
    )

    # 4. memory query + retrieval (external evidence gated by the retrieval module)
    query = MemoryQuery(entity=scenario.entity, attribute=scenario.attribute)
    trace.memory_query = query
    available: List[EvidenceItem] = list(scenario.working_memory)
    if not ablations["no_retrieval"]:
        available += list(scenario.external)
    trace.evidence = EvidenceSet(query=query, items=tuple(available))

    # 5. reason / resolve
    reasoning = resolve(query, available, ablations)
    trace.reasoning = reasoning

    # 6. plan (planning is where the citation requirement is declared)
    must_cite = scenario.requires_citation and not ablations["no_planning"]
    trace.plan = Plan(
        steps=("state resolved value", "attach citation" if must_cite else "state value"),
        must_cite=must_cite,
    )

    # 7. communicate / draft  (renderer does NOT auto-cite; citation is added in revision)
    if reasoning.resolved_value is None:
        draft = Draft(text=f"I don't have a trusted source for the {scenario.attribute} "
                           f"of {scenario.entity}.", claims=(), citations=())
    else:
        draft = Draft(
            text=f"The {scenario.attribute} of {scenario.entity} is {reasoning.resolved_value}.",
            claims=(reasoning.resolved_value,), citations=(),
        )
    trace.draft = draft

    # 8. verify + 9. revise
    if ablations["no_verification"]:
        trace.verification = VerificationReport(
            passes=True, failures=[], supported_claim_fraction=1.0,
            contradiction_fraction=0.0, calibration_error=0.0,
        )
        trace.revision = Revision(changed=False, before=draft.text, after=draft.text,
                                  reason="verification disabled")
        trace.answer = draft.text
        return trace

    report, revised = _verify_and_revise(trace.plan, draft, reasoning)
    trace.verification = report
    trace.revision = revised
    trace.answer = revised.after
    return trace


def _verify_and_revise(plan: Plan, draft: Draft,
                       reasoning: ReasoningState) -> Tuple[VerificationReport, Revision]:
    failures: List[VerificationFailure] = []

    # Authority check: a stated value must be backed by a *trusted* item.
    if reasoning.resolved_value is not None:
        win = reasoning.winning_evidence
        if win is None or not win.trusted:
            failures.append(VerificationFailure(
                category="UNSUPPORTED_CLAIM", target_claim=reasoning.resolved_value,
                severity=1.0, instruction="drop unsupported value",
            ))

    # Citation requirement declared by the plan.
    needs_citation = plan.must_cite and reasoning.resolved_value is not None
    if needs_citation and not draft.citations:
        failures.append(VerificationFailure(
            category="MISSING_REQUIREMENT", target_claim=None, severity=0.5,
            instruction="attach a source citation",
        ))

    after = draft.text
    changed = False
    # Targeted revision: only fix what failed; preserve verified content.
    if needs_citation and not draft.citations and reasoning.winning_evidence is not None:
        win = reasoning.winning_evidence
        after = f"{draft.text} (source: {win.authority}:{win.source_id})"
        changed = True

    passes = not failures or (changed and all(f.category == "MISSING_REQUIREMENT" for f in failures))
    report = VerificationReport(
        passes=passes, failures=failures,
        supported_claim_fraction=1.0 if reasoning.resolved_value and
        reasoning.winning_evidence and reasoning.winning_evidence.trusted else 0.0,
        contradiction_fraction=0.0, calibration_error=0.0,
    )
    reason = "added citation" if changed else ("clean" if passes else "unresolved failures")
    return report, Revision(changed=changed, before=draft.text, after=after, reason=reason)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def grade(scenario: Scenario, trace: CognitiveTrace) -> Tuple[bool, str]:
    """Did the loop produce the correct, appropriately-cited answer?"""
    ans = trace.answer
    if scenario.expected_value is None:
        # Correct behavior is to decline (no trusted source).
        ok = ("don't have a trusted source" in ans) or ("uncertain" in ans.lower())
        return ok, "declined as required" if ok else "asserted an unsupported value"

    if scenario.expected_value.lower() not in ans.lower():
        return False, f"wrong value (expected {scenario.expected_value!r})"
    if scenario.requires_citation and "source:" not in ans:
        return False, "missing required citation"
    return True, "correct value with required citation" if scenario.requires_citation else "correct value"


__all__ = [
    "UNTRUSTED", "ABLATION_FLAGS", "Scenario", "fact",
    "resolve", "run", "grade",
]
