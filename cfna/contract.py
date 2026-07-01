"""The frozen CFNA cognitive contract.

This module pins down the *typed objects that flow between cognitive stages* so the
architecture's claim is testable: intelligence is split across explicit stages

    perceive -> represent meaning -> determine intent -> query/retrieve memory ->
    reason (resolve) -> plan -> communicate (draft) -> verify -> revise

and every stage consumes and produces a named, inspectable structure with
provenance — never just an opaque tensor handed silently from one module to the
next. This makes the system modular and auditable. It does *not* prove a
decoder-only network could not reproduce the same behavior — a decoder could emit
equivalent structures; the value here is inspectability and control, not
irreducibility.

Design rules for this file:

* Pure dataclasses, no neural-backend import — safe to use in tests and on CPU.
* Reuse the existing controlled vocabularies (:data:`AuthorityLevel`,
  :data:`AUTHORITY_ORDER`) and :class:`VerificationReport` from :mod:`cfna.types`
  rather than re-inventing them.
* Every evidence-bearing object records *where it came from* (source, authority,
  timestamp, content hash) so reasoning and verification can be audited.

Subsystem label (per the project's honesty rule): **REAL / HEURISTIC** — these are
real, exercised data structures and resolution logic; they are not learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Dict, List, Optional, Tuple

from .types import (
    AUTHORITY_ORDER,
    AuthorityLevel,
    VerificationReport,
)

__all__ = [
    "content_hash",
    "authority_rank",
    "PerceptionState",
    "Claim",
    "SemanticState",
    "IntentState",
    "MemoryQuery",
    "EvidenceItem",
    "EvidenceSet",
    "ReasoningState",
    "Plan",
    "Draft",
    "Revision",
    "VerificationReport",
    "CognitiveTrace",
]


def content_hash(text: str) -> str:
    """Stable provenance hash for a piece of content."""
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()[:16]


def authority_rank(level: AuthorityLevel) -> int:
    """Lower is more trusted (index into :data:`AUTHORITY_ORDER`)."""
    try:
        return AUTHORITY_ORDER.index(level)
    except ValueError:  # unknown level => least trusted
        return len(AUTHORITY_ORDER)


# --------------------------------------------------------------------------- #
# Stage 1: perceive
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PerceptionState:
    """Raw input turned into a normalized, segmented surface form."""
    raw_input: str
    byte_len: int
    segments: Tuple[str, ...]
    modality: str = "text"

    @staticmethod
    def perceive(raw_input: str) -> "PerceptionState":
        segments = tuple(s.strip() for s in raw_input.splitlines() if s.strip())
        return PerceptionState(
            raw_input=raw_input,
            byte_len=len(raw_input.encode("utf-8")),
            segments=segments or (raw_input.strip(),),
        )


# --------------------------------------------------------------------------- #
# Stage 2: represent meaning
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Claim:
    """A single (entity, attribute, value) assertion extracted from text."""
    entity: str
    attribute: str
    value: str

    def key(self) -> Tuple[str, str]:
        return (self.entity.lower(), self.attribute.lower())


@dataclass(frozen=True)
class SemanticState:
    """Meaning extracted from perception: claims, entities, salient concepts."""
    surface: str
    claims: Tuple[Claim, ...] = ()
    entities: Tuple[str, ...] = ()
    concepts: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Stage 3: determine intent
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IntentState:
    """What the user actually wants: task type, target, and output requirements."""
    task_type: str                       # e.g. "answer_fact", "summarize"
    target_entity: Optional[str] = None
    target_attribute: Optional[str] = None
    requires_citation: bool = False
    constraints: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Stage 4: query / retrieve memory
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MemoryQuery:
    """A structured lookup against typed memory / retrieval."""
    entity: Optional[str]
    attribute: Optional[str]
    keywords: Tuple[str, ...] = ()
    min_authority: Optional[AuthorityLevel] = None


@dataclass(frozen=True)
class EvidenceItem:
    """One recalled/retrieved unit, carrying full provenance for auditing."""
    value: str
    source_id: str
    authority: AuthorityLevel
    timestamp: str                       # ISO-8601; used for supersession
    content_hash: str
    score: float = 0.0
    trusted: bool = True                 # passed the provenance/authority gate
    claim_key: Optional[Tuple[str, str]] = None

    @property
    def rank(self) -> int:
        return authority_rank(self.authority)


@dataclass(frozen=True)
class EvidenceSet:
    """The evidence gathered for one query, in retrieval order."""
    query: MemoryQuery
    items: Tuple[EvidenceItem, ...] = ()

    def trusted_items(self) -> Tuple[EvidenceItem, ...]:
        return tuple(i for i in self.items if i.trusted)


# --------------------------------------------------------------------------- #
# Stage 5: reason (resolve competing evidence)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReasoningState:
    """The resolved belief plus an auditable record of what was rejected/why."""
    resolved_value: Optional[str]
    winning_evidence: Optional[EvidenceItem]
    superseded: Tuple[EvidenceItem, ...] = ()
    rejected_untrusted: Tuple[EvidenceItem, ...] = ()
    rationale: str = ""
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Stage 6: plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Plan:
    """An ordered rendering plan derived from intent + reasoning."""
    steps: Tuple[str, ...]
    must_cite: bool = False


# --------------------------------------------------------------------------- #
# Stage 7: communicate (draft)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Draft:
    """A produced answer plus the claims and citations it commits to."""
    text: str
    claims: Tuple[str, ...] = ()
    citations: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Stage 9: revise
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Revision:
    """A record of a targeted revision (only failed content changes)."""
    changed: bool
    before: str
    after: str
    reason: str = ""


# --------------------------------------------------------------------------- #
# Full trace: every intermediate state, logged for inspection/ablation
# --------------------------------------------------------------------------- #

@dataclass
class CognitiveTrace:
    """The complete record of one pass through the cognitive loop.

    Logging every stage is itself part of the architectural claim: the system's
    decision is explainable from typed intermediate states, not a single opaque
    forward pass.
    """
    perception: Optional[PerceptionState] = None
    semantics: Optional[SemanticState] = None
    intent: Optional[IntentState] = None
    memory_query: Optional[MemoryQuery] = None
    evidence: Optional[EvidenceSet] = None
    reasoning: Optional[ReasoningState] = None
    plan: Optional[Plan] = None
    draft: Optional[Draft] = None
    verification: Optional[VerificationReport] = None
    revision: Optional[Revision] = None
    answer: str = ""
    ablations: Dict[str, bool] = field(default_factory=dict)
