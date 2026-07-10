"""Core runtime data structures for the NUERONCE hybrid foundational model.

These are the typed records that flow between subsystems (ingestion, perception,
embedding, memory, retrieval, workspace, planning, verification). They are pure
dataclasses with no neural-backend dependency, so they are safe to import and
test anywhere.

The ``Tensor`` alias mirrors the design doc: it is a placeholder for whatever
array type the active backend uses (``numpy.ndarray``, ``torch.Tensor``,
``jax.Array``). We keep it as ``Any`` here so the data model never forces a
heavy dependency at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# Placeholder for torch.Tensor / jax.Array / numpy.ndarray.
Tensor = Any

# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #

AuthorityLevel = Literal[
    "system_policy",
    "verified_primary_source",
    "verified_secondary_source",
    "tool_observation",
    "user_provided_fact",
    "model_inference",
    "unverified_external_content",
    "generated_hypothesis",
]

ProvenanceStatus = Literal["verified", "unverified", "failed", "revoked"]

# Ordered from most to least trusted; index is usable as a priority.
AUTHORITY_ORDER: Tuple[AuthorityLevel, ...] = (
    "system_policy",
    "verified_primary_source",
    "verified_secondary_source",
    "tool_observation",
    "user_provided_fact",
    "model_inference",
    "unverified_external_content",
    "generated_hypothesis",
)

MemoryType = Literal["working", "episodic", "semantic", "procedural"]

UnitType = Literal[
    "definition",
    "claim",
    "evidence",
    "method",
    "result",
    "example",
    "equation",
    "code",
    "caveat",
]

RelationType = Literal[
    "support",
    "contradict",
    "unrelated",
    "depends_on",
    "supersedes",
    "quotes",
    "copies",
]

# The typed recurrent-memory channels (see nueronce.memory).
CHANNELS: Tuple[str, ...] = ("sem", "str", "goal", "evid", "unc", "auth", "proc")
K_CHANNELS: int = len(CHANNELS)


# --------------------------------------------------------------------------- #
# Ingestion / provenance
# --------------------------------------------------------------------------- #

@dataclass
class SourceRecord:
    source_id: str
    canonical_url: Optional[str]
    source_type: str
    title: Optional[str]
    authors: List[str]
    publication_date: Optional[str]
    crawl_timestamp: Optional[str]
    review_status: str
    license: str
    commercial_use: Literal["allowed", "prohibited", "unknown"]
    derivatives: Literal["allowed", "prohibited", "unknown"]
    redistribution: Literal["allowed", "prohibited", "unknown"]
    robots_status: Literal["allowed", "blocked", "unknown"]
    terms_status: Literal["approved", "blocked", "review_required"]
    authority_scope: str
    content_hash: str
    issuer_id: Optional[str] = None
    key_id: Optional[str] = None
    signature: Optional[bytes] = None
    authenticity_status: ProvenanceStatus = "unverified"
    verification_timestamp: Optional[str] = None
    revocation_status: str = "not_checked"
    provenance_failure_reason: Optional[str] = None
    lineage_parent_id: Optional[str] = None
    quality_score: float = 0.0
    pii_risk: float = 0.0


@dataclass
class KnowledgeUnit:
    unit_id: str
    source_id: str
    unit_type: UnitType
    text: str
    byte_span: Tuple[int, int]
    section_path: List[str]
    concepts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    contradiction_group: Optional[str] = None
    temporal_scope: Optional[Tuple[str, Optional[str]]] = None
    confidence_target: float = 0.5


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

@dataclass
class CognitiveEmbeddingBundle:
    """One information unit mapped to several *typed* representations.

    The central thesis of NUERONCE: a single cosine space cannot safely encode
    semantic similarity, exact lexical identity, provenance, temporal validity,
    contradiction, and authority as if they were the same relation.
    """

    dense_semantic: Tensor          # [d_sem], e.g. 768
    sparse_lexical: Dict[int, float]
    structural_vec: Tensor          # [d_struct]
    hierarchical_vec: Tensor        # [d_hier]
    temporal_vec: Tensor            # [d_time]
    provenance_vec: Tensor          # [d_prov]
    belief_vec: Tensor              # [3] => support / contradict / unresolved
    uncertainty: float
    authority_mask: Tensor          # [K_CHANNELS]
    source_id: str
    unit_id: str


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

@dataclass
class MemoryRecord:
    memory_id: str
    memory_type: MemoryType
    content: str
    source_ids: List[str]
    embeddings: Dict[str, Any]      # dense / sparse / structural
    structured_repr: Dict[str, Any]
    authority_level: AuthorityLevel
    creation_time: str
    last_verified_time: Optional[str]
    confidence: float
    contradiction_links: List[str] = field(default_factory=list)
    evidence_links: List[str] = field(default_factory=list)
    user_scope: Optional[str] = None
    privacy_scope: str = "session"
    expiration_time: Optional[str] = None
    review_status: str = "unverified"
    consolidation_status: str = "episodic_only"
    issuer_id: Optional[str] = None
    key_id: Optional[str] = None
    content_hash: Optional[str] = None
    signature: Optional[bytes] = None
    authenticity_status: ProvenanceStatus = "unverified"
    verification_timestamp: Optional[str] = None
    revocation_status: str = "not_checked"
    provenance_failure_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Task / reasoning / planning
# --------------------------------------------------------------------------- #

@dataclass
class TaskState:
    literal_request: str
    inferred_goal: str
    expected_output: str
    required_precision: float
    stakes: float
    ambiguity: float
    constraints: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_sources: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)


@dataclass
class WorkspaceSlot:
    slot_id: int
    role: str
    latent_state: Tensor            # [d_model]
    represented_hypothesis: Optional[str] = None
    supporting_evidence: List[str] = field(default_factory=list)
    opposing_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    dependencies: List[int] = field(default_factory=list)
    status: str = "free"


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

@dataclass
class VerificationFailure:
    category: str
    target_claim: Optional[str]
    severity: float
    instruction: str


@dataclass
class VerificationReport:
    passes: bool
    failures: List[VerificationFailure]
    supported_claim_fraction: float
    contradiction_fraction: float
    calibration_error: float
    provenance_statuses: Dict[str, ProvenanceStatus] = field(default_factory=dict)
    rejected_evidence: List[str] = field(default_factory=list)


__all__ = [
    "Tensor",
    "AuthorityLevel",
    "ProvenanceStatus",
    "AUTHORITY_ORDER",
    "MemoryType",
    "UnitType",
    "RelationType",
    "CHANNELS",
    "K_CHANNELS",
    "SourceRecord",
    "KnowledgeUnit",
    "CognitiveEmbeddingBundle",
    "MemoryRecord",
    "TaskState",
    "WorkspaceSlot",
    "VerificationFailure",
    "VerificationReport",
]
