"""Provenance-aware evidence gating — the bridge between ingestion's
``SourceRecord`` contract and the pipelines' ``MemoryRecord`` evidence.

Torch-free on purpose: this is the exact code ``nueronce.pipeline.respond`` uses
to gate what the neural model may see (extracted verbatim from
``pipeline._candidate_to_memory``), factored out so the engine pipeline
(:mod:`nueronce.pipeline_micro`), the integrated Wave-2 evaluation, and the
provenance tests can all use the *same* gate without importing torch.

``gate_hits`` adds the one new capability the integrated loop needs: an
optional ``trust_unsigned`` hook (the learned authority classifier) that
decides whether an *unverified* (unsigned) document is admitted as evidence
or rejected. Signed-but-failed and revoked documents are always rejected,
exactly as before — the classifier is never allowed to override cryptography.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from . import impl
from .ops import now_iso
from .types import MemoryRecord, SourceRecord


def source_to_memory(
    text: str,
    doc_id: str,
    source: Optional[SourceRecord] = None,
    authority_override: Optional[str] = None,
    confidence_override: Optional[float] = None,
) -> MemoryRecord:
    """Convert a retrieved document (+ its provenance record) into evidence.

    Verbatim port of ``nueronce.pipeline._candidate_to_memory`` semantics:
    failed/revoked provenance -> rejected with zero confidence; unverified ->
    restricted at reduced confidence; otherwise verified-secondary. The two
    override arguments let the authority classifier upgrade an unsigned
    document it trusts (used by ``gate_hits``), without touching the
    signed-provenance rules.
    """
    authority_level = "verified_secondary_source"
    review_status = "verified"
    confidence = 0.8
    provenance = {}
    if source is not None:
        if source.authenticity_status in ("failed", "revoked"):
            authority_level = "unverified_external_content"
            review_status = "rejected"
            confidence = 0.0
        elif source.authenticity_status == "unverified":
            authority_level = "unverified_external_content"
            review_status = "restricted"
            confidence = 0.4
        provenance = {
            "issuer_id": source.issuer_id,
            "key_id": source.key_id,
            "content_hash": source.content_hash,
            "signature": source.signature,
            "authenticity_status": source.authenticity_status,
            "verification_timestamp": source.verification_timestamp,
            "revocation_status": source.revocation_status,
            "provenance_failure_reason": source.provenance_failure_reason,
        }
    if authority_override is not None and review_status != "rejected":
        authority_level = authority_override
        review_status = "verified"
    if confidence_override is not None and review_status != "rejected":
        confidence = confidence_override
    return MemoryRecord(
        memory_id=doc_id, memory_type="semantic", content=text,
        source_ids=[doc_id], embeddings={"dense_semantic": impl.embed_text(text)},
        structured_repr={"provenance": provenance} if provenance else {},
        authority_level=authority_level,
        creation_time=now_iso(), last_verified_time=now_iso(), confidence=confidence,
        review_status=review_status, consolidation_status="semantic", **provenance,
    )


def gate_hits(
    hits,
    id_to_text: Dict[str, str],
    id_to_source: Dict[str, SourceRecord],
    trust_unsigned: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[MemoryRecord], List[str]]:
    """Turn retrieval hits into gated evidence: (accepted MemoryRecords,
    rejected doc_ids).

    - failed / revoked signatures: always rejected (crypto is never overridden)
    - unverified (unsigned): admitted restricted by default; if
      ``trust_unsigned`` is provided, True upgrades the record to
      verified-secondary at confidence 0.7, False rejects it outright —
      this is where the learned authority classifier plugs into the loop.
    """
    evidence: List[MemoryRecord] = []
    rejected: List[str] = []
    for h in hits:
        doc_id = h.bundle.source_id
        if doc_id not in id_to_text:
            continue
        src = id_to_source.get(doc_id)
        if src is not None and src.authenticity_status in ("failed", "revoked"):
            rejected.append(doc_id)
            continue
        authority_override = confidence_override = None
        if (src is not None and src.authenticity_status == "unverified"
                and trust_unsigned is not None):
            if trust_unsigned(doc_id):
                authority_override = "verified_secondary_source"
                confidence_override = 0.7
            else:
                rejected.append(doc_id)
                continue
        evidence.append(source_to_memory(
            id_to_text[doc_id], doc_id, src,
            authority_override=authority_override,
            confidence_override=confidence_override,
        ))
    return evidence, rejected


__all__ = ["source_to_memory", "gate_hits"]
