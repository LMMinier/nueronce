"""Policy-aware ingestion and the provenance gate.

The notes require rights-aware acquisition and a provenance ledger: a policy
engine that distinguishes what may be crawled, stored, indexed, used for
retrieval, or used for weight training. The :class:`PolicyGate` logic and the
:class:`IngestionCrawler` orchestration are fully implemented here; the network
fetcher and source vault are injected protocols so the gate logic is testable in
isolation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

from .ops import now_iso, sha256_bytes
from .provenance import Authenticity, KeyRegistry, SignedDocument, verify_document
from .types import SourceRecord


class Fetcher(Protocol):
    def fetch(self, url: str) -> Any:  # returns an object with a .body: bytes
        ...


class SourceVault(Protocol):
    def store_raw(self, record: SourceRecord, raw: Any) -> None: ...
    def record_rejection(self, url: str, reason: str, meta: dict) -> None: ...


class PolicyGate:
    """Decides whether a fetched resource is allowed past the provenance gate."""

    def __init__(self, pii_threshold: float = 0.2):
        self.pii_threshold = pii_threshold

    def allow(self, url: str, meta: dict) -> Tuple[bool, str]:
        if meta.get("robots_status") == "blocked":
            return False, "robots_block"
        if meta.get("terms_status") == "blocked":
            return False, "terms_block"
        if meta.get("commercial_use") == "prohibited":
            return False, "license_block"
        if float(meta.get("pii_risk", 0.0)) > self.pii_threshold:
            return False, "pii_risk"
        return True, "ok"


def score_quality(meta: dict) -> float:
    """Heuristic source-quality score in [0, 1]."""
    score = 0.0
    score += 0.3 if meta.get("review_status") == "peer_reviewed" else 0.0
    score += 0.2 if meta.get("publisher_verified") else 0.0
    score += 0.2 if meta.get("citations_present") else 0.0
    score += 0.1 if meta.get("structured_sections") else 0.0
    score += 0.2 * max(0.0, 1.0 - float(meta.get("spam_risk", 0.0)))
    return min(score, 1.0)


class IngestionCrawler:
    def __init__(
        self,
        policy_gate: PolicyGate,
        source_vault: SourceVault,
        fetcher: Fetcher,
        key_registry: Optional[KeyRegistry] = None,
        verification_as_of: str = "99999999",
    ):
        self.policy_gate = policy_gate
        self.source_vault = source_vault
        self.fetcher = fetcher
        self.key_registry = key_registry
        self.verification_as_of = verification_as_of

    def ingest_url(self, url: str) -> Optional[SourceRecord]:
        raw = self.fetcher.fetch(url)
        meta = self._extract_source_meta(raw)
        ok, reason = self.policy_gate.allow(url, meta)
        if not ok:
            self.source_vault.record_rejection(url, reason, meta)
            return None

        raw_hash = sha256_bytes(raw.body)
        record = self.build_record(
            url, meta, raw_hash, self.key_registry, self.verification_as_of
        )
        self.source_vault.store_raw(record, raw)
        return record

    @staticmethod
    def build_record(
        url: str,
        meta: Dict[str, Any],
        raw_hash: str,
        key_registry: Optional[KeyRegistry] = None,
        verification_as_of: str = "99999999",
    ) -> SourceRecord:
        signed_doc = meta.get("signed_document")
        authenticity_status = "unverified"
        failure_reason = None
        issuer_id = meta.get("issuer_id")
        key_id = meta.get("key_id")
        signature = meta.get("signature")
        revocation_status = "not_checked"

        if isinstance(signed_doc, SignedDocument):
            issuer_id = signed_doc.issuer_id
            key_id = signed_doc.key_id
            signature = signed_doc.signature
            if key_registry is not None:
                result = verify_document(signed_doc, key_registry, verification_as_of)
                authenticity_status = result.authenticity.value
                failure_reason = None if result.reason == "ok" else result.reason
                revocation_status = (
                    "revoked" if result.authenticity is Authenticity.REVOKED else "not_revoked"
                )

        return SourceRecord(
            source_id=f"sha256:{raw_hash}",
            canonical_url=url,
            source_type=meta["source_type"],
            title=meta.get("title"),
            authors=meta.get("authors", []),
            publication_date=meta.get("publication_date"),
            crawl_timestamp=now_iso(),
            review_status=meta.get("review_status", "unknown"),
            license=meta.get("license", "unknown"),
            commercial_use=meta.get("commercial_use", "unknown"),
            derivatives=meta.get("derivatives", "unknown"),
            redistribution=meta.get("redistribution", "unknown"),
            robots_status=meta.get("robots_status", "unknown"),
            terms_status=meta.get("terms_status", "review_required"),
            authority_scope=meta.get("authority_scope", "evidence_only"),
            content_hash=f"sha256:{raw_hash}",
            issuer_id=issuer_id,
            key_id=key_id,
            signature=signature,
            authenticity_status=authenticity_status,
            verification_timestamp=now_iso() if signed_doc is not None else None,
            revocation_status=revocation_status,
            provenance_failure_reason=failure_reason,
            lineage_parent_id=meta.get("lineage_parent_id"),
            quality_score=score_quality(meta),
            pii_risk=float(meta.get("pii_risk", 0.0)),
        )

    def _extract_source_meta(self, raw: Any) -> dict:
        """Parse html/pdf headers, robots, terms, publisher, schema.org, doi,
        citation tags, plus optional license/domain classifiers.

        Override or inject a parser for real crawls; the default expects the
        fetcher to attach a ``.meta`` dict.
        """
        meta = getattr(raw, "meta", None)
        if meta is None:
            raise NotImplementedError(
                "Provide source metadata extraction: attach a .meta dict to the "
                "fetched object, or subclass and override _extract_source_meta."
            )
        return dict(meta)


__all__ = [
    "Fetcher",
    "SourceVault",
    "PolicyGate",
    "IngestionCrawler",
    "score_quality",
]
