"""V3.2 provenance integration and stratified benchmark tests."""

from __future__ import annotations

from cfna.cognition_v2 import Query, Trial, policy
from cfna.contract import EvidenceItem, content_hash
from cfna.impl import default_verifier_hooks
from cfna.ingestion import IngestionCrawler, PolicyGate
from cfna.evidence import source_to_memory as _candidate_to_memory
from cfna.provenance import Authenticity, Issuer, KeyRegistry
from cfna.provenance_v32 import FAMILIES, run
from cfna.types import MemoryRecord
from cfna.verification import IndependentVerifier


class _Raw:
    def __init__(self, body, meta):
        self.body = body
        self.meta = meta


class _Fetcher:
    def __init__(self, raw):
        self.raw = raw

    def fetch(self, url):
        return self.raw


class _Vault:
    def __init__(self):
        self.stored = []
        self.rejected = []

    def store_raw(self, record, raw):
        self.stored.append(record)

    def record_rejection(self, url, reason, meta):
        self.rejected.append((url, reason, meta))


def _meta(**over):
    base = {
        "robots_status": "allowed",
        "terms_status": "approved",
        "commercial_use": "allowed",
        "pii_risk": 0.0,
        "source_type": "policy",
    }
    base.update(over)
    return base


def test_ingestion_records_verified_provenance():
    issuer = Issuer.create("gov.authority", "GOVKEY", seed_int=7)
    reg = KeyRegistry()
    reg.add(issuer.trusted_key())
    signed = issuer.sign("Official policy.", "D1", issued_at="00000050")
    vault = _Vault()

    crawler = IngestionCrawler(
        PolicyGate(), vault, _Fetcher(_Raw(b"Official policy.", _meta(signed_document=signed))),
        key_registry=reg, verification_as_of="00000100",
    )
    rec = crawler.ingest_url("https://example.test/policy")

    assert rec is not None
    assert rec.authenticity_status == "verified"
    assert rec.issuer_id == "gov.authority"
    assert rec.key_id == "GOVKEY"
    assert rec.provenance_failure_reason is None


def test_contract_rejects_failed_provenance_even_with_high_authority():
    item = EvidenceItem(
        value="Xtown",
        source_id="forged",
        authority="verified_primary_source",
        timestamp="00000100",
        content_hash=content_hash("Xtown"),
        claim_key=("zedland", "capital"),
        raw_text="OFFICIAL: capital is Xtown.",
        authenticity_status="failed",
    )
    trial = Trial("prov", Query("Zedland", "capital", "00000100"), [item], gold_value=None)
    verdict = policy(trial)

    assert verdict.answer_value is None
    assert "forged" in verdict.rejected


def test_verifier_flags_rejected_evidence():
    ev = MemoryRecord(
        memory_id="bad",
        memory_type="semantic",
        content="The capital is Xtown.",
        source_ids=["bad"],
        embeddings={},
        structured_repr={},
        authority_level="verified_primary_source",
        creation_time="now",
        last_verified_time="now",
        confidence=1.0,
        authenticity_status="revoked",
    )
    report = IndependentVerifier(default_verifier_hooks()).verify(
        "The capital is Xtown.", {}, [ev], []
    )

    assert not report.passes
    assert "bad" in report.rejected_evidence
    assert report.provenance_statuses["bad"] == "revoked"


def test_pipeline_memory_preserves_source_provenance():
    issuer = Issuer.create("gov.authority", "GOVKEY", seed_int=7)
    signed = issuer.sign("Official policy.", "D1", issued_at="00000050")
    rec = IngestionCrawler.build_record(
        "https://example.test/policy",
        _meta(signed_document=signed),
        "abc123",
        key_registry=KeyRegistry({"GOVKEY": issuer.trusted_key()}),
        verification_as_of="00000100",
    )

    mem = _candidate_to_memory("Official policy.", "doc0", rec)

    assert mem.authenticity_status == "verified"
    assert mem.issuer_id == "gov.authority"
    assert mem.structured_repr["provenance"]["key_id"] == "GOVKEY"


def test_v32_reports_all_required_families_and_metrics():
    res = run(seed=0)
    assert set(res["families"]) == set(FAMILIES)
    assert len(res["families"]) == 19
    for baseline in (
        "classifier_only",
        "metadata_rules_only",
        "signature_gate_only",
        "classifier_plus_provenance",
        "classifier_plus_provenance_contract",
        "full_retrieval_resolution_verifier",
    ):
        metrics = res["baselines"][baseline]["metrics"]
        for key in (
            "verification_accuracy",
            "genuine_acceptance",
            "forgery_rejection",
            "poison_acceptance",
            "false_rejection",
            "abstention",
            "coverage",
            "selective_accuracy",
            "safe_outcome_rate",
            "compromised_key_acceptance_before_revocation",
            "compromised_key_acceptance_after_revocation",
        ):
            assert key in metrics


def test_v32_gate_and_compromised_key_limits_are_explicit():
    res = run(seed=0)
    contract = res["baselines"]["classifier_plus_provenance_contract"]["metrics"]
    classifier = res["baselines"]["classifier_only"]["metrics"]

    assert contract["poison_acceptance"] <= 0.05
    assert contract["compromised_key_acceptance_before_revocation"] == 1.0
    assert contract["compromised_key_acceptance_after_revocation"] == 0.0
    assert contract["safe_outcome_rate"] > classifier["safe_outcome_rate"]
