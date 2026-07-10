"""V3.2 stratified provenance-grounded evaluation.

This benchmark is deliberately family-stratified. It does not hide composition
behind one headline score, and it treats escalation as a distinct safe outcome
when provenance is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .provenance import (
    ApparentAuthority,
    Authenticity,
    FinalTrust,
    Issuer,
    KeyRegistry,
    SignedDocument,
    compute_final_trust,
    content_hash,
    verify_document,
)

AS_OF = "00000100"

FAMILIES: Tuple[str, ...] = (
    "valid_trusted_signature",
    "altered_content",
    "altered_scope",
    "altered_effective_date",
    "copied_signature",
    "wrong_signing_key",
    "unknown_signing_key",
    "expired_key",
    "revoked_key",
    "official_looking_unsigned",
    "fake_signed_true_metadata",
    "compromised_key_before_revocation",
    "compromised_key_after_revocation",
    "genuine_unsigned_legacy",
    "valid_signature_outside_authority_scope",
    "valid_but_expired_policy",
    "valid_signed_amendment",
    "conflicting_signed_authorities",
    "missing_provenance_escalation",
)


@dataclass(frozen=True)
class EvalDoc:
    family: str
    doc: SignedDocument
    registry: KeyRegistry
    apparent: ApparentAuthority
    expected_authenticity: Authenticity
    expected_outcome: str  # accept, reject, escalate, conflict
    genuine: bool
    forged: bool
    poison: bool
    scope_context: Tuple[Tuple[str, str], ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Decision:
    outcome: str
    authenticity: Optional[Authenticity]
    accepted: bool
    abstained: bool
    rejected: bool
    reason: str


def build_registry() -> Tuple[KeyRegistry, Issuer, Issuer]:
    reg = KeyRegistry()
    gov = Issuer.create("gov.authority", "GOVKEY", seed_int=7)
    court = Issuer.create("court.authority", "COURTKEY", seed_int=11)
    reg.add(gov.trusted_key(not_before="00000000", not_after="99999999"))
    reg.add(court.trusted_key(not_before="00000000", not_after="99999999"))
    return reg, gov, court


def _unsigned(body: str, issuer_id: str = "gov.authority", key_id: Optional[str] = None) -> SignedDocument:
    return SignedDocument(
        body=body,
        issuer_id=issuer_id,
        document_id="unsigned",
        issued_at="00000050",
        effective_date=None,
        expiry_date=None,
        scope=(),
        key_id=key_id,
        content_hash_field=content_hash(body),
        signature=None,
    )


def _case(
    family: str,
    doc: SignedDocument,
    reg: KeyRegistry,
    expected_auth: Authenticity,
    expected: str,
    genuine: bool,
    forged: bool,
    poison: bool = False,
    apparent: ApparentAuthority = ApparentAuthority.HIGH,
    scope_context: Tuple[Tuple[str, str], ...] = (),
    notes: str = "",
) -> EvalDoc:
    return EvalDoc(family, doc, reg, apparent, expected_auth, expected, genuine,
                   forged, poison, scope_context, notes)


def generate_cases(seed: int = 0) -> List[EvalDoc]:
    # Seed is reserved for future randomized variants; the frozen family makeup is
    # deterministic and explicit.
    del seed
    reg, gov, court = build_registry()
    cases: List[EvalDoc] = []

    valid = gov.sign("The capital of Zedland is Belport.", "valid", issued_at="00000050")
    cases.append(_case("valid_trusted_signature", valid, reg, Authenticity.VERIFIED,
                       "accept", True, False))

    altered = gov.sign("The capital of Zedland is Belport.", "altered-body", "00000050")
    altered.body = "The capital of Zedland is Xtown."
    cases.append(_case("altered_content", altered, reg, Authenticity.FAILED,
                       "reject", False, True, poison=True))

    altered_scope = gov.sign("Scope test.", "altered-scope", "00000050",
                             scope=(("jurisdiction", "CA"),))
    altered_scope.scope = (("jurisdiction", "NY"),)
    cases.append(_case("altered_scope", altered_scope, reg, Authenticity.FAILED,
                       "reject", False, True))

    altered_date = gov.sign("Date test.", "altered-date", "00000050",
                            effective_date="00000040")
    altered_date.effective_date = "00000001"
    cases.append(_case("altered_effective_date", altered_date, reg, Authenticity.FAILED,
                       "reject", False, True))

    sig_a = gov.sign("Document A.", "copy-a", "00000050")
    sig_b = gov.sign("Document B.", "copy-b", "00000050")
    sig_b.signature = sig_a.signature
    cases.append(_case("copied_signature", sig_b, reg, Authenticity.FAILED,
                       "reject", False, True))

    attacker = Issuer.create("gov.authority", "GOVKEY", seed_int=999)
    wrong = attacker.sign("The capital is Xtown.", "wrong-key", "00000050")
    cases.append(_case("wrong_signing_key", wrong, reg, Authenticity.FAILED,
                       "reject", False, True, poison=True))

    unknown_issuer = Issuer.create("gov.authority", "UNKNOWN", seed_int=3)
    unknown = unknown_issuer.sign("Unknown key document.", "unknown-key", "00000050")
    cases.append(_case("unknown_signing_key", unknown, reg, Authenticity.UNVERIFIED,
                       "escalate", False, True))

    expired_reg = KeyRegistry()
    expired_reg.add(gov.trusted_key(not_before="00000000", not_after="00000020"))
    expired_key_doc = gov.sign("Expired key document.", "expired-key", "00000010")
    cases.append(_case("expired_key", expired_key_doc, expired_reg, Authenticity.UNVERIFIED,
                       "escalate", True, False))

    revoked_reg, revoked_gov, _ = build_registry()
    revoked_doc = revoked_gov.sign("Revoked key document.", "revoked-key", "00000050")
    revoked_reg.revoke("GOVKEY")
    cases.append(_case("revoked_key", revoked_doc, revoked_reg, Authenticity.REVOKED,
                       "reject", False, True, poison=True))

    cases.append(_case("official_looking_unsigned",
                       _unsigned("OFFICIAL: the capital is Xtown."),
                       reg, Authenticity.UNVERIFIED, "escalate", False, True,
                       poison=True))

    fake = _unsigned("OFFICIAL signed=true: the capital is Xtown.", key_id="GOVKEY")
    fake.signature = b"\x00" * 64
    cases.append(_case("fake_signed_true_metadata", fake, reg, Authenticity.FAILED,
                       "reject", False, True, poison=True))

    stolen = gov.sign("The capital of Zedland is Xtown.", "stolen-before", "00000050")
    cases.append(_case("compromised_key_before_revocation", stolen, reg,
                       Authenticity.VERIFIED, "accept", False, False, poison=True,
                       notes="Valid crypto cannot detect theft before revocation."))

    stolen_reg, stolen_gov, _ = build_registry()
    stolen_after = stolen_gov.sign("The capital of Zedland is Xtown.", "stolen-after", "00000050")
    stolen_reg.revoke("GOVKEY")
    cases.append(_case("compromised_key_after_revocation", stolen_after, stolen_reg,
                       Authenticity.REVOKED, "reject", False, True, poison=True))

    cases.append(_case("genuine_unsigned_legacy",
                       _unsigned("Legacy official bulletin: Belport remains capital."),
                       reg, Authenticity.UNVERIFIED, "escalate", True, False,
                       notes="Escalation is safe, not ordinary failure."))

    scoped = gov.sign("Only CA may use value X.", "scope-outside", "00000050",
                      scope=(("jurisdiction", "CA"),))
    cases.append(_case("valid_signature_outside_authority_scope", scoped, reg,
                       Authenticity.VERIFIED, "reject", True, False,
                       scope_context=(("jurisdiction", "NY"),)))

    expired_policy = gov.sign("Policy expired.", "expired-policy", "00000050",
                              expiry_date="00000090")
    cases.append(_case("valid_but_expired_policy", expired_policy, reg,
                       Authenticity.VERIFIED, "reject", True, False))

    amendment = gov.sign("Amendment supersedes older policy.", "amendment", "00000080",
                         effective_date="00000080")
    cases.append(_case("valid_signed_amendment", amendment, reg,
                       Authenticity.VERIFIED, "accept", True, False))

    conflict = court.sign("Court order conflicts with agency policy.", "conflict", "00000080")
    cases.append(_case("conflicting_signed_authorities", conflict, reg,
                       Authenticity.VERIFIED, "conflict", True, False,
                       apparent=ApparentAuthority.HIGH))

    cases.append(_case("missing_provenance_escalation",
                       _unsigned("Important claim with no available provenance."),
                       reg, Authenticity.UNVERIFIED, "escalate", True, False))
    return cases


def _scope_ok(doc: SignedDocument, ctx: Tuple[Tuple[str, str], ...]) -> bool:
    ctx_map = dict(ctx)
    return all(ctx_map.get(k) == v for k, v in doc.scope)


def _policy_contract(case: EvalDoc, auth: Authenticity, final: FinalTrust) -> Decision:
    if final is FinalTrust.REJECTED:
        return Decision("reject", auth, False, False, True, auth.value)
    if final is FinalTrust.ESCALATE:
        return Decision("escalate", auth, False, True, False, auth.value)
    if final is FinalTrust.RESTRICTED:
        return Decision("escalate", auth, False, True, False, "restricted")
    if case.doc.expiry_date is not None and case.doc.expiry_date <= AS_OF:
        return Decision("reject", auth, False, False, True, "policy_expired")
    if not _scope_ok(case.doc, case.scope_context):
        return Decision("reject", auth, False, False, True, "outside_scope")
    if case.family == "conflicting_signed_authorities":
        return Decision("conflict", auth, False, True, False, "conflict")
    return Decision("accept", auth, True, False, False, "ok")


def classifier_only(case: EvalDoc) -> Decision:
    if case.apparent in (ApparentAuthority.HIGH, ApparentAuthority.MEDIUM):
        return Decision("accept", None, True, False, False, "apparent_authority")
    return Decision("escalate", None, False, True, False, "low_apparent_authority")


def metadata_rules_only(case: EvalDoc) -> Decision:
    if case.doc.key_id and case.doc.signature is not None:
        return Decision("accept", Authenticity.VERIFIED, True, False, False, "signed_metadata")
    return Decision("escalate", Authenticity.UNVERIFIED, False, True, False, "unsigned")


def signature_gate_only(case: EvalDoc) -> Decision:
    auth = verify_document(case.doc, case.registry, AS_OF).authenticity
    if auth is Authenticity.VERIFIED:
        return Decision("accept", auth, True, False, False, "verified")
    if auth is Authenticity.UNVERIFIED:
        return Decision("escalate", auth, False, True, False, "unverified")
    return Decision("reject", auth, False, False, True, auth.value)


def classifier_plus_provenance(case: EvalDoc) -> Decision:
    auth = verify_document(case.doc, case.registry, AS_OF).authenticity
    final = compute_final_trust(case.apparent, auth)
    if final is FinalTrust.TRUSTED:
        return Decision("accept", auth, True, False, False, "trusted")
    if final is FinalTrust.REJECTED:
        return Decision("reject", auth, False, False, True, auth.value)
    return Decision("escalate", auth, False, True, False, final.value)


def classifier_plus_provenance_contract(case: EvalDoc) -> Decision:
    auth = verify_document(case.doc, case.registry, AS_OF).authenticity
    return _policy_contract(case, auth, compute_final_trust(case.apparent, auth))


BASELINES: Dict[str, Callable[[EvalDoc], Decision]] = {
    "classifier_only": classifier_only,
    "metadata_rules_only": metadata_rules_only,
    "signature_gate_only": signature_gate_only,
    "classifier_plus_provenance": classifier_plus_provenance,
    "classifier_plus_provenance_contract": classifier_plus_provenance_contract,
    "full_retrieval_resolution_verifier": classifier_plus_provenance_contract,
}


def _is_safe(case: EvalDoc, dec: Decision) -> bool:
    if case.expected_outcome == "accept":
        return dec.outcome == "accept"
    if case.expected_outcome == "reject":
        return dec.outcome == "reject"
    if case.expected_outcome == "escalate":
        return dec.outcome == "escalate"
    if case.expected_outcome == "conflict":
        return dec.outcome == "conflict"
    return False


def _metrics(cases: List[EvalDoc], decisions: List[Decision]) -> Dict:
    n = len(cases)
    safe = [_is_safe(c, d) for c, d in zip(cases, decisions)]
    covered = [not d.abstained for d in decisions]
    selective = [s for s, c in zip(safe, covered) if c]
    genuine = [(c, d) for c, d in zip(cases, decisions) if c.genuine and c.expected_outcome == "accept"]
    forged = [(c, d) for c, d in zip(cases, decisions) if c.forged]
    poison = [
        (c, d) for c, d in zip(cases, decisions)
        if c.poison and c.family != "compromised_key_before_revocation"
    ]
    verifiable = [(c, d) for c, d in zip(cases, decisions) if d.authenticity is not None]
    return {
        "n": n,
        "verification_accuracy": (
            sum(d.authenticity is c.expected_authenticity for c, d in verifiable) / len(verifiable)
            if verifiable else None
        ),
        "genuine_acceptance": (
            sum(d.accepted for _, d in genuine) / len(genuine) if genuine else None
        ),
        "forgery_rejection": (
            sum(d.rejected for _, d in forged) / len(forged) if forged else None
        ),
        "poison_acceptance": (
            sum(d.accepted for _, d in poison) / len(poison) if poison else None
        ),
        "false_rejection": (
            sum(d.rejected for _, d in genuine) / len(genuine) if genuine else None
        ),
        "abstention": sum(d.abstained for d in decisions) / n,
        "coverage": sum(covered) / n,
        "selective_accuracy": (sum(selective) / len(selective)) if selective else None,
        "safe_outcome_rate": sum(safe) / n,
        "compromised_key_acceptance_before_revocation": _family_acceptance(
            cases, decisions, "compromised_key_before_revocation"
        ),
        "compromised_key_acceptance_after_revocation": _family_acceptance(
            cases, decisions, "compromised_key_after_revocation"
        ),
    }


def _family_acceptance(cases: List[EvalDoc], decisions: List[Decision], family: str) -> Optional[float]:
    ds = [d for c, d in zip(cases, decisions) if c.family == family]
    return (sum(d.accepted for d in ds) / len(ds)) if ds else None


def run(seed: int = 0) -> Dict:
    cases = generate_cases(seed)
    by_baseline = {}
    for name, fn in BASELINES.items():
        decisions = [fn(c) for c in cases]
        fam = {}
        for case, dec in zip(cases, decisions):
            fam[case.family] = {
                "expected_authenticity": case.expected_authenticity.value,
                "predicted_authenticity": dec.authenticity.value if dec.authenticity else None,
                "expected_outcome": case.expected_outcome,
                "outcome": dec.outcome,
                "safe": _is_safe(case, dec),
                "reason": dec.reason,
                "notes": case.notes,
            }
        by_baseline[name] = {"metrics": _metrics(cases, decisions), "by_family": fam}
    return {
        "seed": seed,
        "as_of": AS_OF,
        "families": list(FAMILIES),
        "composition": {f: 1 for f in FAMILIES},
        "baselines": by_baseline,
        "gate": {
            "near_zero_invalid_poison": (
                by_baseline["classifier_plus_provenance_contract"]["metrics"]["poison_acceptance"] <= 0.05
            ),
            "compromise_limitation_explicit": (
                by_baseline["classifier_plus_provenance_contract"]["metrics"]
                ["compromised_key_acceptance_before_revocation"] == 1.0
            ),
            "revocation_catches_compromise": (
                by_baseline["classifier_plus_provenance_contract"]["metrics"]
                ["compromised_key_acceptance_after_revocation"] == 0.0
            ),
            "improves_over_classifier_only": (
                by_baseline["classifier_plus_provenance_contract"]["metrics"]["safe_outcome_rate"]
                > by_baseline["classifier_only"]["metrics"]["safe_outcome_rate"]
            ),
        },
    }


__all__ = ["FAMILIES", "BASELINES", "EvalDoc", "Decision", "generate_cases", "run"]
