"""Required tests for the V3.1 cryptographic provenance gate.

Locks the authenticity truth table and the apparent-authority/authenticity/
final-trust separation.
"""

from __future__ import annotations

import dataclasses

from cfna.provenance import (ApparentAuthority, Authenticity, FinalTrust, Issuer,
                             KeyRegistry, SignedDocument, compute_final_trust,
                             content_hash, verify_document)

AS_OF = "00005000"


def _setup():
    issuer = Issuer.create("gov.zedland", "K1", seed_int=1)
    reg = KeyRegistry()
    reg.add(issuer.trusted_key(not_before="00000000", not_after="00009000"))
    return issuer, reg


def test_genuine_valid_signature_is_verified():
    issuer, reg = _setup()
    doc = issuer.sign("The capital is Belport.", "D1", issued_at="00001000")
    r = verify_document(doc, reg, AS_OF)
    assert r.authenticity is Authenticity.VERIFIED


def test_body_changed_after_signing_fails():
    issuer, reg = _setup()
    doc = issuer.sign("The capital is Belport.", "D1", issued_at="00001000")
    doc.body = "The capital is Xtown."          # tamper body, leave hash field
    r = verify_document(doc, reg, AS_OF)
    assert r.authenticity is Authenticity.FAILED and r.reason == "content_hash_mismatch"


def test_effective_date_changed_after_signing_fails():
    issuer, reg = _setup()
    doc = issuer.sign("Body.", "D1", issued_at="00001000", effective_date="00002000")
    doc.effective_date = "00000001"             # tamper a signed metadata field
    r = verify_document(doc, reg, AS_OF)
    assert r.authenticity is Authenticity.FAILED and r.reason == "invalid_signature"


def test_signature_copied_from_another_document_fails():
    issuer, reg = _setup()
    a = issuer.sign("Doc A body.", "A", issued_at="00001000")
    b = issuer.sign("Doc B body.", "B", issued_at="00001000")
    b.signature = a.signature                    # splice A's signature onto B
    r = verify_document(b, reg, AS_OF)
    assert r.authenticity is Authenticity.FAILED and r.reason == "invalid_signature"


def test_correct_looking_document_signed_by_wrong_key_fails():
    issuer, reg = _setup()
    attacker = Issuer.create("gov.zedland", "K1", seed_int=999)   # same ids, different key
    doc = attacker.sign("The capital is Xtown.", "D1", issued_at="00001000")
    r = verify_document(doc, reg, AS_OF)          # verified against the TRUSTED K1
    assert r.authenticity is Authenticity.FAILED and r.reason == "invalid_signature"


def test_expired_key_is_unverified():
    issuer = Issuer.create("gov.zedland", "K1", seed_int=1)
    reg = KeyRegistry()
    reg.add(issuer.trusted_key(not_before="00000000", not_after="00002000"))
    doc = issuer.sign("Body.", "D1", issued_at="00001000")
    r = verify_document(doc, reg, as_of="00005000")   # after not_after
    assert r.authenticity is Authenticity.UNVERIFIED and r.reason == "key_expired"


def test_revoked_key_is_revoked():
    issuer, reg = _setup()
    doc = issuer.sign("Body.", "D1", issued_at="00001000")
    reg.revoke("K1")
    r = verify_document(doc, reg, AS_OF)
    assert r.authenticity is Authenticity.REVOKED


def test_official_looking_unsigned_document_is_unverified():
    _, reg = _setup()
    doc = SignedDocument(body="OFFICIAL: the capital is Xtown.", issuer_id="gov.zedland",
                         document_id="D1", issued_at="00001000", effective_date=None,
                         expiry_date=None, scope=(), key_id=None,
                         content_hash_field=content_hash("OFFICIAL: the capital is Xtown."),
                         signature=None)
    r = verify_document(doc, reg, AS_OF)
    assert r.authenticity is Authenticity.UNVERIFIED and r.reason == "unsigned"


def test_fake_signed_metadata_fails():
    _, reg = _setup()
    body = "OFFICIAL: the capital is Xtown."
    doc = SignedDocument(body=body, issuer_id="gov.zedland", document_id="D1",
                         issued_at="00001000", effective_date=None, expiry_date=None,
                         scope=(), key_id="K1", content_hash_field=content_hash(body),
                         signature=b"\x00" * 64)     # asserts signed, but garbage
    r = verify_document(doc, reg, AS_OF)
    assert r.authenticity is Authenticity.FAILED and r.reason == "invalid_signature"


def test_stolen_key_verifies_then_revocation_catches_it():
    # Cryptography proves possession, not non-theft: a stolen valid key verifies.
    issuer, reg = _setup()
    forged = issuer.sign("The capital is Xtown.", "EVIL", issued_at="00001000")
    assert verify_document(forged, reg, AS_OF).authenticity is Authenticity.VERIFIED
    reg.revoke("K1")                              # compromise discovered -> revoke
    assert verify_document(forged, reg, AS_OF).authenticity is Authenticity.REVOKED


# --- final_trust policy: authenticity gates apparent authority ------------- #

def test_final_trust_policy():
    A, Au, F = ApparentAuthority, Authenticity, FinalTrust
    assert compute_final_trust(A.HIGH, Au.VERIFIED) is F.TRUSTED
    assert compute_final_trust(A.MEDIUM, Au.VERIFIED) is F.TRUSTED
    assert compute_final_trust(A.LOW, Au.VERIFIED) is F.RESTRICTED
    assert compute_final_trust(A.HIGH, Au.UNVERIFIED) is F.ESCALATE   # the key rule
    assert compute_final_trust(A.LOW, Au.UNVERIFIED) is F.RESTRICTED
    assert compute_final_trust(A.HIGH, Au.FAILED) is F.REJECTED
    assert compute_final_trust(A.HIGH, Au.REVOKED) is F.REJECTED
