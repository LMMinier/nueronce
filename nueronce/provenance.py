"""V3.1 cryptographic provenance gate (Phase 1, narrow).

Separates two questions the earlier authority classifier conflated:

  * apparent_authority  — IF this document were genuine, would its issuer have
                          authority? (estimable from text/metadata; learnable)
  * authenticity        — is this document actually from that issuer? (provable
                          only with cryptography, not with a classifier)

The deterministic policy computes ``final_trust`` from the two. Core rule:

    High apparent authority WITHOUT verified authenticity is not trusted authority.

Signatures use Ed25519 from the ``cryptography`` library (a standard construction —
nothing here invents cryptography). A valid signature proves *possession of the
signing key*; it does not prove the key was not stolen. Key compromise is handled
out of band by revocation, expiry, and rotation — represented in the KeyRegistry.

Subsystem label: **REAL / DETERMINISTIC** (verifiable, not learned).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


class ApparentAuthority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Authenticity(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"     # no/unknown signature — cannot establish, not proven fake
    FAILED = "failed"             # a signature was asserted but did not verify
    REVOKED = "revoked"           # verified, but the signing key is revoked


class FinalTrust(str, Enum):
    TRUSTED = "trusted"
    RESTRICTED = "restricted"
    REJECTED = "rejected"
    ESCALATE = "escalate"


def content_hash(body: str) -> str:
    return "sha256:" + sha256(body.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Documents and keys
# --------------------------------------------------------------------------- #

@dataclass
class SignedDocument:
    body: str
    issuer_id: str
    document_id: str
    issued_at: str
    effective_date: Optional[str]
    expiry_date: Optional[str]
    scope: Tuple[Tuple[str, str], ...]
    key_id: Optional[str]
    content_hash_field: str                 # the hash as carried on the document
    signature: Optional[bytes]              # None => unsigned; b"" or garbage => asserted-but-fake

    def signed_payload(self) -> bytes:
        """Canonical bytes the signature must cover — every field except the
        signature itself. Any tamper (body-hash, dates, scope, issuer, key) changes
        this and invalidates the signature."""
        obj = {
            "issuer_id": self.issuer_id, "document_id": self.document_id,
            "issued_at": self.issued_at, "effective_date": self.effective_date,
            "expiry_date": self.expiry_date, "scope": list(self.scope),
            "key_id": self.key_id, "content_hash": self.content_hash_field,
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class TrustedKey:
    key_id: str
    issuer_id: str
    public_key_raw: bytes                   # Ed25519 raw public bytes
    not_before: str
    not_after: str
    revoked: bool = False

    def public_key(self) -> Ed25519PublicKey:
        return Ed25519PublicKey.from_public_bytes(self.public_key_raw)


@dataclass
class KeyRegistry:
    keys: Dict[str, TrustedKey] = field(default_factory=dict)

    def add(self, key: TrustedKey) -> None:
        self.keys[key.key_id] = key

    def get(self, key_id: Optional[str]) -> Optional[TrustedKey]:
        return self.keys.get(key_id) if key_id else None

    def revoke(self, key_id: str) -> None:
        if key_id in self.keys:
            self.keys[key_id].revoked = True


@dataclass(frozen=True)
class AuthenticityResult:
    authenticity: Authenticity
    reason: str
    issuer_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify_document(doc: SignedDocument, registry: KeyRegistry, as_of: str) -> AuthenticityResult:
    """Return a structured authenticity result. Order of checks matters:
    integrity/signature first, then key trust state (expiry/revocation)."""
    # 1. Unsigned: cannot establish authenticity. Not proven fake.
    if doc.signature is None:
        return AuthenticityResult(Authenticity.UNVERIFIED, "unsigned")

    # 2. Content-hash integrity: recompute from body, compare to the (signed) field.
    if content_hash(doc.body) != doc.content_hash_field:
        return AuthenticityResult(Authenticity.FAILED, "content_hash_mismatch")

    # 3. Resolve the claimed key.
    key = registry.get(doc.key_id)
    if key is None:
        return AuthenticityResult(Authenticity.UNVERIFIED, "unknown_key")

    # 4. Verify the real signature over the canonical payload (catches tampered
    #    dates/scope/metadata, copied signatures, wrong-key signatures, fake sigs).
    try:
        key.public_key().verify(doc.signature, doc.signed_payload())
    except InvalidSignature:
        return AuthenticityResult(Authenticity.FAILED, "invalid_signature")

    # 5. The signature is valid. Bind issuer identity to the key.
    if doc.issuer_id != key.issuer_id:
        return AuthenticityResult(Authenticity.FAILED, "issuer_key_mismatch", key.issuer_id)

    # 6. Key trust state. Revocation dominates (compromise discovered).
    if key.revoked:
        return AuthenticityResult(Authenticity.REVOKED, "key_revoked", key.issuer_id)
    if not (key.not_before <= as_of <= key.not_after):
        return AuthenticityResult(Authenticity.UNVERIFIED, "key_expired", key.issuer_id)

    return AuthenticityResult(Authenticity.VERIFIED, "ok", key.issuer_id)


def compute_final_trust(apparent: ApparentAuthority, auth: Authenticity) -> FinalTrust:
    """Deterministic policy: authenticity gates authority.

    High apparent authority without verified authenticity is NOT trusted authority.
    """
    if auth in (Authenticity.FAILED, Authenticity.REVOKED):
        return FinalTrust.REJECTED
    if auth == Authenticity.VERIFIED:
        return FinalTrust.TRUSTED if apparent in (ApparentAuthority.HIGH,
                                                  ApparentAuthority.MEDIUM) else FinalTrust.RESTRICTED
    # UNVERIFIED: authentic origin not established.
    if apparent == ApparentAuthority.HIGH:
        return FinalTrust.ESCALATE          # important-looking but unprovable -> human review
    return FinalTrust.RESTRICTED


# --------------------------------------------------------------------------- #
# Signing helper (for benchmarks/tests — the issuer side)
# --------------------------------------------------------------------------- #

@dataclass
class Issuer:
    issuer_id: str
    key_id: str
    _private: Ed25519PrivateKey

    @staticmethod
    def create(issuer_id: str, key_id: str, seed_int: int = 0) -> "Issuer":
        # Deterministic-ish key per (seed) so tests are reproducible.
        priv = Ed25519PrivateKey.from_private_bytes(sha256(f"{key_id}:{seed_int}".encode()).digest())
        return Issuer(issuer_id, key_id, priv)

    def public_raw(self) -> bytes:
        return self._private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def trusted_key(self, not_before="00000000", not_after="99999999") -> TrustedKey:
        return TrustedKey(self.key_id, self.issuer_id, self.public_raw(), not_before, not_after)

    def sign(self, body: str, document_id: str, issued_at: str,
             effective_date: Optional[str] = None, expiry_date: Optional[str] = None,
             scope: Tuple[Tuple[str, str], ...] = ()) -> SignedDocument:
        doc = SignedDocument(
            body=body, issuer_id=self.issuer_id, document_id=document_id,
            issued_at=issued_at, effective_date=effective_date, expiry_date=expiry_date,
            scope=scope, key_id=self.key_id, content_hash_field=content_hash(body),
            signature=None)
        doc.signature = self._private.sign(doc.signed_payload())
        return doc


__all__ = [
    "ApparentAuthority", "Authenticity", "FinalTrust", "content_hash",
    "SignedDocument", "TrustedKey", "KeyRegistry", "AuthenticityResult",
    "verify_document", "compute_final_trust", "Issuer",
]
