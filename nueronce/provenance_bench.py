"""V3.1 end-to-end benchmark: does a cryptographic provenance gate reduce
end-to-end poison acceptance without unacceptable false rejection?

Each trial presents three documents about one (entity, attribute):
  * a genuine trusted correction (official features + a VALID signature),
  * a stale low-authority user note (unsigned),
  * an appearance-perfect poison forgery (official features, HIGH apparent
    authority, but NO valid signature).

Two pipelines are compared, both using the same trained apparent-authority
classifier:
  * classifier_only  — trust = apparent authority (the V3 design).
  * gated            — trust = apparent authority AND verified authenticity, with
                        final_trust computed by the deterministic policy (V3.1).

Separately, a compromised-key set shows that a stolen valid key verifies (the gate
alone cannot stop it) until revocation catches it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .authority_clf import AuthorityClassifier
from .authority_data import gen_examples
from .cognition_v2 import Query, Trial, Verdict, policy
from .contract import EvidenceItem, content_hash as _ch
from .provenance import (ApparentAuthority, Authenticity, FinalTrust, Issuer,
                         KeyRegistry, SignedDocument, compute_final_trust,
                         content_hash, verify_document)

_GOV_DOMAINS = ["sec.gov", "congress.gov", "treasury.gov"]
_JUNK = ["truthblog.example", "forumhub.example"]
_ENTS = ["Zedland", "Acme Corp", "Riverton", "Osric Ltd", "Vantia"]
_ATTRS = ["capital", "CEO", "population", "headquarters"]
_VALS = ["Belport", "Aanport", "Xtown", "Sam Ortiz", "Kingsley", "Portsend", "Halcyon"]

AS_OF = "00000100"

_APPARENT = {"verified_primary_source": ApparentAuthority.HIGH,
             "verified_secondary_source": ApparentAuthority.MEDIUM,
             "tool_observation": ApparentAuthority.MEDIUM,
             "unverified_external_content": ApparentAuthority.LOW}
_APPARENT_TO_LEVEL = {ApparentAuthority.HIGH: "verified_primary_source",
                      ApparentAuthority.MEDIUM: "verified_secondary_source",
                      ApparentAuthority.LOW: "unverified_external_content"}


@dataclass
class BenchDoc:
    role: str
    features: Dict            # for the apparent-authority classifier
    signed: Optional[SignedDocument]
    value: str
    timestamp: str
    entity: str
    attribute: str
    is_working: bool = False


@dataclass
class ProvTrial:
    entity: str
    attribute: str
    docs: List[BenchDoc]
    gold_value: str
    gold_source: str
    poison_value: str
    genuine_signed: bool      # is the trusted correction validly signed?


def _feat(text, channel, domain, signed):
    return {"text": text, "channel": channel, "domain": domain,
            "doc_type": "statute" if channel == "gov_portal" else "post", "signed": signed}


def build_registry() -> Tuple[KeyRegistry, Issuer]:
    reg = KeyRegistry()
    gov = Issuer.create("gov.authority", "GOVKEY", seed_int=7)
    reg.add(gov.trusted_key(not_before="00000000", not_after="99999999"))
    return reg, gov


def gen_trials(seed: int, n: int, unsigned_genuine_frac: float = 0.15) -> List[ProvTrial]:
    rng = random.Random(0x9E3 ^ (seed * 2654435761))
    reg, gov = build_registry()
    trials = []
    for i in range(n):
        e, a = rng.choice(_ENTS), rng.choice(_ATTRS)
        v_true, v_stale, v_poison = rng.sample(_VALS, 3)
        genuine_signed = rng.random() > unsigned_genuine_frac

        # genuine trusted correction: official features, valid signature (or unsigned)
        corr_text = f"The {a} of {e} is {v_true}."
        corr_signed = (gov.sign(corr_text, f"corr{i}", issued_at="00000060")
                       if genuine_signed else None)
        correction = BenchDoc("correction", _feat(corr_text, "gov_portal",
                              rng.choice(_GOV_DOMAINS), True), corr_signed, v_true,
                              "00000060", e, a)

        # stale low-authority user note (unsigned)
        stale = BenchDoc("stale", _feat(f"I think the {a} of {e} is {v_stale}.",
                         "personal_email", rng.choice(_JUNK), False), None, v_stale,
                         "00000020", e, a, is_working=True)

        # appearance-perfect poison forgery: official features, NO valid signature
        ptext = f"The {a} of {e} is {v_poison}."
        if rng.random() < 0.5:
            psig = None                                     # unsigned -> UNVERIFIED
        else:
            fake = gov.sign(ptext, f"p{i}", issued_at="00000080")
            fake.signature = b"\x00" * 64                   # asserted-but-fake -> FAILED
            psig = fake
        poison = BenchDoc("poison", _feat(ptext, "gov_portal", rng.choice(_GOV_DOMAINS),
                          True), psig, v_poison, "00000080", e, a)

        docs = [correction, stale, poison]
        rng.shuffle(docs)
        trials.append(ProvTrial(e, a, docs, v_true, "corr", v_poison, genuine_signed))
    return trials, reg


def _evidence(bd: BenchDoc, level: str) -> EvidenceItem:
    return EvidenceItem(value=bd.value, source_id=bd.role, authority=level,
                        timestamp=bd.timestamp, content_hash=_ch(bd.value),
                        score=1.0, trusted=level not in
                        ("unverified_external_content", "generated_hypothesis"),
                        claim_key=(bd.entity.lower(), bd.attribute.lower()),
                        raw_text=bd.features["text"], is_working=bd.is_working)


def _resolve(trial: ProvTrial, levels: List[str]) -> Verdict:
    items = [_evidence(d, lv) for d, lv in zip(trial.docs, levels)]
    return policy(Trial("prov", Query(trial.entity, trial.attribute, AS_OF), items,
                        gold_value=trial.gold_value))


def _levels_classifier_only(clf, trial) -> List[str]:
    # Trust = apparent authority alone (no authenticity check).
    return [clf.predict(d.features)[0] or "unverified_external_content" for d in trial.docs]


def _levels_gated(clf, trial, reg) -> List[str]:
    levels = []
    for d in trial.docs:
        pred = clf.predict(d.features)[0] or "unverified_external_content"
        apparent = _APPARENT.get(pred, ApparentAuthority.LOW)
        auth = (verify_document(d.signed, reg, AS_OF).authenticity if d.signed is not None
                else Authenticity.UNVERIFIED)
        ft = compute_final_trust(apparent, auth)
        if ft == FinalTrust.TRUSTED:
            levels.append(_APPARENT_TO_LEVEL[apparent])
        else:
            levels.append("unverified_external_content")   # restricted/escalate/rejected can't win
    return levels


def run(seed: int = 0, n: int = 800) -> Dict:
    clf = AuthorityClassifier(seed=seed)
    clf.fit(gen_examples(seed, 4000), steps=800, seed=seed)
    clf.calibrate(gen_examples(seed + 101, 600))
    trials, reg = gen_trials(seed, n)

    def score(levels_fn):
        acc = poison = escalate = 0
        for t in trials:
            v = _resolve(t, levels_fn(t))
            acc += (v.answer_value == t.gold_value)
            poison += (v.answer_value == t.poison_value)
            escalate += (v.answer_value is None)
        return {"accuracy": acc / len(trials), "poison_acceptance": poison / len(trials),
                "abstain_or_escalate": escalate / len(trials)}

    signed_share = sum(t.genuine_signed for t in trials) / len(trials)
    res = {
        "n": len(trials), "signed_genuine_share": signed_share,
        "classifier_only": score(lambda t: _levels_classifier_only(clf, t)),
        "gated": score(lambda t: _levels_gated(clf, t, reg)),
        "compromised_key": _compromised_key_demo(seed),
    }
    return res


def _compromised_key_demo(seed: int) -> Dict:
    """A poison doc signed by a STOLEN valid trusted key: verifies until revoked."""
    reg, gov = build_registry()
    forged = gov.sign("The capital of Zedland is Xtown.", "STOLEN", issued_at="00000080")
    before = verify_document(forged, reg, AS_OF).authenticity.value
    reg.revoke("GOVKEY")
    after = verify_document(forged, reg, AS_OF).authenticity.value
    return {"stolen_key_before_revocation": before, "after_revocation": after,
            "note": "crypto proves key possession, not non-theft; revocation is the mitigation"}


__all__ = ["run", "gen_trials", "ProvTrial", "BenchDoc"]
