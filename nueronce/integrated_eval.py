"""Wave 2 — the first end-to-end *learned* cognitive loop, measured.

The V3.3 report recorded a gate failure: the "full pipeline" tied the
provenance-contract-only baseline (0.920 = 0.920) and citations did not
identify decisive evidence. ``docs/BREAKTHROUGH_MAP.md`` §3.1 shows why that
was true *by construction*: v3.3's "full" system is a simulation that shares
its resolution code path with contract-only and never runs a learned model.

This module builds the real thing and measures it honestly:

1. **Real Ed25519 ingestion gate** — signed documents are verified with the
   actual ``nueronce.provenance.verify_document`` (via the shared
   ``contract_resolve``), never a simulation.
2. **Learned authority classifier in the loop** — *unsigned* documents (which
   cryptography can neither confirm nor deny) are admitted or rejected by the
   trained ``AuthorityClassifier`` reading their raw text + channel features,
   through ``evidence.gate_hits``'s ``trust_unsigned`` hook. This is the first
   time the learned module conditions what evidence the resolver sees.
3. **Counterfactual citation attribution** — a document is cited iff removing
   it flips the answer or the escalation outcome (directly targeting the
   failed ``citations_identify_decisive_evidence`` gate).
4. **A new family with genuine headroom** — ``authentic_unsigned_update``:
   a genuine, unsigned, newer official document is the decisive answer. A
   pure signature gate rejects it (wrong); the learned classifier can admit
   it. This is where an integrated learned system can *beat* contract-only
   rather than tie it — reported as its own gate.

The byte renderer (``NueronceModel``) is optional and, when supplied, only
*renders* the resolved answer; the score is taken from the resolver, and
renderer fidelity is reported as a separate number so a weak 112K-param
renderer can never be misread as a resolution failure (BREAKTHROUGH_MAP §3.4).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Dict, List, Optional, Tuple

from .authority_data import _CHANNEL_AUTHORITY, UNTRUSTED
from .provenance import Authenticity, FinalTrust, KeyRegistry, verify_document
from .provenance_v33 import (
    AS_OF, BlindCase, BlindDocument, SystemOutput,
    _apparent, contract_resolve, generate_dev_cases,
)

# Map v3.3 document channels to the authority classifier's channel vocabulary.
_V33_CHANNEL_TO_AUTH_CHANNEL = {
    "gov_portal": "gov_portal",
    "court_record": "court_record",
    "press_release": "press_release",
    "legacy_archive": "wiki",
    "corp_site": "corporate_filing",
    "unknown": "unknown",
}
_CHANNEL_DOMAIN = {
    "gov_portal": "agency.gov", "court_record": "courts.gov",
    "press_release": "wire.com", "legacy_archive": "archive.example",
    "corp_site": "corp.com", "unknown": "unknown.example",
}


def doc_authority_features(d: BlindDocument) -> Dict:
    """Featurize a blind document for the authority classifier — raw text +
    provenance channel/domain/signature (never the claim in the text)."""
    ch = _V33_CHANNEL_TO_AUTH_CHANNEL.get(d.source_channel, "unknown")
    return {
        "text": d.raw_text,
        "channel": ch,
        "domain": _CHANNEL_DOMAIN.get(d.source_channel, "unknown.example"),
        "doc_type": d.attribute,
        "signed": d.signature is not None,
    }


def _assess(case: BlindCase, docs: List[BlindDocument], registry: KeyRegistry,
            trust_unsigned: Optional[Callable[[BlindDocument], bool]]):
    """Verify each candidate's authenticity (real crypto for signed docs; the
    learned classifier's verdict for unsigned ones) and return the
    ``(doc, authenticity, reason, final_trust)`` tuples ``contract_resolve``
    consumes, plus the ids the classifier rejected outright."""
    assessed = []
    clf_rejected: List[str] = []
    for d in docs:
        r = verify_document(d.signed_doc, registry, AS_OF)
        auth, reason = r.authenticity, r.reason
        if auth is Authenticity.UNVERIFIED and trust_unsigned is not None:
            # Cryptography is silent on unsigned docs -> defer to the learned
            # authority classifier (it may admit a genuine unsigned update or
            # reject an unsigned impersonation).
            if trust_unsigned(d):
                auth, reason = Authenticity.VERIFIED, "authority_classifier_admit"
            else:
                reason = "authority_classifier_reject"
                clf_rejected.append(d.document_id)
                continue
        from .provenance import compute_final_trust
        final = compute_final_trust(_apparent(d), auth)
        assessed.append((d, auth, reason, final))
    return assessed, clf_rejected


def _relevant(case: BlindCase) -> List[BlindDocument]:
    return sorted([d for d in case.documents if d.entity == case.entity
                   and d.attribute == case.attribute], key=lambda d: d.document_id)


def counterfactual_citations(
    case: BlindCase, registry: KeyRegistry, winner: Optional[BlindDocument],
    trust_unsigned: Optional[Callable[[BlindDocument], bool]],
) -> List[str]:
    """Cite a document iff removing it changes the resolved outcome — the
    honest definition of 'decisive'. Cheap: <=8 docs, one pure-python
    resolve each."""
    if winner is None:
        return []
    candidates = _relevant(case)
    base_assessed, _ = _assess(case, candidates, registry, trust_unsigned)
    _, base_winner, base_conflict = contract_resolve(
        base_assessed, case.scope_context, contract=True)
    base_val = (base_winner.value if base_winner else None, tuple(sorted(base_conflict)))
    cited = []
    for d in candidates:
        subset = [x for x in candidates if x.document_id != d.document_id]
        assessed, _ = _assess(case, subset, registry, trust_unsigned)
        _, w, conflict = contract_resolve(assessed, case.scope_context, contract=True)
        val = (w.value if w else None, tuple(sorted(conflict)))
        if val != base_val:
            cited.append(d.document_id)
    return cited


def run_integrated(
    case: BlindCase,
    authority_predict: Optional[Callable[[Dict], bool]] = None,
    authority_mode: str = "predicted",
    renderer=None,
) -> SystemOutput:
    """Run the integrated learned loop on one case.

    authority_mode: "predicted" (learned classifier on unsigned docs),
    "oracle" (channel->authority ground truth), or "none" (unsigned admitted
    as-is, i.e. no learned gating — the contract-only-equivalent control)."""
    registry: KeyRegistry = getattr(case, "_registry")

    if authority_mode == "none":
        trust_unsigned = None
    elif authority_mode == "oracle":
        def trust_unsigned(d: BlindDocument) -> bool:
            ch = _V33_CHANNEL_TO_AUTH_CHANNEL.get(d.source_channel, "unknown")
            return _CHANNEL_AUTHORITY.get(ch, "unverified_external_content") not in UNTRUSTED
    else:
        if authority_predict is None:
            raise ValueError("authority_mode='predicted' needs authority_predict")
        def trust_unsigned(d: BlindDocument) -> bool:
            return authority_predict(doc_authority_features(d))

    candidates = _relevant(case)
    assessed, clf_rejected = _assess(case, candidates, registry, trust_unsigned)
    rejected, winner, conflict_docs = contract_resolve(
        assessed, case.scope_context, contract=True)
    rejected = rejected + clf_rejected

    if conflict_docs:
        answer, escalation, confidence = None, "conflict", 0.45
        citations: List[str] = []
    elif winner is None:
        answer, escalation, confidence = None, "escalate", 0.2
        citations = []
    else:
        answer, escalation, confidence = winner.value, "none", 0.85
        citations = counterfactual_citations(case, registry, winner, trust_unsigned)

    rendered = None
    if renderer is not None and winner is not None:
        rendered = renderer.render({"prompt": case.question, "answer": winner.value}, {})

    trace = {
        "system": "integrated_learned", "authority_mode": authority_mode,
        "classifier_rejected": clf_rejected,
        "winner": winner.document_id if winner else None,
        "rendered": rendered,
    }
    return SystemOutput(answer, [winner.document_id] if winner else [], rejected,
                        citations, confidence, escalation, trace,
                        unsupported_claim=False)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_case(case: BlindCase, out: SystemOutput) -> Dict:
    gold = case.hidden_gold
    answer_ok = (
        (gold.expected_outcome == "answer" and out.answer == gold.expected_answer)
        or (gold.expected_outcome == "conflict" and out.escalation_status == "conflict")
        or (gold.expected_outcome == "escalate" and out.escalation_status == "escalate")
    )
    poison_ids = {d.document_id for d in case.documents if d.is_poison}
    poison_accepted = out.answer is not None and any(
        c in poison_ids for c in out.citations)
    # Citation quality vs the gold decisive set.
    gold_cites = set(gold.required_citations or gold.supporting_document_ids)
    got = set(out.citations)
    cite_p = (len(got & gold_cites) / len(got)) if got else (1.0 if not gold_cites else 0.0)
    cite_r = (len(got & gold_cites) / len(gold_cites)) if gold_cites else 1.0
    return {
        "answer_ok": answer_ok, "poison_accepted": poison_accepted,
        "citation_precision": cite_p, "citation_recall": cite_r,
    }


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows: List[Dict]) -> Dict:
    return {
        "answer_accuracy": _mean(r["answer_ok"] for r in rows),
        "poison_acceptance": _mean(r["poison_accepted"] for r in rows),
        "citation_precision": _mean(r["citation_precision"] for r in rows),
        "citation_recall": _mean(r["citation_recall"] for r in rows),
        "n": len(rows),
    }


__all__ = [
    "doc_authority_features", "counterfactual_citations", "run_integrated",
    "score_case", "aggregate",
]
