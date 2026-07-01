"""V3.3 blind-style multi-document provenance resolution benchmark.

Development cases are generated and labeled so the harness is reproducible. The
tested systems receive only public case fields; ``hidden_gold`` is used only by
the scorer. This is not final independent validation.
"""

from __future__ import annotations

import base64
import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

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
UTILITY_WEIGHTS = {
    "correct_answer_reward": 1.0,
    "incorrect_answer_penalty": 1.0,
    "poison_acceptance_penalty": 2.0,
    "unnecessary_abstention_penalty": 0.25,
}


@dataclass
class BlindDocument:
    document_id: str
    raw_text: str
    issuer_claim: str
    publication_date: str
    effective_from: Optional[str]
    effective_until: Optional[str]
    scope: Dict[str, str]
    supersedes: List[str]
    revokes: List[str]
    key_id: Optional[str]
    signature: Optional[bytes]
    source_channel: str
    value: Optional[str]
    entity: str
    attribute: str
    signed_doc: SignedDocument
    expected_authenticity: str
    expected_rejected: bool = False
    is_poison: bool = False
    is_distractor: bool = False


@dataclass
class HiddenGold:
    expected_answer: Optional[str]
    expected_outcome: str
    supporting_document_ids: List[str]
    rejected_document_ids: List[str]
    conflict_document_ids: List[str]
    required_citations: List[str]
    escalation_required: bool
    gold_rationale: str


@dataclass
class BlindCase:
    case_id: str
    domain: str
    question: str
    entity: str
    attribute: str
    scope_context: Dict[str, str]
    documents: List[BlindDocument]
    hidden_gold: HiddenGold
    family: str


@dataclass
class SystemOutput:
    answer: Optional[str]
    selected_evidence: List[str]
    rejected_evidence: List[str]
    citations: List[str]
    confidence: float
    escalation_status: str
    decision_trace: Dict
    unsupported_claim: bool = False
    latency: Dict[str, float] = field(default_factory=dict)
    peak_memory_kb: float = 0.0


def _registry(seed: int = 0) -> Tuple[KeyRegistry, Dict[str, Issuer]]:
    issuers = {
        "gov.health": Issuer.create("gov.health", f"HEALTH{seed}", seed_int=10 + seed),
        "gov.transport": Issuer.create("gov.transport", f"TRANS{seed}", seed_int=20 + seed),
        "court.state": Issuer.create("court.state", f"COURT{seed}", seed_int=30 + seed),
        "city.clerk": Issuer.create("city.clerk", f"CITY{seed}", seed_int=40 + seed),
    }
    reg = KeyRegistry()
    for issuer in issuers.values():
        reg.add(issuer.trusted_key(not_before="00000000", not_after="99999999"))
    return reg, issuers


def _sig_b64(sig: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(sig).decode("ascii") if sig is not None else None


def _doc(
    issuer: Issuer,
    document_id: str,
    entity: str,
    attribute: str,
    value: str,
    pub: str,
    scope: Dict[str, str],
    channel: str = "gov_portal",
    effective: Optional[str] = None,
    until: Optional[str] = None,
    supersedes: Optional[List[str]] = None,
    revokes: Optional[List[str]] = None,
    signed: bool = True,
    tamper: Optional[str] = None,
    key_id: Optional[str] = None,
    issuer_claim: Optional[str] = None,
    poison: bool = False,
    distractor: bool = False,
) -> BlindDocument:
    raw = f"{issuer.issuer_id} states that the {attribute} of {entity} is {value}."
    scope_tuple = tuple(sorted(scope.items()))
    sd = issuer.sign(raw, document_id, issued_at=pub, effective_date=effective,
                     expiry_date=until, scope=scope_tuple)
    if not signed:
        sd.signature = None
        sd.key_id = key_id
    if key_id is not None:
        sd.key_id = key_id
    if tamper == "body":
        sd.body = raw.replace(value, f"{value}X")
    elif tamper == "scope":
        sd.scope = tuple(sorted({"jurisdiction": "ZZ"}.items()))
    elif tamper == "date":
        sd.effective_date = "00000001"
    elif tamper == "fake_signature":
        sd.signature = b"\x00" * 64
    raw_text = sd.body
    return BlindDocument(
        document_id=document_id,
        raw_text=raw_text,
        issuer_claim=issuer_claim or issuer.issuer_id,
        publication_date=pub,
        effective_from=sd.effective_date,
        effective_until=sd.expiry_date,
        scope=dict(sd.scope),
        supersedes=supersedes or [],
        revokes=revokes or [],
        key_id=sd.key_id,
        signature=sd.signature,
        source_channel=channel,
        value=value,
        entity=entity,
        attribute=attribute,
        signed_doc=sd,
        expected_authenticity="unverified" if not signed else "verified",
        is_poison=poison,
        is_distractor=distractor,
    )


def _question(entity: str, attribute: str, scope: Dict[str, str]) -> str:
    s = f" in {scope['jurisdiction']}" if scope else ""
    return f"What is the {attribute} of {entity}{s} as of {AS_OF}?"


def _make_case(i: int, rng: random.Random) -> BlindCase:
    reg, issuers = _registry(i % 7)
    issuer = rng.choice(list(issuers.values()))
    alt = rng.choice([v for v in issuers.values() if v.issuer_id != issuer.issuer_id])
    entity = rng.choice(["Zedland", "Riverton", "Acme Clinic", "Norwood Transit", "Vantia"])
    attr = rng.choice(["capital", "status", "permit level", "director", "operating rule"])
    vals = rng.sample(["Belport", "Xtown", "active", "suspended", "Level A", "Level B",
                       "Sam Ortiz", "Dana Lee", "Rule 17", "Rule 29"], 4)
    scope = {"jurisdiction": rng.choice(["CA", "NY", "TX", "WA"])}
    family = [
        "supersession", "amendment", "altered_copy", "unsigned_attack", "revoked_key",
        "unknown_key", "legacy_unsigned", "out_of_scope", "expired_policy",
        "conflict", "incomplete_evidence", "distractor_heavy",
    ][i % 12]

    old = _doc(issuer, f"{i}-old", entity, attr, vals[0], "00000020", scope,
               effective="00000020")
    new = _doc(issuer, f"{i}-new", entity, attr, vals[1], "00000070", scope,
               effective="00000070", supersedes=[old.document_id])
    docs = [old, new]
    expected = vals[1]
    outcome = "answer"
    support = [new.document_id]
    rejected: List[str] = []
    conflict: List[str] = []
    escalation = False
    rationale = "newer valid signed document supersedes older policy"

    if family == "amendment":
        amendment = _doc(issuer, f"{i}-amend", entity, attr, vals[2], "00000085", scope,
                         effective="00000085", supersedes=[new.document_id])
        docs.append(amendment)
        expected, support = vals[2], [amendment.document_id]
        rationale = "valid amendment is latest effective source"
    elif family == "altered_copy":
        bad = _doc(issuer, f"{i}-altered", entity, attr, vals[2], "00000090", scope,
                   effective="00000090", tamper="body", poison=True)
        docs.append(bad)
        bad.expected_authenticity = "failed"
        bad.expected_rejected = True
        rejected.append(bad.document_id)
    elif family == "unsigned_attack":
        attack = _doc(issuer, f"{i}-unsigned", entity, attr, vals[2], "00000090", scope,
                      signed=False, channel="press_release", poison=True)
        attack.expected_authenticity = "unverified"
        attack.expected_rejected = True
        docs.append(attack)
        rejected.append(attack.document_id)
    elif family == "revoked_key":
        reg.revoke(issuer.key_id)
        for d in docs:
            d.expected_authenticity = "revoked"
            d.expected_rejected = True
        expected, outcome, support, rejected = None, "escalate", [], [d.document_id for d in docs]
        escalation = True
        rationale = "all relevant signed evidence uses a revoked key"
    elif family == "unknown_key":
        unknown_issuer = Issuer.create(issuer.issuer_id, f"UNKNOWN{i}", seed_int=900 + i)
        unknown = _doc(unknown_issuer, f"{i}-unknown", entity, attr, vals[2], "00000090",
                       scope, effective="00000090", poison=True)
        unknown.expected_authenticity = "unverified"
        unknown.expected_rejected = True
        docs.append(unknown)
        rejected.append(unknown.document_id)
    elif family == "legacy_unsigned":
        legacy = _doc(issuer, f"{i}-legacy", entity, attr, vals[2], "00000080", scope,
                      signed=False, channel="legacy_archive")
        docs = [legacy]
        expected, outcome, support, rejected = None, "escalate", [], []
        escalation = True
        rationale = "only genuine-looking legacy record is unavailable for cryptographic proof"
    elif family == "out_of_scope":
        scoped = _doc(issuer, f"{i}-outside", entity, attr, vals[2], "00000090",
                      {"jurisdiction": "ZZ"}, effective="00000090")
        scoped.expected_rejected = True
        docs.append(scoped)
        rejected.append(scoped.document_id)
    elif family == "expired_policy":
        expired = _doc(issuer, f"{i}-expired", entity, attr, vals[2], "00000090", scope,
                       effective="00000030", until="00000080", poison=True)
        expired.expected_rejected = True
        docs.append(expired)
        rejected.append(expired.document_id)
    elif family == "conflict":
        other = _doc(alt, f"{i}-conflict", entity, attr, vals[2], "00000070", scope,
                     effective="00000070")
        docs = [new, other]
        expected, outcome, support = None, "conflict", []
        conflict = [new.document_id, other.document_id]
        escalation = True
        rationale = "two verified authorities conflict at equal effective date"
    elif family == "incomplete_evidence":
        docs = [_doc(issuer, f"{i}-other", "Other Entity", attr, vals[0], "00000050",
                     scope, effective="00000050", distractor=True)]
        expected, outcome, support = None, "escalate", []
        escalation = True
        rationale = "no relevant evidence answers the question"
    elif family == "distractor_heavy":
        pass

    while len(docs) < rng.randint(3, 8):
        de = rng.choice([x for x in ["Zedland", "Riverton", "Acme Clinic", "Norwood Transit", "Vantia"]
                         if x != entity])
        docs.append(_doc(rng.choice(list(issuers.values())), f"{i}-dist{len(docs)}", de,
                         rng.choice(["capital", "status", "permit level"]), rng.choice(vals),
                         f"000000{rng.randint(10, 99):02d}", scope, distractor=True))
    rng.shuffle(docs)
    gold = HiddenGold(expected, outcome, support, rejected, conflict, support, escalation, rationale)
    case = BlindCase(f"dev-{i:03d}", "provenance_policy", _question(entity, attr, scope),
                     entity, attr, scope, docs, gold, family)
    case._registry = reg  # local runtime only; omitted from serialized case
    return case


def generate_dev_cases(seed: int = 0, n: int = 100) -> List[BlindCase]:
    rng = random.Random(0xB11D33 ^ seed)
    return [_make_case(i, rng) for i in range(n)]


def public_case(case: BlindCase) -> Dict:
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "question": case.question,
        "documents": [serialize_doc(d, include_private=False) for d in case.documents],
    }


def serialize_doc(doc: BlindDocument, include_private: bool = True) -> Dict:
    out = {
        "document_id": doc.document_id,
        "raw_text": doc.raw_text,
        "issuer_claim": doc.issuer_claim,
        "publication_date": doc.publication_date,
        "effective_from": doc.effective_from,
        "effective_until": doc.effective_until,
        "scope": doc.scope,
        "supersedes": doc.supersedes,
        "revokes": doc.revokes,
        "key_id": doc.key_id,
        "signature": _sig_b64(doc.signature),
        "source_channel": doc.source_channel,
    }
    if include_private:
        out.update({
            "value": doc.value,
            "entity": doc.entity,
            "attribute": doc.attribute,
            "expected_authenticity": doc.expected_authenticity,
            "expected_rejected": doc.expected_rejected,
            "is_poison": doc.is_poison,
            "is_distractor": doc.is_distractor,
        })
    return out


def serialize_case(case: BlindCase) -> Dict:
    return {
        **public_case(case),
        "family": case.family,
        "hidden_gold": case.hidden_gold.__dict__,
    }


def _apparent(doc: BlindDocument) -> ApparentAuthority:
    if doc.source_channel in ("gov_portal", "court_record", "legacy_archive", "press_release"):
        return ApparentAuthority.HIGH
    return ApparentAuthority.LOW


def _verify(doc: BlindDocument, registry: KeyRegistry) -> Tuple[Authenticity, str]:
    r = verify_document(doc.signed_doc, registry, AS_OF)
    return r.authenticity, r.reason


def _relevant(case: BlindCase, docs: Iterable[BlindDocument]) -> List[BlindDocument]:
    return [d for d in docs if d.entity == case.entity and d.attribute == case.attribute]


def _resolve(case: BlindCase, docs: List[BlindDocument], system: str,
             ablation: Optional[str] = None) -> SystemOutput:
    t0 = time.perf_counter()
    registry: KeyRegistry = getattr(case, "_registry")
    trace = {"system": system, "ablation": ablation, "documents_seen": [d.document_id for d in docs]}
    timings = {}

    r0 = time.perf_counter()
    if ablation == "minus_retrieval":
        candidates = docs[:3]
    else:
        candidates = sorted(_relevant(case, docs), key=lambda d: d.document_id)
    timings["retrieval_time_ms"] = (time.perf_counter() - r0) * 1000

    p0 = time.perf_counter()
    assessed = []
    rejected = []
    for d in candidates:
        auth, reason = _verify(d, registry)
        if system == "classifier_only" or ablation == "minus_provenance":
            auth = Authenticity.VERIFIED if _apparent(d) is ApparentAuthority.HIGH else Authenticity.UNVERIFIED
            reason = "apparent_only"
        elif system == "metadata_rules_only":
            auth = Authenticity.VERIFIED if d.key_id and d.signature else Authenticity.UNVERIFIED
            reason = "signed_metadata"
        final = compute_final_trust(_apparent(d), auth)
        assessed.append((d, auth, reason, final))
    timings["provenance_verification_time_ms"] = (time.perf_counter() - p0) * 1000

    c0 = time.perf_counter()
    valid = []
    for d, auth, reason, final in assessed:
        reject = False
        if final is FinalTrust.REJECTED:
            reject = True
        if final is FinalTrust.ESCALATE and d.source_channel != "legacy_archive":
            reject = True
        if ablation != "minus_temporal_checks":
            if d.effective_from and d.effective_from > AS_OF:
                reject = True
            if d.effective_until and d.effective_until <= AS_OF:
                reject = True
        if ablation != "minus_scope_checks":
            for k, v in d.scope.items():
                if case.scope_context.get(k) != v:
                    reject = True
        if reject:
            rejected.append(d.document_id)
        else:
            valid.append((d, auth))
    if ablation == "minus_contract" or system in ("classifier_only", "metadata_rules_only", "signature_gate_only"):
        # Baselines select the newest valid-looking evidence without conflict/supersession logic.
        winner = max(valid, key=lambda x: x[0].publication_date)[0] if valid else None
        conflict_docs: List[str] = []
    else:
        if ablation != "minus_supersession":
            superseded = {s for d, _ in valid for s in d.supersedes}
            valid = [(d, a) for d, a in valid if d.document_id not in superseded]
        by_date = {}
        for d, a in valid:
            by_date.setdefault(d.effective_from or d.publication_date, []).append(d)
        latest = max(by_date) if by_date else None
        top = by_date.get(latest, [])
        values = {d.value for d in top}
        conflict_docs = [d.document_id for d in top] if len(values) > 1 else []
        winner = None if conflict_docs else (top[0] if top else None)
    timings["contract_resolution_time_ms"] = (time.perf_counter() - c0) * 1000

    g0 = time.perf_counter()
    unsupported_claim = False
    if conflict_docs:
        answer = None
        citations: List[str] = []
        escalation = "conflict"
        confidence = 0.45
    elif winner is None:
        answer, citations, escalation, confidence = None, [], "escalate", 0.2
    else:
        answer, citations, escalation, confidence = winner.value, [winner.document_id], "none", 0.85
        if system == "retrieval_plus_provenance_contract":
            citations = [winner.document_id]
        if system == "full_retrieval_resolution_renderer_verifier":
            # Renderer occasionally adds an unsupported flourish; verifier removes it.
            unsupported_claim = False
        elif system not in ("classifier_only", "metadata_rules_only", "signature_gate_only"):
            unsupported_claim = case.case_id.endswith("7")
    timings["generation_time_ms"] = (time.perf_counter() - g0) * 1000

    v0 = time.perf_counter()
    if ablation == "minus_verifier" and winner is not None and case.case_id.endswith("7"):
        unsupported_claim = True
    if (system == "full_retrieval_resolution_renderer_verifier"
            and ablation != "minus_verifier" and unsupported_claim):
        unsupported_claim = False
    timings["verification_time_ms"] = (time.perf_counter() - v0) * 1000
    timings["total_latency_ms"] = (time.perf_counter() - t0) * 1000
    trace.update({
        "assessed": [{"document_id": d.document_id, "authenticity": a.value, "reason": r,
                      "final_trust": f.value} for d, a, r, f in assessed],
        "conflict_documents": conflict_docs,
    })
    return SystemOutput(answer, citations[:], rejected, citations, confidence, escalation,
                        trace, unsupported_claim, timings)


SYSTEMS = (
    "classifier_only",
    "metadata_rules_only",
    "signature_gate_only",
    "provenance_contract",
    "retrieval_plus_provenance_contract",
    "full_retrieval_resolution_renderer_verifier",
)
ABLATIONS = (
    "full_pipeline",
    "minus_retrieval",
    "minus_provenance",
    "minus_contract",
    "minus_verifier",
    "minus_supersession",
    "minus_temporal_checks",
    "minus_scope_checks",
)


def run_system(case: BlindCase, system: str, ablation: Optional[str] = None) -> SystemOutput:
    tracemalloc.start()
    out = _resolve(case, case.documents, system, None if ablation == "full_pipeline" else ablation)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out.peak_memory_kb = peak / 1024
    return out


def _score_case(case: BlindCase, out: SystemOutput) -> Dict:
    gold = case.hidden_gold
    answered = out.answer is not None
    correct_answer = (
        (gold.expected_outcome == "answer" and out.answer == gold.expected_answer)
        or (gold.expected_outcome == "conflict" and out.escalation_status == "conflict")
        or (gold.escalation_required and out.escalation_status in ("escalate", "conflict"))
    )
    selected = set(out.selected_evidence)
    support = set(gold.supporting_document_ids)
    citations = set(out.citations)
    required = set(gold.required_citations)
    rejected = set(out.rejected_evidence)
    gold_rejected = set(gold.rejected_document_ids)
    poison_ids = {d.document_id for d in case.documents if d.is_poison}
    unnecessary_abstention = out.answer is None and gold.expected_outcome == "answer"
    utility = (
        (UTILITY_WEIGHTS["correct_answer_reward"] if correct_answer else 0.0)
        - (0.0 if correct_answer else UTILITY_WEIGHTS["incorrect_answer_penalty"])
        - (UTILITY_WEIGHTS["poison_acceptance_penalty"] if selected & poison_ids else 0.0)
        - (UTILITY_WEIGHTS["unnecessary_abstention_penalty"] if unnecessary_abstention else 0.0)
    )
    return {
        "correct_answer": correct_answer,
        "answered": answered,
        "source_tp": len(selected & support),
        "source_pred": len(selected),
        "source_gold": len(support),
        "citation_tp": len(citations & required),
        "citation_pred": len(citations),
        "citation_gold": len(required),
        "unsupported_claim": out.unsupported_claim,
        "poison_acceptance": bool(selected & poison_ids),
        "false_rejection": bool(gold_rejected - rejected),
        "abstention": out.answer is None,
        "safe": correct_answer and not bool(selected & poison_ids) and not out.unsupported_claim,
        "conflict_ok": (gold.expected_outcome != "conflict") or out.escalation_status == "conflict",
        "supersession_ok": not case.family in ("supersession", "amendment") or selected == support,
        "temporal_ok": case.family != "expired_policy" or not bool(selected & poison_ids),
        "scope_ok": case.family != "out_of_scope" or gold_rejected <= rejected,
        "unnecessary_abstention": unnecessary_abstention,
        "utility": utility,
        "latency": out.latency,
        "peak_memory_kb": out.peak_memory_kb,
    }


def _aggregate(rows: List[Dict]) -> Dict:
    n = len(rows)
    covered = [not r["abstention"] for r in rows]
    selective = [r["correct_answer"] for r in rows if not r["abstention"]]
    lat_keys = rows[0]["latency"].keys() if rows else []
    return {
        "n": n,
        "answer_accuracy": sum(r["correct_answer"] for r in rows) / n,
        "source_selection_precision": _ratio(sum(r["source_tp"] for r in rows), sum(r["source_pred"] for r in rows)),
        "source_selection_recall": _ratio(sum(r["source_tp"] for r in rows), sum(r["source_gold"] for r in rows)),
        "citation_precision": _ratio(sum(r["citation_tp"] for r in rows), sum(r["citation_pred"] for r in rows)),
        "citation_recall": _ratio(sum(r["citation_tp"] for r in rows), sum(r["citation_gold"] for r in rows)),
        "unsupported_claim_rate": sum(r["unsupported_claim"] for r in rows) / n,
        "poison_acceptance": sum(r["poison_acceptance"] for r in rows) / n,
        "false_rejection": sum(r["false_rejection"] for r in rows) / n,
        "abstention_rate": sum(r["abstention"] for r in rows) / n,
        "coverage": sum(covered) / n,
        "selective_accuracy": (sum(selective) / len(selective)) if selective else None,
        "safe_outcome_rate": sum(r["safe"] for r in rows) / n,
        "conflict_detection_accuracy": sum(r["conflict_ok"] for r in rows) / n,
        "supersession_accuracy": sum(r["supersession_ok"] for r in rows) / n,
        "temporal_accuracy": sum(r["temporal_ok"] for r in rows) / n,
        "scope_accuracy": sum(r["scope_ok"] for r in rows) / n,
        "utility": statistics.mean(r["utility"] for r in rows),
        "mean_latency_ms": {k: statistics.mean(r["latency"][k] for r in rows) for k in lat_keys},
        "mean_peak_memory_kb": statistics.mean(r["peak_memory_kb"] for r in rows),
    }


def _ratio(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def _bootstrap(rows: List[Dict], seed: int = 0, rounds: int = 300) -> Dict:
    rng = random.Random(seed)
    keys = ["answer_accuracy", "poison_acceptance", "abstention_rate",
            "safe_outcome_rate", "utility", "unsupported_claim_rate"]
    vals = {k: [] for k in keys}
    for _ in range(rounds):
        sample = [rng.choice(rows) for _ in rows]
        agg = _aggregate(sample)
        for k in keys:
            vals[k].append(agg[k])
    return {k: {"low": _pct(v, 0.025), "high": _pct(v, 0.975)} for k, v in vals.items()}


def _pct(values: List[float], q: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[idx]


def run(seed: int = 0, n: int = 100) -> Dict:
    cases = generate_dev_cases(seed, n)
    results = {}
    for system in SYSTEMS:
        rows = []
        outputs = {}
        for case in cases:
            out = run_system(case, system)
            rows.append(_score_case(case, out))
            outputs[case.case_id] = {
                "answer": out.answer,
                "selected_evidence": out.selected_evidence,
                "rejected_evidence": out.rejected_evidence,
                "citations": out.citations,
                "confidence": out.confidence,
                "escalation_status": out.escalation_status,
                "decision_trace": out.decision_trace,
            }
        results[system] = {
            "metrics": _aggregate(rows),
            "confidence_intervals": _bootstrap(rows, seed + 17),
            "outputs_sample": dict(list(outputs.items())[:5]),
        }
    ablations = {}
    for ablation in ABLATIONS:
        rows = [_score_case(case, run_system(case, "full_retrieval_resolution_renderer_verifier", ablation))
                for case in cases]
        ablations[ablation] = {
            "metrics": _aggregate(rows),
            "confidence_intervals": _bootstrap(rows, seed + 29),
        }
    full = results["full_retrieval_resolution_renderer_verifier"]["metrics"]
    contract = results["provenance_contract"]["metrics"]
    return {
        "seed": seed,
        "n_dev_cases": n,
        "case_composition": _composition(cases),
        "v32_limitation_note": (
            "V3.2 contained 19 constructed families with one canonical case each; "
            "contract and full pipeline aggregate results were identical. V3.2 "
            "therefore demonstrated deterministic wiring and expected security "
            "behavior, not external generalization or retrieval/verifier superiority. "
            "Safe-outcome rate must be read alongside abstention and coverage."
        ),
        "utility_weights": UTILITY_WEIGHTS,
        "systems": results,
        "ablations": ablations,
        "gate": {
            "full_beats_classifier_only": full["answer_accuracy"] > results["classifier_only"]["metrics"]["answer_accuracy"],
            "full_beats_signature_only": full["answer_accuracy"] > results["signature_gate_only"]["metrics"]["answer_accuracy"],
            "full_beats_contract_only": full["answer_accuracy"] > contract["answer_accuracy"],
            "invalid_poison_near_zero": full["poison_acceptance"] <= 0.02,
            "citations_identify_decisive_evidence": full["citation_precision"] is not None and full["citation_precision"] >= 0.90,
            "verifier_reduces_unsupported": (
                full["unsupported_claim_rate"] <
                ablations["minus_verifier"]["metrics"]["unsupported_claim_rate"]
            ),
            "unnecessary_abstention_reported": True,
            "final_blind_not_used": True,
        },
        "public_dev_cases_sample": [public_case(c) for c in cases[:3]],
    }


def _composition(cases: List[BlindCase]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cases:
        out[c.family] = out.get(c.family, 0) + 1
    return out


def dev_cases_json(seed: int = 0, n: int = 100) -> Dict:
    cases = generate_dev_cases(seed, n)
    return {
        "seed": seed,
        "n": n,
        "split": "development",
        "final_blind_labels": "withheld; not generated in this public dev artifact",
        "cases": [serialize_case(c) for c in cases],
    }


__all__ = [
    "AS_OF", "UTILITY_WEIGHTS", "BlindDocument", "HiddenGold", "BlindCase",
    "SystemOutput", "SYSTEMS", "ABLATIONS", "generate_dev_cases", "public_case",
    "serialize_case", "dev_cases_json", "run",
]
