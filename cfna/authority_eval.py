"""Evaluate the authority classifier and measure error propagation into the
deterministic V2 contract (Phase 3 -> Phase 7 bridge).

Two questions:
  1. How well does the learned classifier recover ground-truth authority, and how
     robust/calibrated is it (impersonation acceptance, ECE, abstention)?
  2. When the deterministic resolution policy is fed *predicted* authority instead
     of ground-truth metadata, how much end-to-end accuracy is lost? This is the
     honest cost of removing the "authority labels are given" assumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .authority_clf import AuthorityClassifier
from .authority_data import (Doc, DocTrial, UNTRUSTED, gen_examples,
                             gen_resolution_trials)
from .cognition_v2 import Query, Trial, Verdict, policy
from .contract import EvidenceItem, content_hash

_ATTACK_FAMILIES = {"impersonation", "laundering", "paraphrased_official",
                    "metadata_text_conflict", "spoofed_channel", "poison",
                    "spoof_crude", "spoof_medium", "spoof_perfect",
                    "IMPERSONATION", "AUTHORITY_LAUNDERING", "CITATION_SPOOFING",
                    "PARAPHRASED_OFFICIAL", "UNICODE_CONFUSABLE", "METADATA_TEXT_CONFLICT"}


def load_blind(path: str) -> List[Doc]:
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        docs.append(Doc(text=r["text"], source_id=r.get("source_id", ""), channel=r["channel"],
                        domain=r.get("domain", ""), doc_type=r.get("doc_type", ""),
                        signed=bool(r.get("signed", False)), label_authority=r["label_authority"],
                        label_trusted=bool(r["label_trusted"]), family=r.get("family", "?")))
    return docs


def classifier_metrics(clf: AuthorityClassifier, docs: List[Doc],
                       abstain_below: float = 0.0) -> Dict:
    n = len(docs)
    correct = trusted_correct = answered = answered_correct = 0
    confs, hits = [], []
    imp_total = imp_accepted = 0
    genuine_total = genuine_rejected = 0
    by_fam_total: Dict[str, int] = {}
    by_fam_accept: Dict[str, int] = {}
    for d in docs:
        p = clf.predict_proba(d.features())
        idx = int(p.argmax())
        conf = float(p[idx])
        from .authority_data import AUTHORITY_CLASSES
        pred = AUTHORITY_CLASSES[idx]
        pred_trusted = pred not in UNTRUSTED
        correct += (pred == d.label_authority)
        trusted_correct += (pred_trusted == d.label_trusted)
        confs.append(conf); hits.append(pred == d.label_authority)
        if conf >= abstain_below:
            answered += 1
            answered_correct += (pred == d.label_authority)
        # impersonation acceptance: truly-untrusted attack doc predicted trusted
        if d.family in _ATTACK_FAMILIES and not d.label_trusted:
            imp_total += 1
            imp_accepted += pred_trusted
            by_fam_total[d.family] = by_fam_total.get(d.family, 0) + 1
            by_fam_accept[d.family] = by_fam_accept.get(d.family, 0) + int(pred_trusted)
        # false rejection: genuinely trusted doc predicted untrusted (availability cost)
        if d.label_trusted:
            genuine_total += 1
            genuine_rejected += (not pred_trusted)
    return {
        "n": n,
        "accuracy": correct / n,
        "trusted_binary_accuracy": trusted_correct / n,
        "impersonation_acceptance": (imp_accepted / imp_total) if imp_total else 0.0,
        "impersonation_n": imp_total,
        "false_rejection_rate": (genuine_rejected / genuine_total) if genuine_total else 0.0,
        "false_rejection_n": genuine_total,
        "acceptance_by_family": {k: round(by_fam_accept[k] / by_fam_total[k], 3)
                                 for k in sorted(by_fam_total)},
        "ece": _ece(np.array(confs), np.array(hits)),
        "abstention_rate": 1.0 - answered / n if n else 0.0,
        "accuracy_when_answered": (answered_correct / answered) if answered else 0.0,
    }


def _ece(confs: np.ndarray, hits: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(hits[m].mean() - confs[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# End-to-end error propagation
# --------------------------------------------------------------------------- #

def _item(d: Doc, authority: str) -> EvidenceItem:
    return EvidenceItem(
        value=d.value, source_id=d.source_id, authority=authority, timestamp=d.timestamp,
        content_hash=content_hash(d.text), score=1.0, trusted=authority not in UNTRUSTED,
        claim_key=(d.entity.lower(), d.attribute.lower()), raw_text=d.text,
        is_working=(d.channel == "personal_email"))


def _resolve(trial: DocTrial, authorities: List[str]) -> Verdict:
    items = [_item(d, a) for d, a in zip(trial.docs, authorities)]
    q = Query(trial.entity, trial.attribute, as_of="00000100", requires_citation=False)
    return policy(Trial("doc", q, items, gold_value=trial.gold_value))


def error_propagation(clf: AuthorityClassifier, trials: List[DocTrial]) -> Dict:
    oracle_ok = pred_ok = oracle_poison = pred_poison = 0
    for t in trials:
        oracle_auth = [d.label_authority for d in t.docs]
        pred_auth = [clf.predict(d.features())[0] or "unverified_external_content" for d in t.docs]
        vo = _resolve(t, oracle_auth)
        vp = _resolve(t, pred_auth)
        oracle_ok += (vo.answer_value == t.gold_value)
        pred_ok += (vp.answer_value == t.gold_value)
        oracle_poison += (vo.answer_value == t.poison_value)
        pred_poison += (vp.answer_value == t.poison_value)
    n = len(trials)
    return {"n": n,
            "oracle_accuracy": oracle_ok / n, "predicted_accuracy": pred_ok / n,
            "accuracy_drop": (oracle_ok - pred_ok) / n,
            "oracle_poison_rate": oracle_poison / n, "predicted_poison_rate": pred_poison / n}


def train_and_evaluate(seed: int = 0, n_train: int = 4000, n_test: int = 1200,
                       n_trials: int = 600, blind_path: Optional[str] = None) -> Dict:
    train = gen_examples(seed, n_train)
    val = gen_examples(seed + 101, 600)
    test = gen_examples(seed + 202, n_test)
    clf = AuthorityClassifier(seed=seed)
    clf.fit(train, steps=800, seed=seed)
    temp = clf.calibrate(val)
    out = {"temperature": temp,
           "held_out": classifier_metrics(clf, test),
           "error_propagation": error_propagation(clf, gen_resolution_trials(seed + 303, n_trials))}
    if blind_path and Path(blind_path).exists():
        blind = load_blind(blind_path)
        out["blind"] = classifier_metrics(clf, blind)
        out["blind_n"] = len(blind)
    return out


__all__ = ["classifier_metrics", "error_propagation", "train_and_evaluate", "load_blind"]
