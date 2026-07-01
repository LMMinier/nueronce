"""Evaluation, per-family metrics, gate, and failure taxonomy for Cognitive Suite V2.

Runs every strategy (the full policy, competing baselines, and single-stage
ablations) over the randomized suite and the adversarial holdouts across multiple
seeds, and reports per-family behavior plus a pass/fail milestone gate.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Tuple

from .cognition_v2 import (FAMILIES, STRATEGIES, Trial, Verdict, generate_suite,
                           is_correct)
from .cognitive_holdouts import HOLDOUT_FAMILIES, generate_holdouts


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #

def classify_failure(t: Trial, v: Verdict) -> str:
    if is_correct(t, v):
        return "ok"
    if t.gold_conflict:
        return "missed_conflict"
    if t.gold_value is None:
        return "missed_decline"          # invented an answer when none was warranted
    if t.poison_value is not None and v.answer_value == t.poison_value:
        return "poisoned"
    if t.stale_value is not None and v.answer_value == t.stale_value:
        return "stale"
    if v.answer_value is None:
        return "false_decline"           # declined when a trusted answer existed
    if v.answer_value != t.gold_value:
        return "wrong_value"
    if t.query.requires_citation and not v.cited:
        return "missing_citation"
    if t.query.requires_citation and v.cite_source != t.gold_source:
        return "wrong_citation"
    if t.gold_uncertain and not v.is_uncertain:
        return "lost_uncertainty"
    return "other"


# --------------------------------------------------------------------------- #
# Per-config scoring
# --------------------------------------------------------------------------- #

def _score_rows(trials: List[Trial], strategy: str, seed: int):
    rng = random.Random(seed)
    fn = STRATEGIES[strategy]
    rows = []
    for t in trials:
        v = fn(t, rng)
        rows.append((t, v, is_correct(t, v), classify_failure(t, v)))
    return rows


def _metrics(rows) -> Dict:
    n = len(rows)
    acc = sum(c for _, _, c, _ in rows) / n if n else 0.0
    poison = [(t, v) for t, v, _, _ in rows if t.poison_value is not None]
    poison_rate = (sum(v.answer_value == t.poison_value for t, v in poison) / len(poison)
                   if poison else 0.0)
    asserted = [(t, v) for t, v, _, _ in rows if v.answer_value is not None]
    unsupported_rate = (sum(1 for _, v in asserted if not v.supported) / len(asserted)
                        if asserted else 0.0)
    cite_trials = [(t, v) for t, v, _, _ in rows
                   if t.query.requires_citation and t.gold_value is not None and not t.gold_conflict]
    citation_acc = (sum(v.cited and v.cite_source == t.gold_source for t, v in cite_trials)
                    / len(cite_trials) if cite_trials else 0.0)
    # Value-only accuracy: did it get the fact right, ignoring citation/refusal/conflict?
    # This is the fair comparison to citation-blind baselines.
    value_trials = [(t, v) for t, v, _, _ in rows
                    if t.gold_value is not None and not t.gold_conflict and not t.gold_uncertain]
    value_acc = (sum(v.answer_value == t.gold_value for t, v in value_trials)
                 / len(value_trials) if value_trials else 0.0)
    src_trials = [(t, v) for t, v, _, _ in rows if t.gold_source is not None]
    source_acc = (sum(v.winning_source == t.gold_source for t, v in src_trials)
                  / len(src_trials) if src_trials else 0.0)
    by_family: Dict[str, Dict] = {}
    for fam in set(t.family for t, _, _, _ in rows):
        fr = [(t, v, c) for t, v, c, _ in rows if t.family == fam]
        by_family[fam] = {"n": len(fr), "acc": sum(c for _, _, c in fr) / len(fr)}
    fail_counts: Dict[str, int] = {}
    for _, _, c, cat in rows:
        if not c:
            fail_counts[cat] = fail_counts.get(cat, 0) + 1
    return {"n": n, "accuracy": acc, "value_accuracy": value_acc,
            "source_accuracy": source_acc, "poisoning_rate": poison_rate,
            "unsupported_rate": unsupported_rate, "citation_accuracy": citation_acc,
            "by_family": by_family, "failure_counts": fail_counts}


# --------------------------------------------------------------------------- #
# Full multi-seed run + gate
# --------------------------------------------------------------------------- #

def run(seeds: List[int], n_per_seed: int, holdout_n: int) -> Dict:
    strategies = list(STRATEGIES)
    indist = {s: [] for s in strategies}
    holdout = {s: [] for s in strategies}
    # accumulate per-seed metrics
    for seed in seeds:
        trials = generate_suite(seed, n_per_seed)
        hold = generate_holdouts(seed, holdout_n)
        for s in strategies:
            indist[s].append(_metrics(_score_rows(trials, s, seed)))
            holdout[s].append(_metrics(_score_rows(hold, s, seed)))

    def agg(per_seed):
        accs = [m["accuracy"] for m in per_seed]
        return {
            "accuracy_mean": statistics.mean(accs),
            "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "value_accuracy_mean": statistics.mean(m["value_accuracy"] for m in per_seed),
            "source_accuracy_mean": statistics.mean(m["source_accuracy"] for m in per_seed),
            "poisoning_rate_mean": statistics.mean(m["poisoning_rate"] for m in per_seed),
            "unsupported_rate_mean": statistics.mean(m["unsupported_rate"] for m in per_seed),
            "citation_accuracy_mean": statistics.mean(m["citation_accuracy"] for m in per_seed),
            "by_family_mean": _family_means(per_seed),
            "failure_counts": _sum_failures(per_seed),
        }

    results_indist = {s: agg(indist[s]) for s in strategies}
    results_holdout = {s: agg(holdout[s]) for s in strategies}

    gate = _gate(results_indist, results_holdout)
    return {
        "config": {"seeds": seeds, "n_per_seed": n_per_seed, "holdout_n": holdout_n,
                   "families": list(FAMILIES), "holdout_families": list(HOLDOUT_FAMILIES)},
        "in_distribution": results_indist,
        "adversarial_holdout": results_holdout,
        "gate": gate,
    }


def _family_means(per_seed) -> Dict[str, float]:
    fams = set()
    for m in per_seed:
        fams |= set(m["by_family"])
    out = {}
    for fam in fams:
        vals = [m["by_family"][fam]["acc"] for m in per_seed if fam in m["by_family"]]
        out[fam] = statistics.mean(vals) if vals else 0.0
    return out


def _sum_failures(per_seed) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in per_seed:
        for k, c in m["failure_counts"].items():
            out[k] = out.get(k, 0) + c
    return out


def _gate(indist: Dict, holdout: Dict) -> Dict:
    full = indist["FULL_COGNITIVE_LOOP"]
    full_h = holdout["FULL_COGNITIVE_LOOP"]
    checks = {}
    checks["in_distribution_accuracy>=0.95"] = full["accuracy_mean"] >= 0.95
    checks["adversarial_accuracy>=0.85"] = full_h["accuracy_mean"] >= 0.85
    checks["poisoning_rate<0.05"] = max(full["poisoning_rate_mean"],
                                        full_h["poisoning_rate_mean"]) < 0.05
    # Compare on value-only accuracy (ignores citation) so the margin reflects
    # getting the *fact* right, not the baselines' structural lack of citations.
    checks["beats_NEWEST_FACT_WINS_on_value>=0.10"] = (
        full["value_accuracy_mean"] - indist["NEWEST_FACT_WINS"]["value_accuracy_mean"] >= 0.10)
    checks["beats_KEYWORD_RULE_ENGINE_on_value>=0.10"] = (
        full["value_accuracy_mean"] - indist["KEYWORD_RULE_ENGINE"]["value_accuracy_mean"] >= 0.10)

    # Each module must have >=1 family where removing it drops accuracy >= 0.20.
    module_degradation = {}
    for abl in ("NO_AUTHORITY", "NO_SUPERSESSION", "NO_RETRIEVAL", "NO_PLANNING", "NO_VERIFICATION"):
        worst = 0.0
        worst_fam = None
        for fam, facc in full["by_family_mean"].items():
            aacc = holdout_or_indist_family(indist, abl, fam)
            drop = facc - aacc
            if drop > worst:
                worst, worst_fam = drop, fam
        module_degradation[abl] = {"worst_family": worst_fam, "drop": round(worst, 3),
                                   "meaningful": worst >= 0.20}
    checks["every_module_has_meaningful_degradation"] = all(
        d["meaningful"] for d in module_degradation.values())

    return {"checks": checks, "module_degradation": module_degradation,
            "passed": all(checks.values())}


def holdout_or_indist_family(indist: Dict, strategy: str, fam: str) -> float:
    return indist[strategy]["by_family_mean"].get(fam, 0.0)


__all__ = ["run", "classify_failure"]
