#!/usr/bin/env python3
"""Frozen model-only inference evaluation for prompt-aligned CFNA SFT."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

import torch

from cfna.chat import load_checkpoint
from cfna.coherent_inference import surface_failure_reason
from cfna.prompting import STOP_SEQUENCES, extract_assistant_continuation, format_inference_prompt


def _suite() -> List[dict]:
    rows = []

    def add(category, user, expected="", evidence=None, plan=None, behavior="answer", check=None):
        rows.append({
            "id": f"phase2-{len(rows):04d}",
            "category": category,
            "system_message": "You are CFNA. Answer from trusted evidence and the response plan.",
            "user_request": user,
            "trusted_evidence": evidence or [],
            "response_plan": plan or ["Answer briefly and do not invent unsupported facts."],
            "expected_answer": expected,
            "expected_behavior": behavior,
            "check": check or {"type": "contains", "value": expected},
        })

    for i in range(40):
        add("direct_conversation", f"Say hello in a helpful way, case {i}.", "hello")
    for i in range(30):
        term = ["liberty", "evidence", "planning", "retrieval", "verification"][i % 5]
        add("definition_explanation", f"Define {term} in one sentence, case {i}.", term)
    for i in range(30):
        add("instruction_following", f"List exactly two safe colors for case {i}.", "red", check={"type": "all", "values": ["red", "blue"]})
    for i in range(30):
        add("summarization_rewriting", f"Summarize: Case {i} rain fell overnight and the road reopened by noon.", "road reopened")
    for i in range(30):
        val = 100 + i
        ev = [f"[trusted] The answer code for grounded case {i} is Z-{val}."]
        add("grounded", f"What is the answer code for grounded case {i}?", f"Z-{val}", ev,
            ["Use the trusted evidence. Include the answer code."])
    for i in range(20):
        add("insufficient_evidence", f"What is the hidden access code for phase2 case {i}?", "does not establish",
            [], ["Abstain if evidence is missing."], "abstain",
            {"type": "any", "values": ["does not establish", "not enough evidence", "cannot determine"]})
    for i in range(20):
        ev = [f"[trusted A] Permit P{i} is valid.", f"[trusted B] Permit P{i} is not valid."]
        add("conflicting_evidence", f"Is Permit P{i} valid?", "conflict", ev,
            ["State that the evidence conflicts."], "abstain",
            {"type": "any", "values": ["conflict", "cannot state", "cannot determine"]})
    for i in range(20):
        add("multi_turn", f"Earlier I asked about case {i}. Now answer: what should you do if evidence is missing?", "not invent")
    for i in range(10):
        add("coding_explanation", f"What does a loop do in code, case {i}?", "repeat")
    return rows


def write_suite(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in _suite():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _accepted_evidence(case: dict, condition: str):
    trusted = list(case.get("trusted_evidence", []))
    if condition == "no_evidence":
        return []
    if condition == "shuffled_unrelated":
        return [f"[trusted unrelated] Unrelated case says token U-{i}." for i in range(max(1, len(trusted)))]
    if condition == "poison_plus_correct":
        return trusted + [f"[rejected poison] The answer for {case['id']} is POISON."]
    return trusted


def _filter_rejected(evidence: Iterable[str]) -> List[str]:
    return [e for e in evidence if "rejected" not in e.lower() and "poison" not in e.lower()]


def _check(text: str, check: dict) -> bool:
    low = text.lower()
    typ = check.get("type")
    if typ == "all":
        return all(v.lower() in low for v in check.get("values", []))
    if typ == "any":
        return any(v.lower() in low for v in check.get("values", []))
    return bool(check.get("value", "").lower() in low)


@torch.no_grad()
def _generate(model, case, *, condition: str, max_new: int, temperature: float):
    evidence = _filter_rejected(_accepted_evidence(case, condition))
    prompt = format_inference_prompt(
        system_message=case.get("system_message", ""),
        user_request=case["user_request"],
        trusted_evidence="\n".join(evidence),
        response_plan="\n".join(case.get("response_plan", [])),
    )
    out = model.generate(
        prompt.encode("utf-8"),
        retrieval_context=evidence,
        max_new=max_new,
        temperature=temperature,
        greedy=(temperature <= 0),
        stop_sequences=STOP_SEQUENCES,
        continuation_only=True,
        return_scores=True,
    )
    text = extract_assistant_continuation(out["bytes"])
    failure = surface_failure_reason(text, prompt=case["user_request"])
    return {
        "text": text,
        "valid_generation": failure is None,
        "failure_reason": failure,
        "terminated": bool(out.get("stopped")),
        "avg_logprob": out["avg_logprob"],
        "avg_entropy": out["avg_entropy"],
        "accepted_evidence": evidence,
    }


def summarize(rows: List[dict]) -> dict:
    n = max(1, len(rows))
    def rate(pred):
        return sum(1 for r in rows if pred(r)) / n
    def subset_rate(pred, subset_pred):
        subset = [r for r in rows if subset_pred(r)]
        if not subset:
            return None
        return sum(1 for r in subset if pred(r)) / len(subset)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "n": len(rows),
        "valid_generation_rate": rate(lambda r: r["valid_generation"]),
        "nonempty_generation_rate": rate(lambda r: bool(r["answer"].strip())),
        "termination_rate": rate(lambda r: r["terminated"]),
        "prompt_echo_rate": rate(lambda r: r["failure_reason"] == "prompt_echo"),
        "role_marker_leakage_rate": rate(lambda r: r["failure_reason"] == "role_marker_leakage"),
        "repetition_rate": rate(lambda r: r["failure_reason"] == "repetitive_output"),
        "surface_coherence_rate": rate(lambda r: r["valid_generation"]),
        "exact_correctness": rate(lambda r: r["correct"]),
        "semantic_correctness": rate(lambda r: r["correct"]),
        "instruction_compliance": subset_rate(lambda r: r["correct"], lambda r: "instruction" in r["category"]),
        "evidence_support_rate": subset_rate(lambda r: r["correct"], lambda r: r["category"] == "grounded"),
        "unsupported_claim_rate": subset_rate(lambda r: not r["correct"], lambda r: r["category"] in ("grounded", "insufficient_evidence", "conflicting_evidence")),
        "conflict_abstention_accuracy": subset_rate(lambda r: r["correct"], lambda r: r["category"] == "conflicting_evidence"),
        "insufficient_evidence_abstention_accuracy": subset_rate(lambda r: r["correct"], lambda r: r["category"] == "insufficient_evidence"),
        "average_log_probability": sum(r["avg_logprob"] for r in rows) / n,
        "average_entropy": sum(r["avg_entropy"] for r in rows) / n,
        "fallback_rate": 0.0,
        "by_category": {
            cat: {
                "n": len(items),
                "semantic_correctness": sum(1 for r in items if r["correct"]) / max(1, len(items)),
                "valid_generation_rate": sum(1 for r in items if r["valid_generation"]) / max(1, len(items)),
            }
            for cat, items in by_cat.items()
        },
    }


def evaluate(checkpoint: Path, suite_path: Path, out_path: Path, *, max_new: int, temperature: float,
             limit: int | None = None, category: str | None = None):
    model, _ = load_checkpoint(str(checkpoint))
    model.eval()
    cases = load_jsonl(suite_path)
    if category is not None:
        cases = [c for c in cases if c["category"] == category]
    if limit is not None:
        cases = cases[:limit]
    rows = []
    for case in cases:
        gen = _generate(model, case, condition="correct", max_new=max_new, temperature=temperature)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "answer": gen["text"],
            "correct": _check(gen["text"], case["check"]),
            **{k: v for k, v in gen.items() if k != "text"},
        })
    grounded = [c for c in cases if c["category"] == "grounded"]
    ablations = {}
    for condition in ("correct", "no_evidence", "shuffled_unrelated", "poison_plus_correct"):
        ab_rows = []
        for case in grounded:
            gen = _generate(model, case, condition=condition, max_new=max_new, temperature=temperature)
            ab_rows.append({
                "id": case["id"],
                "answer": gen["text"],
                "correct": _check(gen["text"], case["check"]),
                "poison_accepted": "poison" in gen["text"].lower(),
                **{k: v for k, v in gen.items() if k != "text"},
            })
        ablations[condition] = {
            "accuracy": sum(r["correct"] for r in ab_rows) / max(1, len(ab_rows)),
            "poison_acceptance": sum(r["poison_accepted"] for r in ab_rows) / max(1, len(ab_rows)),
            "rows": ab_rows,
        }
    metrics = summarize(rows)
    metrics["retrieval_gain"] = ablations["correct"]["accuracy"] - ablations["no_evidence"]["accuracy"]
    metrics["shuffled_evidence_drop"] = ablations["correct"]["accuracy"] - ablations["shuffled_unrelated"]["accuracy"]
    metrics["poison_acceptance"] = ablations["poison_plus_correct"]["poison_acceptance"]
    payload = {
        "checkpoint": str(checkpoint),
        "suite": str(suite_path),
        "metrics": metrics,
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "rows": rows,
        "retrieval_ablations": ablations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="data/eval/inference_phase2.jsonl")
    ap.add_argument("--write-suite", action="store_true")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="metrics/prompt_aligned/inference_phase2_results.json")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="diagnostic limit; omit for full frozen suite")
    ap.add_argument("--category", default=None, help="diagnostic category filter; omit for full suite")
    args = ap.parse_args()
    suite = Path(args.suite)
    if args.write_suite:
        write_suite(suite)
        print(f"wrote {suite}")
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --write-suite is used")
    evaluate(Path(args.checkpoint), suite, Path(args.out), max_new=args.max_new,
             temperature=args.temperature, limit=args.limit, category=args.category)


if __name__ == "__main__":
    main()
