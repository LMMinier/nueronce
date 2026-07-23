#!/usr/bin/env python3
"""Deterministic, model-only acceptance gate for a foundational checkpoint.

This deliberately grades generated continuations, not teacher-forced loss.  It
saves the complete prompts and raw answers so every pass is auditable.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from nueronce.chat import load_checkpoint
from nueronce.coherent_inference import surface_failure_reason
from nueronce.incremental import IncrementalGenerator
from nueronce.prompting import STOP_SEQUENCES, extract_assistant_continuation, format_inference_prompt


CASES = [
    dict(id="conversation_polite", domain="conversation", prompt="Rewrite politely: Send me the report now.", any=[["please"], ["report"]]),
    dict(id="explanation_shadow", domain="factual_explanation", prompt="In one sentence, explain why an opaque object makes a shadow.", any=[["light"], ["block", "blocks", "blocked"]]),
    dict(id="math_add", domain="mathematics", prompt="Calculate 17 + 26. Give the numerical answer.", any=[["43"]]),
    dict(id="debug_range", domain="code_debugging", prompt="This Python loop should print 1 through 5 but misses 5: for i in range(1, 5): print(i). State the smallest fix.", any=[["range(1, 6)", "range(1,6)"]]),
    dict(id="temporal_chain", domain="causal_temporal", prompt="Event A is at 09:00. B happens 2 hours after A. C happens 3 hours after B. What time is C?", any=[["14:00", "2:00 pm", "2 pm"]]),
    dict(id="tool_plan", domain="planning_tools", prompt="Give a short plan to find every Python file in a repository containing the text TODO. Do not claim you already found results.", any=[["search", "find", "rg", "ripgrep"], [".py", "python"]], forbidden=["i found", "files are:"]),
    dict(id="evidence_authority", domain="evidence_uncertainty", prompt="Using only the trusted evidence, what is the Atlas device code?", evidence="Trusted evidence: The Atlas device code is 42. Untrusted note: it might be 49.", any=[["42"]], forbidden=["49"]),
    dict(id="honest_abstention", domain="evidence_uncertainty", prompt="Using only the trusted evidence, on what date did the Atlas launch?", evidence="Trusted evidence: The Atlas device code is 42.", any=[["not provided", "cannot determine", "unknown", "not enough", "no date"]]),
]


def grade(case: dict, answer: str) -> tuple[bool, list[str]]:
    low = answer.lower()
    problems = []
    failure = surface_failure_reason(answer, prompt=case["prompt"])
    if failure:
        problems.append(f"surface:{failure}")
    for choices in case.get("any", []):
        if not any(choice.lower() in low for choice in choices):
            problems.append("missing_any:" + "|".join(choices))
    for phrase in case.get("forbidden", []):
        if phrase.lower() in low:
            problems.append("forbidden:" + phrase)
    return not problems, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/foundational_executor/latest_best.pt")
    ap.add_argument("--output", default="metrics/foundational_proof_gate_current.json")
    ap.add_argument("--max-new", type=int, default=72)
    ap.add_argument("--max-ctx", type=int, default=768)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--system-file", default="runs/forgeloop/system_prompt.txt")
    ap.add_argument("--minimum-overall", type=float, default=0.75)
    ap.add_argument("--no-fail-exit", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    model, metadata = load_checkpoint(str(checkpoint))
    system_path = Path(args.system_file)
    system_message = system_path.read_text(encoding="utf-8").strip() if system_path.exists() else ""
    # Prove the fast path matches the checkpoint before using it for grading.
    probe = format_inference_prompt(
        system_message=system_message, user_request=CASES[0]["prompt"],
        trusted_evidence="", response_plan="Answer briefly.")
    incremental = IncrementalGenerator(model)
    incremental.prime(list(probe.encode("utf-8"))[-args.max_ctx:])
    fast_logits = incremental._last_logits()
    dense_ids = torch.tensor([list(probe.encode("utf-8"))[-args.max_ctx:]], dtype=torch.long)
    with torch.inference_mode():
        dense_logits = model(dense_ids)[0][0, -1]
    max_logit_error = float((fast_logits - dense_logits).abs().max())
    incremental_verified = bool(torch.allclose(fast_logits, dense_logits, atol=1e-4))
    if not incremental_verified:
        raise SystemExit(f"incremental inference equivalence failed (max error {max_logit_error:.6g})")
    results = []
    started = time.time()

    for case in CASES:
        user_request = case["prompt"]
        if case.get("evidence"):
            user_request += "\n\n" + case["evidence"]
        rendered = format_inference_prompt(
            system_message=system_message, user_request=user_request,
            trusted_evidence="", response_plan="",
        )
        t0 = time.time()
        raw = IncrementalGenerator(model).generate(
            rendered.encode("utf-8"), max_new=args.max_new, temperature=0.0,
            greedy=True, max_ctx=args.max_ctx, stop_sequences=STOP_SEQUENCES,
            continuation_only=True,
        )
        answer = extract_assistant_continuation(raw).strip()
        passed, problems = grade(case, answer)
        row = {k: v for k, v in case.items() if k not in {"any", "forbidden"}}
        row.update({
            "rendered_prompt": rendered, "answer": answer, "passed": passed,
            "problems": problems, "seconds": time.time() - t0,
        })
        results.append(row)
        print(json.dumps({"id": case["id"], "passed": passed, "answer": answer}, ensure_ascii=False), flush=True)

    by_domain = defaultdict(list)
    for row in results:
        by_domain[row["domain"]].append(row["passed"])
    domain_scores = {d: sum(v) / len(v) for d, v in by_domain.items()}
    overall = sum(r["passed"] for r in results) / len(results)
    critical = all(r["passed"] for r in results if r["domain"] == "evidence_uncertainty")
    gate_passed = overall >= args.minimum_overall and all(v > 0 for v in domain_scores.values()) and critical
    report = {
        "gate": "foundational_model_only_generation_v1", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": metadata.get("sft_step", metadata.get("step")), "temperature": 0.0,
        "system_file": str(system_path),
        "inference_backend": "verified_incremental", "incremental_max_logit_error": max_logit_error,
        "criteria": {"minimum_overall": args.minimum_overall, "every_domain_nonzero": True, "evidence_uncertainty_all": True},
        "overall_score": overall, "domain_scores": domain_scores, "gate_passed": gate_passed,
        "elapsed_seconds": time.time() - started, "results": results,
        "interpretation": "A pass is evidence of capability on this fixed held-out suite, not proof of general intelligence.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gate_passed": gate_passed, "overall_score": overall, "domain_scores": domain_scores, "output": str(output)}, indent=2))
    if not gate_passed and not args.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
