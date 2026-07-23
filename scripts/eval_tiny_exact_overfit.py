#!/usr/bin/env python3
"""Section B of FOUNDATIONAL_GENERATION_RECOVERY.md: evaluate free-running,
greedy, model-only generation on the exact 32 prompts
(scripts/train_tiny_exact_overfit.py) was just trained on.

Uses the exact same serializer/stop-handling as the sealed proof gate
(nueronce.prompting.format_inference_prompt + STOP_SEQUENCES +
NUERONCEModel.generate(greedy=True, continuation_only=True)) so this
diagnoses the shared pipeline, not a simplified stand-in for it. Does not
read or write the sealed proof gate's output file.

Pass condition (per the recovery doc):
  - at least 31/32 exact training prompts reproduced correctly
  - no malformed delimiter fragments ("<|...") leak into the answer
  - state isolation: replaying an earlier prompt after later ones gives the
    same bytes as the first time (no cross-example contamination)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nueronce.model import ModelConfig, NUERONCEModel
from nueronce.prompting import STOP_SEQUENCES, extract_assistant_continuation, format_inference_prompt
from tiny_exact_overfit_examples import CATEGORY_BOUNDARIES, TINY_EXAMPLES


def load_tiny_checkpoint(path: str) -> tuple[NUERONCEModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = NUERONCEModel(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def category_of(index: int) -> str:
    for name, (lo, hi) in CATEGORY_BOUNDARIES.items():
        if lo <= index < hi:
            return name
    return "unknown"


@torch.no_grad()
def run_one(model: NUERONCEModel, system: str, prompt: str, max_new: int, max_ctx: int) -> dict:
    rendered = format_inference_prompt(system_message=system, user_request=prompt,
                                       trusted_evidence="", response_plan="")
    raw = model.generate(rendered.encode("utf-8"), max_new=max_new, temperature=0.0,
                         greedy=True, max_ctx=max_ctx, stop_sequences=STOP_SEQUENCES,
                         continuation_only=True)
    answer = extract_assistant_continuation(raw).strip()
    return {"rendered_prompt": rendered, "raw_bytes_hex": raw.hex(), "answer": answer}


def first_mismatch(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/tiny_exact_overfit/checkpoint.pt")
    ap.add_argument("--output", default="runs/tiny_exact_overfit/eval_report.json")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--min-pass", type=int, default=31)
    ap.add_argument("--no-fail-exit", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint} -- run train_tiny_exact_overfit.py first")
    model, ck = load_tiny_checkpoint(str(checkpoint))
    system = ck.get("sft_system", "")

    results = []
    started = time.time()
    for i, (prompt, target) in enumerate(TINY_EXAMPLES):
        out = run_one(model, system, prompt, args.max_new, args.max_ctx)
        exact = out["answer"] == target
        problems = []
        if not exact:
            problems.append("mismatch")
        if "<|" in out["answer"]:
            problems.append("malformed_delimiter_leak")
        results.append({
            "index": i, "category": category_of(i), "prompt": prompt, "target": target,
            "answer": out["answer"], "exact_match": exact,
            "first_mismatch_char": None if exact else first_mismatch(out["answer"], target),
            "problems": problems, "rendered_prompt": out["rendered_prompt"],
        })
        print(json.dumps({"index": i, "exact_match": exact, "answer": out["answer"]},
                         ensure_ascii=False), flush=True)

    # State isolation: replay item 0 and item 15 (already run above, mid-sequence
    # for item 15) once more, now *after* every other prompt has been evaluated,
    # and require byte-identical answers -- proves no state leaks across calls.
    isolation_checks = []
    for probe_index in (0, 15, 31):
        replay = run_one(model, system, TINY_EXAMPLES[probe_index][0], args.max_new, args.max_ctx)
        original = results[probe_index]["answer"]
        isolation_checks.append({
            "index": probe_index, "original_answer": original,
            "replay_answer": replay["answer"], "identical": replay["answer"] == original,
        })

    exact_count = sum(r["exact_match"] for r in results)
    delimiter_leaks = sum(1 for r in results if "malformed_delimiter_leak" in r["problems"])
    isolation_ok = all(c["identical"] for c in isolation_checks)
    gate_passed = exact_count >= args.min_pass and delimiter_leaks == 0 and isolation_ok

    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["exact_match"])
    category_scores = {c: sum(v) / len(v) for c, v in by_category.items()}

    report = {
        "gate": "tiny_exact_overfit_v1", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": ck.get("sft_step"), "final_train_loss": ck.get("final_loss"),
        "n_examples": len(TINY_EXAMPLES), "exact_match_count": exact_count,
        "exact_match_fraction": exact_count / len(TINY_EXAMPLES),
        "delimiter_leaks": delimiter_leaks, "state_isolation_ok": isolation_ok,
        "isolation_checks": isolation_checks, "category_scores": category_scores,
        "criteria": {"min_exact_matches": args.min_pass, "delimiter_leaks_must_be_zero": True,
                    "state_isolation_required": True},
        "gate_passed": gate_passed, "elapsed_seconds": time.time() - started,
        "results": results,
        "interpretation": (
            "A pass proves the shared serialize/mask/generate pipeline is "
            "mechanically correct on a model that WAS trained to convergence. "
            "It does not by itself prove the sealed proof-gate checkpoint is "
            "healthy -- if this passes but the proof gate still fails 0/8, the "
            "next suspect is training scale/data on the real run, not plumbing."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "gate_passed": gate_passed, "exact_match_count": exact_count,
        "n_examples": len(TINY_EXAMPLES), "delimiter_leaks": delimiter_leaks,
        "state_isolation_ok": isolation_ok, "output": str(output),
    }, indent=2))
    if not gate_passed and not args.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
