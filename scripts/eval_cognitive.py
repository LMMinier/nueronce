#!/usr/bin/env python3
"""Falsification harness for the NUERONCE cognitive architecture.

Runs the controlled task suite under the FULL cognitive loop and under each
single-stage ablation, then reports whether the full architecture solves tasks
that the ablated versions cannot. This is the project's first milestone test:

    On a controlled task suite, the full NUERONCE pipeline performs better than the
    same loop with memory-supersession, authority, retrieval, planning, or
    verification removed.

No language-model training is involved: the result isolates the *cognitive
structure*, so it cannot be explained away by how well the byte model was trained.

Usage:
    python scripts/eval_cognitive.py                 # print report
    python scripts/eval_cognitive.py --json out.json # also write machine metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nueronce.cognition import ABLATION_FLAGS, grade, run
from nueronce.cognitive_suite import default_scenarios


def _run_config(scenarios, ablations):
    rows = []
    for sc in scenarios:
        trace = run(sc, ablations)
        ok, why = grade(sc, trace)
        rows.append({"scenario": sc.name, "passed": ok, "answer": trace.answer, "why": why})
    passed = sum(r["passed"] for r in rows)
    return {"passed": passed, "total": len(rows), "rows": rows}


def _markdown(results, scenarios, milestone) -> str:
    lines = ["# NUERONCE Cognitive Ablation - V1 Report", ""]
    lines.append(f"Scenarios ({len(scenarios)}): " + ", ".join(sc.name for sc in scenarios))
    lines.append("")
    lines.append("| config | score | solves |")
    lines.append("|---|---|---|")
    for name, res in results.items():
        solved = ", ".join(r["scenario"] for r in res["rows"] if r["passed"]) or "(none)"
        lines.append(f"| {name} | {res['passed']}/{res['total']} | {solved} |")
    lines.append("")
    lines.append("## FULL answers")
    lines.append("")
    for r in results["FULL"]["rows"]:
        lines.append(f"- **{r['scenario']}** {'PASS' if r['passed'] else 'FAIL'}: {r['answer']}")
    lines.append("")
    lines.append(f"**Milestone:** {'PASS - full loop strictly beats every ablation' if milestone else 'NOT MET'}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--md", type=str, default="")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded for reproducibility; the V1 loop is deterministic")
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    scenarios = default_scenarios()
    configs = {"FULL": {}}
    for flag in ABLATION_FLAGS:
        configs[flag] = {flag: True}

    results = {name: _run_config(scenarios, abl) for name, abl in configs.items()}

    # ---- human-readable report ----
    full = results["FULL"]
    print("NUERONCE cognitive ablation - first milestone test")
    print("=" * 64)
    print(f"scenarios: {', '.join(sc.name for sc in scenarios)}\n")
    print(f"{'config':<22}{'score':>8}   solves")
    print("-" * 64)
    for name, res in results.items():
        solved = ",".join(r["scenario"] for r in res["rows"] if r["passed"]) or "(none)"
        print(f"{name:<22}{res['passed']}/{res['total']:>3}   {solved}")

    print("\nFULL answers:")
    for r in full["rows"]:
        print(f"  [{'OK ' if r['passed'] else 'XX '}] {r['scenario']}: {r['answer']}")

    print("\nablation failures (what breaks when a stage is removed):")
    for name, res in results.items():
        if name == "FULL":
            continue
        broke = [f"{r['scenario']} ({r['why']})" for r in res["rows"] if not r["passed"]]
        if broke:
            print(f"  - {name}: " + "; ".join(broke))

    milestone = full["passed"] == full["total"] and all(
        res["passed"] < full["passed"] for n, res in results.items() if n != "FULL"
    )
    print("\nMILESTONE:", "PASS - full loop strictly beats every ablation"
          if milestone else "NOT MET")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"seed": args.seed, "results": results, "milestone_passed": milestone},
            indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(_markdown(results, scenarios, milestone), encoding="utf-8")
        print(f"wrote {md}")

    return 0 if milestone else 1


if __name__ == "__main__":
    raise SystemExit(main())
