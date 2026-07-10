#!/usr/bin/env python3
"""Run Cognitive Suite V2: randomized trials + adversarial holdouts, all strategies,
multiple seeds. Writes machine metrics (JSON) and a Markdown report, and reports
the milestone gate.

Usage:
    python scripts/eval_cognitive_v2.py --seeds 1 2 3 4 5 --n 1050 --holdout 480 \
        --json benchmarks/cognitive_v2.json --md docs/reports/COGNITIVE_V2_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nueronce.cognitive_eval import run


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def markdown(res: dict) -> str:
    cfg = res["config"]
    ind = res["in_distribution"]
    hol = res["adversarial_holdout"]
    gate = res["gate"]
    L = ["# NUERONCE Cognitive Suite V2 - Report", ""]
    L.append(f"- seeds: {cfg['seeds']}  |  trials/seed: {cfg['n_per_seed']}  |  "
             f"holdout/seed: {cfg['holdout_n']}")
    L.append(f"- families: {len(cfg['families'])} in-distribution, "
             f"{len(cfg['holdout_families'])} adversarial")
    L.append("")
    L.append("## Strategy accuracy (mean +/- std over seeds)")
    L.append("")
    L.append("`composite` requires the right value AND (where required) the right "
             "citation, refusal, or conflict flag. `value-only` ignores citation/"
             "refusal/conflict, so it is the fair comparison to the citation-blind "
             "baselines.")
    L.append("")
    L.append("| strategy | composite in-dist | value-only | source-sel | adversarial | poison rate | unsupported |")
    L.append("|---|---|---|---|---|---|---|")
    for s in ind:
        L.append(f"| {s} | {_fmt(ind[s]['accuracy_mean'])} +/- {_fmt(ind[s]['accuracy_std'])} "
                 f"| {_fmt(ind[s]['value_accuracy_mean'])} "
                 f"| {_fmt(ind[s]['source_accuracy_mean'])} "
                 f"| {_fmt(hol[s]['accuracy_mean'])} "
                 f"| {_fmt(ind[s]['poisoning_rate_mean'])} "
                 f"| {_fmt(ind[s]['unsupported_rate_mean'])} |")
    L.append("")
    L.append("## FULL policy — per-family accuracy (in-distribution)")
    L.append("")
    L.append("| family | acc |")
    L.append("|---|---|")
    for fam, acc in sorted(res["in_distribution"]["FULL_COGNITIVE_LOOP"]["by_family_mean"].items()):
        L.append(f"| {fam} | {_fmt(acc)} |")
    L.append("")
    L.append("## Module necessity (worst-family drop when removed)")
    L.append("")
    L.append("| removed module | worst family | acc drop | meaningful (>=0.20) |")
    L.append("|---|---|---|---|")
    for mod, d in gate["module_degradation"].items():
        L.append(f"| {mod} | {d['worst_family']} | {_fmt(d['drop'])} | {d['meaningful']} |")
    L.append("")
    L.append("## FULL failure taxonomy (in-distribution, summed over seeds)")
    L.append("")
    fc = ind["FULL_COGNITIVE_LOOP"]["failure_counts"]
    if fc:
        for k, c in sorted(fc.items(), key=lambda kv: -kv[1]):
            L.append(f"- {k}: {c}")
    else:
        L.append("- (no failures)")
    L.append("")
    L.append("## Gate")
    L.append("")
    for k, ok in gate["checks"].items():
        L.append(f"- [{'x' if ok else ' '}] {k}")
    L.append("")
    L.append(f"**GATE: {'PASS' if gate['passed'] else 'FAIL'}**")
    L.append("")
    L.append("## Limitations (read before citing these numbers)")
    L.append("")
    L.append("- **FULL = 1.000 is not evidence of real-world correctness.** The policy "
             "and the gold labels were authored from the *same* provenance rules, so "
             "the full loop matching its own specification is expected. The result "
             "shows internal completeness and consistency, not external validity.")
    L.append("- **Authority labels are given as ground-truth metadata.** The suite does "
             "not test inferring authority/temporal-scope/claims from raw text — the "
             "policy resists text attacks precisely because it ignores item text. That "
             "inference is the deferred, learnable problem.")
    L.append("- **The meaningful signals** are therefore the *contrasts*: value-only "
             "accuracy vs. baselines (poison/stale/expired/scope handling), the "
             "poisoning rate of text-based baselines (KEYWORD), and the per-family "
             "collapse under each ablation — not FULL's absolute score.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n", type=int, default=1050, help="in-distribution trials per seed")
    ap.add_argument("--holdout", type=int, default=480, help="adversarial trials per seed")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--md", type=str, default="")
    args = ap.parse_args()

    res = run(args.seeds, args.n, args.holdout)
    ind = res["in_distribution"]
    hol = res["adversarial_holdout"]
    print("FULL  in-dist acc: %.3f +/- %.3f | adversarial: %.3f | poison: %.3f" % (
        ind["FULL_COGNITIVE_LOOP"]["accuracy_mean"], ind["FULL_COGNITIVE_LOOP"]["accuracy_std"],
        hol["FULL_COGNITIVE_LOOP"]["accuracy_mean"], ind["FULL_COGNITIVE_LOOP"]["poisoning_rate_mean"]))
    print("composite  : NEWEST %.3f | HIGHEST %.3f | KEYWORD %.3f | RANDOM %.3f" % (
        ind["NEWEST_FACT_WINS"]["accuracy_mean"], ind["HIGHEST_AUTHORITY_ONLY"]["accuracy_mean"],
        ind["KEYWORD_RULE_ENGINE"]["accuracy_mean"], ind["RANDOM_CHOICE"]["accuracy_mean"]))
    print("value-only : FULL %.3f | NEWEST %.3f | HIGHEST %.3f | KEYWORD %.3f" % (
        ind["FULL_COGNITIVE_LOOP"]["value_accuracy_mean"], ind["NEWEST_FACT_WINS"]["value_accuracy_mean"],
        ind["HIGHEST_AUTHORITY_ONLY"]["value_accuracy_mean"], ind["KEYWORD_RULE_ENGINE"]["value_accuracy_mean"]))
    print("GATE:", "PASS" if res["gate"]["passed"] else "FAIL")
    for k, ok in res["gate"]["checks"].items():
        print(f"  [{'x' if ok else ' '}] {k}")

    if args.json:
        p = Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", p)
    if args.md:
        p = Path(args.md); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown(res), encoding="utf-8")
        print("wrote", p)
    return 0 if res["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
