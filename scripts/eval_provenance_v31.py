#!/usr/bin/env python3
"""Run the V3.1 cryptographic provenance-gate benchmark.

Usage:
    python scripts/eval_provenance_v31.py --seed 0 --n 800 \
        --json benchmarks/provenance_v31.json \
        --md docs/reports/PROVENANCE_V31_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfna.provenance_bench import run


def markdown(res: dict, seed: int) -> str:
    co = res["classifier_only"]
    gated = res["gated"]
    comp = res["compromised_key"]
    lines = [
        "# Provenance Gate V3.1 - Report",
        "",
        f"- seed: {seed}",
        f"- trials: {res['n']}",
        f"- signed genuine share: {res['signed_genuine_share']:.3f}",
        "",
        "## Milestone",
        "",
        "| pipeline | accuracy | poison acceptance | abstain/escalate |",
        "|---|---|---|---|",
        f"| classifier-only | {co['accuracy']:.3f} | {co['poison_acceptance']:.3f} | {co['abstain_or_escalate']:.3f} |",
        f"| classifier + provenance gate | {gated['accuracy']:.3f} | {gated['poison_acceptance']:.3f} | {gated['abstain_or_escalate']:.3f} |",
        "",
        "The gate tests authenticity separately from apparent authority. In this synthetic",
        "benchmark, appearance-perfect poison documents look official to the classifier",
        "but lack valid cryptographic provenance, so the deterministic final-trust policy",
        "prevents them from winning resolution.",
        "",
        "## Compromised Key Check",
        "",
        f"- before revocation: {comp['stolen_key_before_revocation']}",
        f"- after revocation: {comp['after_revocation']}",
        f"- note: {comp['note']}",
        "",
        "## Limits",
        "",
        "- This is still a synthetic benchmark, not external validation.",
        "- Signature verification proves key possession, not that the key was never stolen.",
        "- Unsigned genuine legacy documents are escalated/restricted, so coverage depends",
        "  on migration, trusted timestamps, revocation, and human review policy.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--md", type=str, default="")
    args = parser.parse_args()

    res = run(seed=args.seed, n=args.n)
    co = res["classifier_only"]
    gated = res["gated"]
    print(f"seed {args.seed} | n {res['n']} | signed genuine {res['signed_genuine_share']:.3f}")
    print("classifier-only: acc %.3f | poison %.3f | abstain/escalate %.3f" % (
        co["accuracy"], co["poison_acceptance"], co["abstain_or_escalate"]))
    print("gated         : acc %.3f | poison %.3f | abstain/escalate %.3f" % (
        gated["accuracy"], gated["poison_acceptance"], gated["abstain_or_escalate"]))
    print("compromised key: before %s -> after revocation %s" % (
        res["compromised_key"]["stolen_key_before_revocation"],
        res["compromised_key"]["after_revocation"]))

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"seed": args.seed, **res}, indent=2), encoding="utf-8")
        print("wrote", p)
    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown(res, args.seed), encoding="utf-8")
        print("wrote", p)

    improved = gated["poison_acceptance"] < co["poison_acceptance"]
    acceptable_false_rejection = gated["abstain_or_escalate"] <= 0.25
    return 0 if improved and acceptable_false_rejection else 1


if __name__ == "__main__":
    raise SystemExit(main())
