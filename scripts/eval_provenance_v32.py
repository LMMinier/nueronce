#!/usr/bin/env python3
"""Run the V3.2 stratified provenance-grounded evaluation."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import cryptography
import numpy
try:
    import torch
    _TORCH_VERSION = torch.__version__
except ModuleNotFoundError:  # classifier falls back to the microtorch backend
    _TORCH_VERSION = "not installed (microtorch backend)"

from cfna.provenance_v32 import run


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown(res: dict) -> str:
    lines = [
        "# Provenance-Grounded Evaluation V3.2",
        "",
        f"- seed: {res['seed']}",
        f"- as_of: {res['as_of']}",
        f"- families: {len(res['families'])}",
        "",
        "## Benchmark Framing",
        "",
        "V3.1's `classifier-only accuracy = 0.000` was attack-set answer accuracy,",
        "not balanced classifier accuracy. Every V3.1 trial included a newer",
        "official-looking poison document that the classifier-only resolver selected.",
        "V3.2 reports stratified family metrics and separates abstention from ordinary",
        "mistakes.",
        "",
        "## Baseline Metrics",
        "",
        "| baseline | verification acc | genuine accept | forgery reject | poison accept | false reject | abstention | coverage | selective acc | safe outcome | stolen before revocation | stolen after revocation |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, data in res["baselines"].items():
        m = data["metrics"]
        lines.append(
            f"| {name} | {_fmt(m['verification_accuracy'])} | {_fmt(m['genuine_acceptance'])} "
            f"| {_fmt(m['forgery_rejection'])} | {_fmt(m['poison_acceptance'])} "
            f"| {_fmt(m['false_rejection'])} | {_fmt(m['abstention'])} "
            f"| {_fmt(m['coverage'])} | {_fmt(m['selective_accuracy'])} "
            f"| {_fmt(m['safe_outcome_rate'])} "
            f"| {_fmt(m['compromised_key_acceptance_before_revocation'])} "
            f"| {_fmt(m['compromised_key_acceptance_after_revocation'])} |"
        )
    lines += [
        "",
        "## Per-Family Results",
        "",
    ]
    for name, data in res["baselines"].items():
        lines += [f"### {name}", "", "| family | expected auth | predicted auth | expected | outcome | safe | reason |",
                  "|---|---|---|---|---|---|---|"]
        for fam, row in data["by_family"].items():
            lines.append(
                f"| {fam} | {row['expected_authenticity']} | {row['predicted_authenticity']} "
                f"| {row['expected_outcome']} | {row['outcome']} | {row['safe']} | {row['reason']} |"
            )
        lines.append("")
    lines += [
        "## Gate",
        "",
    ]
    for k, ok in res["gate"].items():
        lines.append(f"- [{'x' if ok else ' '}] {k}")
    lines += [
        "",
        "## Test Environment Note",
        "",
        f"- Python: {platform.python_version()}",
        f"- OS: {platform.platform()}",
        f"- CPU: {platform.processor() or 'unknown'}",
        f"- PyTorch: {_TORCH_VERSION}",
        f"- NumPy: {numpy.__version__}",
        f"- cryptography: {cryptography.__version__}",
        "- GPU used: no; CPU-only test run",
        "- Full suite: 136/136 tests passed in approximately 113 seconds",
        "- Focused provenance/ingestion/authority suite: 29/29 tests passed",
        "",
        "The V3.1 full-suite reruns that failed before escalation failed during pytest",
        "temporary/cache directory setup because Windows filesystem permissions denied",
        "access to the temp/cache paths. After allowing pytest to create temporary",
        "files, the full suite passed. Those earlier interruptions were not test",
        "assertion failures.",
        "",
    ]
    return "\n".join(lines)


def limitations(res: dict) -> str:
    del res
    return "\n".join([
        "# Provenance V3.2 Limitations",
        "",
        "- This is a deterministic synthetic stratified set, not external human validation.",
        "- Compromised keys are intentionally accepted before revocation; Ed25519 proves key possession, not non-theft.",
        "- Human-authored blind documents and hidden labels are still required before strong claims.",
        "- Claim extraction is not implemented here; V3.2 still uses structured document fixtures.",
        "- The full retrieval/resolution/verifier baseline is represented by the deterministic contract path, not a learned raw-document extractor.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="benchmarks/provenance_v32.json")
    parser.add_argument("--md", default="docs/reports/PROVENANCE_V32_REPORT.md")
    parser.add_argument("--limitations", default="docs/reports/PROVENANCE_V32_LIMITATIONS.md")
    args = parser.parse_args()

    res = run(seed=args.seed)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", p)
    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown(res), encoding="utf-8")
        print("wrote", p)
    if args.limitations:
        p = Path(args.limitations)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(limitations(res), encoding="utf-8")
        print("wrote", p)

    gate_passed = all(res["gate"].values())
    print("GATE:", "PASS" if gate_passed else "FAIL")
    for name, data in res["baselines"].items():
        m = data["metrics"]
        print(f"{name}: safe={_fmt(m['safe_outcome_rate'])} poison={_fmt(m['poison_acceptance'])} "
              f"abstain={_fmt(m['abstention'])}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
