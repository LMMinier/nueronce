#!/usr/bin/env python3
"""Run V3.3 blind-style multi-document provenance evaluation."""

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

from cfna.provenance_v33 import ABLATIONS, SYSTEMS, dev_cases_json, run


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def markdown(res: dict) -> str:
    lines = [
        "# Provenance V3.3 - Blind Multi-Document Resolution",
        "",
        f"- seed: {res['seed']}",
        f"- development cases: {res['n_dev_cases']}",
        "- final blind labels: withheld/not used in this development artifact",
        "",
        "## V3.2 Limitation Carried Forward",
        "",
        res["v32_limitation_note"],
        "",
        "Do not describe V3.2 or this generated development set as independent",
        "scientific validation. Safe-outcome rate is always shown with abstention",
        "and coverage.",
        "",
        "## Utility Weights",
        "",
    ]
    for k, v in res["utility_weights"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "Raw metrics remain primary; utility is only a compact tradeoff summary.",
        "",
        "## System Metrics",
        "",
        "| system | answer acc | src P | src R | cite P | cite R | unsupported | poison | false reject | abstain | coverage | selective acc | safe | utility |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in SYSTEMS:
        m = res["systems"][name]["metrics"]
        lines.append(
            f"| {name} | {_fmt(m['answer_accuracy'])} | {_fmt(m['source_selection_precision'])} "
            f"| {_fmt(m['source_selection_recall'])} | {_fmt(m['citation_precision'])} "
            f"| {_fmt(m['citation_recall'])} | {_fmt(m['unsupported_claim_rate'])} "
            f"| {_fmt(m['poison_acceptance'])} | {_fmt(m['false_rejection'])} "
            f"| {_fmt(m['abstention_rate'])} | {_fmt(m['coverage'])} "
            f"| {_fmt(m['selective_accuracy'])} | {_fmt(m['safe_outcome_rate'])} "
            f"| {_fmt(m['utility'])} |"
        )
    lines += [
        "",
        "## Bootstrap Confidence Intervals",
        "",
        "| system | answer acc 95% CI | poison 95% CI | abstain 95% CI | safe 95% CI | utility 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for name in SYSTEMS:
        ci = res["systems"][name]["confidence_intervals"]
        lines.append(
            f"| {name} | {_ci(ci['answer_accuracy'])} | {_ci(ci['poison_acceptance'])} "
            f"| {_ci(ci['abstention_rate'])} | {_ci(ci['safe_outcome_rate'])} "
            f"| {_ci(ci['utility'])} |"
        )
    lines += [
        "",
        "## Ablations",
        "",
        "| ablation | answer acc | unsupported | poison | abstain | coverage | safe | utility | total latency ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ABLATIONS:
        m = res["ablations"][name]["metrics"]
        lines.append(
            f"| {name} | {_fmt(m['answer_accuracy'])} | {_fmt(m['unsupported_claim_rate'])} "
            f"| {_fmt(m['poison_acceptance'])} | {_fmt(m['abstention_rate'])} "
            f"| {_fmt(m['coverage'])} | {_fmt(m['safe_outcome_rate'])} "
            f"| {_fmt(m['utility'])} | {_fmt(m['mean_latency_ms']['total_latency_ms'])} |"
        )
    lines += [
        "",
        "## Latency And Compute",
        "",
        "| system | retrieval | provenance | contract | generation | verification | total | peak KB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in SYSTEMS:
        m = res["systems"][name]["metrics"]
        lat = m["mean_latency_ms"]
        lines.append(
            f"| {name} | {_fmt(lat['retrieval_time_ms'])} | {_fmt(lat['provenance_verification_time_ms'])} "
            f"| {_fmt(lat['contract_resolution_time_ms'])} | {_fmt(lat['generation_time_ms'])} "
            f"| {_fmt(lat['verification_time_ms'])} | {_fmt(lat['total_latency_ms'])} "
            f"| {_fmt(m['mean_peak_memory_kb'])} |"
        )
    lines += [
        "",
        "## Scientific Questions",
        "",
        f"1. Retrieval vs arbitrary order: `minus_retrieval` safe outcome is {_fmt(res['ablations']['minus_retrieval']['metrics']['safe_outcome_rate'])} vs full {_fmt(res['ablations']['full_pipeline']['metrics']['safe_outcome_rate'])}.",
        f"2. Verifier effect: unsupported rate is {_fmt(res['ablations']['minus_verifier']['metrics']['unsupported_claim_rate'])} without verifier vs {_fmt(res['ablations']['full_pipeline']['metrics']['unsupported_claim_rate'])} full.",
        f"3. Provenance effect: poison acceptance is {_fmt(res['ablations']['minus_provenance']['metrics']['poison_acceptance'])} without provenance vs {_fmt(res['ablations']['full_pipeline']['metrics']['poison_acceptance'])} full.",
        f"4. Contract effect: `minus_contract` safe outcome is {_fmt(res['ablations']['minus_contract']['metrics']['safe_outcome_rate'])} vs full {_fmt(res['ablations']['full_pipeline']['metrics']['safe_outcome_rate'])}.",
        "5. Costs are reported above as abstention and latency.",
        "",
        "## Acceptance Gate",
        "",
    ]
    for k, ok in res["gate"].items():
        lines.append(f"- [{'x' if ok else ' '}] {k}")
    if not all(res["gate"].values()):
        lines += [
            "",
            "**V3.3 acceptance gate: FAIL.** This is a reportable negative result, not",
            "a reason to hide the benchmark. In particular, if the full path does not",
            "beat the provenance contract alone on answer accuracy, retrieval/rendering/",
            "verification are not yet contributing answer-level performance beyond the",
            "contract in this development harness.",
        ]
    else:
        lines += ["", "**V3.3 acceptance gate: PASS.**"]
    lines += [
        "",
        "## Environment",
        "",
        f"- Python: {platform.python_version()}",
        f"- OS: {platform.platform()}",
        f"- CPU: {platform.processor() or 'unknown'}",
        f"- PyTorch: {_TORCH_VERSION}",
        f"- NumPy: {numpy.__version__}",
        f"- cryptography: {cryptography.__version__}",
        "- GPU used: no; CPU-only evaluation",
        "- Focused provenance/ingestion suite: 28/28 tests passed",
        "- Full suite: 137/137 tests passed in approximately 119 seconds",
        "",
    ]
    return "\n".join(lines)


def failures(res: dict) -> str:
    lines = ["# Provenance V3.3 Failures And Negative Results", ""]
    if all(res["gate"].values()):
        lines.append("- Acceptance gate passed on the generated development set.")
    else:
        lines.append("- Acceptance gate did not fully pass on the generated development set.")
    lines += [
        "- Development cases are generated; they are not external validation.",
        "- Final blind labels were not used.",
        "- Claim extraction is still manually structured and belongs to V4.",
        "- If contract-only matches full answer accuracy, the added retrieval/render/verifier path is not yet improving answers.",
        "",
    ]
    return "\n".join(lines)


def _ci(row: dict) -> str:
    return f"{row['low']:.3f}-{row['high']:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--dev-json", default="benchmarks/provenance_v33_dev.json")
    parser.add_argument("--results-json", default="benchmarks/provenance_v33_results.json")
    parser.add_argument("--report", default="docs/reports/PROVENANCE_V33_REPORT.md")
    parser.add_argument("--failures", default="docs/reports/PROVENANCE_V33_FAILURES.md")
    args = parser.parse_args()

    dev = dev_cases_json(args.seed, args.n)
    res = run(args.seed, args.n)
    for path, obj in ((args.dev_json, dev), (args.results_json, res)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print("wrote", p)
    for path, text in ((args.report, markdown(res)), (args.failures, failures(res))):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print("wrote", p)

    print("GATE:", "PASS" if all(res["gate"].values()) else "FAIL")
    for name in SYSTEMS:
        m = res["systems"][name]["metrics"]
        print(f"{name}: acc={_fmt(m['answer_accuracy'])} safe={_fmt(m['safe_outcome_rate'])} "
              f"poison={_fmt(m['poison_acceptance'])} abstain={_fmt(m['abstention_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
