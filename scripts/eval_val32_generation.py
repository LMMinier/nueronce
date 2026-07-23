#!/usr/bin/env python3

import json
import re
from pathlib import Path

import torch

from nueronce.chat import load_checkpoint
from nueronce.incremental import IncrementalGenerator
from nueronce.prompting import (
    STOP_SEQUENCES,
    extract_assistant_continuation,
    format_inference_prompt,
)

CHECKPOINT = Path(r"runs\foundational_recovery_v2_probe500\step1500_best.pt")
DATASET = Path(r"data\foundational_val32\val.jsonl")
SYSTEM_FILE = Path(r"runs\forgeloop\system_prompt.txt")
OUTPUT = Path(r"metrics\foundational_val32_step1500_generation.json")

torch.set_num_threads(8)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def first_divergence(expected: str, actual: str):
    expected_bytes = expected.encode("utf-8")
    actual_bytes = actual.encode("utf-8")

    limit = min(len(expected_bytes), len(actual_bytes))

    for index in range(limit):
        if expected_bytes[index] != actual_bytes[index]:
            return {
                "byte_index": index,
                "expected_byte": expected_bytes[index],
                "actual_byte": actual_bytes[index],
            }

    if len(expected_bytes) != len(actual_bytes):
        return {
            "byte_index": limit,
            "expected_byte": (
                expected_bytes[limit] if limit < len(expected_bytes) else None
            ),
            "actual_byte": (
                actual_bytes[limit] if limit < len(actual_bytes) else None
            ),
        }

    return None


if not CHECKPOINT.exists():
    raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")

if not DATASET.exists():
    raise SystemExit(f"Dataset not found: {DATASET}")

model, metadata = load_checkpoint(str(CHECKPOINT))
model.eval()

system_message = (
    SYSTEM_FILE.read_text(encoding="utf-8").strip()
    if SYSTEM_FILE.exists()
    else ""
)

rows = [
    json.loads(line)
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

results = []

for index, row in enumerate(rows, start=1):
    expected = row["response"].strip()

    rendered = format_inference_prompt(
        system_message=system_message,
        user_request=row["prompt"],
        trusted_evidence="",
        response_plan="",
    )

    max_new = max(
        96,
        min(512, len(expected.encode("utf-8")) + 96),
    )

    raw = IncrementalGenerator(model).generate(
        rendered.encode("utf-8"),
        max_new=max_new,
        temperature=0.0,
        greedy=True,
        max_ctx=768,
        stop_sequences=STOP_SEQUENCES,
        continuation_only=True,
    )

    actual = extract_assistant_continuation(raw).strip()

    exact = actual == expected
    normalized = normalize(actual) == normalize(expected)

    result = {
        "index": index,
        "record_id": row.get("record_id"),
        "domain": row.get("domain"),
        "prompt": row["prompt"],
        "expected": expected,
        "actual": actual,
        "exact_match": exact,
        "normalized_match": normalized,
        "expected_bytes": len(expected.encode("utf-8")),
        "generated_bytes": len(actual.encode("utf-8")),
        "first_divergence": first_divergence(expected, actual),
    }

    results.append(result)

    print(
        json.dumps(
            {
                "index": index,
                "domain": row.get("domain"),
                "exact": exact,
                "normalized": normalized,
                "expected": expected,
                "actual": actual,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

exact_count = sum(row["exact_match"] for row in results)
normalized_count = sum(row["normalized_match"] for row in results)

domain_scores = {}

for domain in sorted({row["domain"] for row in results}):
    domain_rows = [row for row in results if row["domain"] == domain]

    domain_scores[domain] = {
        "rows": len(domain_rows),
        "exact": sum(row["exact_match"] for row in domain_rows),
        "normalized": sum(row["normalized_match"] for row in domain_rows),
    }

report = {
    "checkpoint": str(CHECKPOINT),
    "checkpoint_step": metadata.get("sft_step", metadata.get("step")),
    "best_val_loss": metadata.get("best_val_loss"),
    "total": len(results),
    "exact_matches": exact_count,
    "normalized_matches": normalized_count,
    "exact_rate": exact_count / len(results),
    "normalized_rate": normalized_count / len(results),
    "diagnostic_passed": normalized_count >= 30,
    "domain_scores": domain_scores,
    "results": results,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print(
    json.dumps(
        {
            "checkpoint_step": report["checkpoint_step"],
            "best_val_loss": report["best_val_loss"],
            "total": report["total"],
            "exact_matches": report["exact_matches"],
            "normalized_matches": report["normalized_matches"],
            "exact_rate": report["exact_rate"],
            "normalized_rate": report["normalized_rate"],
            "diagnostic_passed": report["diagnostic_passed"],
            "output": str(OUTPUT),
        },
        indent=2,
    )
)
