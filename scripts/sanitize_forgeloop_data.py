"""Build strict ForgeLoop prompt/response splits from prompt-aligned records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SPACE = re.compile(r"\s+")
SCENARIO = re.compile(r"\s*(?:for\s+)?scenario\s+\d+\.?", re.I)
ADDRESSES = re.compile(r"\s*This addresses scenario\s+\d+\.?", re.I)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")


def norm(text: str) -> str:
    return SPACE.sub(" ", text.casefold()).strip()


def digest(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def clean_text(text: str, *, response: bool = False) -> str:
    text = CONTROL.sub("", text).strip()
    text = ADDRESSES.sub("", text) if response else SCENARIO.sub("", text)
    text = SPACE.sub(" ", text).strip()
    if response and text and text[-1] not in ".!?`)}]":
        text += "."
    return text


def convert(record: dict) -> tuple[dict | None, str | None]:
    required = ("id", "user_request", "assistant_response", "category", "source_group")
    if any(not isinstance(record.get(k), str) or not record[k].strip() for k in required):
        return None, "missing_required_text"
    evidence = record.get("trusted_evidence") or []
    plan = record.get("response_plan") or []
    if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
        return None, "malformed_evidence"
    if not isinstance(plan, list) or not all(isinstance(x, str) for x in plan):
        return None, "malformed_plan"
    user = clean_text(record["user_request"])
    answer = clean_text(record["assistant_response"], response=True)
    evidence = [clean_text(x) for x in evidence if clean_text(x)]
    plan = [clean_text(x) for x in plan if clean_text(x)]
    if len(user) < 3 or len(answer) < 2:
        return None, "too_short"
    if len(user) > 2000 or len(answer) > 2000:
        return None, "too_long"
    if any(marker in user or marker in answer for marker in ROLE_MARKERS):
        return None, "role_marker_injection"
    sections = [f"Request:\n{user}"]
    if evidence:
        sections.append("Trusted evidence:\n" + "\n".join(evidence))
    if plan:
        sections.append("Response requirements:\n" + "\n".join(plan))
    prompt = "\n\n".join(sections)
    return {
        "prompt": prompt,
        "response": answer,
        "kind": "nueronce_prompt_aligned_sanitized",
        "source": "data/sft_prompt_aligned",
        "license": "repository-authored",
        "category": record["category"],
        "source_group": record["source_group"],
        "record_id": record["id"],
        "example_hash": hashlib.sha256((prompt + "\0" + answer).encode("utf-8")).hexdigest(),
    }, None


def read_paths(paths):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        yield {"_invalid": f"{path}:{line_number}:{exc}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/sft_prompt_aligned"))
    p.add_argument("--output", type=Path, default=Path("data/forgeloop_sanitized"))
    args = p.parse_args()
    split_paths = {
        "train": sorted((args.source / "train").glob("shard_*.jsonl")),
        "val": [args.source / "validation.jsonl"],
        "test": [args.source / "test.jsonl"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.source), "output": str(args.output), "splits": {}}
    prompt_hashes = {}
    split_rows = {}
    pair_hashes = set()
    for split, paths in split_paths.items():
        accepted, rejected, categories = [], Counter(), Counter()
        local_pairs = set()
        for record in read_paths(paths):
            converted, reason = convert(record)
            if reason:
                rejected[reason] += 1
                continue
            pair = converted["example_hash"]
            # Weighted duplicate records are useful for sampling weights but are
            # not new knowledge. Keep one exact prompt/answer pair only.
            if pair in local_pairs:
                rejected["exact_duplicate"] += 1
                continue
            local_pairs.add(pair)
            accepted.append(converted)
            categories[converted["category"]] += 1
        out = args.output / f"{split}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for row in accepted:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        prompt_hashes[split] = {digest(row["prompt"]) for row in accepted}
        split_rows[split] = accepted
        pair_hashes.update(row["example_hash"] for row in accepted)
        report["splits"][split] = {
            "input_records": len(accepted) + sum(rejected.values()),
            "accepted_unique": len(accepted),
            "rejected": dict(rejected),
            "categories": dict(categories),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
    # Cleaning deliberately removes synthetic scenario IDs, which can reveal
    # template leakage hidden by different numbers. Holdouts win: remove any
    # cleaned prompt seen in val/test from train, then remove val prompts from
    # test so every evaluation request is genuinely distinct.
    val_prompts = {digest(row["prompt"]) for row in split_rows["val"]}
    test_prompts = {digest(row["prompt"]) for row in split_rows["test"]}
    before_train = len(split_rows["train"])
    split_rows["train"] = [row for row in split_rows["train"]
                           if digest(row["prompt"]) not in val_prompts | test_prompts]
    before_test = len(split_rows["test"])
    split_rows["test"] = [row for row in split_rows["test"]
                          if digest(row["prompt"]) not in val_prompts]
    report["splits"]["train"]["rejected"]["split_prompt_leakage"] = (
        before_train - len(split_rows["train"])
    )
    report["splits"]["test"]["rejected"]["split_prompt_leakage"] = (
        before_test - len(split_rows["test"])
    )
    for split, rows in split_rows.items():
        out = args.output / f"{split}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        prompt_hashes[split] = {digest(row["prompt"]) for row in rows}
        report["splits"][split]["accepted_unique"] = len(rows)
        report["splits"][split]["categories"] = dict(Counter(row["category"] for row in rows))
        report["splits"][split]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    overlaps = {
        "train_val_prompt": len(prompt_hashes["train"] & prompt_hashes["val"]),
        "train_test_prompt": len(prompt_hashes["train"] & prompt_hashes["test"]),
        "val_test_prompt": len(prompt_hashes["val"] & prompt_hashes["test"]),
    }
    report["leakage"] = overlaps
    report["quality_gate_passed"] = all(value == 0 for value in overlaps.values())
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["quality_gate_passed"]:
        raise SystemExit("split leakage detected")


if __name__ == "__main__":
    main()
