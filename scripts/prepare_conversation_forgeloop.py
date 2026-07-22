"""Convert validated conversation SFT shards to strict ForgeLoop JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sanitize_forgeloop_data import clean_text, digest


def convert(record):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None, "malformed_messages"
    if any(not isinstance(m, dict) or m.get("role") not in {"user", "assistant"}
           or not isinstance(m.get("content"), str) or not m["content"].strip()
           for m in messages):
        return None, "malformed_turn"
    history = []
    for message in messages[:-1]:
        role = "User" if message["role"] == "user" else "Assistant"
        history.append(f"{role}: {clean_text(message['content'])}")
    evidence = [clean_text(x) for x in record.get("trusted_evidence", []) if clean_text(x)]
    plan = [clean_text(x) for x in record.get("response_plan", []) if clean_text(x)]
    prompt_parts = ["Conversation:\n" + "\n".join(history)]
    if evidence:
        prompt_parts.append("Trusted evidence:\n" + "\n".join(evidence))
    if plan:
        prompt_parts.append("Response requirements:\n" + "\n".join(plan))
    prompt = "\n\n".join(prompt_parts)
    response = clean_text(messages[-1]["content"], response=True)
    if not prompt or not response or len(prompt) > 4000 or len(response) > 2000:
        return None, "length_or_empty"
    pair_hash = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
    return {
        "prompt": prompt, "response": response,
        "kind": "nueronce_conversation_sanitized",
        "source": record.get("source", "nueronce-generated"),
        "category": record.get("category", "unknown"),
        "record_id": record.get("id", pair_hash[:16]),
        "example_hash": pair_hash,
    }, None


def load(paths):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/conversation_sft_clean"))
    p.add_argument("--output", type=Path, default=Path("data/foundational_sanitized"))
    args = p.parse_args()
    paths = {
        "train": sorted((args.source / "train_shards").glob("shard_*.jsonl")),
        "val": [args.source / "validation.jsonl"],
        "test": [args.source / "test.jsonl"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows_by_split, report = {}, {"source": str(args.source), "splits": {}}
    for split, split_paths in paths.items():
        rows, rejected, seen = [], Counter(), set()
        for record in load(split_paths):
            row, reason = convert(record)
            if reason:
                rejected[reason] += 1; continue
            if row["example_hash"] in seen:
                rejected["duplicate_after_cleaning"] += 1; continue
            seen.add(row["example_hash"]); rows.append(row)
        rows_by_split[split] = rows
        report["splits"][split] = {
            "accepted": len(rows), "rejected": dict(rejected),
            "categories": dict(Counter(r["category"] for r in rows)),
        }
    # Scenario-ID removal can expose templated prompt collisions that the raw
    # source split considered distinct. Evaluation splits win those collisions.
    val_hashes = {digest(r["prompt"]) for r in rows_by_split["val"]}
    test_hashes = {digest(r["prompt"]) for r in rows_by_split["test"]}
    before_train = len(rows_by_split["train"])
    before_test = len(rows_by_split["test"])
    rows_by_split["train"] = [r for r in rows_by_split["train"]
                              if digest(r["prompt"]) not in val_hashes | test_hashes]
    rows_by_split["test"] = [r for r in rows_by_split["test"]
                             if digest(r["prompt"]) not in val_hashes]
    report["splits"]["train"]["rejected"]["split_prompt_leakage"] = (
        before_train - len(rows_by_split["train"])
    )
    report["splits"]["test"]["rejected"]["split_prompt_leakage"] = (
        before_test - len(rows_by_split["test"])
    )
    for split, rows in rows_by_split.items():
        report["splits"][split]["accepted"] = len(rows)
        report["splits"][split]["categories"] = dict(Counter(r["category"] for r in rows))
    hashes = {s: {digest(r["prompt"]) for r in rows} for s, rows in rows_by_split.items()}
    report["leakage"] = {
        "train_val": len(hashes["train"] & hashes["val"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "val_test": len(hashes["val"] & hashes["test"]),
    }
    report["quality_gate_passed"] = not any(report["leakage"].values())
    if not report["quality_gate_passed"]:
        raise RuntimeError(f"cleaned prompt leakage: {report['leakage']}")
    for split, rows in rows_by_split.items():
        path = args.output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
