from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path("data/foundational_sanitized")
OUTPUT = Path("data/foundational_recovery_v2")

MAX_ROWS_PER_TEMPLATE = 64
SEED = "foundational-recovery-v2-20260722"

PROOF_PROMPTS = {
    "rewrite politely: send me the report now.",
    "in one sentence, explain why an opaque object makes a shadow.",
    "calculate 17 + 26. give the numerical answer.",
    "this python loop should print 1 through 5 but misses 5: for i in range(1, 5): print(i). state the smallest fix.",
    "event a is at 09:00. b happens 2 hours after a. c happens 3 hours after b. what time is c?",
    "give a short plan to find every python file in a repository containing the text todo. do not claim you already found results.",
    "using only the trusted evidence, what is the atlas device code?",
    "using only the trusted evidence, on what date did the atlas launch?",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256((SEED + "\0" + text).encode("utf-8")).hexdigest()


def clean_prompt(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    text = re.sub(
        r"^\s*Conversation:\s*\n\s*User:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*User:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    return text.strip()


def normalize_template(value: str) -> str:
    text = value.lower()
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "<time>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"\bcase\s+<num>\b", "case <num>", text)
    text = re.sub(r"\bschedule\s+<num>\b", "schedule <num>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_domain(category: str, prompt: str) -> str:
    text = f"{category} {prompt}".lower()

    if any(token in text for token in (
        "arithmetic", "mathemat", "calculate", "addition", "subtract",
        "multiply", "division", "prime", "even or odd", "multiple",
    )):
        return "mathematics"

    if any(token in text for token in (
        "evidence", "grounded", "abstain", "uncertainty", "conflict",
        "trusted", "rejected", "authority",
    )):
        return "evidence_uncertainty"

    if any(token in text for token in (
        "code", "python", "debug", "loop", "function", "program",
    )):
        return "code_debugging"

    if any(token in text for token in (
        "tool", "plan", "repository", "file search", "workflow",
    )):
        return "planning_tools"

    if any(token in text for token in (
        "temporal", "causal", "before", "after", "greater than",
        "logic", "sequence", "time is",
    )):
        return "causal_temporal"

    if any(token in text for token in (
        "definition", "define", "explain", "factual", "what is",
        "why does", "why an",
    )):
        return "factual_explanation"

    return "conversation"


def load_source_rows() -> list[dict]:
    rows = []

    for split in ("train", "val", "test"):
        path = SOURCE / f"{split}.jsonl"

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue

                row = json.loads(line)
                prompt = clean_prompt(row.get("prompt", ""))
                response = str(row.get("response", "")).strip()

                if not prompt or not response:
                    continue

                category = str(row.get("category") or "conversation")
                domain = map_domain(category, prompt)

                rows.append({
                    **row,
                    "prompt": prompt,
                    "response": response,
                    "category": category,
                    "domain": domain,
                    "source_split": split,
                    "source_line": line_number,
                })

    return rows


rows = load_source_rows()

# Remove exact duplicate prompt/response pairs.
deduped = {}
for row in rows:
    key = sha256_bytes(
        (row["prompt"] + "\0" + row["response"]).encode("utf-8")
    )
    deduped.setdefault(key, row)

rows = list(deduped.values())

# Drop prompts that have contradictory or multiple target answers.
answers_by_prompt = defaultdict(set)
for row in rows:
    answers_by_prompt[row["prompt"]].add(row["response"])

conflicting_prompts = {
    prompt
    for prompt, answers in answers_by_prompt.items()
    if len(answers) > 1
}

rows = [
    row for row in rows
    if row["prompt"] not in conflicting_prompts
]

# Keep proof-gate cases fully held out.
excluded_proof_rows = [
    row for row in rows
    if row["prompt"].strip().lower() in PROOF_PROMPTS
]

rows = [
    row for row in rows
    if row["prompt"].strip().lower() not in PROOF_PROMPTS
]

# Group equivalent synthetic templates.
groups = defaultdict(list)

for row in rows:
    template = normalize_template(row["prompt"])
    template_key = f'{row["domain"]}\0{row["category"]}\0{template}'
    row["template"] = template
    row["template_id"] = stable_hash(template_key)
    groups[row["template_id"]].append(row)

# Deterministically cap massive repeated templates.
capped_rows = []
dropped_by_cap = 0

for template_id, group in groups.items():
    group.sort(
        key=lambda row: stable_hash(
            row["prompt"] + "\0" + row["response"]
        )
    )

    kept = group[:MAX_ROWS_PER_TEMPLATE]
    dropped_by_cap += max(0, len(group) - len(kept))
    capped_rows.extend(kept)

# Assign entire template families to one split.
splits = {"train": [], "val": [], "test": []}

for row in capped_rows:
    bucket = int(row["template_id"][:8], 16) % 100

    if bucket < 10:
        split = "test"
    elif bucket < 20:
        split = "val"
    else:
        split = "train"

    row["split"] = split
    splits[split].append(row)

for split in splits:
    splits[split].sort(
        key=lambda row: stable_hash(
            row["prompt"] + "\0" + row["response"]
        )
    )

OUTPUT.mkdir(parents=True, exist_ok=True)

manifest = {
    "version": 2,
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "seed": SEED,
    "max_rows_per_template": MAX_ROWS_PER_TEMPLATE,
    "input_rows": len(load_source_rows()),
    "deduplicated_rows": len(deduped),
    "conflicting_prompts_dropped": len(conflicting_prompts),
    "proof_rows_excluded": len(excluded_proof_rows),
    "template_cap_rows_dropped": dropped_by_cap,
    "splits": {},
}

prompt_sets = {}
template_sets = {}

for split, split_rows in splits.items():
    path = OUTPUT / f"{split}.jsonl"

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in split_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    data = path.read_bytes()

    prompt_sets[split] = {row["prompt"] for row in split_rows}
    template_sets[split] = {row["template_id"] for row in split_rows}

    manifest["splits"][split] = {
        "rows": len(split_rows),
        "templates": len(template_sets[split]),
        "sha256": sha256_bytes(data),
        "domain_counts": dict(
            sorted(Counter(row["domain"] for row in split_rows).items())
        ),
        "category_counts": dict(
            sorted(Counter(row["category"] for row in split_rows).items())
        ),
        "nested_conversation_prompts": sum(
            row["prompt"].startswith("Conversation:")
            for row in split_rows
        ),
    }

manifest["leakage"] = {
    "train_val_prompt_overlap": len(
        prompt_sets["train"] & prompt_sets["val"]
    ),
    "train_test_prompt_overlap": len(
        prompt_sets["train"] & prompt_sets["test"]
    ),
    "val_test_prompt_overlap": len(
        prompt_sets["val"] & prompt_sets["test"]
    ),
    "train_val_template_overlap": len(
        template_sets["train"] & template_sets["val"]
    ),
    "train_test_template_overlap": len(
        template_sets["train"] & template_sets["test"]
    ),
    "val_test_template_overlap": len(
        template_sets["val"] & template_sets["test"]
    ),
}

all_domains = {
    row["domain"]
    for split_rows in splits.values()
    for row in split_rows
}

manifest["quality_gate_passed"] = (
    all(manifest["splits"][split]["rows"] > 0 for split in splits)
    and all(
        manifest["splits"][split]["nested_conversation_prompts"] == 0
        for split in splits
    )
    and not any(manifest["leakage"].values())
    and all(
        manifest["splits"]["train"]["domain_counts"].get(domain, 0) >= 20
        for domain in all_domains
    )
)

manifest_path = OUTPUT / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(manifest, indent=2, ensure_ascii=False))

if not manifest["quality_gate_passed"]:
    raise SystemExit("FOUNDATIONAL RECOVERY V2 QUALITY GATE FAILED")
