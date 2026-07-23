import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path(
    r"artifacts\foundational_v3_relabel_preview_corrected.csv"
)

OUT_DIR = Path(r"data\foundational_recovery_v3")
REPORT_MD = Path(r"artifacts\foundational_v3_build_report.md")
MANIFEST_JSON = Path(r"artifacts\foundational_v3_manifest.json")

FAMILY_CAP = 32
MIN_EVAL_FAMILIES = 3
SPLITS = ("train", "val", "test")


def sha256_text(*parts):
    text = "\n".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text):
    return " ".join(str(text).lower().split())


def family_id(row):
    template_id = str(row.get("template_id", "")).strip()

    if template_id:
        return template_id

    return "prompt-" + sha256_text(
        normalize_text(row.get("prompt", ""))
    )


def exclusion_reason(row):
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    category = str(row.get("category", "")).strip()

    if not prompt:
        return "empty_prompt"

    if not response:
        return "empty_response"

    # Canonical prompting already provides system/user/assistant markers.
    # Nested transcript labels can teach the wrong inference format.
    if re.search(
        r"(^|\n)\s*(assistant|user)\s*:",
        prompt,
        flags=re.IGNORECASE,
    ):
        return "nested_chat_transcript"

    if re.search(
        r"<\|(assistant|user|system|end)\|>",
        prompt,
        flags=re.IGNORECASE,
    ):
        return "embedded_control_marker"

    # The sampled synthetic syllogisms contain grammar defects,
    # plural disagreement, truncated nouns, and literal placeholder X.
    if category == "logic":
        if re.match(
            r"^\s*(all|if all)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            return "untrusted_synthetic_syllogism"

        if re.search(
            r"\bX is\b",
            response,
            flags=re.IGNORECASE,
        ):
            return "untrusted_synthetic_syllogism"

    # Example: "How do I tie your shoelaces?"
    if re.match(
        r"^\s*how do i\b.*\byour\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        return "procedure_pronoun_mismatch"

    return None


rows = list(
    csv.DictReader(
        SOURCE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        )
    )
)

initial_status_counts = Counter(row["status"] for row in rows)

eligible = [
    row
    for row in rows
    if row["status"] not in {
        "hard_reject",
        "manual_review",
    }
    and row["target_capability"] != "manual_review"
]

excluded_quality = Counter()
quality_approved = []

for row in eligible:
    reason = exclusion_reason(row)

    if reason:
        excluded_quality[reason] += 1
        continue

    quality_approved.append(row)

# Group approved rows by capability and template family.
families_by_capability = defaultdict(lambda: defaultdict(list))

for row in quality_approved:
    capability = row["target_capability"]
    family = family_id(row)

    families_by_capability[capability][family].append(row)

# Cap repeated template families so one pattern cannot dominate training.
capped_families = defaultdict(dict)
capped_rows_removed = Counter()

for capability, families in families_by_capability.items():
    for family, members in families.items():
        ordered = sorted(
            members,
            key=lambda row: sha256_text(
                row.get("prompt", ""),
                row.get("response", ""),
                row.get("record_id", ""),
                row.get("source", ""),
            ),
        )

        kept = ordered[:FAMILY_CAP]
        removed = len(ordered) - len(kept)

        if removed:
            capped_rows_removed[capability] += removed

        capped_families[capability][family] = kept

# Assign entire families to only one split.
family_assignment = {}
split_family_counts = defaultdict(Counter)

for capability in sorted(capped_families):
    families = list(capped_families[capability])

    families.sort(
        key=lambda family: sha256_text(
            "foundational-v3-family-split",
            capability,
            family,
        )
    )

    family_count = len(families)

    if family_count < 3:
        raise RuntimeError(
            f"{capability!r} has only {family_count} families; "
            "cannot create three independent splits."
        )

    eval_count = max(
        MIN_EVAL_FAMILIES,
        round(family_count * 0.10),
    )

    while family_count - (2 * eval_count) < 1:
        eval_count -= 1

    if eval_count < 1:
        raise RuntimeError(
            f"Could not allocate independent splits for {capability!r}."
        )

    test_families = set(families[:eval_count])
    val_families = set(
        families[eval_count : 2 * eval_count]
    )
    train_families = set(
        families[2 * eval_count :]
    )

    assignments = {
        "train": train_families,
        "val": val_families,
        "test": test_families,
    }

    for split, assigned_families in assignments.items():
        split_family_counts[capability][split] = len(
            assigned_families
        )

        for family in assigned_families:
            family_assignment[(capability, family)] = split

# Construct the new rows.
output_rows = {split: [] for split in SPLITS}

for capability in sorted(capped_families):
    for family, members in capped_families[capability].items():
        split = family_assignment[(capability, family)]

        for row in members:
            prompt = str(row["prompt"]).strip()
            response = str(row["response"]).strip()

            example_hash = sha256_text(
                capability,
                prompt,
                response,
                row.get("record_id", ""),
            )

            converted = {
                "kind": "nueronce_foundational_v3",
                "split": split,
                "domain": capability,
                "category": row.get("category", ""),
                "prompt": prompt,
                "response": response,
                "source": row.get("source", ""),
                "source_split": row.get("source_split", ""),
                "original_split": row.get("split", ""),
                "original_domain": row.get("current_domain", ""),
                "record_id": row.get("record_id", ""),
                "template_id": row.get("template_id", ""),
                "template_family": family,
                "example_hash": example_hash,
                "provenance_warnings": [
                    item
                    for item in str(
                        row.get("provenance_warnings", "")
                    ).split("|")
                    if item
                ],
            }

            output_rows[split].append(converted)

# Stable deterministic order within each split.
for split in SPLITS:
    output_rows[split].sort(
        key=lambda row: (
            row["domain"],
            sha256_text(
                "foundational-v3-row-order",
                row["example_hash"],
            ),
        )
    )

# Verify no row, prompt, or family leaks between splits.
family_sets = {
    split: {
        (row["domain"], row["template_family"])
        for row in output_rows[split]
    }
    for split in SPLITS
}

prompt_sets = {
    split: {
        normalize_text(row["prompt"])
        for row in output_rows[split]
    }
    for split in SPLITS
}

content_sets = {
    split: {
        sha256_text(
            normalize_text(row["prompt"]),
            normalize_text(row["response"]),
        )
        for row in output_rows[split]
    }
    for split in SPLITS
}

overlap_report = {}

for left, right in (
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
):
    family_overlap = family_sets[left] & family_sets[right]
    prompt_overlap = prompt_sets[left] & prompt_sets[right]
    content_overlap = content_sets[left] & content_sets[right]

    overlap_report[f"{left}_vs_{right}"] = {
        "family_overlap": len(family_overlap),
        "prompt_overlap": len(prompt_overlap),
        "content_overlap": len(content_overlap),
    }

    if family_overlap:
        raise RuntimeError(
            f"Family leakage detected between {left} and {right}: "
            f"{len(family_overlap)}"
        )

    if prompt_overlap:
        raise RuntimeError(
            f"Prompt leakage detected between {left} and {right}: "
            f"{len(prompt_overlap)}"
        )

    if content_overlap:
        raise RuntimeError(
            f"Content leakage detected between {left} and {right}: "
            f"{len(content_overlap)}"
        )

# Replace only this new V3 directory.
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    destination = OUT_DIR / f"{split}.jsonl"

    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows[split]:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

split_counts = {
    split: len(output_rows[split])
    for split in SPLITS
}

capability_split_counts = defaultdict(Counter)

for split in SPLITS:
    for row in output_rows[split]:
        capability_split_counts[row["domain"]][split] += 1

manifest = {
    "dataset": "foundational_recovery_v3",
    "family_cap": FAMILY_CAP,
    "minimum_eval_families": MIN_EVAL_FAMILIES,
    "source_rows": len(rows),
    "eligible_after_status_filter": len(eligible),
    "quality_approved_before_family_cap": len(quality_approved),
    "final_rows": sum(split_counts.values()),
    "split_counts": split_counts,
    "excluded_quality": dict(excluded_quality),
    "capped_rows_removed": dict(capped_rows_removed),
    "overlap_report": overlap_report,
    "capabilities": {
        capability: {
            "rows": dict(capability_split_counts[capability]),
            "families": dict(split_family_counts[capability]),
        }
        for capability in sorted(capability_split_counts)
    },
    "unsupported_capabilities": [
        "code_debugging",
    ],
}

MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
MANIFEST_JSON.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

lines = [
    "# Foundational Recovery V3 Build Report",
    "",
    "## Build summary",
    "",
    f"- Source rows: **{len(rows)}**",
    f"- Eligible after status filtering: **{len(eligible)}**",
    f"- Approved after additional quality filtering: "
    f"**{len(quality_approved)}**",
    f"- Final rows after family caps: "
    f"**{sum(split_counts.values())}**",
    f"- Family cap: **{FAMILY_CAP} rows**",
    f"- Supported capabilities: "
    f"**{len(capability_split_counts)}**",
    "- Unsupported capability: **code_debugging**",
    "",
    "## New split sizes",
    "",
    "| Split | Rows |",
    "|---|---:|",
]

for split in SPLITS:
    lines.append(f"| {split} | {split_counts[split]} |")

lines.extend([
    "",
    "## Additional quality exclusions",
    "",
    "| Reason | Rows |",
    "|---|---:|",
])

if excluded_quality:
    for reason, count in excluded_quality.most_common():
        lines.append(f"| {reason} | {count} |")
else:
    lines.append("| None | 0 |")

lines.extend([
    "",
    "## Rows removed by family cap",
    "",
    "| Capability | Removed rows |",
    "|---|---:|",
])

if capped_rows_removed:
    for capability, count in sorted(
        capped_rows_removed.items()
    ):
        lines.append(f"| {capability} | {count} |")
else:
    lines.append("| None | 0 |")

lines.extend([
    "",
    "## Capability distribution",
    "",
    "| Capability | Train rows | Val rows | Test rows | "
    "Train families | Val families | Test families |",
    "|---|---:|---:|---:|---:|---:|---:|",
])

for capability in sorted(capability_split_counts):
    row_counts = capability_split_counts[capability]
    family_counts = split_family_counts[capability]

    lines.append(
        f"| {capability} "
        f"| {row_counts['train']} "
        f"| {row_counts['val']} "
        f"| {row_counts['test']} "
        f"| {family_counts['train']} "
        f"| {family_counts['val']} "
        f"| {family_counts['test']} |"
    )

lines.extend([
    "",
    "## Leakage verification",
    "",
    "| Comparison | Family overlap | Prompt overlap | Content overlap |",
    "|---|---:|---:|---:|",
])

for comparison, values in overlap_report.items():
    lines.append(
        f"| {comparison.replace('_', ' ')} "
        f"| {values['family_overlap']} "
        f"| {values['prompt_overlap']} "
        f"| {values['content_overlap']} |"
    )

lines.extend([
    "",
    "## Training restriction",
    "",
    "Do not claim or evaluate code-debugging capability from this dataset. "
    "No genuine code-debugging examples were found.",
    "",
    "Do not start training until the generated V3 files and this report "
    "have been reviewed.",
])

REPORT_MD.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(f"Wrote dataset: {OUT_DIR}")
print(f"Wrote report: {REPORT_MD}")
print(f"Wrote manifest: {MANIFEST_JSON}")
print()
print("Final split counts:")

for split in SPLITS:
    print(f"  {split}: {split_counts[split]}")

print()
print("Capability counts:")

for capability in sorted(capability_split_counts):
    rows_by_split = capability_split_counts[capability]
    families = split_family_counts[capability]

    print(
        f"  {capability}: "
        f"rows train={rows_by_split['train']} "
        f"val={rows_by_split['val']} "
        f"test={rows_by_split['test']}; "
        f"families train={families['train']} "
        f"val={families['val']} "
        f"test={families['test']}"
    )

print()
print("Leakage checks:")

for comparison, values in overlap_report.items():
    print(f"  {comparison}: {values}")

print()
print("Quality exclusions:")

for reason, count in excluded_quality.most_common():
    print(f"  {reason}: {count}")
