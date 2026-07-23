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

OUT_DIR = Path(r"data\foundational_recovery_v3_1")
REPORT = Path(r"artifacts\foundational_v3_1_build_report.md")
MANIFEST = Path(r"artifacts\foundational_v3_1_manifest.json")

FAMILY_CAP = 16
EVAL_FAMILY_FRACTION = 0.15
MIN_EVAL_FAMILIES = 3
SPLITS = ("train", "val", "test")


def digest(*parts):
    text = "\n".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    return " ".join(str(text).lower().split())


def get_family(row):
    template_id = str(row.get("template_id", "")).strip()

    if template_id:
        return template_id

    return "prompt-" + digest(normalize(row.get("prompt", "")))


def quality_exclusion(row):
    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    category = str(row.get("category", "")).strip()

    if not prompt:
        return "empty_prompt"

    if not response:
        return "empty_response"

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

    if re.match(
        r"^\s*how do i\b.*\byour\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        return "procedure_pronoun_mismatch"

    return None


def choose_family_subset(
    candidates,
    required_count,
    target_rows,
    salt,
):
    """
    Select exactly required_count whole families with a combined row
    count as close as possible to target_rows.

    Family sizes are capped at 16, so dynamic programming over sizes
    is small and deterministic.
    """

    if required_count <= 0:
        return set()

    if required_count > len(candidates):
        raise RuntimeError(
            f"Requested {required_count} families from "
            f"{len(candidates)} candidates."
        )

    buckets = defaultdict(list)

    for family, members in candidates.items():
        buckets[len(members)].append(family)

    for size in buckets:
        buckets[size].sort(
            key=lambda family: digest(salt, family)
        )

    # (number_of_families, total_rows) -> selection by family size.
    states = {
        (0, 0): {}
    }

    for size in sorted(buckets):
        available = len(buckets[size])
        new_states = dict(states)

        for (used_count, used_rows), selections in states.items():
            maximum = min(
                available,
                required_count - used_count,
            )

            for amount in range(1, maximum + 1):
                next_count = used_count + amount
                next_rows = used_rows + (size * amount)

                key = (next_count, next_rows)

                if key in new_states:
                    continue

                next_selection = dict(selections)
                next_selection[size] = amount
                new_states[key] = next_selection

        states = new_states

    possible = [
        (row_count, selections)
        for (family_count, row_count), selections in states.items()
        if family_count == required_count
    ]

    if not possible:
        raise RuntimeError(
            "Could not construct an exact family-count allocation."
        )

    selected_rows, selected_sizes = min(
        possible,
        key=lambda item: (
            abs(item[0] - target_rows),
            item[0] > target_rows,
            item[0],
        ),
    )

    selected = set()

    for size, amount in selected_sizes.items():
        selected.update(buckets[size][:amount])

    if len(selected) != required_count:
        raise RuntimeError(
            f"Expected {required_count} selected families, "
            f"found {len(selected)}."
        )

    return selected


rows = list(
    csv.DictReader(
        SOURCE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        )
    )
)

eligible = [
    row
    for row in rows
    if row["status"] not in {
        "hard_reject",
        "manual_review",
    }
    and row["target_capability"] != "manual_review"
]

quality_exclusions = Counter()
approved = []

for row in eligible:
    reason = quality_exclusion(row)

    if reason:
        quality_exclusions[reason] += 1
        continue

    approved.append(row)

families_by_capability = defaultdict(lambda: defaultdict(list))

for row in approved:
    capability = row["target_capability"]
    family = get_family(row)
    families_by_capability[capability][family].append(row)

# Deterministically cap large repeated families.
capped = defaultdict(dict)
cap_removals = Counter()

for capability, families in families_by_capability.items():
    for family, members in families.items():
        ordered = sorted(
            members,
            key=lambda row: digest(
                capability,
                family,
                row.get("prompt", ""),
                row.get("response", ""),
                row.get("record_id", ""),
            ),
        )

        capped[capability][family] = ordered[:FAMILY_CAP]

        removed = len(ordered) - len(capped[capability][family])

        if removed:
            cap_removals[capability] += removed

assignments = {}
allocation_details = {}

for capability in sorted(capped):
    families = capped[capability]
    family_count = len(families)
    total_rows = sum(len(members) for members in families.values())

    if family_count < 7:
        raise RuntimeError(
            f"{capability} has only {family_count} families; "
            "not enough for robust train/val/test allocation."
        )

    eval_family_count = max(
        MIN_EVAL_FAMILIES,
        round(family_count * EVAL_FAMILY_FRACTION),
    )

    while family_count - (2 * eval_family_count) < 1:
        eval_family_count -= 1

    if eval_family_count < 1:
        raise RuntimeError(
            f"Could not allocate independent splits for {capability}."
        )

    target_eval_rows = max(
        eval_family_count,
        round(total_rows * EVAL_FAMILY_FRACTION),
    )

    test_families = choose_family_subset(
        candidates=families,
        required_count=eval_family_count,
        target_rows=target_eval_rows,
        salt=f"{capability}-test",
    )

    remaining_after_test = {
        family: members
        for family, members in families.items()
        if family not in test_families
    }

    val_families = choose_family_subset(
        candidates=remaining_after_test,
        required_count=eval_family_count,
        target_rows=target_eval_rows,
        salt=f"{capability}-val",
    )

    train_families = (
        set(families)
        - test_families
        - val_families
    )

    if not train_families:
        raise RuntimeError(
            f"No training families remain for {capability}."
        )

    for family in train_families:
        assignments[(capability, family)] = "train"

    for family in val_families:
        assignments[(capability, family)] = "val"

    for family in test_families:
        assignments[(capability, family)] = "test"

    allocation_details[capability] = {
        "total_rows_after_cap": total_rows,
        "total_families": family_count,
        "target_eval_rows": target_eval_rows,
        "train_families": len(train_families),
        "val_families": len(val_families),
        "test_families": len(test_families),
    }

output_rows = {
    split: []
    for split in SPLITS
}

for capability in sorted(capped):
    for family, members in capped[capability].items():
        split = assignments[(capability, family)]

        for source_row in members:
            prompt = str(source_row["prompt"]).strip()
            response = str(source_row["response"]).strip()

            output_rows[split].append({
                "kind": "nueronce_foundational_v3_1",
                "split": split,
                "domain": capability,
                "category": source_row.get("category", ""),
                "prompt": prompt,
                "response": response,
                "source": source_row.get("source", ""),
                "source_split": source_row.get("source_split", ""),
                "original_split": source_row.get("split", ""),
                "original_domain": source_row.get(
                    "current_domain",
                    "",
                ),
                "record_id": source_row.get("record_id", ""),
                "template_id": source_row.get("template_id", ""),
                "template_family": family,
                "example_hash": digest(
                    capability,
                    prompt,
                    response,
                    source_row.get("record_id", ""),
                ),
                "provenance_warnings": [
                    warning
                    for warning in str(
                        source_row.get(
                            "provenance_warnings",
                            "",
                        )
                    ).split("|")
                    if warning
                ],
            })

for split in SPLITS:
    output_rows[split].sort(
        key=lambda row: (
            row["domain"],
            digest(
                "v3-1-row-order",
                row["example_hash"],
            ),
        )
    )

family_sets = {
    split: {
        (row["domain"], row["template_family"])
        for row in output_rows[split]
    }
    for split in SPLITS
}

prompt_sets = {
    split: {
        normalize(row["prompt"])
        for row in output_rows[split]
    }
    for split in SPLITS
}

content_sets = {
    split: {
        digest(
            normalize(row["prompt"]),
            normalize(row["response"]),
        )
        for row in output_rows[split]
    }
    for split in SPLITS
}

overlaps = {}

for left, right in (
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
):
    values = {
        "family_overlap": len(
            family_sets[left] & family_sets[right]
        ),
        "prompt_overlap": len(
            prompt_sets[left] & prompt_sets[right]
        ),
        "content_overlap": len(
            content_sets[left] & content_sets[right]
        ),
    }

    overlaps[f"{left}_vs_{right}"] = values

    if any(values.values()):
        raise RuntimeError(
            f"Leakage between {left} and {right}: {values}"
        )

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    path = OUT_DIR / f"{split}.jsonl"

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for row in output_rows[split]:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

row_counts = defaultdict(Counter)
family_counts = defaultdict(Counter)
largest_family_share = defaultdict(dict)

for split in SPLITS:
    grouped = defaultdict(lambda: defaultdict(int))

    for row in output_rows[split]:
        capability = row["domain"]
        family = row["template_family"]

        row_counts[capability][split] += 1
        grouped[capability][family] += 1

    for capability, family_sizes in grouped.items():
        family_counts[capability][split] = len(family_sizes)

        total = sum(family_sizes.values())
        largest = max(family_sizes.values())

        largest_family_share[capability][split] = (
            largest / total
            if total
            else 0.0
        )

warnings = []

for capability in sorted(row_counts):
    for split in ("val", "test"):
        rows_in_split = row_counts[capability][split]
        share = largest_family_share[capability][split]

        if rows_in_split < 10:
            warnings.append(
                f"{capability}/{split} has only "
                f"{rows_in_split} rows"
            )

        if share > 0.60:
            warnings.append(
                f"{capability}/{split} largest family share "
                f"is {share:.1%}"
            )

manifest = {
    "dataset": "foundational_recovery_v3_1",
    "family_cap": FAMILY_CAP,
    "eval_family_fraction": EVAL_FAMILY_FRACTION,
    "minimum_eval_families": MIN_EVAL_FAMILIES,
    "source_rows": len(rows),
    "eligible_rows": len(eligible),
    "approved_rows_before_cap": len(approved),
    "final_rows": sum(
        len(output_rows[split])
        for split in SPLITS
    ),
    "split_counts": {
        split: len(output_rows[split])
        for split in SPLITS
    },
    "quality_exclusions": dict(quality_exclusions),
    "family_cap_removals": dict(cap_removals),
    "overlaps": overlaps,
    "allocation": allocation_details,
    "warnings": warnings,
    "unsupported_capabilities": [
        "code_debugging",
    ],
}

MANIFEST.write_text(
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
    "# Foundational Recovery V3.1 Build Report",
    "",
    "## Summary",
    "",
    f"- Family cap: **{FAMILY_CAP}**",
    f"- Validation family target: "
    f"**{EVAL_FAMILY_FRACTION:.0%} per capability**",
    f"- Final rows: **{manifest['final_rows']}**",
    "- Unsupported capability: **code_debugging**",
    "",
    "## Split totals",
    "",
    "| Split | Rows |",
    "|---|---:|",
]

for split in SPLITS:
    lines.append(
        f"| {split} | {len(output_rows[split])} |"
    )

lines.extend([
    "",
    "## Capability distribution",
    "",
    "| Capability | Train rows | Val rows | Test rows | "
    "Train families | Val families | Test families | "
    "Largest val family | Largest test family |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
])

for capability in sorted(row_counts):
    lines.append(
        f"| {capability} "
        f"| {row_counts[capability]['train']} "
        f"| {row_counts[capability]['val']} "
        f"| {row_counts[capability]['test']} "
        f"| {family_counts[capability]['train']} "
        f"| {family_counts[capability]['val']} "
        f"| {family_counts[capability]['test']} "
        f"| {largest_family_share[capability]['val']:.1%} "
        f"| {largest_family_share[capability]['test']:.1%} |"
    )

lines.extend([
    "",
    "## Leakage checks",
    "",
    "| Comparison | Family | Prompt | Content |",
    "|---|---:|---:|---:|",
])

for comparison, values in overlaps.items():
    lines.append(
        f"| {comparison.replace('_', ' ')} "
        f"| {values['family_overlap']} "
        f"| {values['prompt_overlap']} "
        f"| {values['content_overlap']} |"
    )

lines.extend([
    "",
    "## Review warnings",
    "",
])

if warnings:
    for warning in warnings:
        lines.append(f"- {warning}")
else:
    lines.append("- None")

lines.extend([
    "",
    "## Training restriction",
    "",
    "Do not train until this report is reviewed. "
    "Do not claim code-debugging capability.",
])

REPORT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(f"Wrote dataset: {OUT_DIR}")
print(f"Wrote report: {REPORT}")
print(f"Wrote manifest: {MANIFEST}")
print()

print("Split totals:")

for split in SPLITS:
    print(f"  {split}: {len(output_rows[split])}")

print()
print("Capability distribution:")

for capability in sorted(row_counts):
    print(
        f"  {capability}: "
        f"train={row_counts[capability]['train']}, "
        f"val={row_counts[capability]['val']}, "
        f"test={row_counts[capability]['test']}; "
        f"families="
        f"{family_counts[capability]['train']}/"
        f"{family_counts[capability]['val']}/"
        f"{family_counts[capability]['test']}; "
        f"largest_eval_family="
        f"{largest_family_share[capability]['val']:.1%}/"
        f"{largest_family_share[capability]['test']:.1%}"
    )

print()
print("Leakage checks:")

for comparison, values in overlaps.items():
    print(f"  {comparison}: {values}")

print()
print("Warnings:")

if warnings:
    for warning in warnings:
        print(f"  {warning}")
else:
    print("  none")
