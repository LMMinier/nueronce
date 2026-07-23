import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = Path("data/foundational_sanitized/train.jsonl")
VAL = Path("data/foundational_sanitized/val.jsonl")
OUTPUT = Path("metrics/template_binding_audit.json")


def load(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text):
    text = text.lower()
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "<TIME>", text)
    text = re.sub(r"\b\d+\b", "<NUM>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


train = load(TRAIN)
val = load(VAL)

exact_prompt_answers = defaultdict(set)
template_groups = defaultdict(list)

for row in train:
    prompt = row["prompt"]
    response = row["response"]
    category = row.get("category", "unknown")

    exact_prompt_answers[prompt].add(response)
    key = (category, normalize(prompt))
    template_groups[key].append(row)

exact_conflicts = [
    {
        "prompt": prompt,
        "responses": sorted(responses),
    }
    for prompt, responses in exact_prompt_answers.items()
    if len(responses) > 1
]

train_templates = {
    (row.get("category", "unknown"), normalize(row["prompt"]))
    for row in train
}
val_templates = {
    (row.get("category", "unknown"), normalize(row["prompt"]))
    for row in val
}

overlap = train_templates & val_templates

ambiguous = []
for (category, template), rows in template_groups.items():
    response_shapes = sorted({normalize(row["response"]) for row in rows})

    if len(response_shapes) > 1:
        ambiguous.append({
            "category": category,
            "template": template,
            "rows": len(rows),
            "response_shape_count": len(response_shapes),
            "response_shapes": response_shapes[:10],
            "examples": [
                {
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
                for row in rows[:4]
            ],
        })

ambiguous.sort(
    key=lambda item: (
        item["rows"],
        item["response_shape_count"],
    ),
    reverse=True,
)

report = {
    "train_rows": len(train),
    "val_rows": len(val),
    "train_categories": dict(
        Counter(row.get("category", "unknown") for row in train)
    ),
    "nested_conversation_prompts": sum(
        row["prompt"].startswith("Conversation:\nUser:")
        for row in train
    ),
    "exact_prompt_conflicts": len(exact_conflicts),
    "normalized_train_templates": len(train_templates),
    "normalized_val_templates": len(val_templates),
    "normalized_template_overlap": len(overlap),
    "normalized_val_overlap_fraction": (
        len(overlap) / len(val_templates) if val_templates else 0.0
    ),
    "ambiguous_template_groups": len(ambiguous),
    "top_ambiguous_groups": ambiguous[:12],
    "exact_conflict_examples": exact_conflicts[:10],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("TRAIN ROWS:", report["train_rows"])
print("VAL ROWS:", report["val_rows"])
print("NESTED CONVERSATION PROMPTS:", report["nested_conversation_prompts"])
print("EXACT PROMPT CONFLICTS:", report["exact_prompt_conflicts"])
print("TRAIN TEMPLATE COUNT:", report["normalized_train_templates"])
print("VAL TEMPLATE COUNT:", report["normalized_val_templates"])
print("TRAIN/VAL TEMPLATE OVERLAP:", report["normalized_template_overlap"])
print(
    "VAL TEMPLATE OVERLAP FRACTION:",
    round(report["normalized_val_overlap_fraction"], 4),
)
print("AMBIGUOUS TEMPLATE GROUPS:", report["ambiguous_template_groups"])

for index, group in enumerate(report["top_ambiguous_groups"], 1):
    print("\n" + "=" * 90)
    print("GROUP:", index)
    print("CATEGORY:", group["category"])
    print("ROWS:", group["rows"])
    print("RESPONSE SHAPES:", group["response_shape_count"])
    print("TEMPLATE:", group["template"][:800])

    for example in group["examples"]:
        print("\nPROMPT:", example["prompt"][:500])
        print("RESPONSE:", example["response"][:300])

print("\nFULL REPORT:", OUTPUT)
