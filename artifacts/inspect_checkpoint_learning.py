import json
from pathlib import Path

import torch

from nueronce.chat import load_checkpoint
from nueronce.incremental import IncrementalGenerator
from nueronce.prompting import (
    STOP_SEQUENCES,
    extract_assistant_continuation,
    format_inference_prompt,
)
from nueronce.training.dialogue_data import encode_example, make_sft_batch

torch.set_num_threads(8)

CHECKPOINT = "runs/foundational_executor/latest_best.pt"
MAX_LEN = 768

model, metadata = load_checkpoint(CHECKPOINT)
model.eval()

system = metadata.get("sft_system") or Path(
    "runs/forgeloop/system_prompt.txt"
).read_text(encoding="utf-8").strip()


def load_rows(path, count=2):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= count:
                break
    return rows


def inspect_row(split, index, row):
    prompt = row["prompt"]
    expected = row["response"]

    original_bytes, original_mask = encode_example(
        prompt,
        expected,
        system=system,
    )
    original_first_target = next(
        i for i, flag in enumerate(original_mask) if flag
    )

    batch_np = make_sft_batch(
        [(prompt, expected)],
        system=system,
        max_len=MAX_LEN,
    )
    ids = torch.from_numpy(batch_np["byte_ids"]).long()
    mask = torch.from_numpy(batch_np["target_mask"]).bool()

    shifted_mask = mask[:, 1:]

    with torch.inference_mode():
        logits, _ = model(ids)
        loss = model.masked_token_loss(logits, ids, mask)

        predictions = logits[:, :-1].argmax(dim=-1)
        targets = ids[:, 1:]

        correct = predictions[shifted_mask] == targets[shifted_mask]
        accuracy = float(correct.float().mean()) if correct.numel() else 0.0

    rendered = format_inference_prompt(
        system_message=system,
        user_request=prompt,
        trusted_evidence="",
        response_plan="",
    )

    raw = IncrementalGenerator(model).generate(
        rendered.encode("utf-8"),
        max_new=min(200, max(80, len(expected.encode("utf-8")) + 40)),
        temperature=0.0,
        greedy=True,
        max_ctx=MAX_LEN,
        stop_sequences=STOP_SEQUENCES,
        continuation_only=True,
    )
    generated = extract_assistant_continuation(raw).strip()

    retained_first_target = int(
        torch.nonzero(mask[0], as_tuple=False)[0].item()
    )

    print("\n" + "=" * 90)
    print("SPLIT:", split, "ROW:", index)
    print("CATEGORY:", row.get("category"))
    print("DOMAIN:", row.get("domain"))
    print("ORIGINAL_BYTES:", len(original_bytes))
    print("ORIGINAL_PREFIX_BYTES:", original_first_target)
    print("BATCH_BYTES:", ids.shape[1])
    print("RETAINED_PREFIX_BYTES:", retained_first_target)
    print("TEACHER_FORCED_LOSS:", float(loss))
    print("TEACHER_FORCED_BYTE_ACCURACY:", accuracy)
    print("\nPROMPT:")
    print(prompt[:1000])
    print("\nEXPECTED:")
    print(expected[:1000])
    print("\nGENERATED:")
    print(generated[:1000])


for split, path in [
    ("train", "data/foundational_sanitized/train.jsonl"),
    ("val", "data/foundational_sanitized/val.jsonl"),
]:
    for index, row in enumerate(load_rows(path)):
        inspect_row(split, index, row)
