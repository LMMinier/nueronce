from pathlib import Path
import torch

from nueronce.chat import load_checkpoint
from nueronce.incremental import IncrementalGenerator
from nueronce.prompting import format_inference_prompt

torch.set_num_threads(8)

checkpoint = "runs/foundational_executor/latest_best.pt"
model, _ = load_checkpoint(checkpoint)

system = Path("runs/forgeloop/system_prompt.txt").read_text(
    encoding="utf-8"
).strip()

prompt = format_inference_prompt(
    system_message=system,
    user_request="Calculate 17 + 26. Give the numerical answer.",
    trusted_evidence="",
    response_plan="",
)

prompt_bytes = prompt.encode("utf-8")
max_ctx = 512
max_new = 72

# Fully dense greedy generation: recompute the complete context every byte.
all_ids = list(prompt_bytes)
dense_ids = []

with torch.inference_mode():
    for _ in range(max_new):
        context = all_ids[-max_ctx:]
        tensor = torch.tensor([context], dtype=torch.long)
        logits = model(tensor)[0][0, -1]
        next_id = int(torch.argmax(logits).item())
        dense_ids.append(next_id)
        all_ids.append(next_id)

dense_bytes = bytes(dense_ids)

# Current incremental generation path.
incremental_result = IncrementalGenerator(model).generate(
    prompt_bytes,
    max_new=max_new,
    temperature=0.0,
    greedy=True,
    max_ctx=max_ctx,
    stop_sequences=(),
    continuation_only=True,
)

if isinstance(incremental_result, str):
    incremental_bytes = incremental_result.encode("utf-8", errors="replace")
else:
    incremental_bytes = bytes(incremental_result)

limit = min(len(dense_bytes), len(incremental_bytes))
difference = next(
    (i for i in range(limit) if dense_bytes[i] != incremental_bytes[i]),
    None,
)

if difference is None and len(dense_bytes) != len(incremental_bytes):
    difference = limit

print("DENSE:")
print(repr(dense_bytes.decode("utf-8", errors="replace")))

print("\nINCREMENTAL:")
print(repr(incremental_bytes.decode("utf-8", errors="replace")))

print("\nEXACT_MATCH:", dense_bytes == incremental_bytes)
print("FIRST_DIFFERENCE_INDEX:", difference)
print("DENSE_LENGTH:", len(dense_bytes))
print("INCREMENTAL_LENGTH:", len(incremental_bytes))
