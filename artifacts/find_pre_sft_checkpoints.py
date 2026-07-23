from pathlib import Path
import torch

signals = ("base", "pretrain", "pretrained", "foundation", "large")

paths = sorted(
    set(Path(".").rglob("*.pt")) |
    set(Path(".").rglob("*.pth"))
)

found = 0

for path in paths:
    if any(part in {".git", "__pycache__", "artifacts"} for part in path.parts):
        continue

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        continue

    if not isinstance(checkpoint, dict):
        continue

    sft_step = checkpoint.get("sft_step")
    name_matches = any(word in path.name.lower() for word in signals)

    if sft_step not in (None, 0) and not name_matches:
        continue

    state = checkpoint.get("state_dict")
    params = checkpoint.get("params")

    parameter_count = None
    representation = None

    if isinstance(state, dict):
        representation = "state_dict"
        parameter_count = sum(
            int(value.numel())
            for value in state.values()
            if hasattr(value, "numel")
        )
    elif isinstance(params, list):
        representation = "params_list"
        parameter_count = sum(
            int(value.size if hasattr(value, "size") else value.numel())
            for value in params
        )

    config = checkpoint.get("config") or {}
    meta = checkpoint.get("meta") or {}

    print("\n" + "=" * 100)
    print("PATH:", path)
    print("SIZE:", path.stat().st_size)
    print("FORMAT:", representation)
    print("PARAMETERS:", parameter_count)
    print("STEP:", checkpoint.get("step", meta.get("step")))
    print("SFT_STEP:", sft_step)
    print("STAGE:", meta.get("stage", meta.get("phase")))
    print("PRESET:", config.get("preset", meta.get("preset")))
    print("EXECUTION_DEPTH:", config.get("execution_depth"))
    print("BEST_VAL_LOSS:", checkpoint.get("best_val_loss"))
    print("KEYS:", sorted(checkpoint.keys()))
    found += 1

print("\nCANDIDATES FOUND:", found)
