from pathlib import Path
import torch

paths = [
    Path("runs/forgeloop/latest.pt"),
    Path("runs/forgeloop/latest_best.pt"),
    Path("checkpoints/preflight/latest.pt"),
    Path("checkpoints/preflight/latest_best.pt"),
    Path("cfna_forgeloop_sft_best.pt"),
    Path("runs/foundational_executor/accept_b1_best.pt"),
    Path("runs/foundational_executor/accept_b2_best.pt"),
    Path("runs/foundational_executor/accept_b4_best.pt"),
    Path("runs/foundational_executor/accept_b8_best.pt"),
]

for path in paths:
    print("\n" + "=" * 90)
    print("PATH:", path)

    if not path.exists():
        print("MISSING")
        continue

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        config = checkpoint.get("config") or {}
        print("SIZE:", path.stat().st_size)
        print("KEYS:", sorted(checkpoint.keys()))
        print("STEP:", checkpoint.get("step"))
        print("SFT_STEP:", checkpoint.get("sft_step"))
        print("BEST_VAL_LOSS:", checkpoint.get("best_val_loss"))
        print("BAD_EVALS:", checkpoint.get("bad_evals"))
        print("EXECUTION_DEPTH:", config.get("execution_depth"))
        print("SOURCE_CHECKPOINT:", checkpoint.get("source_checkpoint"))
        print("SOURCE_SHA256:", checkpoint.get("source_checkpoint_sha256"))
        print("SFT_TRAIN_PATH:", checkpoint.get("sft_train_path"))
        print("SFT_SYSTEM:", repr(checkpoint.get("sft_system"))[:300])

    except Exception as error:
        print("LOAD ERROR:", type(error).__name__, error)
