#!/usr/bin/env python3
"""Run the 355M-class NUERONCE through the Nueronce Engine StreamFactor path.

This launcher is intentionally separate from the legacy SFT CLI while the
segmented-autograd work is developed. It provides a real preset, float32 tensor
creation, factorized optimizer state, and resumable generic checkpoints.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--save-dir", default="checkpoints/micro_nueronce_355m")
    ap.add_argument("--metrics-dir", default="metrics/micro_nueronce_355m")
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--examples-per-shard", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--tile-rows", type=int, default=128)
    ap.add_argument("--no-momentum", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    from nueronce.engine.optim import StreamFactor
    from nueronce.engine.scaling import base_355m_config, enable_training_dtype
    import nueronce.training.sharded_sft as trainer

    enable_training_dtype("float32")

    def optimizer_factory(params, lr=1e-3, weight_decay=0.0, **_):
        return StreamFactor(params, lr=lr, weight_decay=weight_decay,
                            tile_rows=args.tile_rows,
                            momentum=not args.no_momentum)

    def save_checkpoint(path, model, opt, meta):
        from nueronce.training.dialogue_data import PROMPT_FORMAT
        payload = {
            "config": vars(model.cfg),
            "params": [p.data.copy() for p in model.parameters()],
            "optimizer": opt.state_dict(),
            "optimizer_name": "streamfactor",
            "opt_lr": opt.lr,
            "meta": {**meta, "prompt_format": meta.get("prompt_format", PROMPT_FORMAT),
                     "dtype": "float32", "preset": "base_355m"},
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(target)

    def apply_checkpoint(payload, model, opt):
        for p, arr in zip(model.parameters(), payload["params"]):
            p.data = arr.copy()
        opt.load_state_dict(payload["optimizer"])

    trainer.AdamW = optimizer_factory
    trainer.save_checkpoint = save_checkpoint
    trainer.apply_checkpoint = apply_checkpoint

    cfg = trainer.ShardedSFTConfig(
        train_dir=args.train_dir, val_path=args.validation, test_path=args.test,
        save_dir=args.save_dir, metrics_dir=args.metrics_dir,
        num_shards=args.num_shards,
        examples_per_shard=args.examples_per_shard,
        batch_size=args.batch, max_len=args.max_len, lr=args.lr,
        grad_accum_steps=args.grad_accum_steps, epochs=args.epochs,
        max_steps=args.max_steps, seed=args.seed, resume=not args.no_resume,
        periodic_val_every=100, periodic_val_examples=64,
        checkpoint_every_steps=100, log_every=10,
    )
    model_cfg = base_355m_config()
    print("Nueronce Engine base_355m + StreamFactor + float32")
    print("Expected parameter count: 352,993,825")
    trainer.run_sharded_sft(model_cfg, cfg)


if __name__ == "__main__":
    main()
