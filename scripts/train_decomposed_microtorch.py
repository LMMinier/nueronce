#!/usr/bin/env python3
"""Smoke/train the decomposed MicroTorch runtime on a small CFNA preset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from cfna.microtorch.runtime import (
    BlockStateManager,
    BlockStreamFactor,
    DecomposedTrainer,
    ExecutionPlan,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--state-dir", default="checkpoints/decomposed_state")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cfg = MicroModelConfig(
        byte_embed_dim=8,
        d_local=12,
        d_model=16,
        p_max=8,
        physical_blocks=1,
        logical_depth=1,
        n_heads=2,
        unit_window=8,
        decoder_window=8,
        decoder_layers=1,
        d_state=4,
        channel_dim=4,
        ret_byte_dim=4,
        min_patch=2,
        max_patch=8,
    )
    model = MicroCFNAModel(cfg)
    plan = ExecutionPlan.from_cfna(model)
    trainer = DecomposedTrainer(
        model,
        plan,
        BlockStateManager(args.state_dir),
        BlockStreamFactor(lr=2e-3, weight_decay=1e-3, tile_rows=16),
    )

    batch = np.array([[ord(c) for c in "logic: 2+3=5\n"]])
    history = []
    for _ in range(args.steps):
        rec = trainer.train_step(lambda: model.lm_loss(batch))
        history.append(rec)
        print(json.dumps(rec))

    assert np.isfinite([x["loss"] for x in history]).all()
    print(
        json.dumps(
            {
                "status": "ok",
                "params": model.num_params(),
                "loss_start": history[0]["loss"],
                "loss_end": history[-1]["loss"],
                "blocks": [b.name for b in plan.blocks],
            }
        )
    )


if __name__ == "__main__":
    main()
