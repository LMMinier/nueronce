#!/usr/bin/env python3
"""Convert a trained PyTorch NUERONCE checkpoint to Nueronce Engine format."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from nueronce.engine.interop import load_torch_state_dict
from nueronce.engine.nueronce_model import NueronceConfig, NueronceModel
from nueronce.engine.optim import StreamFactor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optimizer", choices=("streamfactor", "none"), default="streamfactor")
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    # Torch checkpoints may now carry activation_checkpointing themselves
    # (ModelConfig mirrors the engine config field-for-field); the engine
    # conversion always wants it on, so override rather than duplicate.
    source_config = {k: v for k, v in source["config"].items()
                     if k != "activation_checkpointing"}
    config = NueronceConfig(**source_config, activation_checkpointing=True)
    model = NueronceModel(config)
    report = load_torch_state_dict(model, source["state_dict"])
    optimizer = StreamFactor(model.parameters(), lr=args.lr, weight_decay=0.01,
                             momentum=False) if args.optimizer == "streamfactor" else None
    payload = {
        "format": "nueronce-engine-v1-from-torch",
        "config": vars(config),
        "params": [parameter.data.copy() for parameter in model.parameters()],
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "meta": {
            "step": int(source.get("sft_step", source.get("step", 0))),
            "source_step": int(source.get("step", 0)),
            "source_sft_step": int(source.get("sft_step", 0)),
            "source_sha256": sha256(args.source),
            "source_path": str(args.source),
            "source_best_val_loss": source.get("best_val_loss"),
            "prompt_format": "canonical",
            "conversion": report,
        },
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.destination)
    print(json.dumps({"destination": str(args.destination), "bytes": args.destination.stat().st_size,
                      "params": model.num_params(), **report}, indent=2))


if __name__ == "__main__":
    main()
