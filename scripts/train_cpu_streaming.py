#!/usr/bin/env python3
"""Train CFNA without CUDA using bounded byte windows and local objectives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cfna.model import CFNAModel, ModelConfig
from cfna.training.cpu_streaming import CPUStreamingConfig, CPUStreamingTrainer, stream_file_chunks


def collect_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.append(path)
    if not files:
        raise SystemExit("No readable corpus files found")
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Text/binary files or directories")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps-per-stage", type=int, default=32)
    parser.add_argument("--checkpoint", default="checkpoints/cfna_cpu_streaming.pt")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0,
                        help="PyTorch CPU threads; 0 leaves the runtime default")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA is available but intentionally unused; training device is CPU.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    files = collect_files(args.inputs)
    cfg = CPUStreamingConfig(chunk_size=args.chunk_size,
                             batch_size=args.batch_size,
                             steps_per_stage=args.steps_per_stage)
    model = CFNAModel(ModelConfig())
    trainer = CPUStreamingTrainer(model, cfg)
    print(json.dumps({"files": len(files), **trainer.memory_contract()}, indent=2))

    stream = stream_file_chunks(files, args.chunk_size, overlap=1)
    last = None
    for stats in trainer.train_stream(stream, max_steps=args.steps):
        last = stats
        if int(stats["step"]) % args.log_every == 0:
            print(json.dumps(stats, sort_keys=True))

    trainer.save(args.checkpoint)
    print(json.dumps({"saved": args.checkpoint, "last": last,
                      "memory_contract": trainer.memory_contract()}, indent=2))


if __name__ == "__main__":
    main()
