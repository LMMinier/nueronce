#!/usr/bin/env python3
"""Matched dense/sparse/DCT/RFT NUERONCE training experiment.

Each model variant runs in a fresh subprocess to isolate peak RSS and CUDA
allocator state. The parent process combines worker JSON files into JSON, CSV,
and Markdown tables.

Example quick smoke run:

    python scripts/benchmark_rft_matched.py --steps 20 --seq 48 --batch 2

More meaningful CPU run:

    python scripts/benchmark_rft_matched.py \
      --steps 400 --seq 96 --batch 8 --eval-batches 16 --device cpu

The automatic conversation metrics are mechanical diagnostics, not a human
judgment of coherence. Generated samples are always retained for inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from nueronce.data import corpus_bytes, make_batches
from nueronce.model import ModelConfig, NUERONCEModel
from nueronce.spectral_baselines import replace_core_ffn

VARIANTS = ("dense", "ordinary_sparse", "dct_sparse", "rft_sparse")
PROMPTS = (
    b"User: Hello.\nAssistant:",
    b"User: What is two plus two?\nAssistant:",
    b"User: Explain memory in one sentence.\nAssistant:",
    b"User: I am unsure. Please answer honestly.\nAssistant:",
)


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _model_bytes(model: torch.nn.Module) -> int:
    return sum(_tensor_bytes(p) for p in model.parameters()) + sum(
        _tensor_bytes(b) for b in model.buffers()
    )


def _trainable_bytes(model: torch.nn.Module) -> int:
    return sum(_tensor_bytes(p) for p in model.parameters() if p.requires_grad)


def _optimizer_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += _tensor_bytes(value)
    return total


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in data)
    return printable / len(data)


def _ngram_repetition(data: bytes, n: int = 4) -> float:
    if len(data) < n:
        return 0.0
    grams = [data[index:index+n] for index in range(len(data) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _conversation_diagnostics(prompt: bytes, output: bytes) -> Dict[str, float]:
    lower_prompt = set(prompt.lower().split())
    lower_output = set(output.lower().split())
    overlap = len(lower_prompt & lower_output) / max(1, len(lower_prompt))
    return {
        "output_bytes": float(len(output)),
        "printable_ratio": _printable_ratio(output),
        "unique_byte_ratio": len(set(output)) / max(1, len(output)),
        "fourgram_repetition": _ngram_repetition(output, 4),
        "prompt_word_overlap": overlap,
    }


def _aggregate_probe_score(probes: List[Dict[str, object]]) -> float:
    if not probes:
        return 0.0
    scores = []
    for probe in probes:
        metric = probe["metrics"]
        score = (
            0.35 * metric["printable_ratio"]
            + 0.25 * metric["unique_byte_ratio"]
            + 0.20 * (1.0 - metric["fourgram_repetition"])
            + 0.20 * min(1.0, metric["output_bytes"] / 32.0)
        )
        scores.append(score)
    return float(sum(scores) / len(scores))


def _config(args) -> ModelConfig:
    if args.tiny:
        return ModelConfig(
            byte_embed_dim=16,
            d_local=24,
            d_model=32,
            p_max=8,
            physical_blocks=1,
            logical_depth=1,
            n_heads=4,
            unit_window=8,
            decoder_window=8,
            decoder_layers=1,
            d_state=4,
            channel_dim=4,
            ret_byte_dim=8,
        )
    return ModelConfig()


def run_worker(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data = corpus_bytes(repeat=args.corpus_repeat)
    train_batches = make_batches(
        data, args.seq, args.batch, args.steps, seed=args.seed
    )
    eval_batches = make_batches(
        data, args.seq, args.batch, args.eval_batches, seed=args.seed + 999
    )

    cfg = _config(args)
    model = NUERONCEModel(cfg)
    replace_core_ffn(
        model,
        args.variant,
        fan_in=args.fan_in,
        ffn_mult=args.ffn_mult,
        rft_block_size=args.rft_block_size,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    initial_rss = _rss_mb()
    history = []
    token_count = 0
    started = time.perf_counter()

    model.train()
    for step, cpu_batch in enumerate(train_batches, start=1):
        batch = cpu_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.loss(batch)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        token_count += int(batch.numel())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({
                "step": step,
                "train_bpb": float(parts["bpb"]),
                "loss": float(parts["loss"]),
                "grad_norm": grad_norm,
            })

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started

    model.eval()
    eval_bpbs = []
    with torch.no_grad():
        for cpu_batch in eval_batches:
            _, parts = model.loss(cpu_batch.to(device))
            eval_bpbs.append(float(parts["bpb"]))
    validation_bpb = sum(eval_bpbs) / max(1, len(eval_bpbs))

    probes = []
    for prompt in PROMPTS:
        generated = model.generate(
            prompt,
            max_new=args.probe_bytes,
            greedy=True,
            continuation_only=True,
        )
        output = bytes(generated)
        probes.append({
            "prompt": prompt.decode("utf-8", errors="replace"),
            "output": output.decode("utf-8", errors="replace"),
            "output_hex": output.hex(),
            "metrics": _conversation_diagnostics(prompt, output),
        })

    peak_cuda_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    optimizer_bytes = _optimizer_bytes(optimizer)
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "seq": args.seq,
        "batch": args.batch,
        "tokens_seen": token_count,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_storage_mb": _model_bytes(model) / 1024**2,
        "trainable_parameter_mb": _trainable_bytes(model) / 1024**2,
        "optimizer_state_mb": optimizer_bytes / 1024**2,
        "peak_rss_mb": _rss_mb(),
        "rss_growth_mb": max(0.0, _rss_mb() - initial_rss),
        "peak_cuda_allocated_mb": peak_cuda_mb,
        "train_seconds": train_seconds,
        "steps_per_second": args.steps / max(train_seconds, 1e-12),
        "tokens_per_second": token_count / max(train_seconds, 1e-12),
        "first_train_bpb": history[0]["train_bpb"],
        "final_train_bpb": history[-1]["train_bpb"],
        "validation_bpb": validation_bpb,
        "conversation_proxy_score": _aggregate_probe_score(probes),
        "conversation_probes": probes,
        "history": history,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
        },
    }
    Path(args.worker_out).write_text(json.dumps(result, indent=2))
    return result


def _format_number(value, digits=3):
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def write_tables(results: List[Dict[str, object]], output_dir: Path) -> None:
    columns = [
        "variant",
        "trainable_parameters",
        "model_storage_mb",
        "optimizer_state_mb",
        "peak_rss_mb",
        "peak_cuda_allocated_mb",
        "train_seconds",
        "tokens_per_second",
        "validation_bpb",
        "conversation_proxy_score",
    ]
    with (output_dir / "matched_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result[column] for column in columns})

    header = (
        "| Variant | Trainable params | Model MB | Optimizer MB | Peak RSS MB | "
        "Peak CUDA MB | Train sec | Tokens/s | Val BPB | Conversation proxy |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for result in results:
        rows.append(
            "| {variant} | {params} | {model} | {optim} | {rss} | {cuda} | "
            "{seconds} | {tokens} | {bpb} | {proxy} |".format(
                variant=result["variant"],
                params=_format_number(result["trainable_parameters"], 0),
                model=_format_number(result["model_storage_mb"]),
                optim=_format_number(result["optimizer_state_mb"]),
                rss=_format_number(result["peak_rss_mb"]),
                cuda=_format_number(result["peak_cuda_allocated_mb"]),
                seconds=_format_number(result["train_seconds"]),
                tokens=_format_number(result["tokens_per_second"], 1),
                bpb=_format_number(result["validation_bpb"], 4),
                proxy=_format_number(result["conversation_proxy_score"], 4),
            )
        )
    notes = (
        "\n\n**Interpretation:** lower validation BPB and memory are better; higher "
        "tokens/s is better. The conversation proxy only measures printable output, "
        "diversity, repetition, and non-empty length. Read the saved samples before "
        "making any coherence claim.\n"
    )
    (output_dir / "matched_results.md").write_text(header + "\n".join(rows) + notes)


def run_parent(args) -> List[Dict[str, object]]:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in args.variants:
        worker_out = output_dir / f"{variant}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--variant", variant,
            "--worker-out", str(worker_out),
            "--steps", str(args.steps),
            "--seq", str(args.seq),
            "--batch", str(args.batch),
            "--eval-batches", str(args.eval_batches),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--seed", str(args.seed),
            "--fan-in", str(args.fan_in),
            "--ffn-mult", str(args.ffn_mult),
            "--rft-block-size", str(args.rft_block_size),
            "--corpus-repeat", str(args.corpus_repeat),
            "--device", args.device,
            "--probe-bytes", str(args.probe_bytes),
            "--log-every", str(args.log_every),
        ]
        if args.tiny:
            command.append("--tiny")
        print(f"\n=== {variant} ===", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            results.append({
                "variant": variant,
                "status": "failed",
                "returncode": completed.returncode,
            })
            continue
        results.append(json.loads(worker_out.read_text()))

    successful = [result for result in results if result.get("status") != "failed"]
    (output_dir / "matched_results.json").write_text(json.dumps(results, indent=2))
    if successful:
        write_tables(successful, output_dir)
        print("\n" + (output_dir / "matched_results.md").read_text())
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="dense")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--worker-out", default="worker_result.json")
    parser.add_argument("--out-dir", default="metrics/rft_matched")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fan-in", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=3)
    parser.add_argument("--rft-block-size", type=int, default=64)
    parser.add_argument("--corpus-repeat", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-bytes", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--tiny", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        run_parent(parsed)
