#!/usr/bin/env python3
"""Measure dense, low-rank, FFT, DCT, RFT, and adaptive expert costs.

This compact benchmark is intended for operator-ranking smoke tests before a
full 35M/355M layer-wise ablation. It records trainable parameters, fixed basis
buffers, AdamW state, throughput, validation BPB, and adaptive routing share.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = (
    "NUERONCE separates understanding thinking remembering and speaking. "
    "Typed memory keeps evidence authority and semantic channels apart. "
    "The hybrid core mixes recurrent state attention retrieval and spectral operators. "
    "A verifier checks claims before the answer is shown. "
    "Two plus two is four. Memory carries useful information across time. "
) * 180
DATA = torch.tensor(list(CORPUS.encode()), dtype=torch.long)


def batches(seed: int, steps: int, batch: int, seq: int):
    generator = torch.Generator().manual_seed(seed)
    high = len(DATA) - seq - 1
    for _ in range(steps):
        starts = torch.randint(0, high, (batch,), generator=generator)
        yield torch.stack([DATA[start:start + seq] for start in starts])


def dct_basis(dim: int) -> torch.Tensor:
    n = torch.arange(dim).float()[:, None]
    k = torch.arange(dim).float()[None, :]
    basis = torch.cos(math.pi / dim * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1.0 / dim)
    if dim > 1:
        basis[:, 1:] *= math.sqrt(2.0 / dim)
    return basis


def canonical_rft_basis(dim: int) -> torch.Tensor:
    real_dtype = torch.float64
    n = torch.arange(dim, dtype=real_dtype)[:, None]
    k = torch.arange(1, dim + 1, dtype=real_dtype)[None, :]
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    frequencies = torch.remainder(k * phi, 1.0)
    raw = torch.exp(2j * math.pi * n * frequencies) / math.sqrt(dim)
    gram = raw.conj().T @ raw
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(1e-12).rsqrt().to(torch.complex128))
        @ eigenvectors.conj().T
    )
    return (raw @ inverse_sqrt).to(torch.complex64)


class DenseExpert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class LowRankExpert(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.up(self.down(x))


class SpectralExpert(nn.Module):
    def __init__(self, dim: int, kind: str, keep: float = 0.25):
        super().__init__()
        self.dim = dim
        self.kind = kind
        if kind == "dct":
            self.register_buffer("basis", dct_basis(dim))
        elif kind == "rft":
            self.register_buffer("basis", canonical_rft_basis(dim))
        else:
            self.basis = None
        retained = max(1, int(dim * keep))
        self.index = nn.Parameter(torch.randperm(dim)[:retained], requires_grad=False)
        self.weight = nn.Parameter(torch.ones(retained))
        self.mix = nn.Parameter(torch.randn(retained, retained) / math.sqrt(retained))

    def forward(self, x):
        if self.kind == "fft":
            coefficients = torch.fft.fft(x.float(), dim=-1)
        elif self.kind == "dct":
            coefficients = x @ self.basis
        else:
            coefficients = x.to(torch.complex64) @ self.basis.conj()
        selected = coefficients[..., self.index]
        mixed = (selected @ self.mix.to(selected.dtype)) * self.weight.to(selected.dtype)
        full = torch.zeros_like(coefficients)
        full[..., self.index] = mixed
        if self.kind == "fft":
            output = torch.fft.ifft(full, dim=-1).real
        elif self.kind == "dct":
            output = full @ self.basis.T
        else:
            output = (full @ self.basis.T).real
        return output.to(x.dtype)


class ByteModel(nn.Module):
    def __init__(self, variant: str, dim: int = 48, hidden: int = 96):
        super().__init__()
        self.variant = variant
        self.embedding = nn.Embedding(256, dim)
        self.recurrent = nn.GRU(dim, dim, batch_first=True)
        if variant == "dense":
            self.expert = DenseExpert(dim, hidden)
        elif variant == "lowrank":
            self.expert = LowRankExpert(dim, 12)
        elif variant in {"fft", "dct", "rft"}:
            self.expert = SpectralExpert(dim, variant)
        elif variant == "adaptive":
            self.names = ["dense", "lowrank", "fft", "dct", "rft"]
            self.experts = nn.ModuleList([
                DenseExpert(dim, hidden),
                LowRankExpert(dim, 12),
                SpectralExpert(dim, "fft"),
                SpectralExpert(dim, "dct"),
                SpectralExpert(dim, "rft"),
            ])
            self.router = nn.Linear(dim, len(self.names))
        else:
            raise ValueError(variant)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 256)

    def forward(self, ids):
        x, _ = self.recurrent(self.embedding(ids))
        route = None
        if self.variant == "adaptive":
            routing_logits = self.router(x.mean(dim=1))
            alpha = F.gumbel_softmax(routing_logits, tau=1.0, hard=True, dim=-1)
            outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
            residual = (outputs * alpha[:, :, None, None]).sum(dim=1)
            route = alpha.mean(dim=0)
        else:
            residual = self.expert(x)
        return self.head(self.norm(x + residual)), route


def tensor_cost(model, optimizer):
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    optimizer_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                optimizer_bytes += value.numel() * value.element_size()
    return parameter_bytes, buffer_bytes, optimizer_bytes


def run_variant(variant, seed, steps, batch, seq):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ByteModel(variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    routes = torch.zeros(5)
    token_count = 0
    first_bpb = None
    started = time.perf_counter()
    for input_ids in batches(seed, steps, batch, seq):
        optimizer.zero_grad(set_to_none=True)
        logits, route = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256), input_ids[:, 1:].reshape(-1)
        )
        if first_bpb is None:
            first_bpb = float(loss.detach() / math.log(2.0))
        loss.backward()
        optimizer.step()
        token_count += input_ids.numel()
        if route is not None:
            routes += route.detach().cpu()
    elapsed = time.perf_counter() - started

    validation = []
    model.eval()
    with torch.no_grad():
        for input_ids in batches(seed + 999, 12, batch, seq):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256), input_ids[:, 1:].reshape(-1)
            )
            validation.append(float(loss / math.log(2.0)))

    parameter_bytes, buffer_bytes, optimizer_bytes = tensor_cost(model, optimizer)
    result = {
        "variant": variant,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "parameter_mb": parameter_bytes / 2**20,
        "buffer_mb": buffer_bytes / 2**20,
        "optimizer_mb": optimizer_bytes / 2**20,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "train_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "first_bpb": first_bpb,
        "final_train_bpb": float(loss.detach() / math.log(2.0)),
        "validation_bpb": sum(validation) / len(validation),
    }
    if variant == "adaptive":
        result["route_counts"] = dict(zip(model.names, (routes / steps).tolist()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="metrics/adaptive_spectral_cost")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in (3, 11):
        for variant in ("dense", "lowrank", "fft", "dct", "rft", "adaptive"):
            results.append(run_variant(variant, seed, args.steps, args.batch, args.seq))
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
