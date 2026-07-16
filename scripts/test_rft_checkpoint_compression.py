#!/usr/bin/env python3
"""Test whether trained NUERONCE-like FFN weights compress in RFT coordinates.

This small deterministic harness trains a byte-level recurrent language model,
projects each trained FFN matrix into direct, DCT, and canonical-RFT coordinates,
retains 50/25/10/5 percent of coefficients, reconstructs the dense matrix, and
measures held-out BPB before and after a short recovery fine-tune.

The experiment isolates coordinate-system compressibility from sparse runtime
kernel performance.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PHI = (1 + 5**0.5) / 2
CORPUS = (
    "User: Hello. Assistant: Hello! How can I help you today?\n"
    "User: What is two plus two? Assistant: Two plus two is four.\n"
    "User: Explain memory in one sentence. Assistant: Memory stores information so it can be used later.\n"
    "User: I am unsure. Please answer honestly. Assistant: I will be honest about what I know and what I do not know.\n"
    "NUERONCE separates perception, memory, reasoning, planning, verification, and communication.\n"
    "The resonant Fourier transform maps data into a unitary phi-derived coordinate basis.\n"
) * 100
DATA = torch.tensor(list(CORPUS.encode()), dtype=torch.long)


def batches(seed: int, steps: int, seq: int = 64, batch: int = 8):
    generator = torch.Generator().manual_seed(seed)
    upper = len(DATA) - seq - 1
    for _ in range(steps):
        starts = torch.randint(0, upper, (batch,), generator=generator)
        yield torch.stack([DATA[index:index + seq] for index in starts])


def rft_basis(size: int) -> torch.Tensor:
    n = torch.arange(size, dtype=torch.float64)[:, None]
    k = torch.arange(1, size + 1, dtype=torch.float64)[None, :]
    frequencies = torch.remainder(k * PHI, 1.0)
    raw = torch.exp(2j * math.pi * n * frequencies) / math.sqrt(size)
    gram = raw.conj().T @ raw
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    inverse_root = eigenvectors @ torch.diag(
        eigenvalues.clamp_min(1e-12).rsqrt().to(torch.complex128)
    ) @ eigenvectors.conj().T
    return raw @ inverse_root


def dct_basis(size: int) -> torch.Tensor:
    n = torch.arange(size, dtype=torch.float64)[:, None]
    k = torch.arange(size, dtype=torch.float64)[None, :]
    basis = torch.cos(math.pi / size * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1 / size)
    if size > 1:
        basis[:, 1:] *= math.sqrt(2 / size)
    return basis


class Dense(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))

    def forward(self, x):
        return x @ self.weight.T


class FFN(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 64):
        super().__init__()
        self.up = Dense(dim, hidden)
        self.gate = Dense(dim, hidden)
        self.down = Dense(hidden, dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ByteLM(nn.Module):
    def __init__(self, dim: int = 32, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(256, dim)
        self.recurrent = nn.GRU(dim, dim, batch_first=True)
        self.ffn = FFN(dim, hidden)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 256)

    def forward(self, ids):
        state, _ = self.recurrent(self.embedding(ids))
        state = state + self.ffn(self.norm(state))
        return self.head(state)


def evaluate_bpb(model: nn.Module, seed: int = 999, steps: int = 20) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for ids in batches(seed, steps):
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)
            )
            values.append(float(loss / math.log(2)))
    return sum(values) / len(values)


def train(model: nn.Module, steps: int, seed: int, lr: float = 3e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for ids in batches(seed, steps):
        optimizer.zero_grad()
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
    return model


def prune_top(coefficients: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = coefficients.abs().reshape(-1)
    count = max(1, int(round(flat.numel() * fraction)))
    threshold = torch.topk(flat, count, sorted=False).values.min()
    return torch.where(
        coefficients.abs() >= threshold,
        coefficients,
        torch.zeros_like(coefficients),
    )


def compress_weight(weight: torch.Tensor, basis: str, fraction: float):
    original = weight.detach().double()
    if basis == "direct":
        coefficients = original
        retained = prune_top(coefficients, fraction)
        reconstructed = retained
    elif basis == "dct":
        out_basis = dct_basis(weight.shape[0])
        in_basis = dct_basis(weight.shape[1])
        coefficients = out_basis.T @ original @ in_basis
        retained = prune_top(coefficients, fraction)
        reconstructed = out_basis @ retained @ in_basis.T
    elif basis == "rft":
        out_basis = rft_basis(weight.shape[0])
        in_basis = rft_basis(weight.shape[1])
        coefficients = (
            out_basis.conj().T
            @ original.to(torch.complex128)
            @ in_basis
        )
        retained = prune_top(coefficients, fraction)
        reconstructed = (out_basis @ retained @ in_basis.conj().T).real
    else:
        raise ValueError(basis)

    error = float(
        torch.linalg.norm(original - reconstructed)
        / torch.linalg.norm(original)
    )
    retained_energy = float(
        (retained.abs().square().sum() / coefficients.abs().square().sum()).real
    )
    return reconstructed.float(), error, retained_energy


def main():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = train(ByteLM(), steps=320, seed=5)
    baseline_bpb = evaluate_bpb(model)

    output = Path("metrics/rft_checkpoint_compression")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "dense_checkpoint.pt")

    rows = []
    for basis in ("direct", "dct", "rft"):
        for fraction in (0.50, 0.25, 0.10, 0.05):
            candidate = copy.deepcopy(model)
            errors = []
            energies = []
            for name in ("up", "gate", "down"):
                layer = getattr(candidate.ffn, name)
                reconstructed, error, energy = compress_weight(
                    layer.weight, basis, fraction
                )
                layer.weight.data.copy_(reconstructed)
                errors.append(error)
                energies.append(energy)

            before = evaluate_bpb(candidate)
            recovered = copy.deepcopy(candidate)
            train(recovered, steps=40, seed=77, lr=5e-4)
            after = evaluate_bpb(recovered)
            rows.append({
                "basis": basis,
                "retained_fraction": fraction,
                "baseline_bpb": baseline_bpb,
                "mean_weight_error": sum(errors) / len(errors),
                "mean_energy_retained": sum(energies) / len(energies),
                "bpb_before_finetune": before,
                "bpb_after_40step_finetune": after,
            })

    (output / "results.json").write_text(json.dumps(rows, indent=2))
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print("baseline_bpb", baseline_bpb)


if __name__ == "__main__":
    main()
