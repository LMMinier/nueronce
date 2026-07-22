"""Depth-extrapolation benchmark for the NUERONCE execution register."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from nueronce.reasoning import AddressableExecutionRegister


def make_batch(nodes, batch_size, hops, generator):
    mapping = torch.stack([torch.randperm(nodes, generator=generator) for _ in range(batch_size)])
    start = torch.randint(nodes, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    state = start
    targets = []
    for _ in range(hops):
        state = mapping[rows, state]
        targets.append(state)
    return mapping, start, targets


class Machine(nn.Module):
    def __init__(self, nodes, dim):
        super().__init__()
        self.nodes = nodes
        self.symbol = nn.Embedding(nodes, dim)
        self.executor = AddressableExecutionRegister(dim)

    def forward(self, mapping, start, steps):
        b = mapping.shape[0]
        addresses = torch.arange(self.nodes)[None, :].expand(b, -1)
        keys = self.symbol(addresses)
        values = self.symbol(mapping)
        state, trace = self.executor(self.symbol(start), keys, values, steps)
        # Tied symbolic decoder: the register must land near the addressed symbol.
        logits = [s @ self.symbol.weight.T for s in trace.states]
        return logits[-1], logits, trace


def train(model, steps, batch_size, nodes, seed):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    started = time.perf_counter()
    transitions = 0
    for step in range(steps):
        hops = 1 + step % 4
        mapping, start, targets = make_batch(nodes, batch_size, hops, gen)
        _, logits, trace = model(mapping, start, hops)
        state_loss = torch.stack([nn.functional.cross_entropy(p, y)
                                  for p, y in zip(logits, targets)]).mean()
        expected_addresses = [start] + targets[:-1]
        address_loss = torch.stack([
            nn.functional.nll_loss(weights.clamp_min(1e-9).log(), address)
            for weights, address in zip(trace.read_weights, expected_addresses)
        ]).mean()
        loss = state_loss + 0.5 * address_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        transitions += hops
    return {"optimizer_steps": steps, "transition_steps": transitions,
            "seconds": time.perf_counter() - started, "final_loss": float(loss.detach())}


@torch.no_grad()
def evaluate(model, nodes, hop, batches, batch_size, seed):
    gen = torch.Generator().manual_seed(seed + hop)
    correct = total = 0
    read_correct = read_total = 0
    started = time.perf_counter()
    for _ in range(batches):
        mapping, start, targets = make_batch(nodes, batch_size, hop, gen)
        pred, _, trace = model(mapping, start, hop)
        correct += int((pred.argmax(-1) == targets[-1]).sum())
        total += batch_size
        # The read address at transition t should equal the previous state.
        expected = [start] + targets[:-1]
        for weights, address in zip(trace.read_weights, expected):
            read_correct += int((weights.argmax(-1) == address).sum())
            read_total += batch_size
    elapsed = time.perf_counter() - started
    return {"accuracy": correct / total, "address_accuracy": read_correct / read_total,
            "examples_per_second": total / elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", type=Path, default=Path("metrics/execution_register.json"))
    args = p.parse_args()
    torch.manual_seed(args.seed)
    model = Machine(args.nodes, args.dim)
    training = train(model, args.steps, args.batch_size, args.nodes, args.seed)
    results = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "training": training,
        "train_hops": [1, 4],
        "evaluation": {str(h): evaluate(model, args.nodes, h, args.eval_batches,
                                         args.batch_size, args.seed + 100)
                       for h in range(1, 9)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
