"""Compute-matched test of recurrent deliberation on pointer chasing.

The input defines a random permutation f and asks for f^k(start). Training only
uses short k; evaluation includes longer, unseen reasoning depths. This tests
computation, not memorized language. Results are emitted as JSON.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from nueronce.blocks import HybridCoreStack


class PointerReasoner(nn.Module):
    def __init__(self, nodes: int, dim: int, max_hops: int):
        super().__init__()
        self.nodes = nodes
        self.value = nn.Embedding(nodes + 1, dim)
        self.position = nn.Embedding(nodes + 1, dim)
        self.start = nn.Embedding(nodes, dim)
        self.hops = nn.Embedding(max_hops + 1, dim)
        self.core = HybridCoreStack(
            dim, physical_blocks=1, n_heads=4, local_window=nodes + 1,
            sparse_topk=nodes, d_state=8, ffn_mult=2,
        )
        self.head = nn.Linear(dim, nodes)

    def forward(self, mappings, starts, hops, depth, mode="fixed", halt_epsilon=0.0,
                collect_states=False):
        b, n = mappings.shape
        positions = torch.arange(n, device=mappings.device)[None, :].expand(b, -1)
        table = self.value(mappings) + self.position(positions)
        query = (self.value.weight[self.nodes][None, :]
                 + self.start(starts) + self.hops(hops))
        x = torch.cat([table, query[:, None, :]], dim=1)
        result = self.core(
            x, depth, reasoning_mode=mode, min_depth=2,
            halt_epsilon=halt_epsilon, damping=1.0, collect_states=collect_states,
        )
        if collect_states:
            x, states = result
            return self.head(x[:, -1]), [self.head(s[:, -1]) for s in states]
        return self.head(result[:, -1])


def batch(nodes, batch_size, hop_low, hop_high, generator):
    maps = torch.stack([torch.randperm(nodes, generator=generator) for _ in range(batch_size)])
    starts = torch.randint(nodes, (batch_size,), generator=generator)
    hops = torch.randint(hop_low, hop_high + 1, (batch_size,), generator=generator)
    targets = starts.clone()
    rows = torch.arange(batch_size)
    for i in range(hop_high):
        nxt = maps[rows, targets]
        targets = torch.where(hops > i, nxt, targets)
    return maps, starts, hops, targets


def transition_targets(mappings, starts, max_depth):
    rows = torch.arange(mappings.shape[0])
    state = starts
    targets = []
    for _ in range(max_depth):
        state = mappings[rows, state]
        targets.append(state)
    return targets


def train(model, steps, depths, mode, nodes, batch_size, seed, process_supervision=False):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    gen = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    final_loss = None
    block_steps = 0
    for step in range(steps):
        data = batch(nodes, batch_size, 1, 4, gen)
        depth = depths[step % len(depths)]
        if process_supervision:
            logits, trajectory = model(*data[:3], depth, mode, collect_states=True)
            intermediate = transition_targets(data[0], data[1], depth)
            # Iteration t must execute one mapping transition. For requests
            # shorter than t, supervise the stable requested answer instead.
            losses = []
            for t, step_logits in enumerate(trajectory):
                target = torch.where(data[2] > t, intermediate[t], data[3])
                losses.append(nn.functional.cross_entropy(step_logits, target))
            loss = torch.stack(losses).mean()
        else:
            loss = nn.functional.cross_entropy(model(*data[:3], depth, mode), data[3])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.detach())
        block_steps += depth
    return {"steps": steps, "block_steps": block_steps,
            "seconds": time.perf_counter() - started, "final_loss": final_loss}


@torch.no_grad()
def evaluate(model, nodes, depth, mode, seed, batches=20, batch_size=64,
             halt_epsilon=0.0):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    correct = {h: 0 for h in range(1, 9)}
    total = {h: 0 for h in range(1, 9)}
    executed = []
    started = time.perf_counter()
    for h in range(1, 9):
        for _ in range(batches):
            data = batch(nodes, batch_size, h, h, gen)
            pred = model(*data[:3], depth, mode, halt_epsilon).argmax(-1)
            correct[h] += int((pred == data[3]).sum())
            total[h] += batch_size
            executed.append(model.core.last_reasoning_stats["steps"])
    elapsed = time.perf_counter() - started
    accuracy = {str(h): correct[h] / total[h] for h in correct}
    return {
        "accuracy_by_hops": accuracy,
        "seen_hops_accuracy": sum(correct[h] for h in range(1, 5)) / sum(total[h] for h in range(1, 5)),
        "unseen_hops_accuracy": sum(correct[h] for h in range(5, 9)) / sum(total[h] for h in range(5, 9)),
        "requested_depth": depth,
        "mean_executed_depth": sum(executed) / len(executed),
        "examples_per_second": sum(total.values()) / elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("metrics/recurrent_reasoning.json"))
    args = p.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    prototype = PointerReasoner(args.nodes, args.dim, 8)
    baseline = PointerReasoner(args.nodes, args.dim, 8)
    recurrent = PointerReasoner(args.nodes, args.dim, 8)
    executor = PointerReasoner(args.nodes, args.dim, 8)
    baseline.load_state_dict(prototype.state_dict())
    recurrent.load_state_dict(prototype.state_dict())
    executor.load_state_dict(prototype.state_dict())
    params = sum(p.numel() for p in prototype.parameters())

    # Baseline receives twice as many optimizer steps so both variants execute
    # the same number of recurrent block applications during training.
    base_train = train(baseline, args.steps * 2, [2], "fixed", args.nodes,
                       args.batch_size, args.seed + 1)
    rec_train = train(recurrent, args.steps, [2, 3, 4, 5, 6], "equilibrium",
                      args.nodes, args.batch_size, args.seed + 1)
    exec_train = train(executor, args.steps, [2, 3, 4, 5, 6], "equilibrium",
                       args.nodes, args.batch_size, args.seed + 1,
                       process_supervision=True)
    results = {
        "task": "random-permutation-pointer-chasing",
        "parameters_each": params,
        "train_hops": [1, 4],
        "test_hops": [1, 8],
        "compute_match": {"baseline_block_steps": base_train["block_steps"],
                          "recurrent_block_steps": rec_train["block_steps"]},
        "baseline_train": base_train,
        "recurrent_train": rec_train,
        "executor_train": exec_train,
        "baseline": {}, "recurrent": {}, "transition_supervised": {},
    }
    for depth in (2, 4, 6, 8):
        results["baseline"][str(depth)] = evaluate(
            baseline, args.nodes, depth, "fixed", args.seed + 100,
            args.eval_batches,
        )
        results["recurrent"][str(depth)] = evaluate(
            recurrent, args.nodes, depth, "equilibrium", args.seed + 100,
            args.eval_batches,
        )
        results["transition_supervised"][str(depth)] = evaluate(
            executor, args.nodes, depth, "equilibrium", args.seed + 100,
            args.eval_batches,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
