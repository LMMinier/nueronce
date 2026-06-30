#!/usr/bin/env python3
"""Train briefly, then run the full CFNA inference pipeline on a query.

Demonstrates the whole architecture working together: the trained byte model as
perception+core+renderer, the hybrid retriever, the latent workspace, the planner,
and the independent verifier.

Usage:  python scripts/run_pipeline.py [--steps 300]
"""

from __future__ import annotations

import argparse

import torch

from cfna import data, pipeline
from cfna.model import CFNAModel, ModelConfig


def train(model: CFNAModel, steps: int, seq: int = 96, batch: int = 16, lr: float = 3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    batches = data.make_batches(data.corpus_bytes(repeat=10), seq, batch, steps, seed=0)
    model.train()
    for batch_ids in batches:
        loss, _ = model.loss(batch_ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.detach().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--query", type=str, default="CFNA separates understanding,")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = CFNAModel(ModelConfig())
    print(f"training {model.num_params():,}-param model for {args.steps} steps ...")
    final = train(model, args.steps)
    print(f"final train loss: {final:.3f}")

    corpus = [s.strip() for s in data.CORPUS.split(".") if s.strip()]
    text, report, trace = pipeline.respond(model, args.query, corpus, mode="DELIBERATE", max_rounds=2)

    print("\n=== CFNA pipeline ===")
    print("query     :", args.query)
    print("retrieved :", trace["retrieved"][:3])
    print("reasoning :", trace["reasoning"]["best_hypothesis"],
          "(conf", round(trace["reasoning"]["confidence"], 3), ")")
    print("plan      :", trace["plan_sections"])
    print("verifier  :", trace["verification"])
    print("answer    :", text)


if __name__ == "__main__":
    main()
