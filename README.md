# nueronce

A from-the-ground-up **hybrid foundational model** architecture (**CFNA**) plus a
companion **research MCP server**.

CFNA reorganizes the standard `next_token = MODEL(previous_tokens)` recipe into a
pipeline with explicit division of labor:

```
perceive → represent meaning → determine intent → retrieve evidence →
reason → plan → communicate → verify → revise
```

Perception, typed memory, retrieval, reasoning, planning, surface realization, and
verification are *separate subsystems* rather than one decoder-only network doing
all of it via token continuation. See **[`docs/CFNA_design.md`](docs/CFNA_design.md)**
for the full design and **[`docs/architecture.md`](docs/architecture.md)** for the
module map.

Every operator is **built from scratch** (`cfna/nn.py`): PyTorch is used only as
a tensor/autograd/optimizer substrate — no `nn.Transformer`,
`nn.MultiheadAttention`, fused attention, or any prebuilt state-space (Mamba)
module. The full pipeline **trains end-to-end** and is verified causal. See
[`docs/RESULTS.md`](docs/RESULTS.md) for a recorded run (bits/byte 8.30 → 0.11).

## What's in here

```
cfna/
  nn.py                   From-scratch primitives: Linear, RMSNorm, attention,
                          SelectiveSSM (no stock transformer/Mamba anywhere)
  blocks.py               CFNA operators: byte CNN perception, typed recurrent
                          memory, hybrid SSM/attention block, byte decoder
  segment.py              Dynamic patching → segment ids, pooling, causal masks
  model.py                CFNAModel — the trainable two-level byte LM
  data.py                 Toy corpus + batching
  pipeline.py             End-to-end respond(): retrieve → reason → plan → verify
  impl.py                 Real symbolic-stage components (parsing, retrieval
                          indexes, verifier checkers, planner, consolidation)
  types.py config.py ops.py        Typed records, configs, numeric/hash utils
  ingestion.py parsing.py          Provenance gate, document parser/compiler
  perception.py embeddings.py      Perception adapter, typed embedding heads
  memory.py routers.py core.py     Typed memory, relation routers, hybrid core
  workspace.py planning.py         Latent workspace, planner + renderer
  verification.py tools.py         Verify→revise loop, authority-gated tools
  retrieval.py runtime.py          Hybrid retriever, LoRA + SSD-backed memory
  schemas/  training/              JSON schemas; WPGCP/VGRFT pipelines
docs/                     Design doc, architecture map, RESULTS, research_mcp
scripts/                  train_demo.py, run_pipeline.py
tests/                    70+ tests incl. learning + causality proofs
research_mcp.py           Notion-backed research MCP server (standalone)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                  # numpy + torch (CPU) + pytest
pytest                                   # 70+ tests incl. learning + causality
python scripts/train_demo.py --steps 400 # training curve + sample + metrics json
python scripts/run_pipeline.py           # train, then end-to-end respond()
```

```python
import torch
from cfna.model import CFNAModel, ModelConfig

model = CFNAModel(ModelConfig())                 # ~2M params, all hand-built
ids = torch.randint(0, 256, (2, 96))
loss, parts = model.loss(ids)                    # next-byte CE + boundary loss
loss.backward()
print(parts["bpb"], "bits/byte")
print(model.generate(b"CFNA separates ", max_new=40, greedy=True))
```

## Research MCP server

`research_mcp.py` is an independent, ready-to-run MCP server that stores and
searches project research in a Notion database. Its setup and usage guide is in
**[`docs/research_mcp.md`](docs/research_mcp.md)** (dependencies in
`requirements.txt`).

## Status

v0.2.0 — every operator is a real, hand-built implementation; the model trains
end-to-end and is verified causal. The trained run, the causality check, and a
correctness bug the tests caught are recorded in
[`docs/RESULTS.md`](docs/RESULTS.md). Falsifiable research hypotheses (H1–H8) and
the module map are in [`docs/architecture.md`](docs/architecture.md) and
[`docs/CFNA_design.md`](docs/CFNA_design.md).

> Scale note: the demo trains a ~2M-parameter model on a tiny corpus to prove the
> architecture is wired correctly and *learns*. It is a correctness harness for
> the design, not a web-scale training run.
