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

## What's in here

```
cfna/                     The CFNA architecture scaffold (a Python package)
  types.py                Typed runtime records + controlled vocabularies   [implemented]
  ops.py                  Numeric / hashing utilities                        [implemented]
  config.py               CFNA-350M prototype defaults                       [implemented]
  ingestion.py            Policy-aware ingestion + provenance gate           [gate implemented]
  parsing.py              Document parser + knowledge-unit compiler          [hooks]
  perception.py           Dynamic byte patching + unit formation             [patching implemented]
  embeddings.py           Cognitive embedding bundle compiler                [backend]
  memory.py               Typed recurrent memory + evidence-gated consolidation
  routers.py              Relation-specific routers                          [geometry implemented]
  retrieval.py            Dense + sparse + late-interaction hybrid retriever [fusion implemented]
  core.py                 Hybrid cognitive fabric (SSM/attn/retrieval blocks)
  workspace.py            Latent reasoning workspace
  planning.py             Planner + two-stage renderer
  verification.py         Independent verifier + verify→revise loop          [loop implemented]
  tools.py                Authority-gated tool executor                      [gate implemented]
  runtime.py              LoRA adapters + SSD-backed memory + orchestration  [LoRA implemented]
  schemas/                JSON schemas + worked examples for wire records    [implemented]
  training/               WPGCP curriculum + VGRFT fine-tuning pipelines
docs/                     Design doc, architecture map, research_mcp guide
tests/                    Unit tests for the implemented logic
research_mcp.py           Notion-backed research MCP server (standalone)
```

### Implemented vs. scaffolded

This is a faithful *implementation map*, not a trained model. The data model and
the fully-specified, backend-light logic are implemented and tested:

- provenance gating, source-quality scoring
- dynamic byte patching + information-unit formation
- retrieval score fusion, relation geometry
- evidence-gated consolidation scoring, LoRA low-rank adaptation
- the verify→revise control loop and tool authority gate
- JSON schemas + examples, config defaults

The learned neural components (CNN perception, embedding heads, typed recurrent
cell, hybrid blocks, workspace iteration, the causal renderer, the VGRFT trainer)
expose typed interfaces and raise `cfna.BackendNotConfigured` until a PyTorch/JAX
backend is wired in. Find every seam with:

```bash
grep -rn "needs_backend\|BackendNotConfigured" cfna/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # numpy + pytest
pytest                         # run the test suite
```

```python
import numpy as np
from cfna.perception import dynamic_patching, encode_information_units
from cfna.config import DEFAULT_CONFIG

data = b"Input-dependent transitions improve long-context modeling. See fig. 2."
feats = np.random.default_rng(0).standard_normal((len(data), 8))
spans = dynamic_patching(list(data), feats, np.zeros(len(data)))
units = encode_information_units(list(data), spans, feats)
print(len(units), "dynamic information units")
print("core width:", DEFAULT_CONFIG.core.d_model)
```

## Research MCP server

`research_mcp.py` is an independent, ready-to-run MCP server that stores and
searches project research in a Notion database. Its setup and usage guide is in
**[`docs/research_mcp.md`](docs/research_mcp.md)** (dependencies in
`requirements.txt`).

## Status

Prototype scaffold (v0.1.0). Build order and falsifiable research hypotheses
(H1–H8) are in `docs/architecture.md` and `docs/CFNA_design.md`.
