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
  chat.py                          Conversation loop; real turn-taking after SFT
  schemas/  training/              JSON schemas; WPGCP/VGRFT pipelines
                                    (training/sft.py + dialogue_data.py: real
                                    dialogue SFT, VGRFT stage 1)
  microtorch/               From-scratch (NumPy-only) tensor + autograd engine;
                            cfna_model.py/cfna_blocks.py/segment.py port the
                            *full* CFNA architecture onto it (see below) —
                            the real model, not a smaller stand-in
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
python scripts/train_retrieval.py        # retrieval-augmented training + ablation
python scripts/compare_baselines.py      # CFNA vs Transformer/SSM on held-out bytes
```

### Real corpus → weights → conversation

A license-clean pipeline turns **public-domain** books and US-government speeches
into a trained checkpoint (no blogs, no social media, no scraped web — sources are
curated Project Gutenberg + US-gov texts mirrored by the NLTK data repo):

```bash
python scripts/build_corpus.py           # download + clean + license-check + manifest
python scripts/train_checkpoint.py --minutes 20   # corpus -> CFNA weights (+ held-out bpb)
python scripts/chat_demo.py              # hold a short conversation with the checkpoint
python scripts/train_sft.py              # + dialogue SFT pass (VGRFT stage 1) -> real turn-taking
```

The pipeline is `trusted sources → clean → license check → dedupe → split by
document → UTF-8 bytes → batches → gradient updates → saved weights`, with a
provenance `manifest.jsonl` and license buckets (`safe_commercial/` only for the
first checkpoint). Documents are sampled *per-document-uniform* so no single book
dominates, and whole books are held out for validation. See `cfna/corpus/`.

The base checkpoint is trained with pure next-byte cross-entropy on monologic
prose, which teaches what English looks like, not what to say back when spoken
to. `scripts/train_sft.py` closes that gap: it runs `VGRFTTrainer.supervised_
instruction_tune` (VGRFT stage 1) over a small hand-written (prompt, response)
set in `cfna/training/dialogue_data.py`, masking the loss to response bytes
only. It works with two interchangeable backends — `cfna.training.sft.
TorchSFTBackend` (PyTorch, fine-tunes `CFNAModel`) and `cfna.microtorch.
models.MicroSFTBackend` (the from-scratch, NumPy-only engine described under
`cfna/microtorch/`, so this stage runs with no PyTorch dependency at all).
Stages 2-4 of VGRFT (tool grounding, verifier
training, residual experts) still raise `NotImplementedError` — they need
tool traces / verifier ground truth that don't exist yet.

### The full architecture on the from-scratch engine

`cfna/microtorch/` is a hand-built (NumPy-only) tensor + reverse-mode autograd
engine — not a toy or a fallback for a missing PyTorch install, but its own
foundation: `Tensor.backward()`, matmul, attention, and a selective-scan SSM
all built from primitives, grad-checked against finite differences and
PyTorch parity (`tests/test_microtorch.py`).

`cfna/microtorch/cfna_model.py` (+ `cfna_blocks.py`, `segment.py`) is a
module-for-module port of the *real* `CFNAModel` onto that engine — the same
byte perception CNN, dynamic patching, typed recurrent memory, hybrid
SSM/attention/retrieval core, and retrieval-aware decoder, not a smaller
stand-in. `MicroCFNAModel` matches `CFNAModel`'s interface (`forward`, `loss`,
`masked_token_loss`, `generate`) and its guarantees: exactly 0.0 logit leakage
from future to past bytes, and a verified from-scratch learning curve on a toy
corpus. It also plugs directly into VGRFT stage 1 via
`cfna.microtorch.models.MicroSFTBackend`, so the full production architecture
— not just the smaller `MicroByteLM` demo — can be dialogue-SFT-tuned with
zero PyTorch dependency. See `tests/test_microtorch_cfna_model.py`.

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

v0.3.0 — the **neural sequence core** is real, hand-built, trains end-to-end, and
is verified causal. The **cognitive, ingestion, retrieval-quality, and large-scale
training systems are earlier-stage** (real heuristics, interface-only hooks, or
planned). Read [`docs/STATUS.md`](docs/STATUS.md) for an honest per-component
breakdown (REAL/TRAINABLE · REAL/HEURISTIC · INTERFACE-ONLY · PLANNED) — don't
take "from scratch" to mean the whole CFNA design is built; it isn't yet.

What *is* trained/verified: the byte model (perception → dynamic patching → typed
memory → hybrid core → decoder), trainable segmentation, retrieval-augmented
training, and a matched-baseline **held-out** comparison. Numbers, the causality
check, and a router bug the tests caught are in [`docs/RESULTS.md`](docs/RESULTS.md).
Module map and hypotheses: [`docs/architecture.md`](docs/architecture.md),
[`docs/CFNA_design.md`](docs/CFNA_design.md).

> Scale note: the demos train ~1–2M-parameter models on tiny corpora to prove the
> architecture is wired correctly and *learns*. It is a correctness harness for
> the design, not a web-scale training run.
