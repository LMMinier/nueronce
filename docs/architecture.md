# CFNA module map

How the design (`docs/CFNA_design.md`) maps onto the `cfna` package. As of v0.2.0
every operator is a **real, hand-built implementation** and the full pipeline
trains end-to-end. PyTorch is used **only** as a tensor / autograd / optimizer
substrate — no `nn.Transformer`, `nn.MultiheadAttention`, `nn.Linear`,
`nn.LayerNorm`, `nn.Embedding`, fused `scaled_dot_product_attention`, or any
external state-space (Mamba) package. See `cfna/nn.py` for the from-scratch
primitives.

## From-scratch substrate

| Layer | File | What's hand-built |
|---|---|---|
| Primitives | `cfna/nn.py` | `Linear`, `Embedding`, `RMSNorm`, `MLP`, `GatedMLP`, masked softmax, `LocalAttention`, `SparseGlobalAttention` (top-k causal), `CrossAttention`, `SelectiveSSM` (selective scan) |
| Operators | `cfna/blocks.py` | `BytePerceptionEncoder` (causal byte CNN + boundary head), `TypedRecurrentMemory` (typed gated cell), `HybridBlock`/`HybridCoreStack` (per-position router), `UnitEmbedder`, `ByteDecoder` |
| Patching | `cfna/segment.py` | segment ids from boundaries, mean-pool matrix, byte→unit causal mask, boundary targets |
| Model | `cfna/model.py` | `CFNAModel` — the two-level byte LM tying it all together |
| Training | `cfna/data.py`, `scripts/train_demo.py` | toy corpus, batching, training loop |
| Retrieval training | `cfna/retrieval_train.py`, `scripts/train_retrieval.py` | RETRO-style retrieval-augmented training + with/without ablation |
| Pipeline | `cfna/pipeline.py`, `cfna/impl.py` | end-to-end `respond()` + real symbolic-stage hooks |

## Retrieval is trained, not bolted on

The core's retrieval cross-attention (`HybridBlock.retrieval`) and a byte-decoder
retrieval cross-attention both receive gradients during training. Retrieved
neighbor bytes are encoded as *(left-context key, current-byte value)* so the
reader can match on context and copy the answer (induction-style). The task in
`cfna/retrieval_train.py` makes the answer recoverable *only* via retrieval (fresh
random facts, used once), so the with-vs-without ablation cleanly measures whether
retrieval is used. See `docs/RESULTS.md`.

## Design stage → module (all real)

| Design stage | Module(s) | Notes |
|---|---|---|
| Data model | `cfna/types.py` | typed records + vocabularies |
| Numeric/hash utils | `cfna/ops.py` | cosine, sparse_dot, hash n-grams, sha256 |
| Config | `cfna/config.py`, `cfna/model.py:ModelConfig` | 350M defaults + trainable tiny config |
| Ingestion / provenance gate | `cfna/ingestion.py` | `PolicyGate`, `IngestionCrawler` |
| Parser / KU compiler | `cfna/parsing.py` + `cfna/impl.py` | regex detectors via `default_compiler_hooks()` |
| Perception / patching | `cfna/perception.py` → `cfna/blocks.py` | `ByteCharPerception` adapter + real CNN; `dynamic_patching` (numpy) |
| Cognitive embeddings | `cfna/embeddings.py` | real typed MLP heads |
| Typed recurrent memory | `cfna/memory.py` → `cfna/blocks.py` | real gated cell (step + sequence) + pure consolidation scoring |
| Relation routers | `cfna/routers.py` | geometric scores + real default evidence classifier |
| Hybrid retrieval | `cfna/retrieval.py` + `cfna/impl.py` | score fusion + in-memory dense/sparse indexes |
| Hybrid cognitive fabric | `cfna/core.py` → `cfna/blocks.py` | SSM + local/sparse attention + router |
| Global workspace | `cfna/workspace.py` | real slot self-attention + update + confidence heads |
| Planner / renderer | `cfna/planning.py` + `cfna/pipeline.py` | heuristic planner + model-backed renderer |
| Verifier | `cfna/verification.py` + `cfna/impl.py` | verify→revise loop + real evidence-grounded checkers |
| Tools / authority | `cfna/tools.py` | authority-gated executor |
| Compact runtime | `cfna/runtime.py` | LoRA (numpy) + SSD-backed store |
| WPGCP / VGRFT training | `cfna/training/*` | curriculum, episodes, loss aggregation, VGRFT drivers |

## What "trains end-to-end" means

`CFNAModel` is a two-level byte language model:

```
bytes → causal byte CNN + learned boundary head
      → dynamic patching into variable-length units
      → unit embedding + typed recurrent memory
      → hybrid core (SSM + local/sparse attention), reused over logical depth
      → byte decoder cross-attending to *completed* units (causal)
      → next-byte logits
```

Objective = next-byte cross-entropy + an auxiliary boundary loss (so the patcher
is genuinely learned). Because every path is causal, the model is a valid
autoregressive LM — the test `tests/test_model_learns.py::test_model_is_causal`
asserts that editing a future byte changes **zero** earlier-position logits.

## Reproducing the run

```bash
pip install -e ".[dev]"
pytest                                   # 70+ tests incl. learning + causality
python scripts/train_demo.py --steps 400 # full training curve + sample + metrics
python scripts/run_pipeline.py           # train, then end-to-end respond()
```

See `docs/RESULTS.md` for a recorded run.

## A bug this structure caught

The first hybrid-block router mean-pooled over the **whole** unit sequence to
compute mixing weights, then applied them per position — leaking future units into
past positions' routing (the causality test failed with a ~0.04 logit change
before an edited byte). Fixed by switching to **per-position routing**; the test
now reports exactly `0.0`. This is the kind of subtle correctness issue the
"is it built right?" tests exist to surface.
