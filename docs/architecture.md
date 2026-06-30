# CFNA module map

How the design (`docs/CFNA_design.md`) maps onto the `cfna` package, and what is
**implemented** vs. a **typed stub** awaiting a neural backend.

Legend: ✅ implemented & tested · 🧩 control-flow implemented, leaf ops injected ·
⛔ needs a neural backend (raises `BackendNotConfigured`).

| Design stage | Module | Key symbols | Status |
|---|---|---|---|
| Data model | `cfna/types.py` | `SourceRecord`, `KnowledgeUnit`, `CognitiveEmbeddingBundle`, `MemoryRecord`, `TaskState`, `WorkspaceSlot`, `Verification*` | ✅ |
| Numeric/hash utils | `cfna/ops.py` | `cosine`, `sparse_dot`, `softmax`, `sigmoid`, `gelu`, `hash_ngram_features`, `sha256_bytes`, `now_iso` | ✅ |
| Config | `cfna/config.py` | `CFNAConfig` + per-subsystem configs, `DEFAULT_CONFIG` | ✅ |
| Backend boundary | `cfna/_backend.py` | `BackendNotConfigured`, `needs_backend` | ✅ |
| Ingestion / provenance gate | `cfna/ingestion.py` | `PolicyGate`, `IngestionCrawler`, `score_quality` | 🧩 (gate ✅, fetcher injected) |
| Parser / KU compiler | `cfna/parsing.py` | `DocumentParser`, `KnowledgeUnitCompiler`, `CompilerHooks` | 🧩 |
| Perception / patching | `cfna/perception.py` | `dynamic_patching`, `encode_information_units` ✅; `ByteCharPerception` ⛔ | mixed |
| Cognitive embeddings | `cfna/embeddings.py` | `CognitiveEmbeddingCompiler` | ⛔ |
| Typed recurrent memory | `cfna/memory.py` | `consolidation_score`/`_decision` ✅; `TypedRecurrentMemoryCell` ⛔ | mixed |
| Relation routers | `cfna/routers.py` | semantic/lexical/structural/temporal/authority ✅; `evidence_relation` ⛔ | mixed |
| Hybrid retrieval | `cfna/retrieval.py` | `combine_scores` ✅; `HybridRetriever` 🧩 (indexes injected) | mixed |
| Hybrid cognitive fabric | `cfna/core.py` | `CFNACore.run` 🧩; `HybridBlock.forward` ⛔ | mixed |
| Global workspace | `cfna/workspace.py` | `GlobalWorkspace` (roles ✅, neural ops ⛔) | mixed |
| Planner / renderer | `cfna/planning.py` | `Planner` 🧩; `SemanticRenderer` 🧩; `CausalLanguageRenderer` ⛔ | mixed |
| Verifier | `cfna/verification.py` | `verify_and_revise` 🧩, `IndependentVerifier` 🧩 (checkers injected) | 🧩 |
| Tools / authority | `cfna/tools.py` | `ToolExecutor` (authority gate ✅, runner injected) | 🧩 |
| Compact runtime | `cfna/runtime.py` | `LoRAAdapter` ✅, `SSDBackedMemoryStore` ✅ | mixed |
| Schemas | `cfna/schemas/` | `load_schema`, `load_example`, 4 record schemas + examples | ✅ |
| WPGCP training | `cfna/training/{curriculum,episodes,losses}.py` | scheduler/weights/aggregation ✅; data transforms injected | 🧩 |
| VGRFT training | `cfna/training/vgrft.py` | `ContinualLearner` 🧩; `VGRFTTrainer` ⛔ | mixed |

## The backend boundary

The data model and pure-logic paths run with only `numpy`. Learned components
(`ByteCharPerception`, `CognitiveEmbeddingCompiler`, `TypedRecurrentMemoryCell`,
`HybridBlock`, workspace iteration, `CausalLanguageRenderer`, the VGRFT trainer)
raise `cfna.BackendNotConfigured` until a PyTorch/JAX backend is wired in. This
keeps the whole package importable and the architecture navigable before any
training infrastructure exists. To find every backend seam:

```bash
grep -rn "needs_backend\|BackendNotConfigured" cfna/
```

## Injected-hooks pattern

Subsystems whose control flow is real but whose leaf operations are
corpus/model-specific (parser, knowledge-unit compiler, consolidator, planner,
verifier, retriever, episode generator, continual learner) accept their leaf ops
as injected callables (often grouped in a `*Hooks` dataclass). This makes the
orchestration unit-testable today and lets you drop in real implementations
incrementally.

## Suggested build order

1. Wire a tensor backend behind `cfna/_backend.py` (PyTorch first).
2. Implement `ByteCharPerception` + train the boundary head; validate against
   `dynamic_patching` (H1).
3. Implement `CognitiveEmbeddingCompiler` heads; stand up dense + sparse indexes
   behind `HybridRetriever` (H3).
4. Implement `TypedRecurrentMemoryCell` and `HybridBlock`; run `CFNACore` over
   FAST depth (H2).
5. Add workspace iteration, planner hooks, and the renderer (H4).
6. Train the verifier and residual experts via VGRFT (H5, H8).
7. Turn on authority masks end-to-end (H6) and run the ablation matrix.
