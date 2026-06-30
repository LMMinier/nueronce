# Component status

Honest, per-component maturity. Four labels:

- **REAL / TRAINABLE** — real neural module, trained or trainable, gradients flow,
  verified by tests.
- **REAL / HEURISTIC** — real, working code, but rule/statistics-based, not learned.
- **INTERFACE ONLY** — control flow + types are real; the leaf operation is an
  injected hook or raises until supplied.
- **PLANNED** — described in the design, not yet built.

The neural sequence core is the most complete part. The cognitive, ingestion,
retrieval-quality, and large-scale training systems are earlier-stage.

## Neural core

| Component | File | Status |
|---|---|---|
| Primitives (Linear, RMSNorm, attention, SelectiveSSM, …) | `cfna/nn.py` | REAL / TRAINABLE |
| Byte perception CNN + boundary head | `cfna/blocks.py` | REAL / TRAINABLE |
| Dynamic patching (segments, pooling, causal masks) | `cfna/segment.py` | REAL / TRAINABLE |
| Typed recurrent memory cell | `cfna/blocks.py` | REAL / TRAINABLE (structurally typed; channel *semantics* not yet supervised — see Limitations) |
| Hybrid block (SSM+local+sparse+retrieval, per-position router) | `cfna/blocks.py` | REAL / TRAINABLE |
| Byte decoder (unit + retrieval cross-attn) | `cfna/blocks.py` | REAL / TRAINABLE |
| `CFNAModel` end-to-end byte LM | `cfna/model.py` | REAL / TRAINABLE |
| Trainable segmentation (LM grad → boundary head) | `cfna/model.py` | REAL / TRAINABLE |
| Retrieval-augmented training | `cfna/retrieval_train.py` | REAL / TRAINABLE |
| Baselines (byte Transformer, pure SSM) + held-out eval | `cfna/baselines.py`, `cfna/eval.py` | REAL / TRAINABLE |
| 350M-scale constructor (`large_config`) | `cfna/model.py` | REAL / TRAINABLE (constructs + forwards; not trained) |
| License-clean corpus pipeline (download→clean→dedupe→manifest→buckets) | `cfna/corpus/*` | REAL (public-domain only; trusted sources) |
| Corpus→weights training + checkpoint | `scripts/train_checkpoint.py` | REAL / TRAINABLE (11M model on ~14 MB) |
| Conversation interface | `cfna/chat.py` | REAL (small byte model: English-shaped continuations, not an instruct assistant) |

## Cognitive / knowledge layer

| Component | File | Status |
|---|---|---|
| Provenance policy gate | `cfna/ingestion.py` | REAL / HEURISTIC |
| Web fetcher, robots/terms/license/PII/malware scanners | — | PLANNED |
| Document parser (HTML/PDF/code) | `cfna/parsing.py` | INTERFACE ONLY |
| Regex claim/evidence/code/equation detectors | `cfna/impl.py` | REAL / HEURISTIC |
| Cognitive embedding heads | `cfna/embeddings.py` | REAL (untrained) |
| Retrieval score fusion | `cfna/retrieval.py` | REAL / TRAINABLE-ready |
| Dense/sparse/late retrieval *quality* | `cfna/impl.py` | REAL / HEURISTIC (hashed n-grams, not SBERT/SPLADE/ColBERT) |
| Contradiction / temporal scoring | `cfna/impl.py` | REAL / HEURISTIC (currently constant) |
| Typed memory consolidation scoring | `cfna/memory.py` | REAL / HEURISTIC |
| Memory clustering / persistent stores | `cfna/impl.py` | INTERFACE ONLY |
| Global workspace (slot attention) | `cfna/workspace.py` | REAL but UNTRAINED (frozen, not in the LM objective) |
| Planner | `cfna/planning.py`, `cfna/impl.py` | REAL / HEURISTIC |
| Verifier checkers | `cfna/verification.py`, `cfna/impl.py` | REAL / HEURISTIC (word-overlap, negation, calibration heuristics) |
| Verify→revise loop (removes/​hedges failing claims) | `cfna/impl.py`, `cfna/pipeline.py` | REAL / HEURISTIC |
| Authority-gated tool executor | `cfna/tools.py` | REAL (runner injected) |

## Training program

| Component | File | Status |
|---|---|---|
| WPGCP curriculum / phase weights / loss aggregation | `cfna/training/*` | REAL (scaffolding); episode/data compilers INTERFACE ONLY |
| VGRFT (instruction/tool/verifier/residual training) | `cfna/training/vgrft.py` | INTERFACE ONLY (raise `NotImplementedError`) |
| Continual-learning controller | `cfna/training/vgrft.py` | REAL orchestration (training steps injected) |
| LoRA adapters | `cfna/runtime.py` | REAL (numpy) |
| Quantized / streaming runtime | — | PLANNED |

## Known limitations (see also the design doc)

1. **Typed channels are structurally typed, not yet semantically specialized.**
   No per-channel objectives/probes yet — the names are intent until trained.
2. **"Sparse"/local attention compute the full score matrix then mask/top-k.**
   Sparse *weights*, not sparse *compute*; not yet subquadratic.
3. **Every hybrid path executes every step** (router mixes, doesn't skip). True
   conditional computation (execute only selected branches) is PLANNED.
4. **Generation recomputes the full forward each step** — no incremental
   SSM/conv/decoder state cache yet. Streaming inference is PLANNED.
5. **Retrieval quality is heuristic** (hashed n-grams), not learned dense/sparse
   retrieval; the *training mechanism* is real, the *retriever* is frozen.
6. **Held-out comparison is tiny-scale.** On a ~4 KB corpus at ~1.2M params, CFNA
   is competitive with a matched byte Transformer but does **not** decisively beat
   it; both beat a pure SSM on generalization. No scaling claims. See `RESULTS.md`.
