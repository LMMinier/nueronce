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
| Primitives (Linear, RMSNorm, attention, SelectiveSSM, …) | `nueronce/nn.py` | REAL / TRAINABLE |
| From-scratch autograd engine (tensors, backprop, ops, optimizers) | `nueronce/engine/` | REAL / TRAINABLE (NumPy-only; grad-checked + torch-parity; for correctness/clarity, not speed) |
| `NueronceModel` — the *full* NUERONCE architecture on engine (perception, patching, typed memory, hybrid core, retrieval, decoder), not a smaller stand-in | `nueronce/engine/nueronce_model.py`, `nueronce_blocks.py`, `segment.py` | REAL / TRAINABLE (NumPy-only; same causal shift conventions, gradient- and causality-checked, learns on a toy corpus, plugs into VGRFT stage 1) |
| Byte perception CNN + boundary head | `nueronce/blocks.py` | REAL / TRAINABLE |
| Dynamic patching (segments, pooling, causal masks) | `nueronce/segment.py` | REAL / TRAINABLE |
| Typed recurrent memory cell | `nueronce/blocks.py` | REAL / TRAINABLE (structurally typed; channel *semantics* not yet supervised — see Limitations) |
| Hybrid block (SSM+local+sparse+retrieval, per-position router) | `nueronce/blocks.py` | REAL / TRAINABLE |
| Byte decoder (unit + retrieval cross-attn) | `nueronce/blocks.py` | REAL / TRAINABLE |
| `NUERONCEModel` end-to-end byte LM | `nueronce/model.py` | REAL / TRAINABLE |
| Trainable segmentation (LM grad → boundary head) | `nueronce/model.py` | REAL / TRAINABLE |
| Retrieval-augmented training | `nueronce/retrieval_train.py` | REAL / TRAINABLE |
| Baselines (byte Transformer, pure SSM) + held-out eval | `nueronce/baselines.py`, `nueronce/eval.py` | REAL / TRAINABLE |
| 350M-scale constructor (`large_config`) | `nueronce/model.py` | REAL / TRAINABLE (constructs + forwards; not trained) |
| License-clean corpus pipeline (download→clean→dedupe→manifest→buckets) | `nueronce/corpus/*` | REAL (public-domain only; trusted sources) |
| Corpus→weights training + checkpoint | `scripts/train_checkpoint.py` | REAL / TRAINABLE (11M model on ~14 MB) |
| Conversation interface | `nueronce/chat.py` | REAL (small byte model: English-shaped continuations by default; real turn-taking signal only after `scripts/train_sft.py`) |
| Supervised instruction tuning (dialogue SFT) | `nueronce/training/sft.py`, `nueronce/training/dialogue_data.py`, `nueronce/engine/models.py` | REAL / TRAINABLE (small hand-written prompt→response set; masked-loss fine-tune on top of a pretrained checkpoint; PyTorch `NUERONCEModel` and from-scratch-engine `MicroByteLM`/`NueronceModel` backends, interchangeably) |

## Cognitive / knowledge layer

| Component | File | Status |
|---|---|---|
| Provenance policy gate | `nueronce/ingestion.py` | REAL / HEURISTIC |
| Web fetcher, robots/terms/license/PII/malware scanners | — | PLANNED |
| Document parser (HTML/PDF/code) | `nueronce/parsing.py` | INTERFACE ONLY |
| Regex claim/evidence/code/equation detectors | `nueronce/impl.py` | REAL / HEURISTIC |
| Cognitive embedding heads | `nueronce/embeddings.py` | REAL (untrained) |
| Retrieval score fusion | `nueronce/retrieval.py` | REAL / TRAINABLE-ready |
| Dense/sparse/late retrieval *quality* | `nueronce/impl.py` | REAL / HEURISTIC (hashed n-grams, not SBERT/SPLADE/ColBERT) |
| Contradiction / temporal scoring | `nueronce/impl.py` | REAL / HEURISTIC (currently constant) |
| Typed memory consolidation scoring | `nueronce/memory.py` | REAL / HEURISTIC |
| Memory clustering / persistent stores | `nueronce/impl.py` | INTERFACE ONLY |
| Global workspace (slot attention) | `nueronce/workspace.py` | REAL but UNTRAINED (frozen, not in the LM objective) |
| Planner | `nueronce/planning.py`, `nueronce/impl.py` | REAL / HEURISTIC |
| Verifier checkers | `nueronce/verification.py`, `nueronce/impl.py` | REAL / HEURISTIC (word-overlap, negation, calibration heuristics) |
| Verify→revise loop (removes/​hedges failing claims) | `nueronce/impl.py`, `nueronce/pipeline.py` | REAL / HEURISTIC |
| Authority-gated tool executor | `nueronce/tools.py` | REAL (runner injected) |

## Training program

| Component | File | Status |
|---|---|---|
| WPGCP curriculum / phase weights / loss aggregation | `nueronce/training/*` | REAL (scaffolding); episode/data compilers INTERFACE ONLY |
| VGRFT stage 1: supervised instruction tuning (SFT) | `nueronce/training/vgrft.py` + `nueronce/training/sft.py` | REAL / TRAINABLE (backend-injected; `TorchSFTBackend` / `MicroSFTBackend` both run) |
| VGRFT stages 2-4 (tool grounding/verifier/residual training) | `nueronce/training/vgrft.py` | INTERFACE ONLY (raise `NotImplementedError`; need tool traces / verifier ground truth that don't exist yet) |
| Continual-learning controller | `nueronce/training/vgrft.py` | REAL orchestration (training steps injected) |
| LoRA adapters | `nueronce/runtime.py` | REAL (numpy) |
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
6. **Held-out comparison is tiny-scale.** On a ~4 KB corpus at ~1.2M params, NUERONCE
   is competitive with a matched byte Transformer but does **not** decisively beat
   it; both beat a pure SSM on generalization. No scaling claims. See `RESULTS.md`.
