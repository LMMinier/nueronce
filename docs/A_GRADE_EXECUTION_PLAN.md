# NUERONCE A-Grade Execution Plan

This document turns the project scorecard into falsifiable acceptance gates. A grade changes only when the evidence exists; code volume, parameter count, and planned features do not count as proof.

## Current baseline

| Dimension | Current grade | Reason |
|---|---:|---|
| Architecture capacity | B+ | Full byte-first CFNA is implemented and trainable, but true sparse compute, incremental inference, typed-channel specialization, and neural/symbolic integration are incomplete. |
| Training mechanism | A- | Full forward/backward, response-only SFT, finite-gradient rejection, activation recomputation, split-run backward, and atomic checkpointing work. Long-run scheduling and recovery evidence are still missing. |
| Numerical stability | A- | The known float32 sigmoid and masked-softmax failures are fixed and all 177 gradient tensors were finite in the verified 35M step. Long stress runs and randomized extreme-value coverage are still required. |
| Memory-constrained execution | B+ | A 34.37M model completes training under roughly 1.5–2.5 GiB in tested short sequences. Gradient spooling, longer contexts, and higher-rung evidence remain incomplete. |
| Corpus availability | B | Licensed corpus and prompt-aligned SFT data exist, but quality, deduplication, domain balance, document-held-out validation, and scale need stronger published reports. |
| Learned language knowledge | F | The current fresh 35M line has one strict SFT update and no broad pretraining. |
| Instruction following | F | One four-byte response-only update is insufficient to measure instruction following. |
| Reasoning | Not measurable | No trained checkpoint has passed a held-out reasoning suite. |
| Retrieval use by weights | Not measurable | Retrieval wiring is real, but the current released weights have not demonstrated learned retrieval use. |
| Scientific evidence trail | B+ | The repository records positive and negative results, but external blind validation, multi-seed confidence intervals, and released benchmark artifacts are incomplete. |

## A-level acceptance gates

### 1. Architecture capacity — A

All of the following must pass:

- Full 35M architecture trains end-to-end with byte perception, dynamic patches, typed memory, hybrid core, retrieval input, and byte decoder enabled.
- True sparse/local execution avoids constructing the full score matrix for the sparse path; benchmark reports measured compute and memory reduction against dense masking.
- Incremental generation caches convolution, SSM, and decoder state and demonstrates at least a 3x tokens/second gain over full-prefix recomputation at context 256.
- Authority/provenance metadata is consumed by the learned pipeline, not only by a separate symbolic island.
- Typed-channel specialization is either demonstrated by channel-specific ablation with at least one channel causing a statistically meaningful task-specific degradation, or the typed claim is removed and replaced with an evidence-backed design.
- Matched-parameter ablations isolate dynamic patching, typed memory, hybrid routing, and retrieval.

### 2. Training mechanism — A

All of the following must pass:

- Deterministic resume reproduces the next-step loss and parameter hash from the same checkpoint, batch, and RNG state.
- At least 1,000 consecutive 35M updates complete without checkpoint corruption or manual intervention.
- Data cursor, shuffled epoch state, optimizer state, phase schedule, RNG state, and validation history survive restart.
- Checkpoints are saved atomically, verified by SHA-256, and published as GitHub Release assets with manifests.
- A failed/nonfinite step leaves parameters, optimizer moments, step counter, and last valid checkpoint unchanged.
- Training and evaluation commands run from a clean checkout with documented dependencies.

### 3. Numerical stability — A

All of the following must pass:

- Zero NaN/Inf values across 1,000 consecutive mixed pretraining/SFT updates.
- Randomized float32 stress tests cover extreme logits, fully masked rows, empty masks, long recurrent trajectories, zero-variance normalization inputs, and large/small denominator cases.
- Gradient checks pass on every primitive used by the 35M model, with documented tolerances.
- Gradient norm, activation range, parameter range, and optimizer-stat range are logged per stage.
- Automatic step rejection and rollback are tested by deliberate injected failures.

### 4. Memory-constrained execution — A

All of the following must pass:

- 35M batch-1 sequence-64 training remains below 4 GiB resident memory on CPU.
- 35M batch-1 sequence-256 completes through tiled or staged execution without exceeding the declared memory budget.
- Block-gradient spooling or equivalent streaming update removes the requirement for all full gradients to remain resident simultaneously.
- Peak memory is measured, not estimated, for forward, backward, update, save, and resume.
- A 90M rung completes at least 100 valid steps within an explicitly documented RAM budget.

### 5. Corpus availability — A

All of the following must pass:

- Every source has machine-readable license, provenance, source URL, acquisition date, and commercial-use status.
- Exact and near-duplicate rates are measured before and after deduplication.
- Train/validation/test splits occur by document or source family, never by adjacent byte windows from the same document.
- The released training set includes sufficient clean data for the planned 35M run, with domain distribution and byte counts published.
- Prompt-aligned SFT includes direct, grounded, abstention, correction, multi-turn, retrieval, and tool-use categories with held-out templates and authors.
- Restricted/prohibited sources are automatically rejected by tests.

### 6. Learned language knowledge — A

This grade is relative to a 35M research model, not frontier-model capability.

All of the following must pass on a released checkpoint:

- Broad byte pretraining reaches held-out bits-per-byte <= 1.8 on a document-held-out validation set, or beats matched byte-Transformer and pure-SSM baselines by a statistically supported margin.
- Generated text is readable and coherent across at least 100 held-out prompts without retrieval or tools.
- The model demonstrates knowledge transfer across unseen documents and paraphrased questions.
- Memorization checks show performance is not explained by exact training overlap.
- At least three seeds or checkpoints establish that the result is repeatable.

### 7. Instruction following — A

All of the following must pass:

- At least 500 held-out instructions spanning direct answers, summaries, procedures, formatting constraints, grounded answers, corrections, and abstentions.
- >= 80% task success under a frozen evaluator rubric, with per-category scores published.
- Prompt-template and author-held-out evaluation prevents template memorization.
- Grounded tasks require use of supplied evidence; unsupported additions are scored separately.
- Abstention precision and recall are both reported rather than merged into overall accuracy.

### 8. Reasoning — A

All of the following must pass:

- A frozen held-out suite covers arithmetic, symbolic manipulation, multi-step deduction, state tracking, and counterfactual consistency.
- The trained CFNA beats its own pretrained-only checkpoint and at least one matched-parameter baseline.
- Reasoning accuracy is reported separately from renderer fidelity, retrieval success, and tool assistance.
- At least one ablation identifies which architectural or training component causes the gain.
- Results include bootstrap confidence intervals or multiple-seed variance.

### 9. Retrieval use by weights — A

All of the following must pass:

- On unseen facts unavailable in model pretraining, retrieval improves answer accuracy by at least 10 absolute percentage points over no-retrieval inference.
- Citation precision and decisive-evidence attribution each reach at least 0.85 on a held-out set.
- Counterfactual removal of decisive evidence changes the answer or confidence in the expected direction.
- Dense-only, sparse-only, late-only, fused, and no-retrieval ablations are compared at equal model weights.
- Authority/provenance gating prevents untrusted retrieved text from influencing the answer in adversarial tests.

### 10. Scientific evidence trail — A

All of the following must pass:

- Every headline claim maps to a versioned benchmark file, command, environment, seed, and checkpoint hash.
- Positive, null, and falsified hypotheses remain visible in the repository.
- At least one naturalistic blind benchmark is authored outside the implementation code path.
- Core results reproduce on a second machine or operating system.
- Multi-seed results include uncertainty, not only best-run numbers.
- Model checkpoints, manifests, evaluation data permitted for release, and reports are published as immutable release assets.

## Execution order

The grades cannot all move in parallel. The critical path is:

1. **Release discipline:** publish step-1 checkpoint and manifest as a GitHub Release asset.
2. **Runtime hardening:** deterministic resume test, randomized numerical stress suite, stage telemetry, and gradient spooling.
3. **Broad pretraining:** run stable document-shuffled byte pretraining to the held-out quality gate.
4. **Phased SFT:** direct -> grounded -> abstention/correction -> multi-turn/tool/retrieval.
5. **Frozen evaluation:** language, instruction following, reasoning, retrieval, authority, and renderer fidelity.
6. **Matched baselines and ablations:** byte Transformer, pure SSM, and component removals.
7. **External blind validation and release:** publish final checkpoint, data manifests, benchmark outputs, confidence intervals, and limitations.

## Immediate definition of done

The next milestone is **not** “all A grades.” It is `base35m-pretrain-v1`:

- GitHub Release contains checkpoint + manifest.
- 1,000 uninterrupted valid training updates.
- Zero nonfinite steps.
- Document-held-out validation curve.
- Deterministic resume proof.
- Peak memory report.
- Generated text samples at fixed prompts and seeds.

Only after that milestone should learned language knowledge be regraded above F.
