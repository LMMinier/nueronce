# NUERONCE: A Decomposed CPU-First Training Runtime for Memory-Constrained Language Models

**Luis M. Minier**  
Independent Researcher, Bronx, New York  
July 2026

## Abstract

Training neural language models is commonly organized around a monolithic execution pattern: construct a full forward graph, retain intermediate activations, execute a full reverse pass, retain parameter gradients, and update globally resident optimizer state. That pattern is effective on accelerators with large high-bandwidth memory, but it couples logical model depth to physical memory residency and makes full-parameter experimentation difficult on ordinary computers. NUERONCE investigates a different systems arrangement. Its model architecture, the Cognitive Fractal Neural Architecture (NUERONCE), is implemented in a NumPy-native automatic-differentiation engine called Nueronce Engine. The runtime separates model definition, execution planning, activation residency, optimizer-state ownership, evaluation, and checkpoint recovery. Heavy NUERONCE stages use exact activation recomputation: stage internals are discarded during the original forward pass and reconstructed only when their output gradient arrives. Optimizer state is partitioned temporally by model subsystem and loaded, updated, saved, and released block by block. A factorized tiled optimizer, StreamFactor, replaces full matrix second-moment tensors with row and column statistics and bounds update working sets through tiled in-place processing.

On a 35,301-parameter NUERONCE test model, the activation-checkpointed path produced identical loss and parameter gradients to the resident-graph baseline while reducing the reachable forward graph from 1,064 Tensor nodes to 241, a 77.35% reduction. A decomposed three-step training smoke test reduced next-byte loss from approximately 5.88 to 5.19 while paging optimizer state across eight NUERONCE subsystems. A 352,993,825-parameter configuration is constructable and wired to the float32, activation-recomputed, factorized-optimizer path; however, a complete sustained 355M training run has not yet been demonstrated. The work therefore establishes a functioning research runtime and a falsifiable scaling hypothesis, not a completed claim of efficient 355M convergence or superior intelligence.

## 1. Introduction

The practical memory cost of neural-network training is not determined by parameter count alone. Training commonly retains parameters, activations, gradients, optimizer statistics, temporary update arrays, and framework bookkeeping simultaneously. Adam-style optimization may keep first- and second-moment arrays with the same shape as each parameter. Reverse-mode differentiation typically retains the intermediate values required to compute gradients. As models grow, the resulting memory pressure creates a dependency on GPUs, distributed systems, and mature frameworks whose execution assumptions are optimized around accelerator hardware.

NUERONCE asks a narrower question:

> Can a language-model architecture and its training runtime be reorganized so that logical model scale does not require the full training state to remain physically resident at once?

The project does not claim that computation can be eliminated. A 355M-parameter model still requires substantial arithmetic, memory bandwidth, and training data. The aim is to exchange memory residency for recomputation, temporal state ownership, tiled processing, and explicit scheduling. This is especially relevant for independent researchers who have CPU and disk capacity but lack large GPU memory.

The project contains two related contributions:

1. **NUERONCE**, a byte-first model combining local byte perception, dynamic patching, typed recurrent memory, hybrid state-space and attention processing, retrieval context, and byte decoding.
2. **A decomposed Nueronce Engine training runtime**, which separates the mathematical model from execution, activation lifetime, optimizer-state lifetime, checkpoint recovery, and evaluation.

The strongest verified contribution at the present stage is the training-runtime decomposition. Claims about broad intelligence, superior architecture quality, or inexpensive full 355M convergence remain open experimental questions.

## 2. Historical and Technical Context

### 2.1 Reverse-mode differentiation and activation memory

Reverse-mode automatic differentiation computes gradients efficiently for scalar losses but needs information from the forward computation. A conventional dynamic graph keeps references to the operations and values required for backward propagation. This ties activation memory to graph depth and sequence length.

Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016), formalized the compute-memory trade: selected values can be retained while other activations are recomputed during backward. Their work showed that memory can be reduced below linear growth in network depth by accepting additional forward computation. NUERONCE applies the same general principle through an independently implemented Nueronce Engine checkpoint operator specialized around NUERONCE stage boundaries.

Reference: https://arxiv.org/abs/1604.06174

### 2.2 Reversible computation

Gomez et al., *The Reversible Residual Network* (2017), showed that some layer activations can be reconstructed from later states, allowing activation storage to become largely independent of depth for reversible sections. NUERONCE does not currently claim a reversible NUERONCE implementation. Reversible typed-channel transitions are a future research direction, while exact recomputation checkpoints are the present verified mechanism.

Reference: https://arxiv.org/abs/1707.04585

### 2.3 Adaptive optimizer memory

Adam maintains exponentially smoothed first- and second-moment estimates for each parameter. AdamW later clarified that weight decay should be decoupled from the adaptive gradient update. These methods are effective but create optimizer-state memory proportional to the parameter count.

Adafactor, introduced by Shazeer and Stern in 2018, reduces auxiliary state for matrix parameters by approximating second moments from row and column statistics. It also introduced update clipping, increasing second-moment decay, and parameter-scale-relative updates. StreamFactor adopts the row/column factorization principle and update clipping, but is implemented independently within Nueronce Engine and integrated with block-paged state and tiled CPU updates.

References:

- AdamW: https://arxiv.org/abs/1711.05101
- Adafactor: https://arxiv.org/abs/1804.04235

### 2.4 State partitioning and heterogeneous storage

ZeRO demonstrated that parameters, gradients, and optimizer states need not be redundantly resident across every device. ZeRO-Infinity extended the hierarchy across GPU, CPU, and NVMe storage. NUERONCE targets a different setting—a single CPU-first process—but adopts the transferable systems principle that inactive training state should not occupy active memory. Instead of spatial partitioning across GPUs, NUERONCE performs temporal partitioning across NUERONCE subsystems.

References:

- ZeRO: https://arxiv.org/abs/1910.02054
- ZeRO-Infinity: https://arxiv.org/abs/2104.07857

### 2.5 Quantized and low-rank optimizer state

Dettmers et al. showed that blockwise quantization can compress optimizer statistics while preserving performance, with special stability treatment for embeddings. GaLore later proposed low-rank projection of gradients while retaining full-rank trainable parameters. NUERONCE does not currently include a validated quantized first moment or low-rank gradient projection. These are identified as conditional future optimizations that require measured stability and gradient-rank evidence rather than automatic adoption.

References:

- 8-bit optimizers: https://arxiv.org/abs/2110.02861
- GaLore: https://arxiv.org/abs/2403.03507

## 3. NUERONCE Model Organization

The Nueronce Engine NUERONCE implementation follows this data path:

```text
byte identifiers
→ causal byte perception and boundary prediction
→ dynamic segmentation and patch pooling
→ unit embedding
→ typed recurrent memory
→ hybrid core
   • state-space processing
   • local attention
   • sparse/global attention mechanisms
   • optional retrieval context
→ byte decoder
→ next-byte logits
```

The model is byte-first rather than dependent on a learned subword tokenizer. Dynamic segmentation predicts boundaries and pools local byte features into a bounded number of units. A differentiable boundary feature allows language-model gradients to influence the boundary head even though the discrete segment structure itself is detached. Typed recurrent memory and the hybrid core attempt to separate and combine distinct forms of representation, while the decoder maps unit-level processing back to byte predictions.

The architecture includes named subsystems that also provide natural runtime ownership boundaries:

1. `perception`
2. `unit_embed`
3. `memory`
4. `core`
5. `decoder`
6. `ret_byte_embed`
7. `ret_proj`
8. `boundary_proj`

These boundaries are used by the decomposed runtime for validation, checkpoint policy, optimizer-state ownership, and update scheduling.

## 4. Nueronce Engine and Runtime Decomposition

### 4.1 NumPy-native automatic differentiation

Nueronce Engine is a reverse-mode automatic-differentiation engine implemented with Python and NumPy. Each Tensor stores its numeric data, optional gradient, predecessor references, an operation label, and a backward closure. A topological traversal propagates gradients from the output to graph leaves.

The original implementation accepted only an implicit all-ones seed in `Tensor.backward()`. Activation recomputation requires vector-Jacobian products at non-scalar stage outputs, so backward was extended to accept an explicit gradient array. This permits a recomputed stage to receive the gradient of its checkpoint output and propagate it through the local replay graph.

### 4.2 Execution plans

`ExecutionPlan` assigns every model parameter to exactly one `TrainableBlock`. Validation compares the planned parameter identities against the model’s actual parameters and rejects missing or duplicated ownership. This turns parameter ownership into an explicit invariant rather than an assumption.

Each block receives a residency policy:

```text
KEEP        retain the required value
RELEASE     discard after use
RECOMPUTE   retain a boundary value and replay internals during backward
MEMMAP      retain through mapped storage
REVERSIBLE  reconstruct from a later state (future)
```

The current NUERONCE plan marks heavy single-output stages such as memory, core, decoder, unit projection, retrieval projection, and boundary projection for recomputation.

### 4.3 Exact activation-recomputation checkpoint

The checkpoint operator performs the following forward behavior:

1. Execute the stage under `no_grad()` so its internal operations are not attached to the resident graph.
2. Copy the resulting output into a checkpoint Tensor.
3. Attach the stage inputs and trainable parameters as the checkpoint’s graph predecessors.
4. Install a custom checkpoint backward closure.

When backward reaches the checkpoint:

1. Recreate leaf Tensor copies of the stage inputs.
2. Re-execute the stage with gradient recording enabled.
3. Call seeded backward on the replay output using the incoming checkpoint-output gradient.
4. Accumulate replay-input gradients into the corresponding outer input Tensors.
5. Allow parameter gradients to accumulate directly on the shared stage parameters.
6. Release the replay graph when the closure returns and references become unreachable.

This design does not implement hand-written derivatives for every NUERONCE subsystem. It uses the existing Nueronce Engine operation derivatives while controlling graph lifetime at stage boundaries.

### 4.4 Gradient correctness

Two independently instantiated tiny NUERONCE models were initialized with identical parameter values. One used the original resident graph and the other enabled activation checkpointing. They processed the same byte sequence and computed next-byte cross-entropy.

Observed result:

```text
loss difference:             0.0
maximum parameter-gradient difference: 0.0
mismatched parameter gradients:        0
```

The test checks all model parameters. This establishes exact equivalence for the tested configuration and operations. It does not prove equivalence for every future operation or stochastic layer; all additions should extend the parity suite.

### 4.5 Resident graph reduction

The number of Tensors reachable from the loss before backward was counted for the same baseline and checkpointed tiny model:

```text
resident baseline graph:     1,064 Tensor nodes
resident checkpoint graph:     241 Tensor nodes
reduction:                    77.35%
```

Node count is a structural indicator, not a complete peak-RSS measurement. Tensor sizes vary substantially, and NumPy temporaries may not correspond one-to-one with graph nodes. A full memory profiler and per-stage byte accounting remain required for a publication-grade scaling result.

## 5. Temporal Optimizer-State Partitioning

### 5.1 Block state manager

The runtime persists optimizer state separately for each NUERONCE subsystem. At update time it:

```text
loads one block state
→ updates the block
→ atomically saves the block state
→ deletes the in-memory state reference
→ runs garbage collection
→ continues to the next block
```

The state manager also writes a manifest containing block names, parameter counts, residency policies, update frequencies, runtime version, and tape mode.

The present state format uses Python pickle for prototypes. The production design should use typed array files or NumPy memory maps with a JSON manifest so that individual arrays can be mapped without deserializing an entire Python object graph.

### 5.2 StreamFactor

For a matrix parameter with shape `rows × columns`, conventional Adam stores a full second-moment array of the same shape. StreamFactor stores:

```text
row variance:    rows values
column variance: columns values
```

A local second-moment estimate is reconstructed as an outer product normalized by the mean row statistic. Updates are processed in row tiles. Therefore, the optimizer avoids both a full persistent variance matrix and a full temporary normalized-update matrix.

For vectors and scalars, StreamFactor retains full second-moment statistics because these values represent a comparatively small share of total parameters.

The optimizer includes:

- float32 state,
- factorized matrix second moments,
- tiled update processing,
- RMS update clipping,
- decoupled weight decay,
- optional first-moment momentum in the earlier nonpaged implementation,
- state serialization and resume support.

The block-paged runtime currently uses momentum-free factorized state. Quantized blockwise momentum remains a proposed extension.

## 6. 355M-Class Configuration

A measured Nueronce Engine configuration uses:

```text
byte embedding dimension: 128
local dimension:          524
model dimension:        1,048
physical blocks:            6
logical depth:             12
decoder layers:             6
attention heads:            8
```

The resulting model contains:

```text
352,993,825 trainable parameters
```

The large-scale launcher activates:

- float32 parameter and activation creation,
- activation checkpointing,
- StreamFactor,
- sharded training data,
- gradient accumulation configuration,
- tiled optimizer updates,
- resumable atomic checkpoints.

The model has been constructed successfully. Construction is not equivalent to a completed training step, stable convergence, or practical throughput. A full 355M optimization step still requires authoritative peak-memory instrumentation and bounded block-gradient handling. The current checkpoint mechanism reduces resident activation graphs, while parameter-gradient arrays can still accumulate before the later optimizer loop. The next required systems milestone is a same-version block-gradient spool or equivalent mechanism that releases each block’s parameter gradients before replaying all preceding stages.

## 7. Training Demonstration

A 35,301-parameter NUERONCE model was trained through the decomposed runtime on a small next-byte task. The runtime owned eight model subsystems and paged optimizer state by subsystem.

A representative checkpointed run reported:

```text
step 1 loss: approximately 5.88
step 2 loss: approximately 5.46
step 3 loss: approximately 5.19
```

This demonstration establishes that:

- the recomputation path computes finite loss,
- local replay propagates gradients through stage boundaries,
- block optimizer states can be saved and restored,
- parameters change,
- multiple optimization steps execute.

It does not establish language competence, reasoning, generalization, or favorable convergence at large scale.

## 8. Separating Optimization Convergence from Intelligence

A central finding of the project is that declining token loss can be a false positive for intelligence. A model may learn the format and local statistics of training examples without learning transferable rules. For this reason, NUERONCE should not define completion as training loss approaching zero.

The intended evaluation surface includes:

- held-out bits per byte,
- exact-answer task accuracy,
- novel-template accuracy,
- out-of-range arithmetic,
- logical consistency,
- answer-verification accuracy,
- retrieval attribution,
- repetition and degeneration rates,
- channel utilization and specialization probes,
- peak resident memory,
- tokens per second,
- checkpoint-resume equivalence.

The proposed curriculum is staged:

```text
broad byte-level pretraining
→ structural transformations
→ procedural arithmetic and logic
→ verification and correction
→ retrieval grounding
→ conversational supervised fine-tuning
```

This separation matters scientifically. The runtime contribution addresses whether training can fit within constrained memory. The model-learning contribution must separately establish whether NUERONCE gains useful capabilities under controlled data and compute budgets.

## 9. Novelty and Relationship to Prior Work

None of the individual ingredients should be presented as invented in isolation:

- activation recomputation is established prior work,
- factorized second moments are established prior work,
- state partitioning and offload are established prior work,
- tiled numerical kernels are standard systems practice,
- low-rank and quantized optimizer state are active research areas.

The potentially original contribution is the integrated organization and NUERONCE-specific specialization:

1. A NumPy-native dynamic autograd engine with exact seeded stage replay.
2. Training-stage boundaries derived from a byte-first hybrid architecture’s semantic subsystems.
3. Temporal rather than multi-device optimizer-state ownership.
4. Factorized, tiled updates integrated with subsystem paging.
5. A unified execution plan that treats model structure, activation residency, state residency, and update cadence as separate concerns.
6. A path toward typed-channel-aware update schedules and conditional execution.

Novelty must ultimately be established by literature review, independent replication, and empirical comparison—not by repository complexity alone.

## 10. Limitations

The present system has important limitations:

1. **No completed 355M training campaign.** The 355M path is configured and partially validated, but sustained optimization has not been demonstrated.
2. **No authoritative peak-RSS scaling curve.** Graph-node counts and state-size calculations are informative but insufficient.
3. **Parameter gradients remain a major memory term.** Exact block-gradient spooling or immediate safe consumption is not complete.
4. **Perception is not fully checkpointed.** It produces multiple outputs and needs a tuple-aware checkpoint operator.
5. **Python and NumPy overhead.** Dynamic Python graph construction and unfused NumPy operations may make CPU throughput impractical at large scale.
6. **Pickle-based state prototypes.** A typed memory-mapped format is needed for large, reliable state.
7. **No controlled architecture baseline.** NUERONCE has not yet demonstrated superiority to an equal-parameter Transformer or state-space baseline.
8. **Weak current learned behavior.** Earlier small-model experiments showed memorization, repetition, and poor model-only task performance.
9. **No proof of typed-channel specialization.** The architecture names functional channels, but empirical probes must show that they learn distinct roles.
10. **No energy or economic benchmark.** Claims of efficiency require wall-clock and power measurements.

## 11. Required Experiments

### 11.1 Memory scaling ladder

Run identical short-sequence optimization steps for:

```text
112K
11M
35M
90M
355M
```

Record initialization RSS, forward peak, backward peak, update peak, state bytes, disk traffic, wall time, and finite-gradient status.

### 11.2 Gradient-spool correctness

Compare three paths on the same initialization and batch:

1. resident global gradients,
2. recomputed backward with resident gradients,
3. recomputed backward with block gradient spooling.

All resulting parameter updates should agree within declared numerical tolerance.

### 11.3 Optimizer ablation

Compare AdamW where feasible, momentum-free StreamFactor, float32-momentum StreamFactor, and later int8-momentum StreamFactor under equal tokens and initialization.

### 11.4 Architecture comparison

Compare NUERONCE against equal-parameter Transformer and state-space baselines using the same byte corpus, sequence length, training-token budget, and evaluation suite.

### 11.5 Intelligence and generalization

Use procedurally generated training and untouched test distributions. Distinguish seen-template accuracy, novel-template accuracy, and structural out-of-distribution accuracy.

## 12. Reproducibility Contract

A publishable NUERONCE run should preserve:

- repository commit,
- Python and NumPy versions,
- processor model and RAM,
- operating system,
- thread environment variables,
- complete model configuration,
- execution-plan manifest,
- optimizer configuration,
- data manifest and licenses,
- shard order and cursor,
- random-number states,
- checkpoint schema version,
- evaluation datasets and generators,
- raw metrics logs,
- failure and restart history.

Interrupted and uninterrupted runs should produce the same next-step result within declared floating-point tolerance.

## 13. Conclusion

NUERONCE demonstrates that a custom language-model runtime can separate logical model depth from physical training residency without relying on PyTorch, CUDA, or TensorFlow. Its activation-recomputation operator produces exact tested gradients while substantially reducing the resident graph. Its execution plan validates parameter ownership, and its block state manager and StreamFactor optimizer reduce simultaneous optimizer-state residency and matrix second-moment storage.

The project has crossed the boundary from an architecture sketch into a functioning systems prototype. It has not crossed the boundary into a proven efficient 355M training system or a competitively intelligent foundation model. The next decisive evidence is not more architectural description. It is a complete instrumented optimization step at increasing scales, followed by controlled learning and generalization comparisons.

The central hypothesis remains falsifiable:

> A NUERONCE model can be trained with memory bounded primarily by active stages and parameter storage, rather than by simultaneous residency of the complete activation graph and conventional optimizer state.

The repository now contains enough implementation to test that hypothesis directly.

## References

1. Chen, T., Xu, B., Zhang, C., and Guestrin, C. *Training Deep Nets with Sublinear Memory Cost.* 2016. https://arxiv.org/abs/1604.06174
2. Gomez, A. N., Ren, M., Urtasun, R., and Grosse, R. *The Reversible Residual Network: Backpropagation Without Storing Activations.* 2017. https://arxiv.org/abs/1707.04585
3. Loshchilov, I., and Hutter, F. *Decoupled Weight Decay Regularization.* 2017. https://arxiv.org/abs/1711.05101
4. Shazeer, N., and Stern, M. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* 2018. https://arxiv.org/abs/1804.04235
5. Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* 2019. https://arxiv.org/abs/1910.02054
6. Rajbhandari, S., Ruwase, O., Rasley, J., Smith, S., and He, Y. *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.* 2021. https://arxiv.org/abs/2104.07857
7. Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. *8-bit Optimizers via Block-wise Quantization.* 2021. https://arxiv.org/abs/2110.02861
8. Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., and Tian, Y. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* 2024. https://arxiv.org/abs/2403.03507
