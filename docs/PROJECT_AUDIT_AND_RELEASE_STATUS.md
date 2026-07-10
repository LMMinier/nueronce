# NUERONCE Project Audit and Release Status

**Audit date:** July 2026  
**Project:** NUERONCE / Cognitive Fractal Neural Architecture (CFNA)  
**Author:** Luis M. Minier

## Status labels

- **Verified:** implemented and directly tested in the repository.
- **Implemented:** code exists, but the decisive scale or quality experiment is incomplete.
- **Partial:** a meaningful subset works, but the full claim is not established.
- **Proposed:** design direction only.
- **Falsified/unsupported:** current evidence does not support the claim.

## Breakthrough and claim ledger

| Claim or contribution | Status | Current evidence | Missing proof |
|---|---|---|---|
| NumPy-native reverse-mode autograd engine | Verified | MicroTorch Tensor graph, seeded backward, gradient tests | Broader operation coverage and performance benchmarking |
| Full CFNA architecture runs without PyTorch in its call graph | Verified | MicroCFNAModel implements perception, segmentation, memory, hybrid core, retrieval, decoder | Independent reproduction |
| Byte-first model with dynamic patching | Verified as implementation | Discrete segmentation, patch pooling, boundary head | Demonstrate quality or efficiency advantage over fixed bytes/tokens |
| Differentiable boundary influence despite hard segmentation | Verified as implementation | Boundary feature carries LM gradient through boundary projection | Ablation proving benefit |
| Typed recurrent channels | Implemented | Named channel architecture and recurrent memory code | Probes showing distinct learned semantic roles |
| Hybrid state-space, local attention, sparse/global processing | Implemented | Hybrid core modules and tests | Equal-budget baseline and true operation-count measurement |
| Retrieval-aware generation path | Verified as plumbing | Retrieval context reaches core and decoder | Reliable model-only retrieval use and attribution accuracy |
| Verification/revision mechanism | Verified as plumbing | Revision can be triggered in inference path | Learned verification accuracy and reduced factual error |
| 11M, 35M, 90M MicroTorch presets | Verified as construction/configuration | Parameter-count tests and constructability | Sustained training runs |
| 355M-class MicroTorch configuration | Verified as construction/configuration | 352,993,825-parameter configuration measured and wired | Complete instrumented optimization step and sustained run |
| Float32 large-scale training policy | Implemented | Large-scale launcher selects float32 while tests retain float64 | Full numerical-stability campaign |
| Factorized optimizer second moments | Verified on small models | StreamFactor row/column statistics and update smoke tests | Equal-budget convergence comparison against AdamW/Adafactor |
| Tiled in-place optimizer updates | Verified as implementation | Row-tiled matrix update path | Peak temporary-allocation profiling at scale |
| Block-paged optimizer-state ownership | Verified on small models | Per-subsystem state files, manifest, multi-step resume behavior | Typed memmap format and large-state stress test |
| Exact activation recomputation | Verified on tested model | Identical loss and zero parameter-gradient difference | Multi-output and stochastic-stage coverage |
| Reduced resident forward graph | Verified on tested model | 1,064 to 241 reachable nodes, 77.35% reduction | Peak-RSS and byte-level memory measurements |
| Local-stage backward tape | Verified for checkpointed stages | Stage replay receives seeded output gradient and propagates input/parameter gradients | Immediate gradient disposal across the complete model |
| Full-gradient memory bounded by largest block | Not yet proven | Recompute infrastructure exists | Block-gradient spool or safe immediate consumption |
| No-GPU full 355M training | Unsupported at present | Model and runtime components exist | Complete 355M step, throughput, stability, and convergence evidence |
| Training without heavy compute | Unsupported if interpreted literally | Memory can be traded for recomputation | Arithmetic cost remains substantial and must be measured honestly |
| Superior intelligence/reasoning | Unsupported at present | Small procedural tasks can train; earlier generalization is weak | Untouched OOD reasoning benchmarks and baselines |
| New foundational-model substrate | Partial research claim | Distinct architecture plus custom runtime | Demonstrated capability/efficiency advantage and external review |
| Scientifically novel integrated runtime | Plausible, not established | Unusual integration of CFNA-aware decomposition, temporal state ownership, and MicroTorch replay | Formal literature comparison, publication, replication |

## Verified engineering milestones

1. Clean-checkout MicroTorch imports were repaired.
2. The balanced SFT builder was made directly runnable.
3. The custom MicroTorch backend was restored.
4. A 352.99M parameter CFNA configuration was identified.
5. Large-scale float32 policy was introduced.
6. StreamFactor was implemented.
7. Generic optimizer serialization was introduced.
8. A dedicated 355M launcher was added.
9. Model subsystems were assigned explicit parameter ownership.
10. Block optimizer state was separated and paged.
11. Seeded non-scalar backward was added to MicroTorch.
12. Exact activation recomputation checkpoints were added.
13. Heavy CFNA stages were converted to recomputation boundaries.
14. Gradient equivalence against the resident graph was tested.
15. Resident graph node count was reduced by 77.35% on the test model.
16. Multi-step decomposed MicroTorch training produced declining finite loss.

## Current release classification

### What NUERONCE is now

A legitimate, functioning **research prototype for a byte-first hybrid language model and CPU-first decomposed training runtime**.

### What NUERONCE is not yet

- a trained competitive foundation model,
- a proven low-compute replacement for GPU training,
- a demonstrated general-intelligence system,
- a production-ready training framework,
- a validated scientific breakthrough at 355M scale.

## Release readiness by layer

| Layer | Readiness | Grade |
|---|---:|---:|
| Architecture implementation | Research-complete enough for experiments | B+ |
| MicroTorch autograd correctness | Strong prototype | B+ |
| Low-memory optimizer | Working prototype | B |
| Activation-memory system | Working and exactly tested on small model | A- |
| State/checkpoint system | Prototype; needs memmap schema | B- |
| 355M scalability | Constructable but not trained | C |
| Data quality and breadth | Insufficient for broad intelligence | C- |
| Model behavior/generalization | Weak | D+ |
| Reproducibility | Improving; no complete clean-machine benchmark artifact | B- |
| Scientific evidence | Promising internal evidence, limited external proof | C+ |
| Production readiness | Early | D |

## Mandatory final gates

### Gate 1 — Memory correctness

- tuple/multi-output recomputation,
- block-gradient spooling,
- same-parameter-version replay,
- typed memory-mapped state,
- peak-RSS instrumentation.

### Gate 2 — Scale proof

Complete one step at 11M, 35M, 90M, and 355M with:

- finite loss and gradients,
- peak RAM,
- step time,
- disk traffic,
- checkpoint/resume equivalence.

### Gate 3 — Learning proof

Train a stable 35M model through:

- broad pretraining,
- procedural reasoning,
- verification,
- retrieval,
- SFT.

Pass held-out and novel-template tests.

### Gate 4 — Comparative proof

Compare against equal-parameter Transformer and state-space baselines with identical data and compute accounting.

### Gate 5 — External proof

- archived release,
- paper/preprint,
- reproducible scripts,
- independent run or review.

## Honest completion estimates

| Goal | Estimated completion |
|---|---:|
| Research codebase / architecture lab | 80% |
| Low-memory MicroTorch runtime | 65% |
| Reproducible 355M one-step demonstrator | 45% |
| Stable 35M language model | 40% |
| Competitive small foundation model | 25% |
| Proven new training paradigm | 20% |
| Production framework | 15% |

These percentages are engineering judgments, not measured scientific quantities.

## Release statement

NUERONCE should be released as an **experimental research system**. Public descriptions should use the following language:

> NUERONCE implements and tests a CPU-first, NumPy-native training runtime for CFNA models using exact activation recomputation, factorized tiled optimization, and temporal optimizer-state partitioning. Small-model tests verify gradient equivalence and reduced graph residency. A 352.99M-parameter configuration is constructable and wired to the runtime, but full 355M training efficiency and convergence remain unproven.

That statement is strong, accurate, and defensible.
