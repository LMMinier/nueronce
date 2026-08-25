# NUERONCE Genesis Technical Agent

Experimental continual-learning layer for NUERONCE.

## Current implementation

- persistent semantic/procedural memory outside gradient-trained parameters
- versioned corrections with provenance and confidence
- verification-gated curriculum
- coding, math, physics, and chemistry verifier primitives
- knowledge-gap task planner that refuses to mark a task ready when required skills are unverified

## Architecture contract

The intended neural integration point is after NUERONCE typed recurrent memory and before `HybridCoreStack`. Recalled experience should become a gated latent delta to unit states. Novel lexical symbols should use a copy/pointer-capable decoder path rather than being forced into model weights.

## Current controlled gates

The local development artifact passed 12/12 curriculum checks and 4/4 regression tests before this branch was pushed. Those are engineering smoke/transfer tests, **not** HumanEval, LiveCodeBench, or SWE-bench scores.

## Benchmark rule

Do not report comparisons to major coding models until the same benchmark harness and task set are run. Next targets are HumanEval/MBPP-style isolated code generation and then SWE-bench-style repository repair.
