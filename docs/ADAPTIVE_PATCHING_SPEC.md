# Adaptive (entropy-driven) byte patching — spec & experiment protocol

**Status:** machinery built and tested (`nueronce/adaptive_patching.py`,
`tests/test_adaptive_patching.py`, 7/7 green). **Not yet wired into the model
defaults** — activating it changes the boundary-head training target, which
changes the model and breaks `base_35m` checkpoint resume. Apply the
integration diff below only as a **fork after the in-flight `base_35m` run
lands a clean baseline.**

## The one-line thesis this proves

> Dynamic patching that allocates compute by *information* (next-byte entropy),
> not *syntax* (whitespace), spends fewer units on predictable text at equal
> quality — a directly measured compute win.

Current patcher = SpaceByte-tier (syntactic: boundary at word onset after a
space/punctuation byte, `nueronce/segment.py:boundary_targets`). This upgrade =
BLT-tier (informational: boundary where the model is surprised).

## What is already built (no integration needed to use these)

| Piece | What it does |
|---|---|
| `ByteEntropyHead` | cheap standalone causal byte LM → per-position next-byte logits |
| `predictive_entropy(logits)` | per-position Shannon entropy H_t (nats), AMP-safe |
| `entropy_boundary_targets(logits, mode=...)` | drop-in replacement for `segment.boundary_targets`; `global` (H > θ) or `relative` (H rising > θ, BLT approx-monotonic) |
| `patch_stats(seg_ids, byte_mask)` | bytes/unit + units/byte — the efficiency numbers |

`patch_stats` is usable **right now**, unmodified, to measure the current
run's baseline bytes-per-unit — do that first (see Protocol step 0).

## Post-run integration diff (apply as a fork, do NOT apply mid-run)

The change is deliberately small and confined to two seams. It preserves
default behavior when the new flags are off, so the config-parity test and the
existing presets stay green *only if you mirror the fields in both configs*.

### 1. `nueronce/model.py` — `ModelConfig`, add fields (mirror in `engine/nueronce_model.py`)

```python
# adaptive patching (default off = current syntactic behavior)
boundary_target_mode: str = "syntax"      # "syntax" | "entropy_global" | "entropy_relative"
entropy_head_dim: int = 64
entropy_global_theta: float = 2.5
entropy_relative_theta: float = 0.2
```

> Both `ModelConfig` (torch) and `NueronceConfig` (engine) must get these
> fields identically or `tests/test_config_presets.py` fails. That test is the
> guard; run it after editing.

### 2. `nueronce/model.py` — `NUERONCEModel.__init__`

```python
from .adaptive_patching import ByteEntropyHead
self.entropy_head = (ByteEntropyHead(c.entropy_head_dim)
                     if getattr(c, "boundary_target_mode", "syntax") != "syntax" else None)
```

### 3. `nueronce/model.py` — `NUERONCEModel.loss`, swap the boundary target

```python
mode = getattr(self.cfg, "boundary_target_mode", "syntax")
if mode == "syntax":
    b_target = boundary_targets(byte_ids, self._syntax)          # unchanged default
else:
    from .adaptive_patching import entropy_boundary_targets
    ent_logits = self.entropy_head(byte_ids)
    b_target = entropy_boundary_targets(
        ent_logits,
        mode="global" if mode == "entropy_global" else "relative",
        global_theta=self.cfg.entropy_global_theta,
        relative_theta=self.cfg.entropy_relative_theta,
    )
    total = total + F.cross_entropy(                             # keep the entropy head learning
        ent_logits[:, :-1].reshape(-1, 256), byte_ids[:, 1:].reshape(-1))
```

The boundary head still trains against `b_target` exactly as today; only the
*target* changed. The greedy scan (`segment_ids_from_boundaries`) is untouched.

### 4. Presets — add ONE new preset, do not edit `base_35m`

Add e.g. `base_35m_entropy` = `base_35m` fields + `boundary_target_mode="entropy_global"`.
Keeping `base_35m` byte-identical means the baseline checkpoint stays reusable.

## Experiment protocol (the thesis-defining A/B)

**Step 0 (now, safe):** on the current syntactic checkpoint, log
`patch_stats(...)` over the held-out set → **baseline bytes/unit + held-out bpb**.

**Step 1 (after run):** fresh-train two models, identical everything except
the boundary target:
- **A** — `base_35m` (syntax boundaries), same corpus, same steps, same seed.
- **B** — `base_35m_entropy` (entropy boundaries).

**Step 2 — report the four numbers that make or break the thesis:**

| metric | A (syntax) | B (entropy) | what "win" looks like |
|---|---|---|---|
| held-out bpb | — | — | B ≤ A (quality held) |
| mean bytes/unit | — | — | **B > A** (fewer units = less core compute) |
| core FLOPs / input byte | — | — | **B < A** (the efficiency claim) |
| gate / phase2 pass rate | — | — | B ≥ A (capability held) |

**The fundable result:** B holds quality (bpb, gate) while processing
**materially fewer units per byte** than A. That is "we spend compute only
where the information is," as a measured graph — the compute-vs-quality proof
the efficiency thesis needs, expressed as an architectural feature.

**Honest failure modes to report if they happen:**
- entropy boundaries *hurt* bpb (the cheap head's entropy isn't good enough at
  35M scale) — a real negative result, worth publishing.
- bytes/unit doesn't move (θ mistuned, or text isn't predictable enough at this
  corpus scale) — tune θ on held-out, report the sweep.
- entropy-head cost eats the unit savings — report net FLOPs, not just unit
  count; the head must be << the main model to be worth it.

## Deferred

- Engine (NumPy) port of `ByteEntropyHead` for the CPU-training story — only
  needed once the torch A/B shows a win; spec'd but not built.
- Wiring entropy into the `unc` typed-memory channel (`types.py`) — natural
  next step once boundaries are entropy-driven, but out of scope for the A/B.
