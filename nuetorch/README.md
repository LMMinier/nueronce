# NUETORCH Rust CPU prototype

NUETORCH is a portable, dependency-free Rust execution prototype for testing
selected systems ideas behind the NumPy NUETORCH trainer on commodity CPUs. It
is a byte-transition learner: the current byte selects an embedding, and that
single hidden representation predicts the next byte. It has no recurrence and
cannot use context older than one byte.

This is **not** the complete 34.4M NUERONCE architecture. It now includes real,
trainable entropy, boundary, dynamic-patching, and entropy-conditioned routing
machinery, but it does not include the full perception, typed-memory, SSM,
attention, feed-forward, retrieval, decoder, or rematerialization stack.

## Trainable objective

For a detached next-byte probability distribution `p`, the entropy target is:

```text
H(p) = -sum_i p_i ln(p_i)
```

The entropy head is a scalar linear projection of the current embedding and is
trained with half mean squared error, `0.5 * (predicted_H - detached_H)^2`. The
boundary head is another scalar linear projection, passed through a sigmoid and
trained with positive-class-weighted binary cross-entropy. Its target is
explicitly:

```text
syntax_target OR entropy_target
```

Syntax targets include whitespace, newlines, sentence punctuation, brackets,
quotes, slashes, and backticks. Entropy targets require normalized target
entropy at or above the global threshold and an absolute change in entropy nats
at or above the relative threshold. The predicted entropy boundary candidate
uses the same two-sided absolute-change rule and can compare against an optional
predicted-entropy exponential moving average.

The combined training objective is:

```text
total_loss = next_byte_cross_entropy
           + entropy_loss_weight * entropy_half_mse
           + boundary_loss_weight * weighted_boundary_bce
```

Next-byte cross-entropy remains the primary language-model objective.

## Entropy-conditioned route

The route gate does not receive the target byte. Its formula is:

```text
gate = sigmoid(a * clamp(predicted_entropy / ln(256), 0, 1)
             + b * predicted_boundary_probability
             + c)
routed_hidden[i] = hidden[i] * (1 + gate * route_scale[i])
```

The routed hidden representation feeds the next-byte output projection. The
gate coefficients and per-channel route scales are trainable, receive gradients
from next-byte cross-entropy, and are included in optimizer checkpoints.

## Dynamic patches

Predicted boundary probabilities split the prediction positions into patches.
A predicted split is suppressed until `patch_min`, a split is forced at
`patch_max`, no empty patch is emitted, and the final partial patch is retained.
The patches are currently reported and tested; they do not yet drive a
hierarchical NUERONCE computation.

## Metrics

- `total_loss`: the full weighted objective above.
- `next_loss` and `bpb`: next-byte cross-entropy in nats and bits per byte.
- `entropy_loss`: unweighted half-MSE for the trainable entropy prediction.
- `boundary_loss`: unweighted positive-class-weighted BCE.
- `entropy_nats`, `entropy_normalized`, `entropy_bits`: detached target entropy
  in nats, divided by `ln(256)`, and divided by `ln(2)`.
- `predicted_entropy_nats`: mean scalar entropy-head prediction.
- `boundary_accuracy`, `precision`, `recall`: thresholded boundary metrics;
  zero-denominator precision or recall is reported as zero, never NaN.
- `predicted_rate`, `target_rate`: fractions of predicted and labeled boundary
  positions.
- patch count/average/minimum/maximum length and forced-boundary count: summary
  of dynamic patches. A final partial patch may be shorter than `patch_min`.
- gate average/minimum/maximum, percentage above `0.5`, and routed gradient
  norm: route utilization and the pre-update gradient norm over route parameters.
- `grad_norm` and `applied_grad_norm`: full gradient norms before and after
  global clipping.

## Build, test, and run

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

Train on a text or binary corpus:

```bash
cargo run --release -- train \
  --input path/to/train.txt \
  --width 256 --seq 128 --steps 500 --warmup-steps 0 --lr 0.001 \
  --entropy-global-threshold 0.75 \
  --entropy-relative-threshold 0.10 \
  --entropy-ema-decay 0.95 \
  --entropy-loss-weight 0.1 \
  --boundary-loss-weight 0.1 \
  --boundary-positive-weight 2.0 \
  --boundary-threshold 0.5 \
  --patch-min 2 --patch-max 16 \
  --checkpoint nuetorch.ntrs
```

Use `--entropy-ema-decay off` to disable the EMA and `--disable-routing` to
disable the route. Rerunning resumes from the checkpoint. A resume rejects
conflicting width, learning rate, thresholds, loss weights, patch bounds, or
routing mode. `--override-config` explicitly accepts runtime configuration
changes, except width because checkpoint tensor shapes cannot be changed.

Inspect without training:

```bash
cargo run --release -- inspect --checkpoint nuetorch.ntrs
```

## Checkpoints and memory reporting

Checkpoint version `NUETRS02` stores every parameter and row/column optimizer
factor, step, learning rate, thresholds, EMA state and decay, loss and class
weights, boundary threshold, patch bounds, and routing mode. Version
`NUETRS01` fails with an explicit unsupported-version error; it is never
silently interpreted as the new layout.

`tensor_storage_bytes` counts parameter, gradient, optimizer-factor, and
reusable activation vectors. It is **not** complete process memory and excludes
the allocator, executable, stack, temporary per-step vectors, corpus bytes, and
operating-system overhead. Actual peak RSS should be measured separately.

The reproducible release-mode measurement, including hardware, Rust version,
warm-up, wall time, throughput, peak RSS, and checkpoint size, is recorded in
[`BENCHMARK.md`](BENCHMARK.md).

## Remaining limitations

- This is a one-byte-context execution prototype, not the full NUERONCE model.
- Dynamic patches do not yet feed hierarchical computation.
- Kernels are scalar Rust loops without BLAS, explicit SIMD, or threading.
- Parameters, gradients, and reductions are FP32.
- No PyTorch `.pt` or Python pickle converter exists.
- Numerical parity with the NumPy engine remains **unproven**; no Rust/NumPy
  forward, gradient, or one-step-update parity suite has passed.
- A prototype benchmark must not be extrapolated to a 35M-parameter model.

The future port still requires RMSNorm/linear parity, causal Conv1D perception,
dynamic-patch integration, recurrent typed memory, SSM scan, sparse/local
attention, gated feed-forward blocks, the byte decoder, activation
rematerialization, and declared-tolerance numerical parity tests.
