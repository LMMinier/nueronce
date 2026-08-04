# NUETORCH v0.2.0 CPU benchmark

Measured on 2026-07-26. This is a benchmark of the byte-transition execution
prototype only. It must not be extrapolated to the incomplete 35M architecture.

## Environment and configuration

| Measurement | Value |
|---|---:|
| CPU | Intel Xeon Platinum 8370C @ 2.80 GHz |
| Logical cores available | 3 |
| Rust | `rustc 1.89.0 (29483883e 2025-08-04)` |
| Width | 64 |
| Sequence length | 128 bytes |
| Warm-up steps | 25 |
| Measured steps | 750 |
| Parameters | 33,221 |

## Command

```bash
target/release/nuetorch train \
  --input ../origin.txt --width 64 --seq 128 \
  --steps 750 --warmup-steps 25 --log-every 250 --lr 0.001 \
  --checkpoint /tmp/nuetorch-benchmark-v2.ntrs
```

The command was launched by a Python wrapper using `time.monotonic()` and
`resource.getrusage(resource.RUSAGE_CHILDREN)` because GNU `/usr/bin/time` was
not installed in the test environment.

## Results

| Measurement | Value |
|---|---:|
| Internally timed measured-step wall time | 6.005 s |
| External process wall time | 6.230951 s |
| Warmed-up throughput | 124.886 steps/s |
| Warmed-up byte throughput | 15,985.385 bytes/s |
| Reported tensor storage | 273,240 bytes |
| Peak RSS | 12,216 KiB |
| Checkpoint size | 137,523 bytes |

The internal timer excludes model construction, corpus loading, the 25 warm-up
steps, and checkpoint writing. The external timer includes the complete child
process. `tensor_storage_bytes` is not process memory; peak RSS is the relevant
whole-process observation available from the operating system.
