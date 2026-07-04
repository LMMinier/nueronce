# cfna.cpp — dependency-free C++ inference engine for CFNA

A single-file C++17 engine (`cfna_engine.cpp`, no libraries beyond the C++
standard library) that runs CFNA checkpoints exported from the Python stack.
It is a faithful port of the microtorch reference implementation — every
formula transcribed from `cfna/microtorch/{nn,functional,cfna_blocks,
cfna_model}.py` — and it ships under the project's proof-gate rules:

**Verified claims** (tests/test_cpp_engine.py, 4 tests):
- Dense last-position logits match the NumPy float64 oracle to **≤ 1e-8**,
  on a random model and on the real trained 112K checkpoint.
- Greedy generation is **byte-identical** to the oracle across prompts,
  stop behavior, and the context-window-sliding regime.
- Three-way bench on the trained checkpoint (48 bytes, chat prompt, this
  container's CPU): Python dense 45.8 B/s → Python incremental 162.5 B/s →
  **C++ incremental 657 B/s (14.3× vs dense, including process start and
  model load)**. Re-measure on target hardware before quoting elsewhere.

## Build & run

    make -C cpp
    python scripts/export_cfna_bin.py checkpoints/<ckpt>.pt --out model.bin
    ./cpp/cfna_run model.bin --prompt "User: Hello\nAssistant: " --max-new 64

Flags: `--max-new N` `--max-ctx N` `--temp T` (0 = greedy) `--seed S`
`--logits` (print last-position logits — the parity interface)
`--hex` (hex output, byte-safe) `--bench` (bytes/s to stderr).

## File format (version 1)

Little-endian. Written by `scripts/export_cfna_bin.py`.

    magic   u32  0x414E4643 ("CFNA")     version u32  1
    config  15 × i32   byte_embed_dim, d_local, d_model, p_max,
                       physical_blocks, logical_depth, n_heads, unit_window,
                       decoder_window, decoder_layers, d_state, channel_dim,
                       ret_byte_dim, min_patch, max_patch
    tau f64            trainable_segmentation u8       n_tensors u32
    tensor := ndim u32, dims u32[ndim], data f64[∏dims] (row-major)

Weights are stored in `MicroCFNAModel.parameters()` order (attribute
construction order). The exact sequence the loader expects:

1. perception: byte embed [256,emb]; conv3 w[dl,emb,3]+b; conv7 w[dl,dl,7]+b;
   dilated w[dl,dl,3]+b (dilation 4); norm gain; boundary head fc1 w[128,dl]+b,
   fc2 w[1,128]+b
2. unit embedder: proj w[d,dl]+b; norm gain
3. typed memory: forget/write/read/cand w[S,d+S]+b (S = 7·channel_dim);
   readout w[d,S]+b — the per-channel retention constants
   (0.98,0.97,0.95,0.99,0.90,0.999,0.98) are **not** exported (they are a
   fixed Tensor, not a Parameter) and are hardcoded in the engine
4. per hybrid block × physical_blocks: norm1, norm2 gains; SSM (in_proj
   w[2·di,d]; depthwise conv w[di,1,4]+b; x_proj w[dtr+2n,di]; dt_proj
   w[di,dtr]+b; A_log[di,n]; D[di]; out_proj w[d,di]) with di = 2·d_model,
   dtr = max(1, d_model//16); local attn q,k,v,o w[d,d]; sparse attn q,k,v,o;
   retrieval attn q,k,v,o (loaded, unused — retrieval-free path); gated FFN
   up/gate/down; router fc1 w[d,4d]+b, fc2 w[4,d]+b
5. decoder: byte embed [256,d]; per layer × decoder_layers: norm1, self attn
   qkvo, norm2, cross attn qkvo, norm_r, ret attn qkvo (unused), norm3,
   gated FFN; final norm gain; head w[256,d]+b
6. tail: ret byte embed [256,rbd]; ret proj w[d,dl+rbd]+b; boundary proj
   w[d,1]+b

## Numerics contract (why parity is tight)

Everything is float64, matching the NumPy oracle's default dtype, so parity
is ~1e-9-tolerance rather than the looser bound a float32 build would force.
Two intentional oracle quirks are mirrored exactly: attention's
`masked_softmax` applies **no max-shift** (`exp(score)` directly, masked
entries exactly 0, denominator +1e-30), while the router softmax **does**
subtract the row max; and the sparse top-k keeps score **ties** at the k-th
value, exactly like the reference's `masked >= kth`.

## How generation is fast

The default path is a port of `cfna/microtorch/incremental.py`: the unit
stack (memory + hybrid core) is recomputed only when a patch completes;
per-byte work is a ~25-byte perception window plus a decoder pass over the
last `decoder_layers × decoder_window` bytes (the decoder's exact stacked
receptive field — it has no absolute positions, so this is lossless).
`--logits` bypasses all caching and runs the dense forward for parity runs.

## Honest limitations (documented, not hidden)

- **No retrieval context** — retrieval-conditioned generation stays on the
  Python dense path.
- **Batch = 1**, CPU only, no SIMD beyond compiler auto-vectorization.
- **Sampled (non-greedy) text uses `std::mt19937`**: same distribution as
  the Python path, different random stream — greedy is the parity mode.
- **float64 weights only** (v1). float32/int8 quantized formats are future
  work and, per project rules, ship only with their own parity+quality gate.
- Loads microtorch/`sharded_sft`-format checkpoints via the exporter; the
  torch `CFNAModel` shares the architecture but its checkpoints must first
  be converted to microtorch format (same tensor semantics).
