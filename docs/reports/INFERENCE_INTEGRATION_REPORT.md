# Inference Integration Report

Date: 2026-07-01
Branch: `codex/finalize-coherent-inference`

## Summary

This change replaces the previous effective inference path:

```text
raw prompt -> byte generation -> cleanup/repetition filter -> fallback
```

with a connected path:

```text
user request -> trusted evidence retrieval -> workspace reasoning -> response plan
-> evidence-conditioned model generation -> verification -> at most one targeted regeneration
-> final answer
```

The integration is real at the wiring level: approved evidence is inserted into the canonical prompt and also passed into `NUERONCEModel.generate` as `neighbor_ids` / `neighbor_mask` on every autoregressive decoding step. Reasoning, plan fields, tool outputs, and verifier feedback are now model inputs rather than side-channel trace data.

## Files Changed

- `nueronce/prompting.py`: canonical role-marker prompt format, training/inference/revision formatters, continuation extraction, structured context assembly.
- `nueronce/model.py`: upgraded `NUERONCEModel.generate` with retrieval tensors, continuation-only output, stop sequences, top-k, top-p, repetition penalty, NaN-safe sampling, and optional logprob/entropy scores.
- `nueronce/pipeline.py`: connected retrieval, reasoning, planning, evidence-conditioned generation, structured verification feedback, and one revision pass.
- `nueronce/chat.py`: conversation path now uses canonical prompt assembly and continuation-only generation.
- `nueronce/engine/chat.py`: engine chat context now uses the canonical prompt assembly.
- `nueronce/planning.py`: generic causal renderer now expects continuation-only generation.
- `nueronce/coherent_inference.py`: model-only and tool-assisted probe accounting are separated; surface quality is not reported as correctness.
- `nueronce/training/dialogue_data.py`: SFT examples now use the canonical prompt markers and train on assistant response plus `<|end|>`.
- `nueronce/training/generalization_eval.py`: model-only evaluation uses canonical inference prompts and continuation-only output when available.
- `scripts/run_inference.py`: official CLI entry point for model-only, retrieval pipeline, and tool-assisted inference.
- Tests added/updated under `tests/`.

## Prompt Contract

Canonical format:

```text
<|system|>
{system_message}
<|user|>
{user_request}
<|evidence|>
{trusted_evidence}
<|plan|>
{response_plan}
<|assistant|>
{assistant_response}
<|end|>
```

Training, validation-style encoding, chat, coherent inference, pipeline rendering, and model-only evaluation now share these markers.

## Verification

Focused command:

```bash
python -m pytest tests/test_prompting.py tests/test_model_generate.py tests/test_pipeline_inference_integration.py tests/test_sft.py tests/test_sft_engine.py tests/test_sharded_sft.py tests/test_chat.py tests/test_pipeline_components.py tests/test_coherent_inference.py -q
```

Result: `52 passed` in 41.8 seconds.

Full command:

```bash
python -m pytest -q
```

Result: `258 passed, 2 skipped` out of 260 collected tests in 324.9 seconds.

Environment:

- Python: 3.13.2
- OS: Windows 10.0.19045
- PyTorch: 2.11.0+cpu
- Visible display adapters: Intel HD Graphics 530; NVIDIA NVS 510 with 2 GB adapter RAM
- CUDA in active Python runtime: unavailable (`torch.version.cuda` is `None`; `torch.cuda.is_available()` is false)
- cryptography: 49.0.0

The skipped tests are CUDA/AMP dependent. This means the active PyTorch install is CPU-only, not that the machine has no NVIDIA hardware. The inference integration was verified on the CPU path, which is consistent with NUERONCE's CPU-first research constraint.

## Inference Smoke Runs

Checkpoint used:

```text
checkpoints/nueronce_chat.pt
```

Model-only command:

```bash
python scripts/run_inference.py --checkpoint checkpoints/nueronce_chat.pt --prompt "Explain liberty in one paragraph." --model-only --max-new 40 --json-output --show-trace
```

Observed result:

```json
{
  "answer": "",
  "final_source": "model"
}
```

This does not demonstrate coherent model-only conversation.

Tool-assisted command:

```bash
python scripts/run_inference.py --checkpoint checkpoints/nueronce_chat.pt --prompt "What is 17 plus 25?" --assist-tools --max-new 20 --json-output --show-trace
```

Observed result:

```json
{
  "answer": "17 plus 25 equals 42.",
  "final_source": "tool"
}
```

This is explicitly tool-assisted and is not counted as model reasoning.

Retrieval pipeline smoke:

```bash
python scripts/run_inference.py --checkpoint checkpoints/nueronce_chat.pt --prompt "What does NUERONCE separate?" --use-retrieval --max-new 24 --json-output --show-trace
```

Observed model answer:

```text
the the the the the the
```

The trace showed selected evidence, provenance metadata, reasoning, plan, and verification fields, proving the connected path executes. The answer is not coherent, and the current verifier allowed a repetitive surface-quality failure. That verifier gap remains a limitation.

## What Is Fixed

- Generation no longer returns the full prompt by default.
- Multi-byte stop sequences including `<|end|>` are supported.
- Sampling supports top-k, top-p, repetition penalty, greedy mode, and NaN-safe fallback.
- Retrieval tensors persist through autoregressive decoding.
- Evidence, source IDs, reasoning, plan, and tool outputs are included in renderer prompts.
- Authority-rejected evidence is removed by the evidence gate before generation.
- Verification can trigger exactly one evidence-aware regeneration.
- Model-only and tool-assisted probe metrics are separated.
- Prompt echo regression is covered by tests.

## Known Limitations

- The local `checkpoints/nueronce_chat.pt` checkpoint did not produce coherent model-only output under the canonical prompt format.
- The retrieval smoke run produced repetitive text, despite the path being wired correctly.
- The verifier currently catches evidence/provenance issues but does not reliably reject low-surface-quality repetitive answers.
- The canonical prompt change means strong conversational behavior requires fine-tuning/evaluation on the same prompt format; the existing checkpoint should not be treated as proof of assistant-quality dialogue.
- No new claim-extraction model training was started.
- No claim is made that more parameters or GPU-heavy scaling are necessary yet. The immediate measured bottleneck is prompt-aligned training and evaluation quality, not a proven parameter-count ceiling. The CPU path remains the primary compatibility target.

## Conclusion

The inference architecture is now connected end to end. Evidence, reasoning, planning, retrieval tensors, and verifier feedback are inputs to generation rather than disconnected traces. However, coherent model-only conversation has not been demonstrated with the available checkpoint. The next scientific step is prompt-aligned SFT/evaluation on this exact format, with model-only and tool-assisted results kept separate.
