# Prompt-Aligned Grounded SFT Report

Date: 2026-07-01
Branch: `codex/prompt-aligned-grounded-sft`

## Conclusion

**D. The experiment was inconclusive because training or data integrity failed.**

More precisely: the data and training path were repaired and verified, but the required 500-1,000 step pilot and 10,000 step full run were not feasible in the active runtime. The active PyTorch install is CPU-only (`torch 2.11.0+cpu`, `torch.version.cuda = None`), and a full-NUERONCE CPU training smoke showed that even tiny diagnostic runs are slow enough that the requested phase cannot honestly be completed in this session.

No architecture changes were made. No VGRFT stages 2-4 were started.

## Starting Checkpoint

Chosen checkpoint:

```text
checkpoints/nueronce_chat.pt
```

Audit result:

- Loads as a PyTorch `NUERONCEModel`: yes
- Parameter count: `11,131,477`
- Training step: `923`
- Last recorded history item: step `900`, train bpb `3.4384`, held-out bpb `3.6043`
- SHA-256: `38b01207c72389254166e5e6fa7be6a867412b1b14ecdb6d2e87f7fd25a74164`
- Prompt format metadata: absent
- Predates canonical prompt format: yes
- Corpus identity in checkpoint: not recorded

Other audited PyTorch checkpoints:

- `checkpoints/baseline.pt`: loads, `11,131,477` params, step `26`
- `checkpoints/nueronce_chat_sft_smoke.pt`: loads, `11,131,477` params, older smoke SFT, also predates canonical prompt metadata

Full audit artifact:

```text
metrics/prompt_aligned/checkpoint_audit.json
```

## Preflight Corrections

Implemented before training:

- Empty model output is marked invalid with `failure_reason="empty_output"`.
- Surface-quality verifier failures now catch empty output, prompt echo, marker leakage, mostly nonprintable output, repeated-word loops, and repeated short-phrase loops.
- Repetitive first drafts trigger the existing single revision pass; a repetitive revision remains failed.
- Structured context truncation now measures UTF-8 bytes, not Python character count, and drops old turns before protected current user/evidence/plan content.

## Dataset

Builder:

```text
scripts/build_prompt_aligned_sft.py
```

Generated output:

```text
data/sft_prompt_aligned/
  train/shard_01.jsonl ... shard_04.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
```

Composition:

- Unique records: `800`
- Train unique records: `640`
- Weighted train records: `2,000`
- Validation records: `80`
- Test records: `80`
- Unique curriculum composition:
  - direct: `320`
  - grounded: `320`
  - abstain/conflict/revision: `160`
- Rendered-example leakage:
  - train/validation overlap: `0`
  - train/test overlap: `0`
  - validation/test overlap: `0`
- Rejected duplicates: `0`
- Manifest SHA-256: `38ff84a041a19ebdbc97df6c966a30aa52c23461f6cd75b9e6293f58f98933f1`

All training examples are rendered through `nueronce.prompting.format_training_example`.

## Training Path

Extended:

```text
scripts/train_sft.py --backend torch --model full-nueronce
```

Implemented features:

- Loads an existing PyTorch `NUERONCEModel` checkpoint and fails if missing.
- Uses response-bytes-only loss including `<|end|>`.
- Uses textual evidence and retrieval tensors for records containing evidence.
- Saves `latest.pt` and `best.pt`.
- Saves config, optimizer, scheduler, RNG state, prompt-format version, starting checkpoint hash, and dataset manifest hash.
- Uses AMP only when CUDA is actually available.
- Logs validation loss, bits/byte, byte accuracy, train loss, gradient norm, and learning rate.
- Guards against silent zero-loss training when `max_len` truncates away all assistant targets.

## Diagnostic Training

The requested 500-1,000 step pilot was not completed. A 5-step diagnostic was run to verify that the path trains:

```bash
python scripts/train_sft.py \
  --backend torch \
  --model full-nueronce \
  --ckpt checkpoints/nueronce_chat.pt \
  --train-dir data/sft_prompt_aligned/train \
  --validation data/sft_prompt_aligned/validation.jsonl \
  --test data/sft_prompt_aligned/test.jsonl \
  --save-dir checkpoints/nueronce_prompt_aligned \
  --metrics-dir metrics/prompt_aligned \
  --max-len 512 \
  --batch 1 \
  --grad-accum-steps 1 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --grad-clip 1.0 \
  --checkpoint-every-steps 5 \
  --periodic-val-every 5 \
  --seed 44 \
  --max-steps 5
```

Diagnostic result:

- Pre-training validation loss: `3.6177`
- Step-5 validation loss: `3.5500`
- Pre-training validation byte accuracy: `0.1963`
- Step-5 validation byte accuracy: `0.1988`
- Test loss at diagnostic checkpoint: `3.6004`
- Test byte accuracy at diagnostic checkpoint: `0.1989`

This proves the training path is active, not that the pilot succeeded.

Checkpoint artifacts produced but not committed:

- `checkpoints/nueronce_prompt_aligned/best.pt`
  - SHA-256: `166dad0633637ccdca887594aa6c0362e544ad59637c933070672f24e6e5aad6`
- `checkpoints/nueronce_prompt_aligned/latest.pt`
  - SHA-256: `ddf9f05e09863cf004a8c74a7b96e8a50bbe98a3fc018ef74edfb7bfb7c97f2e`

## Pilot Format Diagnostic

Artifact:

```text
metrics/prompt_aligned/pilot_results.json
```

This is explicitly a **5-step diagnostic**, not the required pilot gate.

Results on five fixed prompts:

| checkpoint | nonempty generation | valid generation | termination | repetition loops |
|---|---:|---:|---:|---:|
| original `nueronce_chat.pt` | `0.0` | `0.0` | `0.0` | `0.0` |
| 5-step diagnostic | `0.0` | `0.0` | `0.0` | `0.0` |

Both checkpoints generated empty continuations under the canonical prompt.

## Frozen Evaluation Suite

Suite:

```text
data/eval/inference_phase2.jsonl
```

Counts:

- 40 direct conversational prompts
- 30 definitions/explanations
- 30 instruction-following prompts
- 30 summarization/rewriting prompts
- 30 grounded evidence prompts
- 20 insufficient-evidence prompts
- 20 conflicting-evidence prompts
- 20 multi-turn prompts
- 10 coding explanation prompts

Evaluator:

```text
scripts/eval_inference_phase2.py
```

Full-suite evaluation was not completed because generation is too slow in the active CPU-only runtime. Diagnostic limited evaluations were saved:

- `metrics/prompt_aligned/inference_phase2_original_diagnostic.json`
- `metrics/prompt_aligned/inference_phase2_5step_diagnostic.json`
- `metrics/prompt_aligned/grounded_original_diagnostic.json`
- `metrics/prompt_aligned/grounded_5step_diagnostic.json`

The 10-case direct diagnostic produced:

- valid generation: `0.0`
- non-empty generation: `0.0`
- semantic correctness: `0.0`

The one-case grounded diagnostic produced:

- evidence support rate: `0.0`
- retrieval gain: `0.0`
- shuffled evidence drop: `0.0`
- poison acceptance: `0.0`, only because no answer was produced

These are not acceptance-gate results.

## Tests

Command:

```bash
python -m pytest -q --junitxml=metrics/prompt_aligned/pytest.xml
```

Result:

- tests: `272`
- failures: `0`
- errors: `0`
- skipped: `2`
- time: `310.398s`

Environment:

- Python: `3.13.2`
- PyTorch: `2.11.0+cpu`
- CUDA available to active runtime: `false`
- GPU name from PyTorch: `None`
- OS: `Windows-10-10.0.19045-SP0`

## Limitations

- The required 500-1,000 step pilot was not completed.
- The required 10,000 step full run was not started.
- Full frozen-suite model-only evaluation was not completed.
- Retrieval ablations were only run as a one-case diagnostic.
- Current diagnostic checkpoint still emits empty continuations.
- The active runtime cannot use CUDA even though the Windows system may expose NVIDIA hardware; this PyTorch install is CPU-only.
- The result does not prove model capacity is the blocker. It proves that the phase remains unresolved without a real prompt-aligned training run.

## Next Action

Run the same implemented training path in an environment where the 11M-parameter PyTorch model can complete at least the required 500-step pilot. If the pilot still produces empty continuations after enough steps, then investigate data diversity, decoding, and checkpoint quality before considering parameter scaling.
