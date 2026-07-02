# CFNA turn format ("the turn contract")

One canonical prompt layout, shared byte-for-byte by training and inference.
Single source of truth: `cfna/prompting.py`. Nothing else may define marker
strings — `cfna.training.dialogue_data` re-exports `USER`/`ASSISTANT` from
there, and `tests/test_prompting.py` + `tests/test_chat_format_drift.py` pin
the invariants below.

## Canonical format

Markers (each on its own line, content on the following line(s)):

| Marker | Purpose |
|---|---|
| `<\|system\|>` | system preamble (may be empty — the block is still emitted) |
| `<\|user\|>` | a user turn |
| `<\|evidence\|>` | trusted retrieved evidence, one `[source_id] content (authority=…, authenticity=…)` row per item |
| `<\|plan\|>` | optional response plan / revision instructions |
| `<\|assistant\|>` | an assistant turn; at inference the prompt **ends** with `<\|assistant\|>\n` and the model continues from there |
| `<\|end\|>` | closes an assistant response |

A single training example (`format_training_example`) is laid out as:

```
<|system|>
{system or empty}
<|user|>
{request}
<|evidence|>
{evidence or empty}
<|plan|>
{plan or empty}
<|assistant|>
{response}
<|end|>
```

Multi-turn conversations (`assemble_conversation_prompt`,
`dialogue_data.encode_messages`) interleave `<|user|>`/`<|assistant|>` blocks
between the system block and the trailing `<|assistant|>\n`; when a byte
budget is given, the **oldest turns are dropped first** — system, current
user, evidence, and plan blocks are never truncated.

### Loss mask (SFT contract)

`mask[i]` is True iff byte `i` belongs to an assistant response **including**
its trailing `\n<|end|>\n`. Everything else — system, user turns, evidence,
plan, role markers — is False. Applied per-turn in multi-turn examples.

### Stop sequences

Generation halts on the first of `prompting.STOP_SEQUENCES`:
`<|end|>`, `\n<|user|>`, or `\nUser:` (the last catches legacy-register
regurgitation). `extract_assistant_continuation` applies the same three stops
when trimming raw model output.

## Legacy format

Everything trained before the canonical markers used:

```
User: {message}\nAssistant: {reply}\n
```

No system line when empty, stop on `\n`. Every `sharded_sft` checkpoint up to
and including `checkpoints/micro_cfna_sft_100k` is legacy.

## Checkpoint stamping — why this file exists

The two formats are **not interchangeable at the byte level**. When the shared
tags were repointed from legacy to canonical, the 100k legacy checkpoint's
teacher-forced byte accuracy read 0.524 under canonical prompts vs 0.906 under
its true legacy format — a 38-point *silent* regression that looked like a
weak model (`docs/reports/MICRO_CFNA_SFT_100K_REPORT.md`).

Rules, enforced by `tests/test_chat_format_drift.py`:

1. `cfna.training.sharded_sft.save_checkpoint` stamps
   `meta["prompt_format"]` (`"canonical"` or `"legacy"`, derived from
   `dialogue_data.PROMPT_FORMAT`) into every new checkpoint.
2. Chat loaders (`cfna.microtorch.chat.MicroConversation.resolve_format`)
   read the stamp; **unstamped checkpoints resolve to `"legacy"`**, because
   every checkpoint that predates the stamp is legacy.
3. The legacy prompt path hardcodes the literal `"User: "`/`"Assistant: "`
   strings — never the module-level tags, which now point at the canonical
   markers.
4. Eval must use the checkpoint's own format. Never compare accuracies across
   formats and call the difference model quality.
