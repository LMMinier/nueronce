# NUERONCE Engineering Manual

*A complete, plain-language guide to everything in this repository: what it
is, how every part works, what has actually been built and proven, what
failed and what we learned, and where it goes next. Written 2026-07-23,
the day the pipeline gate passed 31/32.*

Every claim in this manual maps to a file, a test, or a logged run in this
repo. Where something is unproven or broken, it says so. That honesty rule
is itself part of the engineering system here (see §10).

---

## 1. What this project is

NUERONCE is a **language model built completely from scratch** — not a
fine-tune of someone else's model, not a wrapper around an API. Every layer
of it was written by hand in this repository: the math operations, the
neural network layers, the training loops, the data pipeline, the chat
interface, and the evaluation system.

**The core idea in one sentence**: instead of using the standard recipe
every big AI lab uses (a Transformer over word-pieces called "tokens"),
NUERONCE reads **raw bytes** — the actual 0-255 numbers files are made of —
and learns to group them into meaningful chunks on its own, then thinks
about those chunks with a hybrid engine that mixes four different reasoning
mechanisms.

**The mission** (from `PLAN.md`): take it from "produces English-shaped
text" to a real conversational assistant, climbing a ladder of sizes — 11
million parameters → 35M → 90M → 337M — with a pass/fail gate at every
rung, so no step is taken on hope.

**The philosophy**: this is compute-efficiency research. The project's
whole essence is doing the most with small hardware — free Colab sessions,
a desktop, CPU containers — using careful engineering instead of a GPU
farm. Not cheating physics; using the power of broke.

---

## 2. Why bytes instead of tokens (and what's different here)

Normal LLMs first run your text through a *tokenizer* — a fixed dictionary
that chops "understanding" into pieces like "under", "stand", "ing". That
dictionary is frozen before training ever starts, it's language-biased, and
typos or rare words shatter into junk pieces.

NUERONCE has **no tokenizer**. It sees `h`=104, `e`=101, `l`=108... and
must learn everything from there. That is much harder — the model spends
capacity learning spelling — but it means:

- nothing is out-of-vocabulary, ever (any language, any code, any typo);
- the model *learns* where meaningful boundaries are instead of being told;
- the whole pipeline is simpler and self-contained (no tokenizer files).

The catch, and this repo is honest about it: byte models pay a price in
efficiency at small scale, and the exact-spelling errors you'll see in §9
("aple" instead of "apple") are the visible cost of working at the byte
level.

---

## 3. The architecture, piece by piece

The model (`nueronce/model.py`, class `NUERONCEModel`) is a pipeline of five
stages. Think of it as an assembly line for understanding text.

### 3.1 Byte perception (`nueronce/blocks.py`, `BytePerceptionEncoder`)
A small convolutional network (the same family of nets used in image
recognition) slides over the raw bytes and produces a richer description of
each byte in context — "this byte is probably the start of a word," "this
looks like punctuation," etc.

### 3.2 Dynamic patching (`nueronce/segment.py`)
This is the tokenizer-replacement. A learned "boundary head" looks at each
position and outputs a probability: *is a meaningful unit ending here?*
Where that probability crosses a threshold, the model cuts. The chunks
("patches") between cuts get pooled into single unit vectors. Patch sizes
are bounded (min 3, max 24-ish bytes depending on preset), and the boundary
head is trained two ways: a self-supervision signal from word starts, and —
important — the main language loss itself flows gradient into it, so the
model learns to cut wherever cutting helps it predict better.

### 3.3 Typed recurrent memory (`nueronce/blocks.py`, `TypedRecurrentMemory`)
As units stream past, a memory cell carries a running summary forward. It's
"typed": the state is split into **seven named channels** — semantic,
structural, goal, evidence, uncertainty, authority, procedural — each with
its own fixed retention rate (how slowly it forgets; e.g. the authority
channel retains at 0.999 per step, the uncertainty channel at 0.90). Honest
status: the channels are *structurally* separate, but nothing yet forces
each channel to specialize to its name — that's declared intent, listed in
Known Limitations, not a proven property.

### 3.4 The hybrid core (`nueronce/blocks.py`, `HybridCoreStack`)
This is where "thinking" happens. Each unit flows through **four parallel
paths simultaneously**:

1. **Selective state-space path (SSM)** — a modern recurrent mechanism
   (same family as Mamba) that's cheap and good at long-range flow;
2. **Local windowed attention** — looks carefully at nearby units;
3. **Sparse global attention** — picks the top-k most relevant units from
   anywhere in the context and attends only to those;
4. **Retrieval cross-attention** — a slot where externally retrieved
   documents can be attended to (wired into training, see §6.4).

A **router** learns, per position, how much to trust each path — blending
them with learned weights rather than picking one. And the whole stack is
reused: a model with 3 physical blocks and logical depth 4 runs its blocks
multiple times in a loop, getting 12 layers of "thinking" for 3 layers of
memory cost. That reuse trick is one of the main parameter-efficiency
plays.

One real bug lives in this system's history, and the tests caught it: the
router originally averaged over the *whole* sequence to compute its mixing
weights, which leaked future information into the past. The causality test
(`tests/test_model_learns.py::test_model_is_causal` — edit a future byte,
assert *zero* change in past predictions) failed, the router was rewritten
per-position, and the test now measures exactly 0.0 leakage.

### 3.5 Byte decoder (`nueronce/blocks.py`, `ByteDecoder`)
Finally, a decoder turns unit-level thoughts back into byte predictions —
"given everything so far, the next byte is probably `e`". It cross-attends
to completed units and uses windowed attention over recent bytes. Each
byte is masked away from its own in-progress patch (it can't peek at a
chunk that isn't finished yet), which is also what makes fast incremental
generation possible (§7).

### 3.6 The size ladder (`CONFIG_PRESETS` in `nueronce/model.py`)

| Preset | Parameters | Meaning |
|---|---|---|
| `chat_11m` | 11,131,477 | The current working rung. All real results are here. |
| `base_35m` | ~34.4M | Next rung; unlocks when 11M passes its gates. |
| `base_90m` | ~92.1M | Two rungs out. |
| `large_337m` | ~337M | Builds and runs (verified by test), never trained. |

The parameter counts are verified by construction in
`tests/test_config_presets.py` — the test literally builds each model and
counts.

---

## 4. Two implementations of the same brain

The repo contains the entire architecture **twice**, on purpose:

1. **The PyTorch version** (`nueronce/`) — uses PyTorch only as a fast
   math engine (matrix multiply + automatic gradients). Every layer is
   hand-built; no `nn.Transformer`, no pretrained anything. This is the
   version that trains at real speed and is what all the results below
   come from.

2. **The from-scratch engine** (`nueronce/engine/`) — a complete
   automatic-differentiation framework written in pure NumPy: its own
   `Tensor` class, its own backpropagation, its own optimizers (SGD,
   AdamW, and a memory-light Adafactor-style `BlockStreamFactor`), and the
   full NUERONCE model rebuilt on top of it. Why? Three reasons: proof
   that we understand every gradient (nothing is a black box), portability
   (runs anywhere NumPy runs), and verification — the engine's gradients
   are checked against finite differences (max error ~1e-8) *and* against
   PyTorch on identical inputs. It is slower than PyTorch and is **not**
   used to train the real model; it's the transparent reference, and the
   deployment format (`scripts/convert_torch_checkpoint_to_engine.py`
   converts trained checkpoints into engine `.pkl` files).

   The engine also has a **decomposed CPU runtime**
   (`nueronce/engine/runtime.py`): it can train with only one block's
   optimizer state in memory at a time (rest paged to disk) and rebuild
   activations on demand instead of storing them — designed for training
   big models on small machines.

A sister branch (`claude/cfna-neural-core-verify-ldvgn3`) additionally
carries a zero-dependency **C++ inference engine** and an equal-compute
benchmarking kit under the project's earlier "CFNA" name; the two branches
share the same architecture but have diverged in packaging. This manual
covers the default branch.

---

## 5. The turn contract: how conversations are formatted

Training and chat must speak **exactly** the same format, byte for byte, or
the model gets evaluated in a language it wasn't taught. This repo learned
that the hard way: an early re-pointing of the format tags silently cost a
checkpoint 38 points of accuracy (0.906 → 0.524). The fix became
infrastructure:

- **Canonical markers** (`nueronce/prompting.py`): `<|system|>`,
  `<|user|>`, `<|evidence|>`, `<|plan|>`, `<|assistant|>`, `<|end|>`.
  A full prompt is those blocks in order; the model's job is to continue
  after `<|assistant|>` and stop at `<|end|>`.
- **Response-only loss masking** (`nueronce/training/dialogue_data.py`):
  during instruction tuning, the model is only graded on the bytes *it*
  should produce (the assistant's reply, including the stop marker), never
  on predicting the user's words back.
- **Checkpoint stamping**: every fine-tuned checkpoint records which
  format it was trained in; the chat loader auto-detects and refuses to
  mix formats (`tests/test_chat_format_drift.py`).
- **Stop sequences**: generation halts on `<|end|>`, a bare `<|` (so a
  malformed tag can never leak into a reply), or a new user tag.

The proof that training and inference agree byte-for-byte is now an
automated test (`tests/test_foundational_recovery_pipeline.py`), not a hope.

---

## 6. The training system

### 6.1 Data: license-first corpus building
Everything the model reads is provenance-tracked. `nueronce/corpus/stack.py`
is a registry of ~22 vetted sources (Cosmopedia educational text, open-web
math, permissively-licensed code, Project Gutenberg books, Wikipedia, PMC
medical papers...), each entry carrying its license, URL, and intended
training phase. Sources with problematic licenses are explicitly excluded
with the reason written down. The repo root also carries ~24 MB of
owner-curated public-domain books (Dostoevsky, Cervantes, Locke, Newton,
Darwin, grammar and psychology texts) that double as an **offline corpus**:
`scripts/build_local_corpus.py` turns them into a training set with zero
network access — which is exactly how training runs inside locked-down
containers (it's running that way right now, §9.4).

Corpus rules that came from real failures: no single document may dominate
(documents are sampled uniformly, so a giant book doesn't swamp the mix);
train/validation split is by *whole document* (held-out books the model
never saw, so the validation score measures generalization, not
memorization); and no single content register may exceed ~25% of an
instruction-tuning mix (a run where arithmetic hit 77% poisoned the whole
model — written down, never repeated).

### 6.2 Base pretraining (`scripts/train_checkpoint.py`)
The first stage: the model reads the corpus and learns to predict the next
byte. The score is **bits per byte (bpb)** — how many bits of surprise per
byte, lower is better. Random guessing = 8.0. The trainer is built for
hostile environments: time-budgeted (`--minutes`), fully resumable
(`--resume` restores weights, optimizer, step count, and history),
atomic saves (a crash can never half-write a checkpoint), AMP fp16 support
for GPUs, and — a bug fixed this cycle — the learning rate you pass on
resume actually takes effect (the optimizer used to silently restore the
old one, breaking the ladder in §6.5).

### 6.3 Instruction tuning / SFT (`scripts/train_forgeloop_sft.py`)
The second stage: teach the pretrained model to *respond* rather than just
continue text. It trains on (prompt → response) pairs with response-only
masking (§5). Its safety features were battle-earned: atomic `latest.pt`
every N steps, a separate `best.pt` that only advances when validation
improves, safe Ctrl-C handling that saves before exiting
(`tests/test_safe_interruption.py`), RNG state saved for exact
reproducibility, and — new this cycle — its validation report now includes
**free-run-predictive metrics** (per-byte accuracy, first-8-bytes loss,
and a `gate_ready` flag) because aggregate loss alone was proven unable to
predict whether generation works (§9.2).

The SFT data itself: ~11.5K conversational pairs from OpenAssistant
(post-cleaning — see §9.3), a hand-written turn-taking set, plus
**ForgeLoop** — this project's original software-engineering scaffold
(CONTRACT → MAP → PLAN → ACT → OBSERVE → CRITIQUE → REVISE → VERIFY →
LEDGER → MEMORY), a supervised format that teaches bounded, tool-honest
engineering behavior: never claim a test result that wasn't observed.

### 6.4 Retrieval is trained, not bolted on
The core's retrieval path receives gradients during training
(`nueronce/retrieval_train.py`, verified by
`tests/test_retrieval_training.py`), so using retrieved documents is a
learned skill, not an inference-time hack. The current retriever itself is
heuristic (hashed n-grams, not a learned dense embedder) — Known
Limitation, written down.

### 6.5 The convergence method: the LR ladder
Training runs use a simple, disciplined schedule: train at a learning rate
until the held-out score stops improving for ~18 evaluations (a plateau *at
that rate*), then drop the rate 3× and continue. Converged = a further 10×
drop buys almost nothing. This is automated in
`scripts/sandbox_train_driver.sh` (§8).

### 6.6 The gates (pass/fail, pre-registered)
Nothing advances on vibes. Written before the runs:

- **SFT unlock**: base held-out bpb < 1.8 — because experiments here
  proved SFT *shapes* competence but cannot *create* it (a weak base +
  great SFT data still yields garbage; measured twice).
- **35M rung acceptance**: bpb ≤ 1.5, choice-ranking beats chance by ≥15
  points on 3+ subjects, ≥60% valid stop-terminated answers, 5/5
  grammatical transcripts.
- **The sealed proof gate** (`scripts/eval_foundational_proof_gate.py`):
  eight fixed tasks (polite rewrite, explain a shadow, 17+26, fix a Python
  off-by-one, clock arithmetic, make a search plan, extract a fact from
  trusted-vs-untrusted evidence, and honestly abstain when the evidence
  doesn't contain the answer). Graded on *generated output only*, greedy,
  deterministic. **Sealed** means: nobody may edit these eight cases or
  train on them. Pass = ≥75% overall, every domain nonzero, both
  evidence-honesty cases correct.

---

## 7. Fast generation: the incremental engine

Naively, generating each new byte re-runs the whole model over the whole
context — brutally slow. `nueronce/incremental.py` (`IncrementalGenerator`)
exploits three structural facts — everything is causal; the decoder can't
see in-progress patches, so the heavy stages only need recomputing when a
patch *completes*; and the decoder has no absolute positions, so a bounded
window suffices for exact next-byte scores — to cache state and skip
recomputation. The contract is strict: **byte-exact or nothing.** Tests
require its outputs to match the slow path exactly (logits to ~1e-9 in the
engine twin, and the chat path self-verifies against one dense forward on
first use, permanently falling back if there's ever a mismatch, so the
fast path can never silently corrupt a conversation). Measured: ~17×
wall-clock speedup at a 256-byte prompt. The sealed proof gate runs its
generation through this engine, and re-proves the equivalence at load time,
every time.

---

## 8. The automation: multi-pass training that survives anything

`scripts/sandbox_train_driver.sh` runs the entire program unattended, in
repeated passes, in an ephemeral container:

1. **Stage 0**: wait for/evaluate the pipeline gate (§9.4) and push the
   verdict to GitHub.
2. **Stage 1**: base pretraining in 150-minute passes. Each pass resumes
   from the checkpoint. After each pass the driver reads the training
   history itself, detects plateaus, and walks the LR ladder down
   automatically. State survives restarts.
3. **Stage 2**: the moment held-out bpb ≤ 1.8, SFT on the cleaned data
   starts automatically.
4. **Stage 3**: the sealed proof gate runs, and the verdict is pushed —
   pass or fail.

Between passes it pushes training history to git every pass and a slim
resume checkpoint (weights without optimizer, small enough for GitHub)
every two hours. A container being reclaimed costs at most one partial
pass, and *any* machine — Codespace, Colab, desktop — can pick up the run
with `--resume`. This exists because real training here happens on
machines that disappear: Colab reclaims runtimes, sessions idle out, a
Stop button once SIGINT-killed an overnight run (fixed with process-group
detachment), and the single best checkpoint of an early run was lost
because backup wasn't automatic yet (fixed with timed auto-backup; the
lesson is now infrastructure).

---

## 9. What has actually happened: the experimental record

### 9.1 Early proofs (before this cycle)
- A 2.04M-parameter demo trained on a 5KB corpus: bits/byte 8.30 → 0.11,
  proving the architecture learns (`docs/RESULTS.md` top section).
- The full pipeline demo: retrieval finds the right document, the planner
  orders a response, the verifier checks claims against evidence, and a
  revision loop actually removes unsupported claims
  (`scripts/run_pipeline.py`, `tests/test_revise.py`).
- A 112,709-parameter micro-model instruction-tuned on 100K synthetic
  dialogues reached 0.906 teacher-forced byte accuracy — proof the whole
  SFT loop works end-to-end (`docs/reports/MICRO_NUERONCE_SFT_100K_REPORT.md`).
- The 11M `chat_11m` was pretrained on real corpora across Colab sessions
  (best recorded held-out bpb on the larger corpus run: **1.377**
  mid-run; an earlier session banked 1.785).
- Baseline comparison at ~1.2M params on a tiny corpus: NUERONCE is
  competitive with a matched byte-Transformer, doesn't decisively beat
  it; both beat a pure SSM. No scaling claims made. The harness is the
  point (`nueronce/baselines.py`, `docs/RESULTS.md`).
- A detour wired an external spectral-transform research idea (RFT) into
  the stack (PRs #42-43), and was then deliberately and fully reverted
  (PR #45) to keep one canonical architecture. The verification harness
  survived; the speculation didn't. That's the system working.

### 9.2 The crisis: "trained" but broken (the 0/8 investigation)
An 11.85M checkpoint SFT'd on a curriculum reached beautiful validation
loss (0.844 → later 0.44) and then scored **0/8** on the sealed proof
gate — every answer fluent-ish gibberish. Worse, probing the *base*
checkpoint underneath (73,199 steps, zero SFT) produced repetitive garbage
too. Seven hypotheses were opened (`FOUNDATIONAL_GENERATION_RECOVERY.md`):
prompt misalignment, loss masking, EOS handling, state contamination,
checkpoint mismatch, memorization failure, undertraining.

**The resolution — it was arithmetic, not a bug.** Exact reproduction of a
response is the *product* of per-byte accuracies. A response loss of 0.844
corresponds to roughly 85% per-byte argmax accuracy, and 0.85³⁰ ≈ 0.8% —
so a ~30-byte answer almost never comes out exactly right, even though
teacher-forced numbers look great. "Low loss" was low for language
modeling and catastrophically high for exact task completion. Verdicts:
four hypotheses cleared by direct test, one (a config-field drift between
the two implementations) found and fixed, and undertraining confirmed as
the primary cause. Aggregate validation loss was **banned** as the sole
selection metric; the trainers now report per-byte accuracy and
first-8-bytes loss instead (the first 8 assistant bytes decide whether an
answer starts on-topic at all).

Supporting tools built for this: `scripts/eval_loss_generation_curve.py`
measures the loss→generation curve directly (on a deliberately tiny model
it showed teacher-forced accuracy climbing 66%→95% while exact free-run
reproduction stayed at 0-1 of 32 — the trap, photographed), and 9 new
pipeline-contract tests (`tests/test_foundational_recovery_pipeline.py`)
that permanently pin byte-prefix equality, mask boundaries, the
one-byte-shift in the loss, EOS-inside-the-mask, delimiter-leak
prevention, cross-call state isolation, and loud failure on architecture
mismatch.

### 9.3 The data audit
The committed instruction-tuning set was found contaminated: row 1 paired
a question about labor-market monopsony with an answer about Minecraft
modding ("Yes, that's correct. Keeping the code for the TESR..."). Two
classes of broken rows — *orphaned context* (an assistant reply to a
deeper conversation turn, glued to a root prompt it never answered) and
*role-swapped* (the "response" is literally another user's request).
`scripts/clean_sft_pairs.py` detects both (plus an exact-verification mode
against the source dataset for machines with network access). Result: 155
provably broken pairs dropped from the committed files (111 orphaned + 36
role-swapped), with a full per-row report in `artifacts/clean_sft/`. Rows
like those directly teach a model that responses needn't relate to
prompts.

### 9.4 The gate pass (today)
Section B of the recovery plan demanded proof the pipeline itself is
sound: train a fresh real `chat_11m` on 32 fixed short tasks (copying,
arithmetic, polite rewriting, evidence extraction, honest abstention,
debugging, clock math, one-step planning) until near-zero response loss,
then check whether it can reproduce them **free-running** — greedy, no
teacher forcing, production serializer, production generation path.

- **Round 1**: memorized to loss 0.046 in 103 steps (the crippled smoke
  model had needed 700 steps to reach only 0.15 — capacity confirmed).
  Free-run: 24/32. Every miss a byte-stutter on a *known* answer
  ("helllo world", "provideded", "10:0."), zero delimiter leaks, state
  isolation clean. And the miss rate matched prediction: measured 98.72%
  per-byte accuracy predicts 81% exact vs 75% observed — within noise.
  Same arithmetic as §9.2, one rung down the curve.
- **Round 2**: resumed to loss 0.0045. **31/32 exact — GATE PASSED.**
  The one miss: "apple" → "aple", a double-letter stutter, exactly the
  1-in-32 tolerance the pass condition anticipates.

What this proves: the train → serialize → mask → generate pipeline is
mechanically correct on the real architecture. The historical 0/8 was
never plumbing. What it does *not* prove: generalization — reproducing
trained examples is necessary, not sufficient. That's what base
pretraining scale is for.

**Right now** (as this manual is written): the multi-pass driver is in
stage 1, base-pretraining `chat_11m` from scratch on the 23.8MB offline
book corpus in this container — held-out bpb 3.83 → below 3.0 and falling
in the first hour — pushing history and slim checkpoints to GitHub as it
goes, walking the LR ladder toward the 1.8 unlock.

---

## 10. The test suite: what ~340 automated checks defend

Run with `pytest` (torch-dependent tests skip themselves if torch is
absent). The important families:

| Family | What it proves |
|---|---|
| Gradient checks (`test_nueronce_engine.py`) | Every hand-written operation differentiates correctly (finite-difference + PyTorch parity, ~1e-8) |
| Causality (`test_model_learns.py` and engine twins) | The future cannot influence the past — measured 0.0 after the router fix |
| Preset parity (`test_config_presets.py`) | Torch and engine configs agree field-for-field; parameter counts in documented ranges |
| Format contract (`test_prompting.py`, `test_chat_format_drift.py`) | Round-trip format integrity; checkpoints refuse to run in the wrong prompt format |
| Pipeline contract (`test_foundational_recovery_pipeline.py`) | Train/inference byte-prefix equality, mask boundaries, loss shift, EOS in mask, delimiter-leak prevention, state isolation, loud arch-mismatch failure |
| Safe interruption (`test_safe_interruption.py`) | Ctrl-C during training saves atomically and exits cleanly |
| Incremental equivalence (`test_incremental_torch.py` + engine tests) | Fast generation is byte-identical to slow generation |
| SFT loop (`test_sft*.py`, `test_sharded_sft.py`) | Masked-loss fine-tuning works on both backends, sharded and resumable |
| Retrieval training (`test_retrieval_training.py`) | Gradients actually reach the retrieval pathways |
| Scaling (`test_scaling.py`) | The 337M configuration constructs and forwards |

One deliberate *absence* is also part of the system: the sealed proof
gate's eight cases are not in any training set and may not be edited —
"proof gate remaining unchanged" is item 10 of the recovery plan's
required checks.

---

## 11. Known limitations (the honest list)

From `docs/STATUS.md` and the investigation, current and real:

1. Typed memory channels are structurally separate but not yet proven to
   specialize to their names.
2. "Sparse" attention computes the full score matrix then masks — sparse
   *weights*, not yet sparse *compute*.
3. The router blends all four paths every step; true skip-computation is
   future work.
4. The retriever quality is heuristic (hashed n-grams), not learned.
5. The cognitive layer (planner, verifier, contradiction scoring) is real
   working code but rule-based, not learned.
6. At tiny scale NUERONCE does not decisively beat a matched Transformer —
   no superiority claim is made anywhere.
7. Byte-level spelling stutters ("aple") persist until loss is driven very
   low; exact-output tasks need response loss ≈ 0.005, not 0.05.
8. `generate()`'s default context window (256 bytes) silently truncates
   long prompts — explicit `max_ctx` is passed everywhere that matters,
   but the default remains a footgun.
9. The biggest one: **generalization requires scale the current corpus
   doesn't have.** A few thousand SFT rows on top of tens of megabytes of
   pretraining cannot generalize to unseen phrasings. The fix ladder ends
   at "scale pretraining data 100-1000×," which is a compute-acquisition
   problem, not a research problem.

---

## 12. How to run everything (quick reference)

```bash
# install (torch + numpy), then sanity-check
pip install -e . && pytest -q

# build a corpus with zero network access (uses the committed books)
python scripts/build_local_corpus.py --out corpus_local

# base-pretrain the 11M, resumable, time-budgeted
python scripts/train_checkpoint.py --preset chat_11m --corpus corpus_local \
    --minutes 150 --seq 192 --batch 16 --lr 5e-4 --resume \
    --out checkpoints/chat_11m_base_local.pt

# the pipeline gate (prove train->generate works before believing anything)
python scripts/train_tiny_exact_overfit.py --out runs/tiny_exact_overfit/checkpoint.pt
python scripts/eval_tiny_exact_overfit.py                    # need >= 31/32

# clean instruction data, then SFT once base bpb < 1.8
python scripts/clean_sft_pairs.py train.jsonl val.jsonl
python scripts/train_forgeloop_sft.py --base checkpoints/chat_11m_base_local.pt \
    --train train.jsonl --val val.jsonl --out checkpoints/assistant.pt \
    --system-file runs/forgeloop/system_prompt.txt

# the sealed verdict (do not edit this script's cases, ever)
python scripts/eval_foundational_proof_gate.py --checkpoint checkpoints/assistant_best.pt

# or: all of the above, unattended, multi-pass, self-pushing
bash scripts/sandbox_train_driver.sh

# chat with any checkpoint
python -c "
from nueronce.chat import Conversation, load_checkpoint
m, ck = load_checkpoint('checkpoints/assistant_best.pt')
print(Conversation(m, system='You are NUERONCE.').say('Hello!'))"
```

---

## 13. Where it goes from here

1. **Finish the 11M loop** — the driver is walking base pretraining down
   the ladder now; at bpb ≤ 1.8 it SFTs on the cleaned data and takes the
   sealed gate. First nonzero sealed-gate score is the next milestone.
2. **Scale data 100-1000×** on the license-clean pipeline (the registry
   and downloaders already exist; this is GPU/bandwidth, not new code).
3. **Climb the ladder** — 35M with its pre-registered acceptance gates,
   then 90M, then the 337M config that already builds and forwards.
4. **The architecture question stays open, honestly**: if at matched
   compute NUERONCE doesn't beat the transformer baseline, that result
   gets published in this repo like every other one — and the
   verification system, which is the part nobody else has, survives
   either answer.

---

## 14. Glossary

- **Byte**: the raw 0-255 numbers text is stored as. This model's only
  vocabulary.
- **bpb (bits per byte)**: how surprised the model is per byte; 8.0 =
  random guessing, lower = better. The pretraining score.
- **Teacher forcing**: scoring the model while feeding it the correct
  previous bytes — measures knowledge, but hides compounding errors.
- **Free-running**: the model eats its own output byte by byte — how
  generation actually works, and where compounding errors live.
- **Argmax accuracy**: how often the model's top guess for the next byte
  is right. Exact reproduction ≈ this accuracy raised to the power of the
  answer length — the single most important equation in this repo's
  recent history.
- **SFT (supervised fine-tuning)**: teaching a pretrained model to respond
  to instructions using (prompt → response) pairs.
- **Response-only masking**: grading only the reply bytes during SFT.
- **LR ladder**: train until plateau, drop learning rate 3×, repeat.
- **Sealed gate**: a fixed evaluation nobody may edit or train against.
- **ForgeLoop**: this project's bounded software-engineering scaffold
  (CONTRACT → ... → MEMORY), taught via SFT.
- **Patch/unit**: a learned chunk of bytes, this model's replacement for a
  token.
- **SSM**: state-space model, the recurrent path in the hybrid core.
- **AMP**: mixed-precision (fp16) training on GPUs, ~2× faster.
- **Atomic save**: write to a temp file, then rename — a crash can never
  produce a half-written checkpoint.
