# Desktop runbook: run the compute stages locally

*For the Claude Code session on the owner's home desktop. The cloud session
cannot run torch (no GPU, proxy blocks pytorch.org and huggingface.co), so it
writes code + datasets + this runbook; the desktop (or the Colab notebook,
`notebooks/cfna_colab_training.ipynb`, which is the same pipeline cell by
cell) runs everything below and pushes `metrics/` back. Pull this branch
first.*

## 0. One-time setup

```bash
git pull
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build, not +cpu
pip install datasets numpy pytest cryptography cffi
python -m pytest tests/test_gpu_amp.py tests/test_prompting.py tests/test_chat_format_drift.py -q
```

`tests/test_gpu_amp.py` green is the gate for any `--amp` run. If
`torch.version.cuda` prints `None`, the install is wrong — that exact mistake
cost the previous local session its entire run budget.

## 1. Corpus (once, ~400 MB)

```bash
python scripts/dump_corpus_stack.py --out corpus_stack --phase 2 \
    --target-bytes 400000000 --val-every 20
```

## 2. Base pretrain (the long stage — corpus bytes → parameters)

```bash
python scripts/train_checkpoint.py --preset base_35m --corpus corpus_stack \
    --minutes 170 --seq 192 --batch 16 --lr 3e-4 --amp --resume \
    --out checkpoints/cfna_base_35m.pt
```

Re-run the same command after any interruption (`--resume`). Repeat sessions
until held-out bpb plateaus. **Gate: do not start step 4 until held-out
bpb < 1.8** (target ≤ 1.5). Push the `.json` history file after each session.

## 3. Conversational SFT dataset (deterministic, ~10 s, no network)

```bash
python scripts/build_conversation_sft.py --out-dir data/conversation_sft
```

54,685 records / ≤25% per register / leakage-checked. The manifest copy lands
in `metrics/conversation_sft_manifest.json` (tracked); regen on any machine
reproduces it bit-for-bit from the same seed.

## 4. Conversational SFT (response-masked, canonical format)

```bash
python scripts/train_conversation.py --data data/conversation_sft \
    --preset base_35m --init-from checkpoints/cfna_base_35m.pt \
    --loss response --out-dir checkpoints/conv_35m \
    --minutes 60 --batch 16 --lr 1e-4 --amp
# later sessions: replace --init-from with --resume
```

`best.pt` is best-by-val (never last-step) and stamps
`meta.prompt_format="canonical"` per `docs/FORMAT.md`. Training logs append to
`metrics/conversation_train.jsonl` — commit and push after every session.

## 5. Eval + transcripts

```bash
python scripts/eval_inference_phase2.py --write-suite --checkpoint checkpoints/conv_35m/best.pt
python - <<'EOF'
from cfna.chat import Conversation, load_checkpoint
model, ck = load_checkpoint("checkpoints/conv_35m/best.pt")
convo = Conversation(model, system="You are CFNA.", temperature=0.0)
for q in ["Hello", "What is two plus two?", "What is the capital of France?",
          "Is 8 even or odd?", "Thank you"]:
    print(f"User: {q}\nCFNA: {convo.say(q)}\n")
EOF
```

Report per checkpoint (commit to `metrics/` + a short file in
`docs/reports/`): base bpb curve, SFT val curve + byte accuracy,
choice-ranking accuracy vs chance (`cfna.training.mcq_sft.evaluate_mcq`),
phase-2 pass rate, and 5 **unedited** transcripts. Do not discard checkpoints
on generative exact-match alone — structure generalizes before content.

## Acceptance (35M rung) — unchanged from PLAN.md

bpb ≤ 1.5 · choice-ranking +15 pts over chance on ≥3 subjects · ≥60% valid
non-echo stop-terminated answers · 5/5 grammatical transcripts. Then
`--preset base_90m` unlocks.
