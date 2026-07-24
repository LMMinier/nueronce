#!/usr/bin/env bash
# Multi-pass sandbox training driver: the whole fix ladder, resumable forever.
#
#  stage 0: wait for the tiny-exact-overfit pipeline gate, eval it, push verdict
#  stage 1: base pretraining in REPEATED PASSES (each pass resumes from the
#           checkpoint), walking the LR ladder automatically -- after each
#           pass, if the held-out bpb history shows >=18 evals since the best,
#           drop LR 3x; stop the ladder at LR < 1.2e-5 or bpb <= BPB_GATE
#  stage 2: once bpb <= BPB_GATE, curriculum SFT on the CLEANED rows
#           (train.jsonl/val.jsonl post scripts/clean_sft_pairs.py) with the
#           free-run-predictive metrics, resumable the same way
#  stage 3: the sealed proof gate, unmodified, verdict pushed either way
#
# Every pass pushes history; every 2h a slim no-optimizer checkpoint is
# pushed, so a reclaimed container costs at most one pass and any machine
# resumes with --resume. Safe to re-run this script at any time -- every
# stage picks up where it left off.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

LOG_DIR=logs; mkdir -p "$LOG_DIR" checkpoints metrics
BASE_CKPT=checkpoints/chat_11m_base_local.pt
SFT_CKPT=checkpoints/chat_11m_assistant_local.pt
BRANCH=claude/new-session-tkjr3b
PASS_MIN=${PASS_MIN:-150}          # minutes per pass
BPB_GATE=${BPB_GATE:-1.8}          # SFT unlocks below this held-out bpb
LR_FILE="$LOG_DIR/base_lr.txt"     # ladder state survives driver restarts
LAST_CKPT_PUSH_FILE="$LOG_DIR/last_ckpt_push.txt"

push_history() {
  python3 - <<'PYEOF'
import json, torch
from pathlib import Path
ck_path = Path("checkpoints/chat_11m_base_local.pt")
if ck_path.exists():
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    hist = ck.get("history", [])
    Path("metrics/chat_11m_base_local_history.json").write_text(
        json.dumps({"step": ck.get("step"), "history": hist}, indent=1))
    if hist:
        print("latest:", json.dumps(hist[-1]))
PYEOF
  git add metrics/chat_11m_base_local_history.json 2>/dev/null
  git commit -q -m "base pretrain progress (sandbox multi-pass): history update" 2>/dev/null
  git push -q origin "$BRANCH" 2>/dev/null
}

push_slim_maybe() {
  local now last
  now=$(date +%s); last=$(cat "$LAST_CKPT_PUSH_FILE" 2>/dev/null || echo 0)
  if [ $((now - last)) -ge 7200 ] && [ -f "$BASE_CKPT" ]; then
    python3 - <<'PYEOF'
import torch
ck = torch.load("checkpoints/chat_11m_base_local.pt", map_location="cpu", weights_only=False)
torch.save({"state_dict": ck["state_dict"], "config": ck["config"],
            "step": ck.get("step"), "history": ck.get("history", [])[-20:],
            "note": "slim resume checkpoint (no optimizer), sandbox multi-pass run"},
           "checkpoints/chat_11m_base_local_slim.pt")
print("slim checkpoint written")
PYEOF
    git add -f checkpoints/chat_11m_base_local_slim.pt 2>/dev/null
    git commit -q -m "base pretrain: slim resume checkpoint (sandbox multi-pass)" 2>/dev/null
    git push -q origin "$BRANCH" 2>/dev/null
    echo "$now" > "$LAST_CKPT_PUSH_FILE"
  fi
}

# ---- stage 0: pipeline gate ------------------------------------------------
echo "[driver] stage 0: waiting for tiny-exact-overfit training (if running)..."
while pgrep -f "train_tiny_exact_overfit" > /dev/null; do sleep 60; done
if [ -f runs/tiny_exact_overfit/checkpoint.pt ] && [ ! -f runs/tiny_exact_overfit/eval_report.json ]; then
  echo "[driver] running the 31/32 gate eval..."
  python3 scripts/eval_tiny_exact_overfit.py \
    --checkpoint runs/tiny_exact_overfit/checkpoint.pt \
    --output runs/tiny_exact_overfit/eval_report.json --no-fail-exit \
    > "$LOG_DIR/tiny_gate_eval.log" 2>&1
  git add runs/tiny_exact_overfit/eval_report.json 2>/dev/null
  git commit -q -m "Tiny exact-overfit gate result on the real chat_11m (sandbox CPU run)" 2>/dev/null
  git push -q origin "$BRANCH" 2>/dev/null
fi

# ---- stage 1: base pretraining, multi-pass LR ladder -----------------------
[ -f "$LR_FILE" ] || echo "5e-4" > "$LR_FILE"
export BPB_GATE
PASS=0
while true; do
  # current best/plateau state from the checkpoint's own history
  STATE=$(python3 - <<'PYEOF'
import json, os, torch
from pathlib import Path
gate = float(os.environ["BPB_GATE"])
ck_path = Path("checkpoints/chat_11m_base_local.pt")
if not ck_path.exists():
    print(json.dumps({"best": None, "flat": 0, "done": False})); raise SystemExit
h = torch.load(ck_path, map_location="cpu", weights_only=False).get("history", [])
if not h:
    print(json.dumps({"best": None, "flat": 0, "done": False})); raise SystemExit
bpbs = [r["heldout_bpb"] for r in h]
best = min(bpbs)
flat = len(bpbs) - 1 - max(i for i, b in enumerate(bpbs) if b == best)
print(json.dumps({"best": best, "flat": flat, "done": best <= gate}))
PYEOF
)
  BEST=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['best'])")
  FLAT=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['flat'])")
  DONE=$(echo "$STATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['done'])")
  LR=$(cat "$LR_FILE")
  echo "[driver] pass $PASS state: best=$BEST flat=$FLAT lr=$LR done=$DONE"

  if [ "$DONE" = "True" ]; then
    echo "[driver] bpb gate ($BPB_GATE) reached -- advancing to SFT"; break
  fi
  if [ "$FLAT" -ge 18 ]; then
    LR=$(python3 -c "import sys; print('{:.2e}'.format(float(sys.argv[1])/3))" "$LR")
    echo "$LR" > "$LR_FILE"
    echo "[driver] plateau ($FLAT evals since best) -> LR dropped to $LR"
    SMALL=$(python3 -c "import sys; print(1 if float(sys.argv[1]) < 1.2e-5 else 0)" "$LR")
    if [ "$SMALL" = "1" ]; then
      echo "[driver] LR ladder exhausted at bpb=$BEST -- this corpus/scale floor; stopping stage 1"
      break
    fi
  fi

  echo "[driver] pass $PASS: training $PASS_MIN min at lr=$LR (resume)..."
  nice python3 -u scripts/train_checkpoint.py --preset chat_11m \
    --corpus corpus_local --minutes "$PASS_MIN" --seq 192 --batch 16 --lr "$LR" \
    --resume --out "$BASE_CKPT" >> "$LOG_DIR/base_local_train.log" 2>&1
  push_history
  push_slim_maybe
  PASS=$((PASS + 1))
done
push_history

# ---- stage 2: curriculum SFT on cleaned rows (only if gate reached) --------
if [ "$DONE" = "True" ]; then
  echo "[driver] stage 2: SFT on cleaned rows..."
  python3 -u scripts/train_forgeloop_sft.py \
    --base "$BASE_CKPT" --train train.jsonl --val val.jsonl \
    --out "$SFT_CKPT" --system-file runs/forgeloop/system_prompt.txt \
    --batch 4 --max-len 768 --lr 2e-5 --eval-every 25 --patience 20 \
    >> "$LOG_DIR/sft_local_train.log" 2>&1
  git add "$LOG_DIR/sft_local_train.log" 2>/dev/null

  # ---- stage 3: the sealed gate, unmodified --------------------------------
  BEST_SFT=$(ls "${SFT_CKPT%.*}_best.pt" 2>/dev/null || echo "$SFT_CKPT")
  echo "[driver] stage 3: sealed proof gate on $BEST_SFT"
  python3 scripts/eval_foundational_proof_gate.py \
    --checkpoint "$BEST_SFT" \
    --output metrics/foundational_proof_gate_sandbox.json --no-fail-exit \
    > "$LOG_DIR/proof_gate_sandbox.log" 2>&1
  git add metrics/foundational_proof_gate_sandbox.json 2>/dev/null
  git commit -q -m "sealed proof gate verdict (sandbox multi-pass run)" 2>/dev/null
  git push -q origin "$BRANCH" 2>/dev/null
fi

echo "[driver] done"
