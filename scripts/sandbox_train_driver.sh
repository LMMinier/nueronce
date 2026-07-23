#!/usr/bin/env bash
# Sandbox training driver: chains the pipeline gate into base pretraining.
#  1. wait for the running tiny-exact-overfit train (the pipeline proof) to exit
#  2. run the 31/32 free-running eval on its checkpoint
#  3. commit+push the gate verdict
#  4. launch resumable chat_11m base pretraining on corpus_local (offline corpus)
#  5. every 30 min: push training history; every 2 h: push a slim (no-optimizer)
#     best checkpoint so any machine can resume if this container is reclaimed
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

LOG_DIR=logs; mkdir -p "$LOG_DIR" checkpoints
BASE_CKPT=checkpoints/chat_11m_base_local.pt
BASE_LOG="$LOG_DIR/base_local_train.log"
BRANCH=claude/new-session-tkjr3b

echo "[driver] waiting for tiny-exact-overfit training to finish..."
while pgrep -f "train_tiny_exact_overfit" > /dev/null; do sleep 60; done

if [ -f runs/tiny_exact_overfit/checkpoint.pt ]; then
  echo "[driver] running the 31/32 gate eval..."
  python3 scripts/eval_tiny_exact_overfit.py \
    --checkpoint runs/tiny_exact_overfit/checkpoint.pt \
    --output runs/tiny_exact_overfit/eval_report.json --no-fail-exit \
    > "$LOG_DIR/tiny_gate_eval.log" 2>&1
  git add runs/tiny_exact_overfit/eval_report.json "$LOG_DIR/tiny_gate_eval.log" 2>/dev/null
  git commit -m "Tiny exact-overfit gate result on the real chat_11m (sandbox CPU run)" 2>/dev/null
  git push origin "$BRANCH" 2>/dev/null
else
  echo "[driver] WARNING: tiny checkpoint missing; skipping gate eval"
fi

echo "[driver] launching base pretraining (resumable, 600 min budget)..."
nohup nice python3 -u scripts/train_checkpoint.py --preset chat_11m \
  --corpus corpus_local --minutes 600 --seq 192 --batch 16 --lr 5e-4 \
  --resume --out "$BASE_CKPT" > "$BASE_LOG" 2>&1 &
TRAIN_PID=$!
echo "[driver] training PID $TRAIN_PID"

LAST_CKPT_PUSH=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 1800
  # metrics push (cheap): history json lives next to the checkpoint after each save
  python3 - <<'PYEOF'
import json, torch
from pathlib import Path
ck_path = Path("checkpoints/chat_11m_base_local.pt")
if ck_path.exists():
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    hist = ck.get("history", [])
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/chat_11m_base_local_history.json").write_text(
        json.dumps({"step": ck.get("step"), "history": hist}, indent=1))
    if hist:
        print("latest:", hist[-1])
PYEOF
  git add metrics/chat_11m_base_local_history.json 2>/dev/null
  git commit -m "base pretrain progress (sandbox CPU): update history" 2>/dev/null
  git push origin "$BRANCH" 2>/dev/null
  NOW=$(date +%s)
  if [ $((NOW - LAST_CKPT_PUSH)) -ge 7200 ] && [ -f "$BASE_CKPT" ]; then
    python3 - <<'PYEOF'
import torch
from pathlib import Path
ck = torch.load("checkpoints/chat_11m_base_local.pt", map_location="cpu", weights_only=False)
slim = {"state_dict": ck["state_dict"], "config": ck["config"],
        "step": ck.get("step"), "history": ck.get("history", [])[-20:],
        "note": "slim resume checkpoint (no optimizer) pushed from sandbox CPU run"}
torch.save(slim, "checkpoints/chat_11m_base_local_slim.pt")
print("slim checkpoint written")
PYEOF
    git add -f checkpoints/chat_11m_base_local_slim.pt 2>/dev/null
    git commit -m "base pretrain: slim resume checkpoint (sandbox CPU)" 2>/dev/null
    git push origin "$BRANCH" 2>/dev/null
    LAST_CKPT_PUSH=$NOW
  fi
done

echo "[driver] training exited; final push"
git add metrics/chat_11m_base_local_history.json "$BASE_LOG" 2>/dev/null
git commit -m "base pretrain (sandbox CPU): final history + log" 2>/dev/null
git push origin "$BRANCH" 2>/dev/null
echo "[driver] done"
