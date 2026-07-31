#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIR="${RUN_DIR:-runs/p256_microtorch_resume}"
BASE_CKPT="${BASE_CKPT:-$RUN_DIR/train_latest.pkl}"
SFT_CKPT="${SFT_CKPT:-$RUN_DIR/foundational_sft_latest.pkl}"
BEST_SFT="${BEST_SFT:-$RUN_DIR/foundational_sft_best.pkl}"
STATE="${STATE:-$RUN_DIR/foundational_supervisor_state.json}"
PID_FILE="${PID_FILE:-$RUN_DIR/foundational_supervisor.pid}"
LOG_JSONL="${LOG_JSONL:-$RUN_DIR/foundational_supervisor_metrics.jsonl}"
MAX_BASE_STEP="${MAX_BASE_STEP:-50000}"
BASE_CHUNK="${BASE_CHUNK:-500}"
BASE_GATE_BPB="${BASE_GATE_BPB:-1.5}"
MAX_REPEAT_4GRAM_FRACTION="${MAX_REPEAT_4GRAM_FRACTION:-0.20}"
MAX_SFT_STEPS="${MAX_SFT_STEPS:-5000}"
SFT_CHUNK="${SFT_CHUNK:-100}"
SFT_PATIENCE_CHUNKS="${SFT_PATIENCE_CHUNKS:-10}"

mkdir -p "$RUN_DIR"
echo "$$" > "$PID_FILE"
exec 9>"$RUN_DIR/foundational_supervisor.lock"
flock -n 9 || { echo "another foundational supervisor holds the lock" >&2; exit 2; }

checkpoint_step() {
  python - "$1" <<'PY'
import pickle, sys
with open(sys.argv[1], "rb") as handle:
    payload = pickle.load(handle)
print(int((payload.get("meta") or {}).get("step", 0)))
PY
}

write_state() {
  python - "$STATE" "$1" "$2" <<'PY'
import json, os, sys, time
path, stage, detail = sys.argv[1:]
payload = {
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stage": stage,
    "detail": detail,
    "supervisor_pid": os.getppid(),
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
os.replace(tmp, path)
PY
}

evaluate_base() {
  local step="$1"
  python scripts/train_nueronce_engine_35m_split_pretrain.py prepare \
    --checkpoint "$BASE_CKPT" --plan "$RUN_DIR/eval_alice.plan.pkl" \
    --document corpus_local/text/alice.txt.txt --seq-len 1024 --offset 134392 \
    --seed 20260731 --lr 5e-5 --max-grad-norm 1.0 >/dev/null
  python scripts/train_nueronce_engine_35m_split_pretrain.py prepare \
    --checkpoint "$BASE_CKPT" --plan "$RUN_DIR/eval_origin.plan.pkl" \
    --document corpus_local/text/origin.txt.txt --seq-len 1024 --offset 200000 \
    --seed 20260731 --lr 5e-5 --max-grad-norm 1.0 >/dev/null
  python - "$RUN_DIR/eval_alice.plan.pkl" "$RUN_DIR/eval_origin.plan.pkl" "$step" "$LOG_JSONL" <<'PY'
import json, math, pickle, sys, time
plans, step, log = sys.argv[1:3], int(sys.argv[3]), sys.argv[4]
losses = []
for path in plans:
    with open(path, "rb") as handle:
        losses.append(float(pickle.load(handle)["expected_loss"]))
avg_loss = sum(losses) / len(losses)
record = {
    "event": "base_validation", "step": step, "losses": losses,
    "mean_loss": avg_loss, "mean_bpb": avg_loss / math.log(2),
    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
print(record["mean_bpb"])
PY
}

write_state waiting_safety "waiting for step-1200 generation probe"
while [[ ! -s "$RUN_DIR/generation_step1200.json" ]]; do
  sleep 30
done

write_state base_training "continuing strict p256/s1024 base training"
while true; do
  step="$(checkpoint_step "$BASE_CKPT")"
  bpb="$(evaluate_base "$step")"
  if python - "$bpb" "$BASE_GATE_BPB" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
  then
    repetition_report="$RUN_DIR/repetition_gate_step${step}.json"
    python scripts/probe_base_continuation_engine.py \
      --checkpoint "$BASE_CKPT" --out "$repetition_report" --max-new 64
    repeat_fraction="$(python - "$repetition_report" <<'PY'
import json, sys
print(float(json.load(open(sys.argv[1], encoding="utf-8"))["max_repeated_4gram_fraction"]))
PY
)"
    if python - "$repeat_fraction" "$MAX_REPEAT_4GRAM_FRACTION" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
    then
      break
    fi
    write_state base_repetition_gate "BPB passed at $bpb but repeated-4gram fraction $repeat_fraction exceeds $MAX_REPEAT_4GRAM_FRACTION"
  fi
  if (( step >= MAX_BASE_STEP )); then
    write_state base_gate_failed "max base step reached with mean held-out BPB $bpb"
    exit 3
  fi
  target=$(( step + BASE_CHUNK ))
  (( target > MAX_BASE_STEP )) && target="$MAX_BASE_STEP"
  bash scripts/run_p256_microtorch_safety_probe.sh "$target"
  python scripts/probe_base_continuation_engine.py \
    --checkpoint "$BASE_CKPT" --out "$RUN_DIR/generation_step${target}.json"
done

write_state sft_training "base gate passed; starting clean canonical response-only SFT"
if [[ ! -f "$SFT_CKPT" ]]; then
  cp --reflink=auto "$BASE_CKPT" "$SFT_CKPT"
fi

best="inf"
bad=0
while true; do
  read -r sft_step val_loss < <(python - "$SFT_CKPT" <<'PY'
import pickle, sys
with open(sys.argv[1], "rb") as handle:
    payload = pickle.load(handle)
meta = payload.get("meta") or {}
print(int(meta.get("engine_sft_step", 0)), meta.get("val_loss", "inf"))
PY
  )
  (( sft_step >= MAX_SFT_STEPS )) && break
  python scripts/train_forgeloop_engine_sft.py \
    --checkpoint "$SFT_CKPT" --train train.jsonl --val val.jsonl \
    --system-file runs/forgeloop/system_prompt.txt --max-len 768 \
    --lr 2e-5 --steps "$SFT_CHUNK" --eval-every 25 --eval-examples 16 \
    --save-every 25 --seed 91 --dtype float32
  read -r sft_step val_loss < <(python - "$SFT_CKPT" <<'PY'
import pickle, sys
with open(sys.argv[1], "rb") as handle:
    payload = pickle.load(handle)
meta = payload.get("meta") or {}
print(int(meta.get("engine_sft_step", 0)), float(meta.get("val_loss", "inf")))
PY
  )
  if python - "$val_loss" "$best" <<'PY'
import sys
v, b = map(float, sys.argv[1:])
raise SystemExit(0 if v < b - 0.0005 else 1)
PY
  then
    best="$val_loss"
    bad=0
    cp --reflink=auto "$SFT_CKPT" "$BEST_SFT.tmp"
    mv "$BEST_SFT.tmp" "$BEST_SFT"
  else
    bad=$(( bad + 1 ))
  fi
  printf '{"event":"sft_chunk","step":%s,"val_loss":%s,"best":%s,"bad_chunks":%s}\n' \
    "$sft_step" "$val_loss" "$best" "$bad" >> "$LOG_JSONL"
  (( bad >= SFT_PATIENCE_CHUNKS )) && break
done

FINAL_CKPT="$SFT_CKPT"
[[ -f "$BEST_SFT" ]] && FINAL_CKPT="$BEST_SFT"
python scripts/probe_nueronce_engine_chat.py \
  --checkpoint "$FINAL_CKPT" --out "$RUN_DIR/final_chat_probe.json" \
  --temperature 0 --max-new 96 --max-ctx 1024
write_state completed "training stages completed; inspect final_chat_probe.json before claiming coherence"
