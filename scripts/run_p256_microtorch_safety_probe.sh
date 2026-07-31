#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-runs/p256_microtorch_resume/train_latest.pkl}"
RUN_DIR="${RUN_DIR:-runs/p256_microtorch_resume}"
TARGET_STEP="${1:-1200}"
LR="${LR:-5e-5}"
SEED="${SEED:-20260731}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-corpus_local/manifest.jsonl}"
TRAINER="scripts/train_nueronce_engine_35m_split_pretrain.py"

mapfile -t DOCS < <(python - "$TRAIN_MANIFEST" <<'PY'
import json, pathlib, sys
manifest = pathlib.Path(sys.argv[1])
for line in manifest.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    if record.get("split") == "train":
        print(manifest.parent / record["path"])
PY
)
(( ${#DOCS[@]} > 0 )) || { echo "no training documents in $TRAIN_MANIFEST" >&2; exit 2; }

mkdir -p "$RUN_DIR"

current_step() {
  python - "$CHECKPOINT" <<'PY'
import pickle, sys
with open(sys.argv[1], "rb") as handle:
    payload = pickle.load(handle)
print(int(payload["meta"]["step"]))
PY
}

while (( $(current_step) < TARGET_STEP )); do
  step=$(( $(current_step) + 1 ))
  index=$(( (step - 701) % ${#DOCS[@]} ))
  document="${DOCS[$index]}"
  plan="$RUN_DIR/current_train.plan.pkl"

  python "$TRAINER" prepare \
    --checkpoint "$CHECKPOINT" \
    --plan "$plan" \
    --document "$document" \
    --seq-len 1024 \
    --seed "$SEED" \
    --lr "$LR" \
    --max-grad-norm 1.0 \
    --include-boundary-loss
  python "$TRAINER" backward --plan "$plan"

  if (( step == 800 || step == 950 || step == 1200 )); then
    python "$TRAINER" prepare \
      --checkpoint "$CHECKPOINT" \
      --plan "$RUN_DIR/validation_step${step}.plan.pkl" \
      --document corpus_local/text/alice.txt.txt \
      --seq-len 1024 \
      --offset 134392 \
      --seed "$SEED" \
      --lr "$LR" \
      --max-grad-norm 1.0
    python scripts/probe_base_continuation_engine.py \
      --checkpoint "$CHECKPOINT" \
      --out "$RUN_DIR/generation_step${step}.json"
  fi
done
