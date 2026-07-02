# Reproducibility Guide — Cognitive Contract V2 Baseline

This guide lets a fresh machine reproduce the frozen V2 research baseline.

- **Frozen commit:** `d712c36`
- **Branch:** `research/cognitive-contract-v1` (NOT merged to the default branch)
- **Tag:** `v0.2-cognitive-contract` (annotated tag pointing at `d712c36`)
- **Machine-readable manifest:** [`docs/experiment_manifest.json`](experiment_manifest.json)

## Environment (as observed for the frozen baseline)

| Field | Value |
|---|---|
| Python | 3.13.2 |
| torch | 2.11.0+cpu |
| numpy | 2.2.3 |
| OS | Windows-10 (10.0.19045-SP0) |
| Device | **CPU-only** (no CUDA/GPU required) |
| Logical CPU cores | 8 |

The evaluation is fully CPU-only. No GPU, CUDA toolkit, or accelerator is needed.
Results are deterministic given the recorded seeds.

## Step-by-step

### 1. Clone and check out the frozen baseline

```bash
git clone <repo-url> nueronce
cd nueronce
git checkout v0.2-cognitive-contract   # annotated tag at commit d712c36
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install the package and pinned dependencies

```bash
pip install -e .
pip install -r requirements.lock
```

`requirements.lock` is the exact `pip freeze` of the baseline environment. Use it
to reproduce the exact dependency set; `pip install -e .` installs the `cfna`
package itself in editable mode.

### 4. Run the test suite

```bash
python -m pytest -p no:cacheprovider
```

Expected result on the frozen baseline: **111 passed in ~100 s** (0 failed,
0 skipped). Runtime scales with CPU; ~100 s on the reference 8-core Windows box.

### 5. Regenerate the evaluation artifacts

V1 (deterministic; `--seed` is recorded but the V1 loop has no randomness):

```bash
python scripts/eval_cognitive.py --seed 0 \
    --json benchmarks/cognitive_v1.json \
    --md docs/reports/COGNITIVE_V1_REPORT.md
```

V2 (randomized suite: 5 seeds × 1050 in-distribution trials + 480 adversarial
holdout trials per seed):

```bash
python scripts/eval_cognitive_v2.py --seeds 1 2 3 4 5 --n 1050 --holdout 480 \
    --json benchmarks/cognitive_v2.json \
    --md docs/reports/COGNITIVE_V2_REPORT.md
```

Expected V2 headline (FULL_COGNITIVE_LOOP): in-distribution composite/value
accuracy = 1.000, poisoning rate = 0.000; adversarial-holdout accuracy = 1.000,
poisoning rate = 0.000; gate `passed = true`. See
[`docs/KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for what FULL = 1.000 does and
does not mean (it reflects policy ≡ spec, i.e. internal consistency, not external
validity).

### 6. Verify outputs against the recorded checksums

`benchmarks/CHECKSUMS.txt` lists a SHA-256 for each committed artifact under
`benchmarks/` and `docs/reports/`, format `<sha256>  <relative/path>`.

```bash
# Portable verification (matches how CHECKSUMS.txt was generated):
python - <<'PY'
import hashlib, pathlib
for line in pathlib.Path("benchmarks/CHECKSUMS.txt").read_text().splitlines():
    if not line.strip():
        continue
    expected, rel = line.split("  ", 1)
    actual = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
    print(("OK  " if actual == expected else "FAIL") + " " + rel)
PY
```

All lines should report `OK`. A `FAIL` means the regenerated artifact differs
from the frozen baseline (check Python/torch/numpy versions and seeds first).

## Notes

- Do not merge `research/cognitive-contract-v1` to the default branch as part of
  reproduction; it is intentionally an unmerged research branch.
- The byte language-model renderer is not part of this evaluation and does not
  affect these results (see `docs/KNOWN_LIMITATIONS.md`).
