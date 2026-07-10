# Local GPU training playbook — multi-subject corpus → parameters → accurate answers

The repo-side machinery is built; this is the order to run it on the local
CUDA machine. Every stage is resumable; push `metrics/` and manifests back to
git after each stage so the cloud session can analyze between runs.

## 0. Sync

```bash
git pull
pip install -r requirements.txt && pip install -e . datasets tqdm
pytest tests/test_gpu_amp.py -v        # AMP-safety green light (2 CUDA tests)
```

## 1. Pull the multi-subject corpus (license-gated stack)

`nueronce.corpus.stack.CORPUS_STACK` now covers the requested subjects:
math/physics (`open_web_math`), code in ~30 languages (`the_stack_smol`,
filter rows to the permissive-license allowlist), literature/fantasy
(`project_gutenberg`, `standard_ebooks`), health (`pmc_oa_comm`), psychology/
sociology/finance/textbooks (`cosmopedia_100k`, `libretexts`,
`open_textbook_library`), general (`english_wikipedia_latest`,
`fineweb_edu_sample_10bt`). Deliberately excluded: SciQ (CC BY-NC), MMLU
(unclear aggregation terms). ARC is CC BY-SA — share-alike propagates to
derived SFT files; keep them under the same terms.

```bash
python scripts/dump_corpus_stack.py --out corpus_large \
  --sources cosmopedia_100k,open_web_math,the_stack_smol,project_gutenberg,english_wikipedia_latest,pmc_oa_comm \
  --target-bytes 400000000 --val-every 20
```

Balance rule from the 100K-SFT report: one register at 77% poisoned the whole
mix. Cap any single source near ~25% of bytes.

## 2. Base pretraining at the next parameter rung

Presets in `nueronce.model.CONFIG_PRESETS`, parameter counts verified by
construction (test_config_presets.py):

| preset | params | GPU guidance |
|---|---|---|
| `chat_11m` | 11.1M | current checkpoint's rung |
| `base_35m` | 34.4M | 8 GB VRAM, AMP, seq 192 x batch 16, LR ~3e-4 |
| `base_90m` | 92.1M | 12–24 GB VRAM, AMP + grad accumulation, LR ~2e-4 |
| `large_337m` | 337M | multi-GPU / long runs only; do not start here |

The GPT-3 lesson applies at every rung: scale data *with* parameters — an
undertrained 90M loses to a data-matched 35M at equal wall-clock. 400 MB of
corpus comfortably feeds 35M for the first serious run; go 90M only after
35M's held-out bits/byte stalls with data left over.

Point the notebook/trainer's `chat_config()` at the chosen preset (or import
it directly) and resume from the 11M checkpoint where shapes allow — fresh
init otherwise.

## 3. Instruction tuning: subjects + MCQs

```python
from nueronce.corpus.stack import get_entry
from nueronce.training.mcq_sft import load_and_convert
import json
with open("data/subject_sft/train_raw.jsonl", "w") as f:
    for sid in ["arc_easy", "arc_challenge", "openbookqa", "commonsense_qa", "math_qa", "gsm8k"]:
        for rec in load_and_convert(get_entry(sid), split="train"):
            f.write(json.dumps(rec) + "\n")
```

Then the existing pipeline unchanged: `nueronce.training.dataset_prep`
(validate/dedupe/split/shard, keeps val+test frozen) → `scripts/train_sft.py`
/ the forgeloop SFT trainer over the shards, mixed with OASST1/Dolly and the
built-in turn-taking set.

## 4. Evaluate knowledge two ways (this is the accuracy story)

1. **Choice ranking** (`nueronce.training.mcq_sft.evaluate_mcq`): scores each MCQ
   option by masked answer loss — measures *knowledge* even while generation
   is imperfect, and reports chance level alongside. Run it on the held-out
   MCQ split before and after SFT; this is the number that shows subjects
   entering the parameters.
2. **Generative**: `scripts/eval_generalization.py` + chat probes — measures
   whether knowledge *renders*. Expect ranking accuracy to move before
   generative exact-match does (structure-before-content, per the 100K
   report); report both, never only the better one.

## 5. Push results back

```bash
git add metrics/ benchmarks/ corpus_large/manifest.jsonl data/subject_sft/manifest.json
git commit -m "local run: <preset> <corpus bytes> <steps>" && git push
```

The cloud session picks it up from there: curve analysis, next-rung
go/no-go, and preset/LR adjustments.
