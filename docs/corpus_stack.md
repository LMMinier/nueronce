# NUERONCE corpus stack

This is the staged corpus order requested for NUERONCE. The first runnable dump path
is Phase 1: TinyStories followed by Cosmopedia-100k. Later phases are recorded
with their exact pages and extraction notes so their metadata can be preserved
before they are mixed into training.

## Phase 1: prove the architecture

Target: **100-500 MB** of clean text.

1. **TinyStories** — CDLA-Sharing 1.0
   - Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
   - Files: https://huggingface.co/datasets/roneneldan/TinyStories/tree/main
   - Loader: `load_dataset("roneneldan/TinyStories", split="train")`

2. **Cosmopedia-100k** — Apache-2.0
   - Dataset: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k
   - Files: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k/tree/main
   - Loader: `load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")`

## Phase 2: build general language

Target: **1-5 GB**, depending on model size and compute.

3. **FineWeb-Edu** streamed subset
   - Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
   - Files: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main
   - Loader: `load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)`
   - Do not download the full dataset. Stream a bounded document or byte count.

4. **English Wikipedia official dumps**
   - All dumps: https://dumps.wikimedia.org/
   - Latest English dump: https://dumps.wikimedia.org/enwiki/latest/
   - Article dump: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
   - Index: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2
   - Extract a subset; do not train on the entire dump for the current model.
   - Common extractor: https://github.com/attardi/wikiextractor

5. **Project Gutenberg**
   - Main library: https://www.gutenberg.org/
   - Bookshelves: https://www.gutenberg.org/ebooks/bookshelf/
   - Machine-readable catalogs: https://www.gutenberg.org/ebooks/offline_catalogs.html
   - Bulk access policy: https://www.gutenberg.org/policy/robot_access.html
   - RDF catalog: https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2
   - Use plain UTF-8 text when available and record jurisdiction assumptions.

## Phase 3: technical knowledge

6. **Open Textbook Library**
   - Main library: https://open.umn.edu/opentextbooks
   - Subjects: https://open.umn.edu/opentextbooks/subjects
   - Record each book's exact license in the manifest.

7. **LibreTexts**
   - Main: https://libretexts.org/
   - Math: https://math.libretexts.org/
   - Computer science: https://eng.libretexts.org/Bookshelves/Computer_Science
   - Engineering: https://eng.libretexts.org/
   - Physics: https://phys.libretexts.org/
   - Chemistry: https://chem.libretexts.org/
   - Biology: https://bio.libretexts.org/
   - Statistics: https://stats.libretexts.org/
   - Social sciences: https://socialsci.libretexts.org/
   - Humanities: https://human.libretexts.org/
   - Check the license shown on each book or page.

8. **PMC Open Access commercial collection**
   - Official OA subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
   - FTP docs: https://pmc.ncbi.nlm.nih.gov/tools/ftp/
   - Commercial collection: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/
   - Noncommercial collection: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/
   - Other-license collection: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/
   - Start with `oa_comm` and prefer structured JATS XML.

9. **SmolLM-Corpus Python-Edu**
   - Dataset: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
   - Files: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main
   - Loader: `load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)`
   - Preserve repository and license metadata.

## Phase 4: assistant behavior

Do this after the base model can already generate coherent text.

10. **OpenAssistant OASST1** — Apache-2.0
    - Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
    - Files: https://huggingface.co/datasets/OpenAssistant/oasst1/tree/main
    - Loader: `load_dataset("OpenAssistant/oasst1", split="train")`
    - Filter to English: `row.get("lang") == "en"`.

11. **Databricks Dolly 15K** — CC-BY-SA-3.0
    - Dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k
    - File page: https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl
    - Raw JSONL: https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
    - Loader: `load_dataset("databricks/databricks-dolly-15k", split="train")`

## Recommended mix

First respectable NUERONCE base model:

```text
30% FineWeb-Edu
25% Cosmopedia
15% TinyStories
10% Wikipedia
10% public-domain books
7% open textbooks/scientific writing
3% educational Python code
```

Separate instruction tuning:

```text
70% OpenAssistant English conversations
30% Dolly 15K
```

Do not heavily mix instruction data into initial pretraining. First teach the
network language; then teach it how to answer users.

## Runnable phase-1 dump and training

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/dump_corpus_stack.py \
  --out corpus_stack \
  --sources tinystories,cosmopedia_100k \
  --target-bytes 250000000
PYTHONPATH=. python scripts/train_checkpoint.py \
  --corpus corpus_stack \
  --minutes 20 \
  --out checkpoints/nueronce_stack_phase1.pt
PYTHONPATH=. python scripts/chat_demo.py \
  --ckpt checkpoints/nueronce_stack_phase1.pt
```
