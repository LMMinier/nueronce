LMMinier/nueronce — downloader & preprocessing for public-domain books

This directory adds tooling to fetch public-domain books (Project Gutenberg) and prepare them as training-ready shards for a foundational model.

Files added
- scripts/gutendex_downloader.py  -- downloads plain-text/epub formats from Gutendex (https://gutendex.com)
- scripts/preprocess_to_jsonl.py  -- basic cleaning and writes JSONL shards
- requirements.txt                -- runtime dependencies for the scripts
- .gitignore (updated)            -- ignore data/ and DVC state

Recommended workflow
1. Install dependencies
   pip install -r requirements.txt

2. Download books (example)
   # download a range of Gutenberg IDs into data/raw
   python scripts/gutendex_downloader.py --start-id 1342 --end-id 1345 --output data/raw

   # or search by author/title
   python scripts/gutendex_downloader.py --query "Jane Austen" --limit 50 --output data/raw

3. Preprocess into shards
   python scripts/preprocess_to_jsonl.py --input data/raw --output data/processed --shard-size-mb 200

4. Manage large files outside git
   - I recommend using DVC or storing data on an external object store (S3/GCS) and keeping only pointers in the repo.
   - Example: dvc init; dvc add data/processed; dvc remote add -d myremote s3://...; dvc push

Legal & ethical notes
- These scripts fetch public-domain works from Gutendex / Project Gutenberg. Ensure you follow the licenses of any source you use.
- Do not use these scripts to fetch copyrighted material you are not permitted to redistribute.

If you want, I can now:
- add DVC pipeline files and a sample dvc.yaml that tracks data/processed (requires you to configure a remote later),
- add tokenization / sharding tailored to your model (e.g., byte-pair encoding, SentencePiece, TFRecord output), or
- expand the downloader to also fetch Kiwix/ZIM or mirror repositories.

Tell me which of the above you'd like next and I will add it to the repo.