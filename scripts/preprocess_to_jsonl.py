#!/usr/bin/env python3
"""Preprocess downloaded Gutenberg texts into JSONL shards suitable for model training.

This script:
 - removes simple Gutenberg headers/footers
 - normalizes whitespace
 - filters short documents
 - writes JSONL shards with metadata and text

Usage:
  python scripts/preprocess_to_jsonl.py --input data/raw --output data/processed --min-chars 1000 --shard-size-mb 100
"""

import argparse
import io
import json
import os
import re
from pathlib import Path

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\n", re.IGNORECASE | re.DOTALL)
GUTENBERG_END_RE = re.compile(r"\n\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg_header_footer(text):
    # Try to find the common Project Gutenberg start/end markers
    start_match = re.search(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\n", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    end_match = re.search(r"\n\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    return text


def normalize_whitespace(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def process_file(path):
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    text = strip_gutenberg_header_footer(text)
    text = normalize_whitespace(text)
    return text


def iter_raw_texts(indir):
    p = Path(indir)
    for f in sorted(p.glob('*.txt')):
        meta = None
        # try to read accompanying metadata if present
        meta_path = f.with_suffix('.meta.json')
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception:
                meta = None
        yield f.name, f, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--min-chars', type=int, default=1000)
    parser.add_argument('--shard-size-mb', type=int, default=100)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    shard_bytes = 0
    shard_file = None

    def open_shard(idx):
        p = outdir / f'shard_{idx:04d}.jsonl'
        return open(p, 'w', encoding='utf-8')

    shard_file = open_shard(shard_index)

    for name, text_path, meta in iter_raw_texts(args.input):
        try:
            text = process_file(text_path)
            if len(text) < args.min_chars:
                print(f"Skipping {name}: too short ({len(text)} chars)")
                continue
            doc = {
                'id': name.replace('.txt',''),
                'filename': name,
                'meta': meta or {},
                'text': text,
            }
            line = json.dumps(doc, ensure_ascii=False)
            b = line.encode('utf-8')
            shard_file.write(line + '\n')
            shard_bytes += len(b)
            if shard_bytes >= args.shard_size_mb * 1024 * 1024:
                shard_file.close()
                shard_index += 1
                shard_file = open_shard(shard_index)
                shard_bytes = 0
        except Exception as e:
            print(f"Error processing {name}: {e}")
    shard_file.close()
    print(f"Wrote {shard_index+1} shards to {outdir}")


if __name__ == '__main__':
    main()
