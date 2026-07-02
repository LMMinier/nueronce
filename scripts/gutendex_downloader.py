#!/usr/bin/env python3
"""Gutendex downloader

Usage examples:
  # Download by Gutenberg ID range
  python scripts/gutendex_downloader.py --start-id 1342 --end-id 1345 --output data/raw

  # Download by query (author or title)
  python scripts/gutendex_downloader.py --query "Charlotte Bronte" --limit 50 --output data/raw

This script uses the Gutendex API (https://gutendex.com) to fetch metadata and download plain-text formats when available.
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE = "https://gutendex.com/books"


def fetch_books_by_query(query, page_size=32, limit=None):
    books = []
    page = 1
    while True:
        params = {"search": query, "page": page}
        r = requests.get(API_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        books.extend(data.get("results", []))
        if limit and len(books) >= limit:
            return books[:limit]
        if not data.get("next"):
            break
        page += 1
        time.sleep(0.2)
    return books


def fetch_book_by_id(book_id):
    r = requests.get(f"{API_BASE}/{book_id}", timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def choose_text_url(formats):
    # Prefer plain text UTF-8, then ASCII, then others (epub fallback)
    priorities = [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
        ".txt",
        "application/epub+zip",
    ]
    for p in priorities:
        for k, v in formats.items():
            if p in k or k.endswith(p) or p in k:
                return v
    # fallback: first url-looking entry
    for v in formats.values():
        if isinstance(v, str) and v.startswith("http"):
            return v
    return None


def download_file(url, outpath, chunk_size=8192):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        if total is None:
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        else:
            total = int(total)
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
    return outpath


def sanitize_filename(s):
    keep = " ._-()[]{}" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join((c if c in keep else "_") for c in s)[:200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", help="Search query for Gutendex (title/author)")
    p.add_argument("--ids", help="Comma-separated Gutenberg IDs to download, e.g. 1342,84")
    p.add_argument("--start-id", type=int, help="Start Gutenberg ID (inclusive)")
    p.add_argument("--end-id", type=int, help="End Gutenberg ID (inclusive)")
    p.add_argument("--limit", type=int, help="Max books to download for queries")
    p.add_argument("--output", default="data/raw", help="Output directory")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests")
    p.add_argument("--skip-existing", action="store_true", help="Skip download if file exists")
    args = p.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    to_fetch = []
    if args.ids:
        for sid in args.ids.split(','):
            sid = sid.strip()
            if sid:
                to_fetch.append(int(sid))
    if args.start_id and args.end_id:
        to_fetch.extend(list(range(args.start_id, args.end_id + 1)))
    if args.query:
        books = fetch_books_by_query(args.query, limit=args.limit)
        for b in books:
            to_fetch.append(b.get('id'))

    # dedupe
    seen = set()
    to_fetch = [x for x in to_fetch if x not in seen and not seen.add(x)]

    print(f"Found {len(to_fetch)} book ids to process")

    for bid in tqdm(to_fetch):
        try:
            meta = fetch_book_by_id(bid)
            if not meta:
                tqdm.write(f"Book {bid} not found")
                continue
            title = meta.get('title') or f"gutenberg_{bid}"
            authors = ", ".join([a.get('name','') for a in meta.get('authors',[])])
            formats = meta.get('formats', {})
            txt_url = choose_text_url(formats)
            if not txt_url:
                tqdm.write(f"No text/epub format found for {bid} - {title}")
                continue
            safe_title = sanitize_filename(f"{bid}_{title}")
            out_txt = outdir / f"{safe_title}.txt"
            meta_file = outdir / f"{safe_title}.meta.json"
            if args.skip_existing and out_txt.exists():
                tqdm.write(f"Skipping existing {out_txt}")
                continue
            download_file(txt_url, out_txt)
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump({
                    "id": bid,
                    "title": title,
                    "authors": meta.get('authors', []),
                    "language": meta.get('languages', []),
                    "formats": formats,
                    "download_url": txt_url,
                }, mf, ensure_ascii=False, indent=2)
            time.sleep(args.sleep)
        except Exception as e:
            tqdm.write(f"Error downloading {bid}: {e}")


if __name__ == '__main__':
    main()
