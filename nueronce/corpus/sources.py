"""Trusted source registry.

Every entry is a curated, openly-licensed collection from a reputable archive —
never a blog, social-media feed, or scraped web page. The first checkpoint uses
public-domain material so the data lineage is fully defensible.

These collections are mirrored by the NLTK project's data repository (a trusted,
widely-used, curated archive) and are reachable in restricted environments via
``raw.githubusercontent.com``. The underlying texts are public-domain books
(Project Gutenberg editions) and public-domain US-government speeches.

To expand the model's command of language, the registry also pulls a curated set
of individual **public-domain** books directly from Project Gutenberg's canonical
``www.gutenberg.org`` cache. These are full-length literary works (US public
domain) and remain commercial-safe with no attribution requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

_NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora"
_GUTENBERG = "https://www.gutenberg.org/cache/epub"  # canonical PD book cache


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    collection: str
    document_type: str          # book | speech | nonfiction
    license: str                # human label
    license_id: str             # SPDX-ish / "public-domain-us"
    commercial_use: bool
    attribution_required: bool
    bucket: str                 # safe_commercial | restricted_research | share_alike_review
    note: str = ""
    # Optional metadata for single-document (non-zip) book sources. When set, the
    # builder records these directly instead of parsing an NLTK-style header.
    title: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[int] = None


# Public-domain, trusted, commercial-safe. These go in safe_commercial/public_domain.
TRUSTED_SOURCES: List[Source] = [
    Source(
        source_id="gutenberg_classics",
        url=f"{_NLTK}/gutenberg.zip",
        collection="project_gutenberg (via nltk_data)",
        document_type="book",
        license="public domain (US)",
        license_id="public-domain-us",
        commercial_use=True,
        attribution_required=False,
        bucket="safe_commercial",
        note="Austen, Carroll, Melville, Shakespeare, Milton, Whitman, Chesterton, KJV, etc.",
    ),
    Source(
        source_id="us_inaugural_addresses",
        url=f"{_NLTK}/inaugural.zip",
        collection="US presidential inaugural addresses (via nltk_data)",
        document_type="speech",
        license="public domain (US federal government work)",
        license_id="public-domain-usgov",
        commercial_use=True,
        attribution_required=False,
        bucket="safe_commercial",
        note="Formal rhetoric / speech register, 1789-present.",
    ),
    Source(
        source_id="us_state_of_the_union",
        url=f"{_NLTK}/state_union.zip",
        collection="US State of the Union addresses (via nltk_data)",
        document_type="speech",
        license="public domain (US federal government work)",
        license_id="public-domain-usgov",
        commercial_use=True,
        attribution_required=False,
        bucket="safe_commercial",
        note="Modern formal/explanatory government prose.",
    ),
]

# Curated full-length public-domain books pulled directly from Project Gutenberg's
# canonical cache. Each is a US public-domain literary work — broad, well-edited
# prose that gives the byte model a far richer command of English than the small
# NLTK gutenberg sample alone. (id, title, author, year)
_GUTENBERG_BOOKS = [
    (1342, "Pride and Prejudice", "Jane Austen", 1813),
    (158, "Emma", "Jane Austen", 1815),
    (161, "Sense and Sensibility", "Jane Austen", 1811),
    (84, "Frankenstein", "Mary Wollstonecraft Shelley", 1818),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", 1892),
    (2701, "Moby Dick; or, The Whale", "Herman Melville", 1851),
    (98, "A Tale of Two Cities", "Charles Dickens", 1859),
    (1400, "Great Expectations", "Charles Dickens", 1861),
    (345, "Dracula", "Bram Stoker", 1897),
    (174, "The Picture of Dorian Gray", "Oscar Wilde", 1890),
    (76, "Adventures of Huckleberry Finn", "Mark Twain", 1884),
    (74, "The Adventures of Tom Sawyer", "Mark Twain", 1876),
    (120, "Treasure Island", "Robert Louis Stevenson", 1883),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson", 1886),
    (768, "Wuthering Heights", "Emily Bronte", 1847),
    (1260, "Jane Eyre", "Charlotte Bronte", 1847),
    (2554, "Crime and Punishment", "Fyodor Dostoevsky", 1866),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas", 1844),
    (215, "The Call of the Wild", "Jack London", 1903),
    (2542, "A Doll's House", "Henrik Ibsen", 1879),
    (1232, "The Prince", "Niccolo Machiavelli", 1532),
    (5200, "Metamorphosis", "Franz Kafka", 1915),
    (25344, "The Scarlet Letter", "Nathaniel Hawthorne", 1850),
    (16, "Peter Pan", "J. M. Barrie", 1911),
    (209, "The Turn of the Screw", "Henry James", 1898),
]


def _gutenberg_book_sources() -> List[Source]:
    sources: List[Source] = []
    for book_id, title, author, year in _GUTENBERG_BOOKS:
        sources.append(Source(
            source_id="gutenberg_books",
            url=f"{_GUTENBERG}/{book_id}/pg{book_id}.txt",
            collection="project_gutenberg (canonical cache)",
            document_type="book",
            license="public domain (US)",
            license_id="public-domain-us",
            commercial_use=True,
            attribution_required=False,
            bucket="safe_commercial",
            note=f"{title} — {author}",
            title=title,
            author=author,
            publication_year=year,
        ))
    return sources


TRUSTED_SOURCES += _gutenberg_book_sources()


# Explicitly NOT acquired for the first checkpoint (kept here as a guardrail so
# the exclusion is visible and intentional).
EXCLUDED_KINDS = (
    "blogs", "social_media", "forums", "webtext", "chat_logs", "news_scrape",
    "youtube_transcripts", "podcast_transcripts", "song_lyrics", "reddit",
    "common_crawl", "copyrighted_books", "share_alike_without_review",
)


def safe_commercial_sources() -> List[Source]:
    return [s for s in TRUSTED_SOURCES if s.bucket == "safe_commercial"]


__all__ = ["Source", "TRUSTED_SOURCES", "EXCLUDED_KINDS", "safe_commercial_sources"]
