"""Trusted source registry.

Every entry is a curated, openly-licensed collection from a reputable archive —
never a blog, social-media feed, or scraped web page. The first checkpoint uses
public-domain material so the data lineage is fully defensible.

These collections are mirrored by the NLTK project's data repository (a trusted,
widely-used, curated archive) and are reachable in restricted environments via
``raw.githubusercontent.com``. The underlying texts are public-domain books
(Project Gutenberg editions) and public-domain US-government speeches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

_NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora"


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
