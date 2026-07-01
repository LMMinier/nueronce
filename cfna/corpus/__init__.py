"""Open, license-clean corpus acquisition for CFNA's first real checkpoint.

Pipeline (per the build plan):

    trusted sources -> download -> clean text -> license check -> deduplicate
    -> split by document -> UTF-8 bytes -> training batches -> gradient updates
    -> saved CFNA weights

Only trusted, curated, openly-licensed sources are used — public-domain books and
US-government public-domain speeches. No blogs, no social media, no scraped web.
Documents are bucketed by license; the first checkpoint trains only from
``safe_commercial`` (public domain / CC0 / CC BY).
"""

from __future__ import annotations

import importlib


def __getattr__(name):
    if name in {"sources", "build", "dataset", "stack"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'cfna.corpus' has no attribute {name!r}")

__all__ = ["sources", "build", "dataset", "stack"]
