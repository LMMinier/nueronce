"""Balanced SFT dataset construction.

The 100K synthetic SFT run proved the training stack works, but its category
mix was heavily arithmetic/classification dominated. This module builds a
smaller balanced curriculum from the same self-authored generators so
underrepresented conversational skills get enough gradient signal to test
coherence without changing the architecture.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .synthetic_dialogue import GENERATORS, Record


def _records_for_category(name: str) -> List[Record]:
    for cat, gen_fn in GENERATORS:
        if cat == name:
            return list(gen_fn())
    raise KeyError(f"unknown SFT category: {name}")


def balanced_records(
    examples_per_category: int = 500,
    categories: Optional[Sequence[str]] = None,
    repeat_small_categories: bool = True,
) -> Iterator[Record]:
    """Yield records with an equal budget per category.

    Some categories intentionally have fewer unique template records than the
    requested budget. When ``repeat_small_categories`` is true, those categories
    are cycled with new ids so they still receive a balanced training weight.
    The repeated records should be understood as weighting, not new independent
    data.
    """
    wanted = list(categories) if categories is not None else [name for name, _ in GENERATORS]
    for name in wanted:
        records = _records_for_category(name)
        if not records:
            continue
        if repeat_small_categories:
            iterator: Iterable[Record] = itertools.islice(itertools.cycle(records), examples_per_category)
        else:
            iterator = records[:examples_per_category]
        for i, rec in enumerate(iterator):
            out = dict(rec)
            out["id"] = f"balanced-{name}-{i:05d}-{rec['id']}"
            out["source"] = f"{rec.get('source', 'unknown')}|balanced-weighted"
            yield out


def category_counts(records: Iterable[Record]) -> Dict[str, int]:
    return dict(Counter(str(r.get("category", "unknown")) for r in records))


__all__ = ["balanced_records", "category_counts"]
