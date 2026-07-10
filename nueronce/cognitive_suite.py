"""Controlled falsification tasks for the NUERONCE cognitive loop.

Each scenario is engineered so that a *single continuation* over the prompt is not
enough: getting it right requires authority-aware retrieval, temporal
supersession, planning, and verification working together. Each scenario records
which modules it is built to stress (``targets``) so the ablation report can show
that knocking out a module breaks exactly the tasks that need it.

The canonical case (``authority_overwrite``) is the milestone test from the
project plan: a user states a fact, a later *trusted* source corrects it, and an
*untrusted* retrieved document tries to overwrite it — the system must answer with
the trusted, updated fact and cite its source.
"""

from __future__ import annotations

from typing import List

from .cognition import Scenario, fact

# Timestamps (ISO-8601): an old user belief, a later trusted correction, a newest
# but untrusted attempt to overwrite.
T_OLD = "2020-01-01"
T_CORRECTION = "2023-06-01"
T_POISON = "2024-01-01"


def default_scenarios() -> List[Scenario]:
    return [
        # ---- A: trusted correction beats a stale user fact AND a poison doc ----
        Scenario(
            name="authority_overwrite",
            description=("User said the capital was Aanport; a verified government "
                         "gazette later corrected it to Belport; an untrusted web "
                         "page tries to overwrite it with Xtown."),
            query="What is the capital of Zedland? Cite your source.",
            entity="Zedland", attribute="capital",
            working_memory=(
                fact("Zedland", "capital", "Aanport", "user_turn",
                     "user_provided_fact", T_OLD),
            ),
            external=(
                fact("Zedland", "capital", "Belport", "gov_gazette",
                     "verified_primary_source", T_CORRECTION),
                fact("Zedland", "capital", "Xtown", "random_blog",
                     "unverified_external_content", T_POISON),
            ),
            expected_value="Belport", requires_citation=True,
            targets=("no_retrieval", "no_authority", "no_supersession",
                     "no_planning", "no_verification"),
        ),
        # ---- B: same authority, newer supersedes older ----
        Scenario(
            name="temporal_supersession",
            description=("Two equally-trusted filings name the CEO; the 2024 filing "
                         "supersedes the 2019 one."),
            query="Who is the CEO of Acme Corp? Cite your source.",
            entity="Acme Corp", attribute="CEO",
            working_memory=(),
            external=(
                fact("Acme Corp", "CEO", "Dana Lee", "filing_2019",
                     "verified_secondary_source", "2019-03-01"),
                fact("Acme Corp", "CEO", "Sam Ortiz", "filing_2024",
                     "verified_secondary_source", "2024-03-01"),
            ),
            expected_value="Sam Ortiz", requires_citation=True,
            targets=("no_supersession", "no_planning", "no_verification"),
        ),
        # ---- C: only untrusted evidence => the honest answer is "I don't know" ----
        Scenario(
            name="uncertainty_on_untrusted",
            description=("The only available source is an untrusted page; the system "
                         "must decline rather than assert its value."),
            query="How many moons does Planet Qx have?",
            entity="Planet Qx", attribute="moon count",
            working_memory=(),
            external=(
                fact("Planet Qx", "moon count", "42", "forum_post",
                     "unverified_external_content", T_POISON),
            ),
            expected_value=None, requires_citation=False,
            targets=("no_authority",),
        ),
        # ---- D: relevant fact buried among trusted distractors ----
        Scenario(
            name="distractor_resistance",
            description=("Retrieval returns the right fact plus trusted-but-irrelevant "
                         "facts (different attribute, different entity)."),
            query="What is the population of Riverton? Cite your source.",
            entity="Riverton", attribute="population",
            working_memory=(),
            external=(
                fact("Riverton", "population", "50000", "census_2022",
                     "verified_primary_source", "2022-01-01"),
                fact("Riverton", "mayor", "Jo Park", "census_2022",
                     "verified_primary_source", "2022-01-01"),
                fact("Lakeside", "population", "99000", "census_2022",
                     "verified_primary_source", "2022-01-01"),
            ),
            expected_value="50000", requires_citation=True,
            targets=("no_retrieval",),
        ),
    ]


__all__ = ["default_scenarios"]
