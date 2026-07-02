"""Recommended staged corpus stack for CFNA training.

This registry is intentionally explicit: every source has a page URL, file URL,
license label, intended training phase, and notes about whether it should be used
for base-language pretraining or later instruction tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CorpusStackEntry:
    source_id: str
    name: str
    phase: int
    role: str
    dataset_page: str
    files_page: str
    license: str
    loader: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    text_fields: Tuple[str, ...] = ("text",)
    document_template: Optional[str] = None
    target_notes: str = ""
    extra_links: Dict[str, str] = field(default_factory=dict)


CORPUS_STACK: List[CorpusStackEntry] = [
    CorpusStackEntry(
        source_id="tinystories",
        name="TinyStories",
        phase=1,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/roneneldan/TinyStories",
        files_page="https://huggingface.co/datasets/roneneldan/TinyStories/tree/main",
        license="CDLA-Sharing 1.0",
        loader="huggingface",
        dataset_name="roneneldan/TinyStories",
        text_fields=("text",),
        target_notes="Start here to prove spelling, grammar, sentence structure, and short narratives.",
    ),
    CorpusStackEntry(
        source_id="cosmopedia_100k",
        name="Cosmopedia-100k",
        phase=1,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k",
        files_page="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k/tree/main",
        license="Apache-2.0",
        loader="huggingface",
        dataset_name="HuggingFaceTB/cosmopedia-100k",
        text_fields=("text", "markdown", "content"),
        target_notes="Second corpus: educational chapters, explanations, stories, and articles.",
    ),
    CorpusStackEntry(
        source_id="fineweb_edu_sample_10bt",
        name="FineWeb-Edu sample-10BT",
        phase=2,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        files_page="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main",
        license="ODC-By 1.0 (dataset card; verify downstream terms before release)",
        loader="huggingface",
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        streaming=True,
        text_fields=("text",),
        target_notes="Stream only a bounded byte count; do not download the full dataset.",
    ),
    CorpusStackEntry(
        source_id="smollm_cosmopedia_v2",
        name="SmolLM-Corpus Cosmopedia v2",
        phase=3,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
        files_page="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main",
        license="Mixed; preserve subset/source metadata",
        loader="huggingface",
        dataset_name="HuggingFaceTB/smollm-corpus",
        dataset_config="cosmopedia-v2",
        streaming=True,
        text_fields=("text", "content"),
    ),
    CorpusStackEntry(
        source_id="smollm_fineweb_edu_dedup",
        name="SmolLM-Corpus FineWeb-Edu-Dedup",
        phase=3,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
        files_page="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main",
        license="Mixed; preserve subset/source metadata",
        loader="huggingface",
        dataset_name="HuggingFaceTB/smollm-corpus",
        dataset_config="fineweb-edu-dedup",
        streaming=True,
        text_fields=("text", "content"),
    ),
    CorpusStackEntry(
        source_id="smollm_python_edu",
        name="SmolLM-Corpus Python-Edu",
        phase=3,
        role="base_pretraining_code",
        dataset_page="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
        files_page="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main",
        license="Per-repository mixed code licenses; preserve repository/license metadata",
        loader="huggingface",
        dataset_name="HuggingFaceTB/smollm-corpus",
        dataset_config="python-edu",
        streaming=True,
        text_fields=("text", "content", "code"),
        target_notes="Educational Python code; keep repository/license provenance.",
    ),
    CorpusStackEntry(
        source_id="english_wikipedia_latest",
        name="English Wikipedia latest dump",
        phase=2,
        role="base_pretraining",
        dataset_page="https://dumps.wikimedia.org/enwiki/latest/",
        files_page="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2",
        license="CC BY-SA / GFDL; preserve attribution/share-alike requirements",
        loader="wikimedia_dump",
        text_fields=("text",),
        extra_links={
            "all_dumps": "https://dumps.wikimedia.org/",
            "index": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2",
            "wikiextractor": "https://github.com/attardi/wikiextractor",
        },
    ),
    CorpusStackEntry(
        source_id="project_gutenberg",
        name="Project Gutenberg",
        phase=2,
        role="base_pretraining_books",
        dataset_page="https://www.gutenberg.org/",
        files_page="https://www.gutenberg.org/ebooks/offline_catalogs.html",
        license="Public domain in the United States for selected works; verify territory per item",
        loader="catalog_manual",
        text_fields=("text",),
        extra_links={
            "bookshelves": "https://www.gutenberg.org/ebooks/bookshelf/",
            "bulk_policy": "https://www.gutenberg.org/policy/robot_access.html",
            "rdf_catalog": "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2",
        },
    ),
    CorpusStackEntry(
        source_id="standard_ebooks",
        name="Standard Ebooks",
        phase=3,
        role="base_pretraining_books",
        dataset_page="https://standardebooks.org/ebooks",
        files_page="https://standardebooks.org/opds/all",
        license="Public-domain source works plus CC0 Standard Ebooks production work",
        loader="catalog_manual",
        text_fields=("text",),
        extra_links={"github": "https://github.com/standardebooks"},
    ),
    CorpusStackEntry(
        source_id="open_textbook_library",
        name="Open Textbook Library",
        phase=3,
        role="technical_pretraining",
        dataset_page="https://open.umn.edu/opentextbooks",
        files_page="https://open.umn.edu/opentextbooks/subjects",
        license="Per-title Creative Commons; record exact license in manifest",
        loader="catalog_manual",
        text_fields=("text",),
    ),
    CorpusStackEntry(
        source_id="libretexts",
        name="LibreTexts",
        phase=3,
        role="technical_pretraining",
        dataset_page="https://libretexts.org/",
        files_page="https://libretexts.org/",
        license="Per-page/per-book license; record exact license in manifest",
        loader="catalog_manual",
        text_fields=("text",),
        extra_links={
            "math": "https://math.libretexts.org/",
            "computer_science": "https://eng.libretexts.org/Bookshelves/Computer_Science",
            "engineering": "https://eng.libretexts.org/",
            "physics": "https://phys.libretexts.org/",
            "chemistry": "https://chem.libretexts.org/",
            "biology": "https://bio.libretexts.org/",
            "statistics": "https://stats.libretexts.org/",
            "social_sciences": "https://socialsci.libretexts.org/",
            "humanities": "https://human.libretexts.org/",
        },
    ),
    CorpusStackEntry(
        source_id="pmc_oa_comm",
        name="PMC Open Access commercial collection",
        phase=3,
        role="technical_pretraining_science",
        dataset_page="https://pmc.ncbi.nlm.nih.gov/tools/openftlist/",
        files_page="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/",
        license="Per-article OA commercial license; preserve article license metadata",
        loader="pmc_oa_bulk",
        text_fields=("text",),
        extra_links={
            "ftp_docs": "https://pmc.ncbi.nlm.nih.gov/tools/ftp/",
            "noncommercial": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/",
            "other": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/",
        },
    ),
    CorpusStackEntry(
        source_id="oasst1_en",
        name="OpenAssistant OASST1 English",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/OpenAssistant/oasst1",
        files_page="https://huggingface.co/datasets/OpenAssistant/oasst1/tree/main",
        license="Apache-2.0",
        loader="huggingface",
        dataset_name="OpenAssistant/oasst1",
        text_fields=("text",),
        document_template="oasst1",
        target_notes="Use after base pretraining; filter rows where lang == 'en'.",
    ),
    CorpusStackEntry(
        source_id="dolly_15k",
        name="Databricks Dolly 15K",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        files_page="https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/main",
        license="CC-BY-SA-3.0",
        loader="huggingface",
        dataset_name="databricks/databricks-dolly-15k",
        text_fields=("instruction", "context", "response"),
        document_template="dolly",
        extra_links={
            "raw_jsonl": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        },
        target_notes="Use after the base model can already generate coherent text.",
    ),
    # ------------------------------------------------------------------ #
    # Subject expansion: math / physics / multi-language code (phase 2-3)
    # ------------------------------------------------------------------ #
    CorpusStackEntry(
        source_id="open_web_math",
        name="OpenWebMath",
        phase=2,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/open-web-math/open-web-math",
        files_page="https://huggingface.co/datasets/open-web-math/open-web-math/tree/main",
        license="ODC-By 1.0 (verify downstream obligations before release)",
        loader="huggingface",
        dataset_name="open-web-math/open-web-math",
        streaming=True,
        text_fields=("text",),
        target_notes="Mathematical/physics web text with LaTeX preserved. Stream a bounded "
                     "byte budget only; this is the math/quantum register the model needs.",
    ),
    CorpusStackEntry(
        source_id="the_stack_smol",
        name="The Stack (smol, permissive code)",
        phase=3,
        role="base_pretraining",
        dataset_page="https://huggingface.co/datasets/bigcode/the-stack-smol",
        files_page="https://huggingface.co/datasets/bigcode/the-stack-smol/tree/main",
        license="Permissive per-record (MIT/Apache/BSD...; filter on the per-row license field)",
        loader="huggingface",
        dataset_name="bigcode/the-stack-smol",
        streaming=True,
        text_fields=("content",),
        target_notes="~30 programming languages, 10k files each. Filter rows to an explicit "
                     "permissive-license allowlist (mit, apache-2.0, bsd-2/3-clause, isc, "
                     "cc0-1.0, unlicense) exactly as the forgeloop notebook does.",
    ),
    CorpusStackEntry(
        source_id="gsm8k",
        name="GSM8K grade-school math word problems",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/openai/gsm8k",
        files_page="https://huggingface.co/datasets/openai/gsm8k/tree/main",
        license="MIT",
        loader="huggingface",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        text_fields=("question", "answer"),
        document_template="qa",
        target_notes="Step-by-step math answers; convert via cfna.training.mcq_sft (qa mode).",
    ),
    # ------------------------------------------------------------------ #
    # MCQ instruction tuning (phase 4) — convert via cfna.training.mcq_sft
    # ------------------------------------------------------------------ #
    CorpusStackEntry(
        source_id="arc_easy",
        name="AI2 ARC (Easy)",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/allenai/ai2_arc",
        files_page="https://huggingface.co/datasets/allenai/ai2_arc/tree/main",
        license="CC-BY-SA-4.0 (share-alike: keep derived SFT files under the same terms)",
        loader="huggingface",
        dataset_name="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        text_fields=("question",),
        document_template="mcq",
        target_notes="Science MCQs; fields question/choices{text,label}/answerKey.",
    ),
    CorpusStackEntry(
        source_id="arc_challenge",
        name="AI2 ARC (Challenge)",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/allenai/ai2_arc",
        files_page="https://huggingface.co/datasets/allenai/ai2_arc/tree/main",
        license="CC-BY-SA-4.0 (share-alike: keep derived SFT files under the same terms)",
        loader="huggingface",
        dataset_name="allenai/ai2_arc",
        dataset_config="ARC-Challenge",
        text_fields=("question",),
        document_template="mcq",
        target_notes="Harder science MCQs; same schema as ARC-Easy.",
    ),
    CorpusStackEntry(
        source_id="openbookqa",
        name="OpenBookQA",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/allenai/openbookqa",
        files_page="https://huggingface.co/datasets/allenai/openbookqa/tree/main",
        license="Apache-2.0",
        loader="huggingface",
        dataset_name="allenai/openbookqa",
        dataset_config="main",
        text_fields=("question_stem",),
        document_template="mcq",
        target_notes="Open-book science MCQs; fields question_stem/choices/answerKey.",
    ),
    CorpusStackEntry(
        source_id="commonsense_qa",
        name="CommonsenseQA",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/tau/commonsense_qa",
        files_page="https://huggingface.co/datasets/tau/commonsense_qa/tree/main",
        license="MIT",
        loader="huggingface",
        dataset_name="tau/commonsense_qa",
        text_fields=("question",),
        document_template="mcq",
        target_notes="Commonsense MCQs; fields question/choices/answerKey.",
    ),
    CorpusStackEntry(
        source_id="math_qa",
        name="MathQA",
        phase=4,
        role="instruction_tuning",
        dataset_page="https://huggingface.co/datasets/allenai/math_qa",
        files_page="https://huggingface.co/datasets/allenai/math_qa/tree/main",
        license="Apache-2.0",
        loader="huggingface",
        dataset_name="allenai/math_qa",
        text_fields=("Problem",),
        document_template="mcq",
        target_notes="Math word-problem MCQs; fields Problem/options/correct/Rationale.",
    ),
]
# NOTE deliberately excluded: SciQ (CC BY-NC — noncommercial), MMLU (terms of the
# aggregated sources are unclear). Cosmopedia/LibreTexts/PMC already cover the
# health / psychology / sociology / finance registers with clean licenses.


def entries_for_phase(max_phase: int) -> List[CorpusStackEntry]:
    return [entry for entry in CORPUS_STACK if entry.phase <= max_phase]


def get_entry(source_id: str) -> CorpusStackEntry:
    for entry in CORPUS_STACK:
        if entry.source_id == source_id:
            return entry
    raise KeyError(source_id)


__all__ = ["CorpusStackEntry", "CORPUS_STACK", "entries_for_phase", "get_entry"]
