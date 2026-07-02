# NUERONCE Extensive Corpus Catalog

Generated: 2026-07-02

## What is included

- **153 collection-level sources** across books, encyclopedias, philosophy,
  psychology, sociology, science, health, mathematics, physics, quantum computing,
  speeches, government records, writing, grammar, dictionaries, scholarly articles,
  multilingual corpora, and programming documentation.
- **289 title-level book/text recommendations**.
- **10 explicit exclusion rules** for proprietary, mixed-rights, unsafe,
  or legally ambiguous content.
- CSV, JSONL, and SQLite formats.
- A repository-ready Python catalog loader.

## License buckets

- **A_PD_CC0** — 14 sources. Public-domain, U.S. federal work, or CC0. Best fit for the current NUERONCE loader.
- **B_ATTRIBUTION** — 25 sources. Commercially reusable open license, but attribution/provenance must be preserved.
- **C_SHARE_ALIKE** — 14 sources. Open but share-alike/GFDL obligations require a separate legal and release workflow.
- **D_NONCOMMERCIAL** — 2 sources. Research-only or noncommercial license. Never mix into a commercial checkpoint.
- **E_PER_ITEM** — 83 sources. Collection is mixed. Resolve and store the license for every individual document.
- **F_REFERENCE_ONLY** — 15 sources. Do not bulk ingest without a negotiated license or explicit permission.

## Critical finding about the current repository

The existing NUERONCE corpus builder is deliberately narrow. It accepts only a small
host allowlist and only public-domain/CC0-style records with no attribution obligation.
That means OpenStax, Wikipedia/Wikibooks, most open-access journals, official programming
documentation, arXiv, and mixed repositories cannot simply be added to the current
`TRUSTED_SOURCES` list.

The correct expansion is an adapter architecture:

1. **Catalog discovery** — find records without downloading training text.
2. **Rights resolver** — capture the exact item-level license and a copy/hash of the
   rights statement.
3. **Source adapter** — API, XML, EPUB, PDF, Git, HTML, WARC, or repository-specific parser.
4. **Content normalizer** — preserve code blocks, equations, headings, citations, tables,
   and language metadata instead of flattening everything into plain prose.
5. **Quality and safety gate** — dedupe, language ID, OCR score, retraction checks,
   medical-risk labels, PII filtering, and historical-bias labels.
6. **License-separated storage** — never merge public domain, attribution, share-alike,
   noncommercial, and per-item corpora into one undifferentiated directory.
7. **Mixture manifest** — every checkpoint records source weights and document hashes.

## Recommended directory layout

```text
corpus/
  catalog/
    collections.csv
    works.csv
    exclusions.csv
  A_PD_CC0/
  B_ATTRIBUTION/
  C_SHARE_ALIKE/
  D_NONCOMMERCIAL/
  E_PER_ITEM/
  quarantine/
  manifests/
    documents.jsonl
    licenses.jsonl
    checkpoint_mixtures.jsonl
```

## Recommended first production mixture

Do not train on everything at equal weight.

- 25% public-domain literature and essays
- 15% writing, grammar, rhetoric, and speeches
- 15% open textbooks: math, physics, biology, psychology, sociology
- 20% permissively licensed programming documentation and examples
- 10% U.S. government science, policy, and technical reports
- 10% peer-reviewed open-access science/health articles
- 5% dictionaries, multilingual parallel data, and structured reference

Use document-uniform sampling within collections, then cap any single author, site,
journal, programming language, or government agency.

## Medical and science rules

- Never present historical medical books as current health guidance.
- Store publication date, retraction/correction status, peer-review status, and evidence type.
- Keep preprints separate from peer-reviewed literature.
- Use current CDC, NIH, FDA, and systematic-review sources for retrieval grounding.
- Train generation style separately from factual retrieval so stale facts are not baked in
  as unquestioned truth.

## Britannica rule

Current Britannica website content is proprietary and belongs in **F_REFERENCE_ONLY**.
The **1911 Encyclopaedia Britannica** is a public-domain historical reference candidate,
but it must be labeled as historical because many entries are obsolete or biased.

## Public-domain edition rule

A public-domain author does not guarantee a public-domain file. Modern translations,
introductions, annotations, critical editions, illustrations, and audiobook performances
may be separately copyrighted. The manifest must identify the exact edition and rights basis.

## Files

- `nueronce_corpus_catalog.sqlite`
- `nueronce_corpus_collections.csv`
- `nueronce_corpus_collections.jsonl`
- `nueronce_public_domain_and_open_books.csv`
- `nueronce_public_domain_and_open_books.jsonl`
- `nueronce_corpus_exclusions.csv`
- `expanded_source_catalog.py`
