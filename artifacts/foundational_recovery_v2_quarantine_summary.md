# Foundational Recovery V2 Quarantine Summary

- Total rows: **6751**
- Rows carrying at least one warning: **5261**
- Rows without these detected warnings: **1490**

These warnings are triage signals, not automatic deletion decisions.

## Warning counts

| Warning | Rows |
|---|---:|
| synthetic_template_source | 4377 |
| resplit_from_original_source | 1994 |
| code_definition_not_debugging | 209 |
| malformed_plural_agreement | 39 |
| malformed_article_a_this | 18 |
| domain_mismatch_summarization | 15 |
| malformed_article_a_a | 2 |

## Flagged rows by split

| Split | Flagged rows |
|---|---:|
| train | 3788 |
| val | 705 |
| test | 768 |

## Flagged rows by assigned domain

| Domain | Flagged rows |
|---|---:|
| causal_temporal | 288 |
| code_debugging | 412 |
| conversation | 1439 |
| evidence_uncertainty | 911 |
| factual_explanation | 702 |
| mathematics | 1441 |
| planning_tools | 68 |
