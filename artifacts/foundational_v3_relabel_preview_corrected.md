# Foundational V3 Corrected Relabel Preview

No training files were created.

## Status summary

| Status | Rows |
|---|---:|
| hard_reject | 41 |
| manual_review | 37 |
| candidate_with_provenance | 5211 |
| clean_candidate | 1462 |

## Proposed capability distribution

| Capability | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| evidence_grounding | 1433 | 989 | 130 | 314 |
| mathematics | 1088 | 1024 | 64 | 0 |
| instruction_following | 1034 | 780 | 101 | 153 |
| uncertainty_abstention | 935 | 658 | 220 | 57 |
| logic_reasoning | 555 | 420 | 67 | 68 |
| factual_knowledge | 488 | 392 | 44 | 52 |
| code_fundamentals | 412 | 345 | 2 | 65 |
| factual_explanation | 268 | 212 | 28 | 28 |
| conversation | 194 | 154 | 23 | 17 |
| rewriting | 134 | 101 | 21 | 12 |
| summarization | 49 | 39 | 4 | 6 |
| planning_procedures | 49 | 42 | 2 | 5 |
| manual_review | 37 | 30 | 5 | 2 |
| safety_refusal | 34 | 30 | 3 | 1 |

## Category mapping

| Category | Capability | Rows |
|---|---|---:|
| pa_grounded_evidence | evidence_grounding | 1433 |
| arithmetic | mathematics | 1088 |
| instruction_following | instruction_following | 1030 |
| facts | factual_knowledge | 488 |
| coding | code_fundamentals | 408 |
| classification | logic_reasoning | 320 |
| definitions | factual_explanation | 236 |
| logic | logic_reasoning | 235 |
| pa_abstain_conflict_rejected | uncertainty_abstention | 200 |
| pa_abstain_conflict_qualified | uncertainty_abstention | 200 |
| pa_abstain_conflict_conflict | uncertainty_abstention | 200 |
| pa_abstain_conflict_insufficient | uncertainty_abstention | 200 |
| rewriting | rewriting | 130 |
| multiturn | conversation | 126 |
| uncertainty | uncertainty_abstention | 67 |
| greetings | conversation | 64 |
| pa_abstain_conflict_revision | uncertainty_abstention | 64 |
| summarization | summarization | 45 |
| procedures | planning_procedures | 45 |
| handwritten | manual_review | 37 |
| refusals | safety_refusal | 30 |
| explanations | factual_explanation | 24 |
| pa_direct_explanation | factual_explanation | 4 |
| pa_direct_rewrite | rewriting | 4 |
| pa_direct_procedure | planning_procedures | 4 |
| pa_direct_greeting | conversation | 4 |
| pa_direct_instruction | instruction_following | 4 |
| pa_direct_summary | summarization | 4 |
| pa_direct_uncertainty | uncertainty_abstention | 4 |
| pa_direct_refusal | safety_refusal | 4 |
| pa_direct_coding | code_fundamentals | 4 |
| pa_direct_definition | factual_explanation | 4 |

## Candidate code-debugging examples

Detected examples: **0**

