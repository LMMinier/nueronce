# Foundational Recovery V3 Build Report

## Build summary

- Source rows: **6751**
- Eligible after status filtering: **6673**
- Approved after additional quality filtering: **6540**
- Final rows after family caps: **4834**
- Family cap: **32 rows**
- Supported capabilities: **13**
- Unsupported capability: **code_debugging**

## New split sizes

| Split | Rows |
|---|---:|
| train | 3718 |
| val | 511 |
| test | 605 |

## Additional quality exclusions

| Reason | Rows |
|---|---:|
| nested_chat_transcript | 126 |
| untrusted_synthetic_syllogism | 4 |
| procedure_pronoun_mismatch | 3 |

## Rows removed by family cap

| Capability | Removed rows |
|---|---:|
| code_fundamentals | 182 |
| evidence_grounding | 397 |
| instruction_following | 7 |
| logic_reasoning | 256 |
| mathematics | 544 |
| uncertainty_abstention | 320 |

## Capability distribution

| Capability | Train rows | Val rows | Test rows | Train families | Val families | Test families |
|---|---:|---:|---:|---:|---:|---:|
| code_fundamentals | 222 | 4 | 4 | 36 | 4 | 4 |
| conversation | 54 | 7 | 7 | 54 | 7 | 7 |
| evidence_grounding | 841 | 97 | 98 | 34 | 4 | 4 |
| factual_explanation | 214 | 27 | 27 | 214 | 27 | 27 |
| factual_knowledge | 390 | 49 | 49 | 390 | 49 | 49 |
| instruction_following | 831 | 98 | 98 | 783 | 98 | 98 |
| logic_reasoning | 223 | 5 | 67 | 37 | 5 | 5 |
| mathematics | 352 | 96 | 96 | 11 | 3 | 3 |
| planning_procedures | 36 | 5 | 5 | 36 | 5 | 5 |
| rewriting | 108 | 13 | 13 | 108 | 13 | 13 |
| safety_refusal | 28 | 3 | 3 | 28 | 3 | 3 |
| summarization | 39 | 5 | 5 | 39 | 5 | 5 |
| uncertainty_abstention | 380 | 102 | 133 | 70 | 9 | 9 |

## Leakage verification

| Comparison | Family overlap | Prompt overlap | Content overlap |
|---|---:|---:|---:|
| train vs val | 0 | 0 | 0 |
| train vs test | 0 | 0 | 0 |
| val vs test | 0 | 0 | 0 |

## Training restriction

Do not claim or evaluate code-debugging capability from this dataset. No genuine code-debugging examples were found.

Do not start training until the generated V3 files and this report have been reviewed.
