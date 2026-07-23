# Foundational Recovery V3.1 Build Report

## Summary

- Family cap: **16**
- Validation family target: **15% per capability**
- Final rows: **3684**
- Unsupported capability: **code_debugging**

## Split totals

| Split | Rows |
|---|---:|
| train | 2574 |
| val | 555 |
| test | 555 |

## Capability distribution

| Capability | Train rows | Val rows | Test rows | Train families | Val families | Test families | Largest val family | Largest test family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code_fundamentals | 90 | 22 | 22 | 30 | 7 | 7 | 72.7% | 72.7% |
| conversation | 48 | 10 | 10 | 48 | 10 | 10 | 10.0% | 10.0% |
| evidence_grounding | 480 | 96 | 96 | 30 | 6 | 6 | 16.7% | 16.7% |
| factual_explanation | 188 | 40 | 40 | 188 | 40 | 40 | 2.5% | 2.5% |
| factual_knowledge | 342 | 73 | 73 | 342 | 73 | 73 | 1.4% | 1.4% |
| instruction_following | 715 | 147 | 147 | 685 | 147 | 147 | 0.7% | 0.7% |
| logic_reasoning | 123 | 22 | 22 | 33 | 7 | 7 | 72.7% | 72.7% |
| mathematics | 176 | 48 | 48 | 11 | 3 | 3 | 33.3% | 33.3% |
| planning_procedures | 32 | 7 | 7 | 32 | 7 | 7 | 14.3% | 14.3% |
| rewriting | 94 | 20 | 20 | 94 | 20 | 20 | 5.0% | 5.0% |
| safety_refusal | 24 | 5 | 5 | 24 | 5 | 5 | 20.0% | 20.0% |
| summarization | 35 | 7 | 7 | 35 | 7 | 7 | 14.3% | 14.3% |
| uncertainty_abstention | 227 | 58 | 58 | 62 | 13 | 13 | 27.6% | 27.6% |

## Leakage checks

| Comparison | Family | Prompt | Content |
|---|---:|---:|---:|
| train vs val | 0 | 0 | 0 |
| train vs test | 0 | 0 | 0 |
| val vs test | 0 | 0 | 0 |

## Review warnings

- code_fundamentals/val largest family share is 72.7%
- code_fundamentals/test largest family share is 72.7%
- logic_reasoning/val largest family share is 72.7%
- logic_reasoning/test largest family share is 72.7%
- planning_procedures/val has only 7 rows
- planning_procedures/test has only 7 rows
- safety_refusal/val has only 5 rows
- safety_refusal/test has only 5 rows
- summarization/val has only 7 rows
- summarization/test has only 7 rows

## Training restriction

Do not train until this report is reviewed. Do not claim code-debugging capability.
