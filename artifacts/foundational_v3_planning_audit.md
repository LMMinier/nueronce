# Foundational V3 Planning Audit

This audit separates actual content defects from provenance warnings.

## Disposition summary

| Disposition | Rows | Meaning |
|---|---:|---|
| hard_reject | 41 | Contains a detected content or grammar defect. |
| relabel_required | 410 | Potentially useful, but assigned to the wrong capability. |
| provenance_review | 4810 | No detected hard defect, but source or split lineage needs review. |
| clean_candidate | 1490 | No current rule detected a defect or provenance concern. |

## Hard-reject reasons

| Reason | Rows |
|---|---:|
| malformed_plural_agreement | 39 |
| malformed_article_a_this | 18 |
| malformed_article_a_a | 2 |

## Relabel reasons

| Reason | Rows |
|---|---:|
| code_definition_not_debugging | 395 |
| summarization_not_causal_temporal | 15 |

## Provenance warnings

| Warning | Rows |
|---|---:|
| synthetic_template_source | 4377 |
| resplit_from_original_source | 1994 |

## Dispositions by current domain

| Domain | Hard reject | Relabel | Provenance review | Clean candidate |
|---|---:|---:|---:|---:|
| causal_temporal | 39 | 15 | 234 | 2 |
| code_debugging | 0 | 395 | 17 | 0 |
| conversation | 0 | 0 | 1439 | 18 |
| evidence_uncertainty | 0 | 0 | 911 | 1457 |
| factual_explanation | 0 | 0 | 702 | 11 |
| mathematics | 0 | 0 | 1441 | 0 |
| planning_tools | 2 | 0 | 66 | 2 |

## Largest category/domain combinations

| Domain | Category | Rows |
|---|---|---:|
| evidence_uncertainty | pa_grounded_evidence | 1433 |
| mathematics | arithmetic | 1088 |
| conversation | instruction_following | 729 |
| code_debugging | coding | 394 |
| conversation | facts | 375 |
| mathematics | classification | 320 |
| factual_explanation | instruction_following | 276 |
| causal_temporal | logic | 268 |
| factual_explanation | definitions | 236 |
| evidence_uncertainty | pa_abstain_conflict_rejected | 200 |
| evidence_uncertainty | pa_abstain_conflict_qualified | 200 |
| evidence_uncertainty | pa_abstain_conflict_conflict | 200 |
| evidence_uncertainty | pa_abstain_conflict_insufficient | 200 |
| conversation | rewriting | 115 |
| factual_explanation | facts | 105 |
| evidence_uncertainty | uncertainty | 67 |
| evidence_uncertainty | pa_abstain_conflict_revision | 64 |
| conversation | greetings | 59 |
| conversation | multiturn | 54 |
| factual_explanation | multiturn | 51 |
| conversation | refusals | 30 |
| conversation | summarization | 27 |
| conversation | procedures | 24 |
| planning_tools | explanations | 24 |
| conversation | handwritten | 20 |
| mathematics | instruction_following | 18 |
| factual_explanation | handwritten | 15 |
| mathematics | multiturn | 15 |
| causal_temporal | summarization | 15 |
| factual_explanation | coding | 14 |
| factual_explanation | procedures | 12 |
| planning_tools | rewriting | 10 |
| planning_tools | procedures | 9 |
| planning_tools | logic | 8 |
| planning_tools | facts | 8 |
| planning_tools | instruction_following | 7 |
| code_debugging | multiturn | 6 |
| causal_temporal | greetings | 5 |
| code_debugging | rewriting | 5 |
| planning_tools | pa_direct_explanation | 4 |

## Source distribution

| Source | Rows |
|---|---:|
| nueronce-synthetic-template-v1 | 4377 |
| nueronce-prompt-aligned-v1 | 2337 |
| nueronce-handwritten-v1 | 37 |

## Samples: hard_reject

### Example 1

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
All triangles are shapes. This triangle is an example of triangle. Is this triangle a shape?
```

**Response**

```text
Yes, since all triangles are shapes, X is a shapes.
```

### Example 2

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_article_a_this|malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
If all squares are rectangles, and X is a this square, is X a rectangles?
```

**Response**

```text
Yes, since all squares are rectangles, X is a rectangles.
```

### Example 3

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
All violins are instruments. This violin is an example of violin. Is this violin a instrument?
```

**Response**

```text
Yes, since all violins are instruments, X is a instruments.
```

### Example 4

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
All doctors are professionals. This doctor is an example of doctor. Is this doctor a professional?
```

**Response**

```text
Yes, since all doctors are professionals, X is a professionals.
```

### Example 5

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
All teachers are professionals. This teacher is an example of teacher. Is this teacher a professional?
```

**Response**

```text
Yes, since all teachers are professionals, X is a professionals.
```

### Example 6

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
All dogs are mammals. Poodle is an example of dog. Is poodle a mammal?
```

**Response**

```text
Yes, since all dogs are mammals, X is a mammals.
```

### Example 7

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_article_a_this|malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
If all triangles are shapes, and X is a this triangle, is X a shapes?
```

**Response**

```text
Yes, since all triangles are shapes, X is a shapes.
```

### Example 8

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: `malformed_plural_agreement`
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
All books are objects. This novel is an example of book. Is this novel a object?
```

**Response**

```text
Yes, since all books are objects, X is a objects.
```


## Samples: relabel_required

### Example 1

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source`

**Prompt**

```text
What does the following code print? print(3 + 2)
```

**Response**

```text
It prints 5.
```

### Example 2

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source`

**Prompt**

```text
What does the following code print? print(8 * 2)
```

**Response**

```text
It prints 16.
```

### Example 3

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source|resplit_from_original_source`

**Prompt**

```text
What does the following code print? print(16 - 1)
```

**Response**

```text
It prints 15.
```

### Example 4

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source`

**Prompt**

```text
What does the following code print? print(4 + 4)
```

**Response**

```text
It prints 8.
```

### Example 5

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source`

**Prompt**

```text
What is the value of 19 - 2 in Python?
```

**Response**

```text
The value is 17.
```

### Example 6

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source`

**Prompt**

```text
What does the following code print? print(6 * 5)
```

**Response**

```text
It prints 30.
```

### Example 7

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source`

**Prompt**

```text
What is the value of 12 - 4 in Python?
```

**Response**

```text
The value is 8.
```

### Example 8

- Domain: `code_debugging`
- Category: `coding`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: `code_definition_not_debugging`
- Provenance: `synthetic_template_source|resplit_from_original_source`

**Prompt**

```text
What is the value of 16 - 2 in Python?
```

**Response**

```text
The value is 14.
```


## Samples: provenance_review

### Example 1

- Domain: `mathematics`
- Category: `classification`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
Is 3479 a prime number?
```

**Response**

```text
3479 is not a prime number.
```

### Example 2

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
Is 174 greater than 143?
```

**Response**

```text
Yes, 174 is greater than 143.
```

### Example 3

- Domain: `mathematics`
- Category: `arithmetic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
What is 418 divided by 19?
```

**Response**

```text
418 divided by 19 equals 22.
```

### Example 4

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
Is 102 greater than 66?
```

**Response**

```text
Yes, 102 is greater than 66.
```

### Example 5

- Domain: `mathematics`
- Category: `arithmetic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
What is 14 plus 80?
```

**Response**

```text
14 plus 80 equals 94.
```

### Example 6

- Domain: `conversation`
- Category: `instruction_following`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
How many letters are in the word pilot?
```

**Response**

```text
The word pilot has 5 letters.
```

### Example 7

- Domain: `causal_temporal`
- Category: `logic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
Which is bigger, 117 or 165?
```

**Response**

```text
165 is bigger.
```

### Example 8

- Domain: `mathematics`
- Category: `arithmetic`
- Source: `nueronce-synthetic-template-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: `synthetic_template_source`

**Prompt**

```text
What is 35 times 15?
```

**Response**

```text
35 times 15 equals 525.
```


## Samples: clean_candidate

### Example 1

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
What does the North Library allow after 17:00 for cohort 215?

Trusted evidence:
[decisive:library] The North Library allows quiet study after 17:00 for cohort 215.
[distractor:0] The west elevator is under inspection.
[distractor:1] The river path is closed after storms.
[distractor:2] The cafe serves soup at noon.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
The North Library allows quiet study after 17:00 for cohort 215.
```

### Example 2

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
Where does Policy I-208 permit bicycle parking?

Trusted evidence:
[decisive:policy] Policy I-208 permits bicycle parking only in marked racks.
[distractor:0] The river path is closed after storms.
[distractor:1] The cafe serves soup at noon.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
Policy I-208 permits bicycle parking only in marked racks.
```

### Example 3

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
When does Dock Seven close on weekdays during schedule 166?

Trusted evidence:
[decisive:harbor] Dock Seven closes at 20:00 on weekdays during schedule 166.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
Dock Seven closes at 20:00 on weekdays during schedule 166.
```

### Example 4

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
When does Dock Seven close on weekdays during schedule 2?

Trusted evidence:
[decisive:harbor] Dock Seven closes at 20:00 on weekdays during schedule 2.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
Dock Seven closes at 20:00 on weekdays during schedule 2.
```

### Example 5

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
What does the North Library allow after 18:00 for cohort 251?

Trusted evidence:
[decisive:library] The North Library allows quiet study after 18:00 for cohort 251.
[distractor:0] The west elevator is under inspection.
[distractor:1] The river path is closed after storms.
[distractor:2] The cafe serves soup at noon.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
The North Library allows quiet study after 18:00 for cohort 251.
```

### Example 6

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
What protection does Lab Nova-247 require?

Trusted evidence:
[decisive:lab] Lab Nova-247 requires eye protection during chemical handling.
[distractor:0] The west elevator is under inspection.
[distractor:1] The river path is closed after storms.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
Lab Nova-247 requires eye protection during chemical handling.
```

### Example 7

- Domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
How can researchers access the museum letter archive case 58?

Trusted evidence:
[decisive:museum] The museum letter archive case 58 opens to researchers by appointment only.
[distractor:0] The river path is closed after storms.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
Researchers can access the museum letter archive case 58 by appointment only.
```

### Example 8

- Domain: `evidence_uncertainty`
- Category: `pa_abstain_conflict_rejected`
- Source: `nueronce-prompt-aligned-v1`
- Hard reasons: ``
- Relabel reasons: ``
- Provenance: ``

**Prompt**

```text
What does the rejected notice C prove?

Trusted evidence:
[rejected] This notice C for case 122 is not trusted evidence and must not be used.

Response requirements:
Do not use rejected evidence.
```

**Response**

```text
The provided trusted evidence does not establish case 122; rejected evidence cannot support the answer.
```

