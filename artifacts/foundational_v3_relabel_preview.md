# Foundational V3 Relabel Preview

No training files were created by this preview.

## Status summary

| Status | Rows |
|---|---:|
| hard_reject | 41 |
| manual_review | 69 |
| candidate_with_provenance | 5181 |
| clean_candidate | 1460 |

## Proposed capability distribution

| Capability | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| evidence_grounding | 1433 | 989 | 130 | 314 |
| mathematics | 1088 | 1024 | 64 | 0 |
| instruction_following | 1030 | 777 | 101 | 152 |
| uncertainty_abstention | 931 | 655 | 220 | 56 |
| logic_reasoning | 555 | 420 | 67 | 68 |
| factual_knowledge | 488 | 392 | 44 | 52 |
| code_fundamentals | 410 | 343 | 2 | 65 |
| factual_explanation | 264 | 209 | 28 | 27 |
| conversation | 190 | 150 | 23 | 17 |
| rewriting | 130 | 98 | 20 | 12 |
| manual_review | 69 | 57 | 7 | 5 |
| summarization | 45 | 35 | 4 | 6 |
| planning_procedures | 45 | 38 | 2 | 5 |
| safety_refusal | 30 | 27 | 2 | 1 |
| code_debugging | 2 | 2 | 0 | 0 |

## Current-domain to proposed-capability mapping

| Current domain | Proposed capability | Rows |
|---|---|---:|
| evidence_uncertainty | evidence_grounding | 1433 |
| mathematics | mathematics | 1088 |
| evidence_uncertainty | uncertainty_abstention | 931 |
| conversation | instruction_following | 729 |
| code_debugging | code_fundamentals | 397 |
| conversation | factual_knowledge | 375 |
| mathematics | logic_reasoning | 320 |
| factual_explanation | instruction_following | 276 |
| factual_explanation | factual_explanation | 236 |
| causal_temporal | logic_reasoning | 229 |
| conversation | rewriting | 115 |
| conversation | conversation | 113 |
| factual_explanation | factual_knowledge | 105 |
| factual_explanation | conversation | 51 |
| conversation | manual_review | 44 |
| conversation | safety_refusal | 30 |
| planning_tools | factual_explanation | 28 |
| conversation | summarization | 27 |
| conversation | planning_procedures | 24 |
| factual_explanation | manual_review | 19 |
| mathematics | instruction_following | 18 |
| mathematics | conversation | 15 |
| causal_temporal | summarization | 15 |
| factual_explanation | code_fundamentals | 13 |
| factual_explanation | planning_procedures | 12 |
| planning_tools | rewriting | 10 |
| planning_tools | planning_procedures | 9 |
| planning_tools | factual_knowledge | 8 |
| planning_tools | instruction_following | 7 |
| planning_tools | logic_reasoning | 6 |
| code_debugging | conversation | 6 |
| causal_temporal | conversation | 5 |
| code_debugging | rewriting | 5 |
| evidence_uncertainty | manual_review | 4 |
| code_debugging | summarization | 3 |
| causal_temporal | manual_review | 2 |
| code_debugging | code_debugging | 1 |
| factual_explanation | code_debugging | 1 |

## Category mapping

| Category | Proposed capability | Rows |
|---|---|---:|
| pa_grounded_evidence | evidence_grounding | 1433 |
| arithmetic | mathematics | 1088 |
| instruction_following | instruction_following | 1030 |
| facts | factual_knowledge | 488 |
| coding | code_fundamentals | 406 |
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
| pa_direct_rewrite | manual_review | 4 |
| pa_direct_procedure | manual_review | 4 |
| pa_direct_greeting | manual_review | 4 |
| pa_direct_instruction | manual_review | 4 |
| pa_direct_summary | manual_review | 4 |
| pa_direct_uncertainty | manual_review | 4 |
| pa_direct_refusal | manual_review | 4 |
| pa_direct_coding | code_fundamentals | 4 |
| pa_direct_definition | manual_review | 4 |
| coding | code_debugging | 2 |

## Source distribution by proposed capability

| Source | Proposed capability | Rows |
|---|---|---:|
| nueronce-prompt-aligned-v1 | evidence_grounding | 1433 |
| nueronce-synthetic-template-v1 | mathematics | 1088 |
| nueronce-synthetic-template-v1 | instruction_following | 1030 |
| nueronce-prompt-aligned-v1 | uncertainty_abstention | 864 |
| nueronce-synthetic-template-v1 | logic_reasoning | 555 |
| nueronce-synthetic-template-v1 | factual_knowledge | 488 |
| nueronce-synthetic-template-v1 | code_fundamentals | 406 |
| nueronce-synthetic-template-v1 | factual_explanation | 260 |
| nueronce-synthetic-template-v1 | conversation | 190 |
| nueronce-synthetic-template-v1 | rewriting | 130 |
| nueronce-synthetic-template-v1 | uncertainty_abstention | 67 |
| nueronce-synthetic-template-v1 | summarization | 45 |
| nueronce-synthetic-template-v1 | planning_procedures | 45 |
| nueronce-handwritten-v1 | manual_review | 37 |
| nueronce-prompt-aligned-v1 | manual_review | 32 |
| nueronce-synthetic-template-v1 | safety_refusal | 30 |
| nueronce-prompt-aligned-v1 | factual_explanation | 4 |
| nueronce-prompt-aligned-v1 | code_fundamentals | 4 |
| nueronce-synthetic-template-v1 | code_debugging | 2 |

## Samples: code_debugging

### Example 1

- Current domain: `code_debugging`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is an exception in programming?
```

**Response**

```text
An exception is an error that occurs during program execution.
```

### Example 2

- Current domain: `factual_explanation`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Explain what an exception is in coding.
```

**Response**

```text
An exception is an error that occurs during program execution.
```


## Samples: code_fundamentals

### Example 1

- Current domain: `code_debugging`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What does the following code print? print(3 + 2)
```

**Response**

```text
It prints 5.
```

### Example 2

- Current domain: `code_debugging`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What does the following code print? print(8 * 2)
```

**Response**

```text
It prints 16.
```

### Example 3

- Current domain: `code_debugging`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What does the following code print? print(16 - 1)
```

**Response**

```text
It prints 15.
```

### Example 4

- Current domain: `code_debugging`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What does the following code print? print(4 + 4)
```

**Response**

```text
It prints 8.
```

### Example 5

- Current domain: `code_debugging`
- Category: `coding`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is the value of 19 - 2 in Python?
```

**Response**

```text
The value is 17.
```


## Samples: conversation

### Example 1

- Current domain: `factual_explanation`
- Category: `multiturn`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Greetings
Assistant: Greetings! What do you need?
User: What is ten minus four?
```

**Response**

```text
Ten minus four equals six.
```

### Example 2

- Current domain: `conversation`
- Category: `multiturn`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Hi there
Assistant: Hi! What can I do for you?
User: How many days are in a week?
Assistant: There are seven days in a week.
User: What about five times five?
```

**Response**

```text
Five times five equals twenty five.
```

### Example 3

- Current domain: `factual_explanation`
- Category: `multiturn`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Hi there
Assistant: Hi! What can I do for you?
User: What is two plus two?
Assistant: Two plus two equals four.
User: What about five times five?
```

**Response**

```text
Five times five equals twenty five.
```

### Example 4

- Current domain: `conversation`
- Category: `greetings`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Thanks a lot
```

**Response**

```text
You are very welcome!
```

### Example 5

- Current domain: `conversation`
- Category: `multiturn`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Hello
Assistant: Hello! How can I help you today?
User: How do I make a cup of tea?
```

**Response**

```text
1. Boil water. 2. Add a tea bag to a cup. 3. Pour hot water over the tea bag. 4. Let it steep for a few minutes. 5. Remove the tea bag and enjoy.
```


## Samples: evidence_grounding

### Example 1

- Current domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

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

- Current domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

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

- Current domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Status: `candidate_with_provenance`
- Source: `nueronce-prompt-aligned-v1`

**Prompt**

```text
What protection does Lab Vega-45 require?

Trusted evidence:
[decisive:lab] Lab Vega-45 requires eye protection during chemical handling.
[distractor:0] The west elevator is under inspection.
[distractor:1] The river path is closed after storms.

Response requirements:
Use decisive evidence only.
```

**Response**

```text
Lab Vega-45 requires eye protection during chemical handling.
```

### Example 4

- Current domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

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

### Example 5

- Current domain: `evidence_uncertainty`
- Category: `pa_grounded_evidence`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

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


## Samples: factual_explanation

### Example 1

- Current domain: `factual_explanation`
- Category: `definitions`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is the meaning of the word reliable?
```

**Response**

```text
Reliable means consistently good in quality and able to be trusted.
```

### Example 2

- Current domain: `factual_explanation`
- Category: `definitions`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What does adamant mean?
```

**Response**

```text
Adamant means refusing to change one's mind or position.
```

### Example 3

- Current domain: `factual_explanation`
- Category: `definitions`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is the meaning of the word vast?
```

**Response**

```text
Vast means extremely large in area or size.
```

### Example 4

- Current domain: `factual_explanation`
- Category: `definitions`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What does diverse mean?
```

**Response**

```text
Diverse means showing a great deal of variety.
```

### Example 5

- Current domain: `factual_explanation`
- Category: `definitions`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Define humble.
```

**Response**

```text
Humble means having a modest view of one's own importance.
```


## Samples: factual_knowledge

### Example 1

- Current domain: `conversation`
- Category: `facts`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Name the capital city of Jamaica.
```

**Response**

```text
The capital of Jamaica is Kingston.
```

### Example 2

- Current domain: `conversation`
- Category: `facts`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Which continent is Sweden located on?
```

**Response**

```text
Sweden is in Europe.
```

### Example 3

- Current domain: `conversation`
- Category: `facts`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Which city is the capital of Nigeria?
```

**Response**

```text
The capital of Nigeria is Abuja.
```

### Example 4

- Current domain: `conversation`
- Category: `facts`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What continent is Jamaica in?
```

**Response**

```text
Jamaica is in North America.
```

### Example 5

- Current domain: `conversation`
- Category: `facts`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Tell me the capital of Romania.
```

**Response**

```text
The capital of Romania is Bucharest.
```


## Samples: instruction_following

### Example 1

- Current domain: `conversation`
- Category: `instruction_following`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
How many letters are in the word pilot?
```

**Response**

```text
The word pilot has 5 letters.
```

### Example 2

- Current domain: `planning_tools`
- Category: `instruction_following`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is the first letter of the word eggplant?
```

**Response**

```text
The first letter of eggplant is E.
```

### Example 3

- Current domain: `factual_explanation`
- Category: `instruction_following`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is the first letter of the word noodle?
```

**Response**

```text
The first letter of noodle is N.
```

### Example 4

- Current domain: `conversation`
- Category: `instruction_following`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Reverse this word: meadow
```

**Response**

```text
wodaem.
```

### Example 5

- Current domain: `mathematics`
- Category: `instruction_following`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
List the first five multiples of 7.
```

**Response**

```text
7x1=7, 7x2=14, 7x3=21, 7x4=28, 7x5=35.
```


## Samples: logic_reasoning

### Example 1

- Current domain: `mathematics`
- Category: `classification`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Is 3479 a prime number?
```

**Response**

```text
3479 is not a prime number.
```

### Example 2

- Current domain: `causal_temporal`
- Category: `logic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Is 174 greater than 143?
```

**Response**

```text
Yes, 174 is greater than 143.
```

### Example 3

- Current domain: `causal_temporal`
- Category: `logic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Is 102 greater than 66?
```

**Response**

```text
Yes, 102 is greater than 66.
```

### Example 4

- Current domain: `causal_temporal`
- Category: `logic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Which is bigger, 117 or 165?
```

**Response**

```text
165 is bigger.
```

### Example 5

- Current domain: `causal_temporal`
- Category: `logic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Which is bigger, 24 or 121?
```

**Response**

```text
121 is bigger.
```


## Samples: manual_review

### Example 1

- Current domain: `conversation`
- Category: `handwritten`
- Status: `manual_review`
- Source: `nueronce-handwritten-v1`

**Prompt**

```text
Is water wet?
```

**Response**

```text
Water is commonly described as wet.
```

### Example 2

- Current domain: `factual_explanation`
- Category: `handwritten`
- Status: `manual_review`
- Source: `nueronce-handwritten-v1`

**Prompt**

```text
What is your name?
```

**Response**

```text
I do not have a personal name, but you can call me Assistant.
```

### Example 3

- Current domain: `conversation`
- Category: `pa_direct_rewrite`
- Status: `manual_review`
- Source: `nueronce-prompt-aligned-v1`

**Prompt**

```text
Rewrite this formally: I need this fixed fast. Please answer in plain language.

Response requirements:
Rewrite without adding facts. Keep the answer in plain language
```

**Response**

```text
Please resolve this promptly.
```

### Example 4

- Current domain: `conversation`
- Category: `handwritten`
- Status: `manual_review`
- Source: `nueronce-handwritten-v1`

**Prompt**

```text
Are you a robot?
```

**Response**

```text
I am a software program, not a physical robot.
```

### Example 5

- Current domain: `conversation`
- Category: `pa_direct_procedure`
- Status: `manual_review`
- Source: `nueronce-prompt-aligned-v1`

**Prompt**

```text
How do I make tea? Please answer briefly.

Response requirements:
Give simple steps. Keep the answer briefly
```

**Response**

```text
Boil water, steep the tea, remove the leaves or bag, and serve it safely.
```


## Samples: mathematics

### Example 1

- Current domain: `mathematics`
- Category: `arithmetic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is 418 divided by 19?
```

**Response**

```text
418 divided by 19 equals 22.
```

### Example 2

- Current domain: `mathematics`
- Category: `arithmetic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is 14 plus 80?
```

**Response**

```text
14 plus 80 equals 94.
```

### Example 3

- Current domain: `mathematics`
- Category: `arithmetic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What is 35 times 15?
```

**Response**

```text
35 times 15 equals 525.
```

### Example 4

- Current domain: `mathematics`
- Category: `arithmetic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Subtract 32 from 182.
```

**Response**

```text
182 minus 32 equals 150.
```

### Example 5

- Current domain: `mathematics`
- Category: `arithmetic`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What do you get if you add 176 to 200?
```

**Response**

```text
176 plus 200 equals 376.
```


## Samples: planning_procedures

### Example 1

- Current domain: `conversation`
- Category: `procedures`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What are the steps to set up a tent?
```

**Response**

```text
1. Choose a flat spot. 2. Lay out the tent fabric. 3. Assemble the poles. 4. Insert the poles into the tent. 5. Stake down the corners.
```

### Example 2

- Current domain: `conversation`
- Category: `procedures`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What are the steps to change a light bulb?
```

**Response**

```text
1. Turn off the power. 2. Remove the old bulb. 3. Insert the new bulb. 4. Turn the power back on.
```

### Example 3

- Current domain: `factual_explanation`
- Category: `procedures`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Explain how to make a cup of tea.
```

**Response**

```text
1. Boil water. 2. Add a tea bag to a cup. 3. Pour hot water over the tea bag. 4. Let it steep for a few minutes. 5. Remove the tea bag and enjoy.
```

### Example 4

- Current domain: `factual_explanation`
- Category: `procedures`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Explain how to do laundry.
```

**Response**

```text
1. Sort clothes by color. 2. Load the washing machine. 3. Add detergent. 4. Start the wash cycle. 5. Dry the clothes when finished.
```

### Example 5

- Current domain: `conversation`
- Category: `procedures`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What are the steps to boil an egg?
```

**Response**

```text
1. Place the egg in a pot of water. 2. Bring the water to a boil. 3. Boil for about ten minutes. 4. Remove the egg and let it cool.
```


## Samples: rewriting

### Example 1

- Current domain: `conversation`
- Category: `rewriting`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Make this sentence shorter: We should probably talk about this later.
```

**Response**

```text
We should probably talk about this later.
```

### Example 2

- Current domain: `conversation`
- Category: `rewriting`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Simplify this sentence: That was a fun and useful workshop.
```

**Response**

```text
In simple terms: That was a fun and useful workshop.
```

### Example 3

- Current domain: `conversation`
- Category: `rewriting`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Turn this sentence into a question: I think the team did a great job.
```

**Response**

```text
I think the team did a great job?
```

### Example 4

- Current domain: `conversation`
- Category: `rewriting`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Simplify this sentence: That was a very good presentation.
```

**Response**

```text
In simple terms: That was a very good presentation.
```

### Example 5

- Current domain: `conversation`
- Category: `rewriting`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Rewrite this sentence more formally: The project is coming along nicely.
```

**Response**

```text
Formally stated: The project is coming along nicely.
```


## Samples: safety_refusal

### Example 1

- Current domain: `conversation`
- Category: `refusals`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Tell me how to make a dangerous weapon.
```

**Response**

```text
I can't help with that because it could cause harm or break the law. If there's a safe or legal version of what you're trying to do, I'm happy to help with that instead.
```

### Example 2

- Current domain: `conversation`
- Category: `refusals`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Please, how can I hack into someone's email account?
```

**Response**

```text
I can't help with that because it could cause harm or break the law. If there's a safe or legal version of what you're trying to do, I'm happy to help with that instead.
```

### Example 3

- Current domain: `conversation`
- Category: `refusals`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
I really need to know: How can I hack into someone's email account?
```

**Response**

```text
I can't help with that because it could cause harm or break the law. If there's a safe or legal version of what you're trying to do, I'm happy to help with that instead.
```

### Example 4

- Current domain: `conversation`
- Category: `refusals`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
I really need to know: Tell me how to steal a car.
```

**Response**

```text
I can't help with that because it could cause harm or break the law. If there's a safe or legal version of what you're trying to do, I'm happy to help with that instead.
```

### Example 5

- Current domain: `conversation`
- Category: `refusals`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Please, how can I forge someone's signature?
```

**Response**

```text
I can't help with that because it could cause harm or break the law. If there's a safe or legal version of what you're trying to do, I'm happy to help with that instead.
```


## Samples: summarization

### Example 1

- Current domain: `conversation`
- Category: `summarization`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Summarize this text: The museum added a new exhibit about ancient civilizations. Visitors can see artifacts and read about daily life long ago.
```

**Response**

```text
The museum's new exhibit covers ancient civilizations and daily life.
```

### Example 2

- Current domain: `conversation`
- Category: `summarization`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Summarize this text: The restaurant changed its menu to include more vegetarian options. Customers have responded positively to the new dishes.
```

**Response**

```text
The restaurant's new vegetarian menu has been well received.
```

### Example 3

- Current domain: `conversation`
- Category: `summarization`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Summarize: The airline announced a new direct route between the two cities. Tickets go on sale next week.
```

**Response**

```text
A new direct flight route launches, with tickets on sale next week.
```

### Example 4

- Current domain: `conversation`
- Category: `summarization`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Give a short summary of the following: The library will be closed this weekend for renovations. It is expected to reopen on Monday with new reading rooms.
```

**Response**

```text
The library is closed this weekend for renovations and reopens Monday.
```

### Example 5

- Current domain: `conversation`
- Category: `summarization`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
Give a short summary of the following: The airline announced a new direct route between the two cities. Tickets go on sale next week.
```

**Response**

```text
A new direct flight route launches, with tickets on sale next week.
```


## Samples: uncertainty_abstention

### Example 1

- Current domain: `evidence_uncertainty`
- Category: `pa_abstain_conflict_rejected`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

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

### Example 2

- Current domain: `evidence_uncertainty`
- Category: `pa_abstain_conflict_qualified`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

**Prompt**

```text
Can visitors enter the lab annex?

Trusted evidence:
[docA] Visitors may enter the lab annex with staff escort in case 134.

Response requirements:
Give qualified answer.
```

**Response**

```text
Visitors may enter the lab annex only with staff escort in case 134.
```

### Example 3

- Current domain: `evidence_uncertainty`
- Category: `uncertainty`
- Status: `candidate_with_provenance`
- Source: `nueronce-synthetic-template-v1`

**Prompt**

```text
What time is it in Manila?
```

**Response**

```text
I do not have access to the current time in Manila.
```

### Example 4

- Current domain: `evidence_uncertainty`
- Category: `pa_abstain_conflict_qualified`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

**Prompt**

```text
Can visitors enter the lab annex?

Trusted evidence:
[docA] Visitors may enter the lab annex with staff escort in case 90.

Response requirements:
Give qualified answer.
```

**Response**

```text
Visitors may enter the lab annex only with staff escort in case 90.
```

### Example 5

- Current domain: `evidence_uncertainty`
- Category: `pa_abstain_conflict_qualified`
- Status: `clean_candidate`
- Source: `nueronce-prompt-aligned-v1`

**Prompt**

```text
Can visitors enter the lab annex?

Trusted evidence:
[docA] Visitors may enter the lab annex with staff escort in case 106.

Response requirements:
Give qualified answer.
```

**Response**

```text
Visitors may enter the lab annex only with staff escort in case 106.
```

