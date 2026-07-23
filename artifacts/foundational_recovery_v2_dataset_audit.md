# Foundational Recovery V2 Dataset Audit

## Split summary

| Split | Rows | Domains | Templates | Exact duplicate rows |
|---|---:|---:|---:|---:|
| train | 5245 | 7 | 1982 | 0 |
| val | 722 | 7 | 260 | 0 |
| test | 784 | 7 | 273 | 0 |

## Domain distribution

| Domain | Train | Validation | Test |
|---|---:|---:|---:|
| causal_temporal | 266 | 13 | 11 |
| code_debugging | 345 | 2 | 65 |
| conversation | 1129 | 135 | 193 |
| evidence_uncertainty | 1647 | 350 | 371 |
| factual_explanation | 559 | 82 | 72 |
| mathematics | 1244 | 131 | 66 |
| planning_tools | 55 | 9 | 6 |

## Cross-split content overlap

Metadata fields such as example hashes and split labels are excluded.

| Comparison | Shared content rows |
|---|---:|
| train vs val | 0 |
| train vs test | 0 |
| val vs test | 0 |

## Available fields

### train

```text
category
domain
example_hash
kind
prompt
record_id
response
source
source_line
source_split
split
template
template_id
```

### val

```text
category
domain
example_hash
kind
prompt
record_id
response
source
source_line
source_split
split
template
template_id
```

### test

```text
category
domain
example_hash
kind
prompt
record_id
response
source
source_line
source_split
split
template
template_id
```

## Validation examples by assigned domain

These examples are shown without modification so domain quality can be reviewed.

### causal_temporal

Validation rows available: **13**

#### Example 1

```json
{
  "category": "logic",
  "domain": "causal_temporal",
  "example_hash": "a23b85cea406acda1fe7ddab16ef86ddcf8316719a143819764dcc5fa8fae592",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "If all books are objects, and X is a this novel, is X a objects?",
  "record_id": "syllogism-18",
  "response": "Yes, since all books are objects, X is a objects.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 2569,
  "source_split": "train",
  "split": "val",
  "template": "if all books are objects, and x is a this novel, is x a objects?",
  "template_id": "bdbc51bdf210b8d58f3754fd5cccb3953907951bb8a8e516295afb0b9ef7b10f"
}
```

#### Example 2

```json
{
  "category": "logic",
  "domain": "causal_temporal",
  "example_hash": "5962d1843d23c8b6667b2799b5f4ff63e1a07cc33495c3cfd597e783b672332a",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Suppose you exercise regularly. What is the likely result?",
  "record_id": "cond-81",
  "response": "If you exercise regularly, then you tend to get healthier.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 4476,
  "source_split": "train",
  "split": "val",
  "template": "suppose you exercise regularly. what is the likely result?",
  "template_id": "c4e1e4d6c51d0d4da9888da8a894157bc131d6eec1ab809f0d4642ad98d17a0d"
}
```

#### Example 3

```json
{
  "category": "summarization",
  "domain": "causal_temporal",
  "example_hash": "221fa665d16fb116ee565064d81011604884cc118af88d4a5827eec2261601b2",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Summarize: The hiking trail was recently repaired after storm damage. Visitors can now safely reach the summit again.",
  "record_id": "summary-21",
  "response": "The repaired trail again lets visitors safely reach the summit.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 5764,
  "source_split": "train",
  "split": "val",
  "template": "summarize: the hiking trail was recently repaired after storm damage. visitors can now safely reach the summit again.",
  "template_id": "a231c4b4788c92070cb16501d073ca5b4ed2c28df5d1e5079a948e04192bbb51"
}
```

#### Example 4

```json
{
  "category": "logic",
  "domain": "causal_temporal",
  "example_hash": "dd58167582f4f94f2bf8ffe04c963119d8decb4420d4a2e5e16cef259b21b6bf",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "If all teachers are professionals, and X is a this teacher, is X a professionals?",
  "record_id": "syllogism-26",
  "response": "Yes, since all teachers are professionals, X is a professionals.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 570,
  "source_split": "val",
  "split": "val",
  "template": "if all teachers are professionals, and x is a this teacher, is x a professionals?",
  "template_id": "7a2319ca65e9ab761fee9ffa66021109e73f51b0845884752decb1cb3627473e"
}
```

#### Example 5

```json
{
  "category": "summarization",
  "domain": "causal_temporal",
  "example_hash": "d18ebbb1ae671afcae807967e504e9a933378d4212910a59bac211ee9aa45e76",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Give a short summary of the following: The hiking trail was recently repaired after storm damage. Visitors can now safely reach the summit again.",
  "record_id": "summary-22",
  "response": "The repaired trail again lets visitors safely reach the summit.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 14142,
  "source_split": "train",
  "split": "val",
  "template": "give a short summary of the following: the hiking trail was recently repaired after storm damage. visitors can now safely reach the summit again.",
  "template_id": "bc9111845792faf6dd2b40eedf30efbe4f5b4b444ca954d7de615005f6bfeff6"
}
```

#### Example 6

```json
{
  "category": "logic",
  "domain": "causal_temporal",
  "example_hash": "67af050243f3f53c1fa88f33c346be568215deeccbfd3bff2ec26ab7b7669783",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "If all cats are animals, and X is a a kitten, is X a animals?",
  "record_id": "syllogism-14",
  "response": "Yes, since all cats are animals, X is a animals.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 3900,
  "source_split": "train",
  "split": "val",
  "template": "if all cats are animals, and x is a a kitten, is x a animals?",
  "template_id": "d16e196a3053f2e1c02712aa7c91be3e5412c35560bbb5a7baa575b8936b638c"
}
```

### code_debugging

Validation rows available: **2**

#### Example 1

```json
{
  "category": "coding",
  "domain": "code_debugging",
  "example_hash": "4474482ee9a434c9def11bbb6a568ddaf22f7020fd566426fe0befac39bcaaeb",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is an if statement in programming?",
  "record_id": "code-concept-450",
  "response": "An if statement runs a block of code only when a condition is true.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 10927,
  "source_split": "train",
  "split": "val",
  "template": "what is an if statement in programming?",
  "template_id": "fc2edc6600b886633a61c05626a1ed36eeb30ddc4e2f43db2d6333184d364666"
}
```

#### Example 2

```json
{
  "category": "pa_direct_coding",
  "domain": "code_debugging",
  "example_hash": "dacb0916c36cb5d36ac1a30442b174b52b43804a3c3d0a14ce362b3c9c693aff",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is a loop in programming? Please answer without adding extra claims.\n\nResponse requirements:\nExplain the concept. Keep the answer without adding extra claims",
  "record_id": "pa-01236",
  "response": "A loop repeats a block of code while a condition or sequence requires it.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 503,
  "source_split": "val",
  "split": "val",
  "template": "what is a loop in programming? please answer without adding extra claims. response requirements: explain the concept. keep the answer without adding extra claims",
  "template_id": "370bebae167947594619be9cd10d4189700e993c7ebcf9401b4e43752c8a3aff"
}
```

### conversation

Validation rows available: **135**

#### Example 1

```json
{
  "category": "rewriting",
  "domain": "conversation",
  "example_hash": "bacda195dde56261f1ad2a97f705d5575869d1bcb900f8141f39e51bd6b7d370",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Make this sentence shorter: I think we are almost done here.",
  "record_id": "rewrite-short-86",
  "response": "I think we are almost done here.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 159,
  "source_split": "train",
  "split": "val",
  "template": "make this sentence shorter: i think we are almost done here.",
  "template_id": "a0f483eda335771a153b2104998e663b2b9611d4b1b728e3fc5d766f12ed7951"
}
```

#### Example 2

```json
{
  "category": "rewriting",
  "domain": "conversation",
  "example_hash": "8942a8e659055ea1093d236be54e8091ef308b34fe4ca8b316e4b308db7b0c58",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Turn this sentence into a question: That was a fun and useful workshop.",
  "record_id": "rewrite-question-93",
  "response": "That was a fun and useful workshop?",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 51,
  "source_split": "train",
  "split": "val",
  "template": "turn this sentence into a question: that was a fun and useful workshop.",
  "template_id": "41a03f589db0e1f6c12f55c1253a955f34783dc6a150f1cd91f1802d67c559ca"
}
```

#### Example 3

```json
{
  "category": "instruction_following",
  "domain": "conversation",
  "example_hash": "655ffa98731606c593ed0d24cf04cc0cf972a0f152ecc8d0f73e79c12689494f",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Spell the word battery.",
  "record_id": "instr-spell-405",
  "response": "b a t t e r y.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 156,
  "source_split": "train",
  "split": "val",
  "template": "spell the word battery.",
  "template_id": "e21a9f9f05ce5c41fe3236cc8e5f8fdd16ff70566669dc91de38bba86f459634"
}
```

#### Example 4

```json
{
  "category": "facts",
  "domain": "conversation",
  "example_hash": "452ca33d287ee09fb5fe6bcf8d1fa599231198b19408338409ab82342ad06f7f",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Tell me the capital of Serbia.",
  "record_id": "capital-320",
  "response": "The capital of Serbia is Belgrade.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 11375,
  "source_split": "train",
  "split": "val",
  "template": "tell me the capital of serbia.",
  "template_id": "2e6fc46f0670a1d1a8a892a5db26e7851758012e9bcd976cbaa2b11b06fb8ec3"
}
```

#### Example 5

```json
{
  "category": "instruction_following",
  "domain": "conversation",
  "example_hash": "74b83624f331823fa9eb656d5b6855d71f7753df7b296167975349f3ccf05aa8",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "How many letters are in the word market?",
  "record_id": "instr-count-44",
  "response": "The word market has 6 letters.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 7954,
  "source_split": "train",
  "split": "val",
  "template": "how many letters are in the word market?",
  "template_id": "9aef26149d246403e8304bb0df42f15f8c419e2432b871d91f331e9e2fbb8d6b"
}
```

#### Example 6

```json
{
  "category": "instruction_following",
  "domain": "conversation",
  "example_hash": "f80ae2a35b630c1f6f2048a895d7338c6e712708b83167e538103b69f77721d5",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Spell the word gardener.",
  "record_id": "instr-spell-909",
  "response": "g a r d e n e r.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 2509,
  "source_split": "train",
  "split": "val",
  "template": "spell the word gardener.",
  "template_id": "86a03f54e3e1264a877e5e8e51e7426356fef70fbdfef06ce3899bd09eda6369"
}
```

### evidence_uncertainty

Validation rows available: **350**

#### Example 1

```json
{
  "category": "pa_grounded_evidence",
  "domain": "evidence_uncertainty",
  "example_hash": "8239a43b6335fd1b62df91a04d6e1246ceb701765013376a4c62b2b447f836e6",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What protection does Lab Lyra-262 require?\n\nTrusted evidence:\n[decisive:lab] Lab Lyra-262 requires eye protection during chemical handling.\n\nResponse requirements:\nUse decisive evidence only.",
  "record_id": "pa-03576",
  "response": "Lab Lyra-262 requires eye protection during chemical handling.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 14818,
  "source_split": "train",
  "split": "val",
  "template": "what protection does lab lyra-<num> require? trusted evidence: [decisive:lab] lab lyra-<num> requires eye protection during chemical handling. response requirements: use decisive evidence only.",
  "template_id": "d4ddb83b46d698e57b0aafcec9146ff22168516871bdb3afb3703943e68daf90"
}
```

#### Example 2

```json
{
  "category": "pa_abstain_conflict_revision",
  "domain": "evidence_uncertainty",
  "example_hash": "8a4a6a9d4b736428bee4dfb86b53dd8646a84192684857eeba227efb9b563c08",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Revise the draft about bridge opening case 66.\n\nTrusted evidence:\n[docA] The bridge opens at 07:00 after inspection.\n\nResponse requirements:\nFirst draft claimed 06:00 without support.\nCorrect the time using trusted evidence.",
  "record_id": "pa-04334",
  "response": "The bridge opens at 07:00 after inspection.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 11123,
  "source_split": "train",
  "split": "val",
  "template": "revise the draft about bridge opening case <num>. trusted evidence: [doca] the bridge opens at <time> after inspection. response requirements: first draft claimed <time> without support. correct the time using trusted evidence.",
  "template_id": "46c4e95eb7547baef873c371f6e3aef3b36a80896664417e512748098950887c"
}
```

#### Example 3

```json
{
  "category": "pa_abstain_conflict_revision",
  "domain": "evidence_uncertainty",
  "example_hash": "8bdaf108eb150086197bda4e0e49b2a134bcbbdee709047ac9cb68d999cf4e19",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Revise the draft about bridge opening case 122.\n\nTrusted evidence:\n[docA] The bridge opens at 08:00 after inspection.\n\nResponse requirements:\nFirst draft claimed 07:00 without support.\nCorrect the time using trusted evidence.",
  "record_id": "pa-04614",
  "response": "The bridge opens at 08:00 after inspection.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 12352,
  "source_split": "train",
  "split": "val",
  "template": "revise the draft about bridge opening case <num>. trusted evidence: [doca] the bridge opens at <time> after inspection. response requirements: first draft claimed <time> without support. correct the time using trusted evidence.",
  "template_id": "46c4e95eb7547baef873c371f6e3aef3b36a80896664417e512748098950887c"
}
```

#### Example 4

```json
{
  "category": "pa_abstain_conflict_revision",
  "domain": "evidence_uncertainty",
  "example_hash": "fdfa6cc415a5dd2f5dcea8b29905a2a9f647cd4727ff742972862d8e8f9e1338",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Revise the draft about bridge opening case 118.\n\nTrusted evidence:\n[docA] The bridge opens at 09:00 after inspection.\n\nResponse requirements:\nFirst draft claimed 08:00 without support.\nCorrect the time using trusted evidence.",
  "record_id": "pa-04594",
  "response": "The bridge opens at 09:00 after inspection.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 11179,
  "source_split": "train",
  "split": "val",
  "template": "revise the draft about bridge opening case <num>. trusted evidence: [doca] the bridge opens at <time> after inspection. response requirements: first draft claimed <time> without support. correct the time using trusted evidence.",
  "template_id": "46c4e95eb7547baef873c371f6e3aef3b36a80896664417e512748098950887c"
}
```

#### Example 5

```json
{
  "category": "pa_abstain_conflict_revision",
  "domain": "evidence_uncertainty",
  "example_hash": "227f90add4c45f618a228aa9600b9eebdc44192c754befd0459568fd5af34f75",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Revise the draft about bridge opening case 10.\n\nTrusted evidence:\n[docA] The bridge opens at 06:00 after inspection.\n\nResponse requirements:\nFirst draft claimed 05:00 without support.\nCorrect the time using trusted evidence.",
  "record_id": "pa-04054",
  "response": "The bridge opens at 06:00 after inspection.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 8002,
  "source_split": "train",
  "split": "val",
  "template": "revise the draft about bridge opening case <num>. trusted evidence: [doca] the bridge opens at <time> after inspection. response requirements: first draft claimed <time> without support. correct the time using trusted evidence.",
  "template_id": "46c4e95eb7547baef873c371f6e3aef3b36a80896664417e512748098950887c"
}
```

#### Example 6

```json
{
  "category": "pa_abstain_conflict_revision",
  "domain": "evidence_uncertainty",
  "example_hash": "946193dfe3b4ec1c6c4b07c86ff11e1f3abc3a67a949283cb6ac84cef324c2f3",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Revise the draft about bridge opening case 140.\n\nTrusted evidence:\n[docA] The bridge opens at 06:00 after inspection.\n\nResponse requirements:\nFirst draft claimed 05:00 without support.\nCorrect the time using trusted evidence.",
  "record_id": "pa-04704",
  "response": "The bridge opens at 06:00 after inspection.",
  "source": "nueronce-prompt-aligned-v1",
  "source_line": 14736,
  "source_split": "train",
  "split": "val",
  "template": "revise the draft about bridge opening case <num>. trusted evidence: [doca] the bridge opens at <time> after inspection. response requirements: first draft claimed <time> without support. correct the time using trusted evidence.",
  "template_id": "46c4e95eb7547baef873c371f6e3aef3b36a80896664417e512748098950887c"
}
```

### factual_explanation

Validation rows available: **82**

#### Example 1

```json
{
  "category": "facts",
  "domain": "factual_explanation",
  "example_hash": "dc899c9436d82a475936bfa15a842d5eb41808d3837fc3436f285080dcfb395e",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is the capital of Brazil?",
  "record_id": "capital-42",
  "response": "The capital of Brazil is Brasilia.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 13279,
  "source_split": "train",
  "split": "val",
  "template": "what is the capital of brazil?",
  "template_id": "53b75b64cbe26ec5cbc269f752f3f55980e67492f381cf7e876ba0cb4af1b61e"
}
```

#### Example 2

```json
{
  "category": "instruction_following",
  "domain": "factual_explanation",
  "example_hash": "dcbf093b2c6ed45be3e2e58faa32585c0dc19b6ff862b406b026888c926ebcb7",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is the first letter of the word pilot?",
  "record_id": "instr-first-920",
  "response": "The first letter of pilot is P.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 12642,
  "source_split": "train",
  "split": "val",
  "template": "what is the first letter of the word pilot?",
  "template_id": "63dab91636cfaf1dce0409e987e09491ddae1642511b8413d85c08c705d7be98"
}
```

#### Example 3

```json
{
  "category": "instruction_following",
  "domain": "factual_explanation",
  "example_hash": "f12998c1c134480fd63a5579a5600668ff32899ef7d46594d181b3923e6e4f31",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is the first letter of the word chipmunk?",
  "record_id": "instr-first-822",
  "response": "The first letter of chipmunk is C.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 3357,
  "source_split": "train",
  "split": "val",
  "template": "what is the first letter of the word chipmunk?",
  "template_id": "24497b7b4a16971c56da564eb683f4694b0c3521b106cbc5bc9f5faa10672a1b"
}
```

#### Example 4

```json
{
  "category": "instruction_following",
  "domain": "factual_explanation",
  "example_hash": "255723207555567b0ccfebd6b6eba3a0cc2da859fbff2a1f5c0edce1259e107b",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is the first letter of the word carpet?",
  "record_id": "instr-first-269",
  "response": "The first letter of carpet is C.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 6706,
  "source_split": "train",
  "split": "val",
  "template": "what is the first letter of the word carpet?",
  "template_id": "056f46df3a9068b17c743b4c48f21df2072bfa09b895b5d2474bc5a0904d7906"
}
```

#### Example 5

```json
{
  "category": "handwritten",
  "domain": "factual_explanation",
  "example_hash": "0d07288230e9ef7ebe3fe9a4f1757bf1367a73a3b2a5d2c8277a83a9681ba7c5",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is the opposite of up?",
  "record_id": "hw-0037",
  "response": "The opposite of up is down.",
  "source": "nueronce-handwritten-v1",
  "source_line": 12193,
  "source_split": "train",
  "split": "val",
  "template": "what is the opposite of up?",
  "template_id": "69f29a6262cc1ff9344941876a399a83b5194bd17f10f334805b6333815c3e8b"
}
```

#### Example 6

```json
{
  "category": "facts",
  "domain": "factual_explanation",
  "example_hash": "4ce362858633a69c0d86b62db3b87eb01c99aa0ce79005ea492a1f149b8da31b",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "What is the chemical symbol for Carbon?",
  "record_id": "elem-sym-396",
  "response": "The chemical symbol for Carbon is C.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 492,
  "source_split": "train",
  "split": "val",
  "template": "what is the chemical symbol for carbon?",
  "template_id": "d5916f447f3f83c53752d8e32b71b2aecf4a81efaa2c8453b621fbd11099d8ef"
}
```

### mathematics

Validation rows available: **131**

#### Example 1

```json
{
  "category": "classification",
  "domain": "mathematics",
  "example_hash": "b580763176fada8b736ccb028f9776b55ba4ebdc5ac28e3b8c710768a8d241a3",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Determine whether 744 is prime.",
  "record_id": "prime-14744",
  "response": "744 is not a prime number.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 5888,
  "source_split": "train",
  "split": "val",
  "template": "determine whether <num> is prime.",
  "template_id": "9e7599cf114bc50da3460d1fc42ac2156a37c6f8d42ae21f61cab9b862c0565f"
}
```

#### Example 2

```json
{
  "category": "arithmetic",
  "domain": "mathematics",
  "example_hash": "238ac7829ea485a8e0f68dd5e920ac8f16e010d59fbbb2d9ce4ffc9feeb46b18",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Calculate 108 + 64.",
  "record_id": "add-16488",
  "response": "108 plus 64 equals 172.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 19,
  "source_split": "test",
  "split": "val",
  "template": "calculate <num> + <num>.",
  "template_id": "b0152b26fd2fef5219ea92c0633eb039c9c720f66a398e2d2f387284c6ed17fe"
}
```

#### Example 3

```json
{
  "category": "classification",
  "domain": "mathematics",
  "example_hash": "6616ec179d755660d787349bf9104ec00ec91c217805e58012302fe2b3711b7d",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Determine whether 577 is prime.",
  "record_id": "prime-14577",
  "response": "577 is a prime number.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 6618,
  "source_split": "train",
  "split": "val",
  "template": "determine whether <num> is prime.",
  "template_id": "9e7599cf114bc50da3460d1fc42ac2156a37c6f8d42ae21f61cab9b862c0565f"
}
```

#### Example 4

```json
{
  "category": "arithmetic",
  "domain": "mathematics",
  "example_hash": "d9e23c391585f0a781a721bcf52a7a3612b13b96eac35fc618e362a4f2680bd9",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Calculate 50 + 168.",
  "record_id": "add-14832",
  "response": "50 plus 168 equals 218.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 12010,
  "source_split": "train",
  "split": "val",
  "template": "calculate <num> + <num>.",
  "template_id": "b0152b26fd2fef5219ea92c0633eb039c9c720f66a398e2d2f387284c6ed17fe"
}
```

#### Example 5

```json
{
  "category": "arithmetic",
  "domain": "mathematics",
  "example_hash": "02b8b27b86b815a1533520a1c88bcf3d1cdb0b68bf1859840fbfd282c8a4995a",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Calculate 222 + 112.",
  "record_id": "add-19806",
  "response": "222 plus 112 equals 334.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 772,
  "source_split": "val",
  "split": "val",
  "template": "calculate <num> + <num>.",
  "template_id": "b0152b26fd2fef5219ea92c0633eb039c9c720f66a398e2d2f387284c6ed17fe"
}
```

#### Example 6

```json
{
  "category": "arithmetic",
  "domain": "mathematics",
  "example_hash": "878aea829d2735a72eb28ba3f584db3cd470fcbb6bf2812f49ced1dbe1f2b986",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Calculate 46 + 32.",
  "record_id": "add-14682",
  "response": "46 plus 32 equals 78.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 7575,
  "source_split": "train",
  "split": "val",
  "template": "calculate <num> + <num>.",
  "template_id": "b0152b26fd2fef5219ea92c0633eb039c9c720f66a398e2d2f387284c6ed17fe"
}
```

### planning_tools

Validation rows available: **9**

#### Example 1

```json
{
  "category": "explanations",
  "domain": "planning_tools",
  "example_hash": "0b9ca2819003bdf22d06f66eb52cf75c2afc7b6329820bed0fcc54d2a0d31103",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Explain why the ocean is salty.",
  "record_id": "explain-18",
  "response": "The ocean is salty because rivers carry dissolved minerals into it over time.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 6617,
  "source_split": "train",
  "split": "val",
  "template": "explain why the ocean is salty.",
  "template_id": "f90a0ad5060e5832cd49c6e9fe3d24c96f37d845a45fc6ee8370ae17341235b4"
}
```

#### Example 2

```json
{
  "category": "explanations",
  "domain": "planning_tools",
  "example_hash": "bb6fe09659a6ea085a42c44aff4216d37a80bcda682037d79c9a7387242ac05d",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Can you explain why metal feels cold?",
  "record_id": "explain-16",
  "response": "Metal feels cold because it conducts heat away from your hand quickly.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 14647,
  "source_split": "train",
  "split": "val",
  "template": "can you explain why metal feels cold?",
  "template_id": "23b7cc926867acbd6238829079d9d659a191af1d277d650bf564a9b03b080689"
}
```

#### Example 3

```json
{
  "category": "facts",
  "domain": "planning_tools",
  "example_hash": "92a0332c2d2d30b289ab31f293067b128025ccc0dab289f7d59ed75550b05489",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Which planet is the 4th from the sun?",
  "record_id": "planet-457",
  "response": "Mars is the 4th planet from the sun.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 10062,
  "source_split": "train",
  "split": "val",
  "template": "which planet is the 4th from the sun?",
  "template_id": "623c05cf8826633b75ab9954333b05209e691eb65dc3f830062f2fff88ade614"
}
```

#### Example 4

```json
{
  "category": "explanations",
  "domain": "planning_tools",
  "example_hash": "d6d5400d0bc2ee502e33e23e01100313d54ef6c6e42dc98ce2580e55d2734bfc",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Explain why the moon has phases.",
  "record_id": "explain-12",
  "response": "The moon has phases because we see different amounts of its sunlit side as it orbits Earth.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 2546,
  "source_split": "train",
  "split": "val",
  "template": "explain why the moon has phases.",
  "template_id": "1f1b66ec4fdd011adb68ad37f3c9f84aef22cc1d8c20a9019e3b320ac89e7381"
}
```

#### Example 5

```json
{
  "category": "explanations",
  "domain": "planning_tools",
  "example_hash": "ba7ee4bb9b41c66339de1b00a104cef072733579b1f245cd6709a19719a62a66",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "Explain why metal feels cold.",
  "record_id": "explain-15",
  "response": "Metal feels cold because it conducts heat away from your hand quickly.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 15282,
  "source_split": "train",
  "split": "val",
  "template": "explain why metal feels cold.",
  "template_id": "bbacfd9ec31d1215c2d47bc6e75a7574934e8041e40c8ea9a3526d05d38b8203"
}
```

#### Example 6

```json
{
  "category": "logic",
  "domain": "planning_tools",
  "example_hash": "5103c810603bf91e5ee88b2d19d59e19d57ab31ec5715c5c45c1544b29186886",
  "kind": "nueronce_conversation_sanitized",
  "prompt": "If all trees are plants, and X is a this oak tree, is X a plants?",
  "record_id": "syllogism-16",
  "response": "Yes, since all trees are plants, X is a plants.",
  "source": "nueronce-synthetic-template-v1",
  "source_line": 9736,
  "source_split": "train",
  "split": "val",
  "template": "if all trees are plants, and x is a this oak tree, is x a plants?",
  "template_id": "57e959fb0abe0ea8212b0a8f89ad6ba0f515e457c0b0e7325ab52525579ac9ed"
}
```
