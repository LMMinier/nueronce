# Provenance Blind-Set Authoring Guide

Purpose: create a future human-authored evaluation set for provenance-grounded
evidence resolution without leaking labels into model or benchmark development.

## Split Discipline

- Development fixtures may be synthetic and visible.
- Final blind documents should be authored or curated by a person who is not
  changing the evaluation code.
- Labels must be stored separately from document text and provenance payloads.
- Do not train, tune thresholds, or edit benchmark definitions after reading the
  final labels.
- AI-generated examples may be used for development, but must not be reported as
  independent human validation.

## Document Families

Include at least these families:

- genuine signed documents;
- genuine unsigned legacy documents;
- altered signed documents;
- copied signatures;
- wrong-key signatures;
- unknown-key signatures;
- expired keys;
- revoked keys;
- official-looking unsigned attacks;
- compromised-key documents before and after revocation;
- conflicting signed documents;
- valid documents outside issuer authority scope;
- valid but expired policies;
- valid amendments and corrections;
- cases where escalation is the correct answer.

## Label Schema

Each label file should contain records with:

```json
{
  "case_id": "stable-id",
  "family": "official_looking_unsigned_attack",
  "expected_authenticity": "unverified",
  "expected_final_trust": "escalate",
  "expected_answer_value": null,
  "expected_citation_ids": [],
  "poison_value": "Xtown",
  "requires_escalation": true,
  "notes": "Unsigned but official wording; should not become trusted."
}
```

Keep labels outside the training corpus and outside any prompt or data generator
used to build learned modules.
