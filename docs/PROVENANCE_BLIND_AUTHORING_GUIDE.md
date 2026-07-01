# Provenance Blind Multi-Document Authoring Guide

V3.3 development cases are generated fixtures. They test deterministic wiring,
provenance behavior, retrieval ordering, scope, dates, supersession, conflict
handling, citations, and verifier effects. They are not independent scientific
validation.

For the final blind set, another human should write or materially edit the cases
without seeing the resolver implementation. Labels must remain hidden until code,
thresholds, and metrics are frozen.

## Case Shape

Each case has:

- `case_id`
- `domain`
- `question`
- `documents[]`
- `hidden_gold`

Each document has:

- `document_id`
- `raw_text`
- `issuer_claim`
- `publication_date`
- `effective_from`
- `effective_until`
- `scope`
- `supersedes`
- `revokes`
- `key_id`
- `signature`
- `source_channel`

Hidden gold has:

- `expected_answer`
- `expected_outcome`
- `supporting_document_ids`
- `rejected_document_ids`
- `conflict_document_ids`
- `required_citations`
- `escalation_required`
- `gold_rationale`

## Final Blind Rules

- Do not use final blind labels for training, prompt development, or threshold tuning.
- Do not regenerate or edit the final blind set after viewing results.
- Include ambiguous and imperfect cases where escalation is correct.
- Include 3-8 documents per case, with unrelated distractors.
- Include signed, unsigned legacy, altered, unknown-key, revoked-key, expired,
  out-of-scope, amendment, supersession, conflict, and incomplete-evidence cases.
- Vary names, dates, scopes, wording, document order, and distractor count.
- AI may help format records, but AI-generated cases must not be called
  independent human validation.

Keep final blind labels outside public development artifacts until the evaluation
is frozen.
