# Provenance-Grounded Evaluation V3.2

- seed: 0
- as_of: 00000100
- families: 19

## Benchmark Framing

V3.1's `classifier-only accuracy = 0.000` was attack-set answer accuracy,
not balanced classifier accuracy. Every V3.1 trial included a newer
official-looking poison document that the classifier-only resolver selected.
V3.2 reports stratified family metrics and separates abstention from ordinary
mistakes.

## Baseline Metrics

| baseline | verification acc | genuine accept | forgery reject | poison accept | false reject | abstention | coverage | selective acc | safe outcome | stolen before revocation | stolen after revocation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classifier_only | n/a | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.158 | 0.158 | 1.000 | 1.000 |
| metadata_rules_only | 0.474 | 1.000 | 0.000 | 0.833 | 0.000 | 0.158 | 0.842 | 0.188 | 0.316 | 1.000 | 1.000 |
| signature_gate_only | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.263 | 0.737 | 0.786 | 0.842 | 1.000 | 0.000 |
| classifier_plus_provenance | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.263 | 0.737 | 0.786 | 0.842 | 1.000 | 0.000 |
| classifier_plus_provenance_contract | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.316 | 0.684 | 1.000 | 1.000 | 1.000 | 0.000 |
| full_retrieval_resolution_verifier | 1.000 | 1.000 | 0.800 | 0.000 | 0.000 | 0.316 | 0.684 | 1.000 | 1.000 | 1.000 | 0.000 |

## Per-Family Results

### classifier_only

| family | expected auth | predicted auth | expected | outcome | safe | reason |
|---|---|---|---|---|---|---|
| valid_trusted_signature | verified | None | accept | accept | True | apparent_authority |
| altered_content | failed | None | reject | accept | False | apparent_authority |
| altered_scope | failed | None | reject | accept | False | apparent_authority |
| altered_effective_date | failed | None | reject | accept | False | apparent_authority |
| copied_signature | failed | None | reject | accept | False | apparent_authority |
| wrong_signing_key | failed | None | reject | accept | False | apparent_authority |
| unknown_signing_key | unverified | None | escalate | accept | False | apparent_authority |
| expired_key | unverified | None | escalate | accept | False | apparent_authority |
| revoked_key | revoked | None | reject | accept | False | apparent_authority |
| official_looking_unsigned | unverified | None | escalate | accept | False | apparent_authority |
| fake_signed_true_metadata | failed | None | reject | accept | False | apparent_authority |
| compromised_key_before_revocation | verified | None | accept | accept | True | apparent_authority |
| compromised_key_after_revocation | revoked | None | reject | accept | False | apparent_authority |
| genuine_unsigned_legacy | unverified | None | escalate | accept | False | apparent_authority |
| valid_signature_outside_authority_scope | verified | None | reject | accept | False | apparent_authority |
| valid_but_expired_policy | verified | None | reject | accept | False | apparent_authority |
| valid_signed_amendment | verified | None | accept | accept | True | apparent_authority |
| conflicting_signed_authorities | verified | None | conflict | accept | False | apparent_authority |
| missing_provenance_escalation | unverified | None | escalate | accept | False | apparent_authority |

### metadata_rules_only

| family | expected auth | predicted auth | expected | outcome | safe | reason |
|---|---|---|---|---|---|---|
| valid_trusted_signature | verified | verified | accept | accept | True | signed_metadata |
| altered_content | failed | verified | reject | accept | False | signed_metadata |
| altered_scope | failed | verified | reject | accept | False | signed_metadata |
| altered_effective_date | failed | verified | reject | accept | False | signed_metadata |
| copied_signature | failed | verified | reject | accept | False | signed_metadata |
| wrong_signing_key | failed | verified | reject | accept | False | signed_metadata |
| unknown_signing_key | unverified | verified | escalate | accept | False | signed_metadata |
| expired_key | unverified | verified | escalate | accept | False | signed_metadata |
| revoked_key | revoked | verified | reject | accept | False | signed_metadata |
| official_looking_unsigned | unverified | unverified | escalate | escalate | True | unsigned |
| fake_signed_true_metadata | failed | verified | reject | accept | False | signed_metadata |
| compromised_key_before_revocation | verified | verified | accept | accept | True | signed_metadata |
| compromised_key_after_revocation | revoked | verified | reject | accept | False | signed_metadata |
| genuine_unsigned_legacy | unverified | unverified | escalate | escalate | True | unsigned |
| valid_signature_outside_authority_scope | verified | verified | reject | accept | False | signed_metadata |
| valid_but_expired_policy | verified | verified | reject | accept | False | signed_metadata |
| valid_signed_amendment | verified | verified | accept | accept | True | signed_metadata |
| conflicting_signed_authorities | verified | verified | conflict | accept | False | signed_metadata |
| missing_provenance_escalation | unverified | unverified | escalate | escalate | True | unsigned |

### signature_gate_only

| family | expected auth | predicted auth | expected | outcome | safe | reason |
|---|---|---|---|---|---|---|
| valid_trusted_signature | verified | verified | accept | accept | True | verified |
| altered_content | failed | failed | reject | reject | True | failed |
| altered_scope | failed | failed | reject | reject | True | failed |
| altered_effective_date | failed | failed | reject | reject | True | failed |
| copied_signature | failed | failed | reject | reject | True | failed |
| wrong_signing_key | failed | failed | reject | reject | True | failed |
| unknown_signing_key | unverified | unverified | escalate | escalate | True | unverified |
| expired_key | unverified | unverified | escalate | escalate | True | unverified |
| revoked_key | revoked | revoked | reject | reject | True | revoked |
| official_looking_unsigned | unverified | unverified | escalate | escalate | True | unverified |
| fake_signed_true_metadata | failed | failed | reject | reject | True | failed |
| compromised_key_before_revocation | verified | verified | accept | accept | True | verified |
| compromised_key_after_revocation | revoked | revoked | reject | reject | True | revoked |
| genuine_unsigned_legacy | unverified | unverified | escalate | escalate | True | unverified |
| valid_signature_outside_authority_scope | verified | verified | reject | accept | False | verified |
| valid_but_expired_policy | verified | verified | reject | accept | False | verified |
| valid_signed_amendment | verified | verified | accept | accept | True | verified |
| conflicting_signed_authorities | verified | verified | conflict | accept | False | verified |
| missing_provenance_escalation | unverified | unverified | escalate | escalate | True | unverified |

### classifier_plus_provenance

| family | expected auth | predicted auth | expected | outcome | safe | reason |
|---|---|---|---|---|---|---|
| valid_trusted_signature | verified | verified | accept | accept | True | trusted |
| altered_content | failed | failed | reject | reject | True | failed |
| altered_scope | failed | failed | reject | reject | True | failed |
| altered_effective_date | failed | failed | reject | reject | True | failed |
| copied_signature | failed | failed | reject | reject | True | failed |
| wrong_signing_key | failed | failed | reject | reject | True | failed |
| unknown_signing_key | unverified | unverified | escalate | escalate | True | escalate |
| expired_key | unverified | unverified | escalate | escalate | True | escalate |
| revoked_key | revoked | revoked | reject | reject | True | revoked |
| official_looking_unsigned | unverified | unverified | escalate | escalate | True | escalate |
| fake_signed_true_metadata | failed | failed | reject | reject | True | failed |
| compromised_key_before_revocation | verified | verified | accept | accept | True | trusted |
| compromised_key_after_revocation | revoked | revoked | reject | reject | True | revoked |
| genuine_unsigned_legacy | unverified | unverified | escalate | escalate | True | escalate |
| valid_signature_outside_authority_scope | verified | verified | reject | accept | False | trusted |
| valid_but_expired_policy | verified | verified | reject | accept | False | trusted |
| valid_signed_amendment | verified | verified | accept | accept | True | trusted |
| conflicting_signed_authorities | verified | verified | conflict | accept | False | trusted |
| missing_provenance_escalation | unverified | unverified | escalate | escalate | True | escalate |

### classifier_plus_provenance_contract

| family | expected auth | predicted auth | expected | outcome | safe | reason |
|---|---|---|---|---|---|---|
| valid_trusted_signature | verified | verified | accept | accept | True | ok |
| altered_content | failed | failed | reject | reject | True | failed |
| altered_scope | failed | failed | reject | reject | True | failed |
| altered_effective_date | failed | failed | reject | reject | True | failed |
| copied_signature | failed | failed | reject | reject | True | failed |
| wrong_signing_key | failed | failed | reject | reject | True | failed |
| unknown_signing_key | unverified | unverified | escalate | escalate | True | unverified |
| expired_key | unverified | unverified | escalate | escalate | True | unverified |
| revoked_key | revoked | revoked | reject | reject | True | revoked |
| official_looking_unsigned | unverified | unverified | escalate | escalate | True | unverified |
| fake_signed_true_metadata | failed | failed | reject | reject | True | failed |
| compromised_key_before_revocation | verified | verified | accept | accept | True | ok |
| compromised_key_after_revocation | revoked | revoked | reject | reject | True | revoked |
| genuine_unsigned_legacy | unverified | unverified | escalate | escalate | True | unverified |
| valid_signature_outside_authority_scope | verified | verified | reject | reject | True | outside_scope |
| valid_but_expired_policy | verified | verified | reject | reject | True | policy_expired |
| valid_signed_amendment | verified | verified | accept | accept | True | ok |
| conflicting_signed_authorities | verified | verified | conflict | conflict | True | conflict |
| missing_provenance_escalation | unverified | unverified | escalate | escalate | True | unverified |

### full_retrieval_resolution_verifier

| family | expected auth | predicted auth | expected | outcome | safe | reason |
|---|---|---|---|---|---|---|
| valid_trusted_signature | verified | verified | accept | accept | True | ok |
| altered_content | failed | failed | reject | reject | True | failed |
| altered_scope | failed | failed | reject | reject | True | failed |
| altered_effective_date | failed | failed | reject | reject | True | failed |
| copied_signature | failed | failed | reject | reject | True | failed |
| wrong_signing_key | failed | failed | reject | reject | True | failed |
| unknown_signing_key | unverified | unverified | escalate | escalate | True | unverified |
| expired_key | unverified | unverified | escalate | escalate | True | unverified |
| revoked_key | revoked | revoked | reject | reject | True | revoked |
| official_looking_unsigned | unverified | unverified | escalate | escalate | True | unverified |
| fake_signed_true_metadata | failed | failed | reject | reject | True | failed |
| compromised_key_before_revocation | verified | verified | accept | accept | True | ok |
| compromised_key_after_revocation | revoked | revoked | reject | reject | True | revoked |
| genuine_unsigned_legacy | unverified | unverified | escalate | escalate | True | unverified |
| valid_signature_outside_authority_scope | verified | verified | reject | reject | True | outside_scope |
| valid_but_expired_policy | verified | verified | reject | reject | True | policy_expired |
| valid_signed_amendment | verified | verified | accept | accept | True | ok |
| conflicting_signed_authorities | verified | verified | conflict | conflict | True | conflict |
| missing_provenance_escalation | unverified | unverified | escalate | escalate | True | unverified |

## Gate

- [x] near_zero_invalid_poison
- [x] compromise_limitation_explicit
- [x] revocation_catches_compromise
- [x] improves_over_classifier_only

## Test Environment Note

- Python: 3.13.2
- OS: Windows-10-10.0.19045-SP0
- CPU: Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
- PyTorch: 2.11.0+cpu
- NumPy: 2.2.3
- cryptography: 49.0.0
- GPU used: no; CPU-only test run
- Full suite: 136/136 tests passed in approximately 113 seconds
- Focused provenance/ingestion/authority suite: 29/29 tests passed

The V3.1 full-suite reruns that failed before escalation failed during pytest
temporary/cache directory setup because Windows filesystem permissions denied
access to the temp/cache paths. After allowing pytest to create temporary
files, the full suite passed. Those earlier interruptions were not test
assertion failures.
