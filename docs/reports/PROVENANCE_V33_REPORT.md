# Provenance V3.3 - Blind Multi-Document Resolution

- seed: 0
- development cases: 100
- final blind labels: withheld/not used in this development artifact

## V3.2 Limitation Carried Forward

V3.2 contained 19 constructed families with one canonical case each; contract and full pipeline aggregate results were identical. V3.2 therefore demonstrated deterministic wiring and expected security behavior, not external generalization or retrieval/verifier superiority. Safe-outcome rate must be read alongside abstention and coverage.

Do not describe V3.2 or this generated development set as independent
scientific validation. Safe-outcome rate is always shown with abstention
and coverage.

## Utility Weights

- correct_answer_reward: 1.0
- incorrect_answer_penalty: 1.0
- poison_acceptance_penalty: 2.0
- unnecessary_abstention_penalty: 0.25

Raw metrics remain primary; utility is only a compact tradeoff summary.

## System Metrics

| system | answer acc | src P | src R | cite P | cite R | unsupported | poison | false reject | abstain | coverage | selective acc | safe | utility |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classifier_only | 0.500 | 0.457 | 0.618 | 0.457 | 0.618 | 0.000 | 0.260 | 0.340 | 0.080 | 0.920 | 0.457 | 0.500 | -0.520 |
| metadata_rules_only | 0.590 | 0.554 | 0.750 | 0.554 | 0.750 | 0.000 | 0.170 | 0.250 | 0.080 | 0.920 | 0.554 | 0.590 | -0.160 |
| signature_gate_only | 0.840 | 0.810 | 1.000 | 0.810 | 1.000 | 0.000 | 0.000 | 0.000 | 0.160 | 0.840 | 0.810 | 0.840 | 0.680 |
| provenance_contract | 0.920 | 0.895 | 1.000 | 0.895 | 1.000 | 0.090 | 0.000 | 0.000 | 0.240 | 0.760 | 0.895 | 0.830 | 0.840 |
| retrieval_plus_provenance_contract | 0.920 | 0.895 | 1.000 | 0.895 | 1.000 | 0.090 | 0.000 | 0.000 | 0.240 | 0.760 | 0.895 | 0.830 | 0.840 |
| full_retrieval_resolution_renderer_verifier | 0.920 | 0.895 | 1.000 | 0.895 | 1.000 | 0.000 | 0.000 | 0.000 | 0.240 | 0.760 | 0.895 | 0.920 | 0.840 |

## Bootstrap Confidence Intervals

| system | answer acc 95% CI | poison 95% CI | abstain 95% CI | safe 95% CI | utility 95% CI |
|---|---|---|---|---|---|
| classifier_only | 0.400-0.600 | 0.180-0.340 | 0.030-0.140 | 0.400-0.600 | -0.840--0.180 |
| metadata_rules_only | 0.490-0.690 | 0.110-0.240 | 0.030-0.140 | 0.490-0.690 | -0.460-0.100 |
| signature_gate_only | 0.770-0.900 | 0.000-0.000 | 0.100-0.230 | 0.770-0.900 | 0.540-0.800 |
| provenance_contract | 0.860-0.970 | 0.000-0.000 | 0.170-0.320 | 0.750-0.900 | 0.720-0.940 |
| retrieval_plus_provenance_contract | 0.860-0.970 | 0.000-0.000 | 0.170-0.320 | 0.750-0.900 | 0.720-0.940 |
| full_retrieval_resolution_renderer_verifier | 0.860-0.970 | 0.000-0.000 | 0.170-0.320 | 0.860-0.970 | 0.720-0.940 |

## Ablations

| ablation | answer acc | unsupported | poison | abstain | coverage | safe | utility | total latency ms |
|---|---|---|---|---|---|---|---|---|
| full_pipeline | 0.920 | 0.000 | 0.000 | 0.240 | 0.760 | 0.920 | 0.840 | 0.529 |
| minus_retrieval | 0.480 | 0.000 | 0.000 | 0.050 | 0.950 | 0.480 | -0.040 | 0.765 |
| minus_provenance | 0.580 | 0.000 | 0.260 | 0.160 | 0.840 | 0.580 | -0.360 | 0.520 |
| minus_contract | 0.840 | 0.000 | 0.000 | 0.160 | 0.840 | 0.840 | 0.680 | 0.538 |
| minus_verifier | 0.920 | 0.090 | 0.000 | 0.240 | 0.760 | 0.830 | 0.840 | 0.579 |
| minus_supersession | 0.920 | 0.000 | 0.000 | 0.240 | 0.760 | 0.920 | 0.840 | 0.609 |
| minus_temporal_checks | 0.920 | 0.000 | 0.000 | 0.240 | 0.760 | 0.920 | 0.840 | 0.546 |
| minus_scope_checks | 0.840 | 0.000 | 0.000 | 0.240 | 0.760 | 0.840 | 0.680 | 0.559 |

## Latency And Compute

| system | retrieval | provenance | contract | generation | verification | total | peak KB |
|---|---|---|---|---|---|---|---|
| classifier_only | 0.007 | 0.449 | 0.027 | 0.006 | 0.002 | 0.523 | 2.102 |
| metadata_rules_only | 0.007 | 0.445 | 0.026 | 0.006 | 0.002 | 0.515 | 2.001 |
| signature_gate_only | 0.007 | 0.489 | 0.029 | 0.007 | 0.002 | 0.566 | 2.001 |
| provenance_contract | 0.007 | 0.426 | 0.060 | 0.006 | 0.002 | 0.530 | 2.210 |
| retrieval_plus_provenance_contract | 0.009 | 0.507 | 0.067 | 0.012 | 0.002 | 0.628 | 2.210 |
| full_retrieval_resolution_renderer_verifier | 0.007 | 0.432 | 0.057 | 0.006 | 0.002 | 0.533 | 2.210 |

## Scientific Questions

1. Retrieval vs arbitrary order: `minus_retrieval` safe outcome is 0.480 vs full 0.920.
2. Verifier effect: unsupported rate is 0.090 without verifier vs 0.000 full.
3. Provenance effect: poison acceptance is 0.260 without provenance vs 0.000 full.
4. Contract effect: `minus_contract` safe outcome is 0.840 vs full 0.920.
5. Costs are reported above as abstention and latency.

## Acceptance Gate

- [x] full_beats_classifier_only
- [x] full_beats_signature_only
- [ ] full_beats_contract_only
- [x] invalid_poison_near_zero
- [ ] citations_identify_decisive_evidence
- [x] verifier_reduces_unsupported
- [x] unnecessary_abstention_reported
- [x] final_blind_not_used

**V3.3 acceptance gate: FAIL.** This is a reportable negative result, not
a reason to hide the benchmark. In particular, if the full path does not
beat the provenance contract alone on answer accuracy, retrieval/rendering/
verification are not yet contributing answer-level performance beyond the
contract in this development harness.

## Environment

- Python: 3.13.2
- OS: Windows-10-10.0.19045-SP0
- CPU: Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
- PyTorch: 2.11.0+cpu
- NumPy: 2.2.3
- cryptography: 49.0.0
- GPU used: no; CPU-only evaluation
- Focused provenance/ingestion suite: 28/28 tests passed
- Full suite: 137/137 tests passed in approximately 119 seconds
