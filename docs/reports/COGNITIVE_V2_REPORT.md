# CFNA Cognitive Suite V2 - Report

- seeds: [1, 2, 3, 4, 5]  |  trials/seed: 1050  |  holdout/seed: 480
- families: 15 in-distribution, 8 adversarial

## Strategy accuracy (mean +/- std over seeds)

`composite` requires the right value AND (where required) the right citation, refusal, or conflict flag. `value-only` ignores citation/refusal/conflict, so it is the fair comparison to the citation-blind baselines.

| strategy | composite in-dist | value-only | source-sel | adversarial | poison rate | unsupported |
|---|---|---|---|---|---|---|
| FULL_COGNITIVE_LOOP | 1.000 +/- 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| NEWEST_FACT_WINS | 0.034 +/- 0.003 | 0.500 | 0.538 | 0.000 | 1.000 | 0.310 |
| HIGHEST_AUTHORITY_ONLY | 0.034 +/- 0.003 | 0.750 | 0.769 | 0.000 | 0.000 | 0.034 |
| KEYWORD_RULE_ENGINE | 0.034 +/- 0.003 | 0.583 | 0.615 | 0.000 | 0.750 | 0.241 |
| NO_AUTHORITY | 0.634 +/- 0.003 | 0.667 | 0.692 | 0.125 | 1.000 | 0.310 |
| NO_SUPERSESSION | 0.800 +/- 0.000 | 0.833 | 0.846 | 0.875 | 0.000 | 0.000 |
| NO_RETRIEVAL | 0.067 +/- 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| NO_PLANNING | 0.200 +/- 0.000 | 1.000 | 1.000 | 0.125 | 0.000 | 0.000 |
| NO_VERIFICATION | 0.200 +/- 0.000 | 1.000 | 1.000 | 0.125 | 0.000 | 0.000 |
| RANDOM_CHOICE | 0.034 +/- 0.003 | 0.560 | 0.593 | 0.000 | 0.460 | 0.161 |

## FULL policy — per-family accuracy (in-distribution)

| family | acc |
|---|---|
| AUTHORITY_IMPERSONATION | 1.000 |
| CITATION_REQUIREMENT | 1.000 |
| CONFLICT_SURFACING | 1.000 |
| EQUAL_AUTHORITY_TEMPORAL_UPDATE | 1.000 |
| EXPIRED_FACT | 1.000 |
| FUTURE_EFFECTIVE_CORRECTION | 1.000 |
| HIGHER_AUTHORITY_CORRECTION | 1.000 |
| IRRELEVANT_HIGH_AUTHORITY | 1.000 |
| LOWER_AUTHORITY_POISON | 1.000 |
| MISSING_EVIDENCE | 1.000 |
| MULTIPLE_COMPETING_FACTS | 1.000 |
| PARAPHRASED_POISON | 1.000 |
| SCOPE_LIMITED_FACT | 1.000 |
| SOURCE_REVOCATION | 1.000 |
| UNCERTAIN_TRUSTED_SOURCE | 1.000 |

## Module necessity (worst-family drop when removed)

| removed module | worst family | acc drop | meaningful (>=0.20) |
|---|---|---|---|
| NO_AUTHORITY | CONFLICT_SURFACING | 1.000 | True |
| NO_SUPERSESSION | CONFLICT_SURFACING | 1.000 | True |
| NO_RETRIEVAL | CONFLICT_SURFACING | 1.000 | True |
| NO_PLANNING | IRRELEVANT_HIGH_AUTHORITY | 1.000 | True |
| NO_VERIFICATION | IRRELEVANT_HIGH_AUTHORITY | 1.000 | True |

## FULL failure taxonomy (in-distribution, summed over seeds)

- (no failures)

## Gate

- [x] in_distribution_accuracy>=0.95
- [x] adversarial_accuracy>=0.85
- [x] poisoning_rate<0.05
- [x] beats_NEWEST_FACT_WINS_on_value>=0.10
- [x] beats_KEYWORD_RULE_ENGINE_on_value>=0.10
- [x] every_module_has_meaningful_degradation

**GATE: PASS**

## Limitations (read before citing these numbers)

- **FULL = 1.000 is not evidence of real-world correctness.** The policy and the gold labels were authored from the *same* provenance rules, so the full loop matching its own specification is expected. The result shows internal completeness and consistency, not external validity.
- **Authority labels are given as ground-truth metadata.** The suite does not test inferring authority/temporal-scope/claims from raw text — the policy resists text attacks precisely because it ignores item text. That inference is the deferred, learnable problem.
- **The meaningful signals** are therefore the *contrasts*: value-only accuracy vs. baselines (poison/stale/expired/scope handling), the poisoning rate of text-based baselines (KEYWORD), and the per-family collapse under each ablation — not FULL's absolute score.
