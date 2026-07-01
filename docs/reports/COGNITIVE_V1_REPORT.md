# CFNA Cognitive Ablation - V1 Report

Scenarios (4): authority_overwrite, temporal_supersession, uncertainty_on_untrusted, distractor_resistance

| config | score | solves |
|---|---|---|
| FULL | 4/4 | authority_overwrite, temporal_supersession, uncertainty_on_untrusted, distractor_resistance |
| no_retrieval | 1/4 | uncertainty_on_untrusted |
| no_authority | 2/4 | temporal_supersession, distractor_resistance |
| no_supersession | 2/4 | uncertainty_on_untrusted, distractor_resistance |
| no_planning | 1/4 | uncertainty_on_untrusted |
| no_verification | 1/4 | uncertainty_on_untrusted |

## FULL answers

- **authority_overwrite** PASS: The capital of Zedland is Belport. (source: verified_primary_source:gov_gazette)
- **temporal_supersession** PASS: The CEO of Acme Corp is Sam Ortiz. (source: verified_secondary_source:filing_2024)
- **uncertainty_on_untrusted** PASS: I don't have a trusted source for the moon count of Planet Qx.
- **distractor_resistance** PASS: The population of Riverton is 50000. (source: verified_primary_source:census_2022)

**Milestone:** PASS - full loop strictly beats every ablation
