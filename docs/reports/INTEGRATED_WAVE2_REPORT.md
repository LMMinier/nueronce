# Integrated Wave 2 — End-to-End Learned Cognitive Loop

- seed: 20260701 | cases: 100 | authority backend: `nueronce.engine.authority_clf`

## The question

V3.3 recorded two gate failures: the "full" pipeline tied contract-only (0.920 = 0.920) and citations did not identify decisive evidence. `docs/BREAKTHROUGH_MAP.md` §3.1 showed the tie was true by construction (v3.3's "full" is a simulation sharing contract-only's code path). This runs the *real* learned loop: Ed25519 gate → trained authority classifier on unsigned docs → shared `contract_resolve` → counterfactual citations.

## Systems (frozen V3.3 cases)

| system | answer acc | poison accept | cite P | cite R |
|---|---|---|---|---|
| provenance_contract | 0.920 | 0.000 | 0.920 | 1.000 |
| integrated_learned_none | 0.920 | 0.000 | 0.920 | 1.000 |
| integrated_learned_oracle | 0.830 | 0.170 | 0.830 | 0.830 |
| integrated_learned_predicted | 0.830 | 0.170 | 0.830 | 0.830 |

## Citation quality on answerable cases

- decisive-citation precision: 1.000
- decisive-citation recall: 1.000
- (n answerable = 68)

## Gates

- full_beats_contract_only: False
- citations_identify_decisive_evidence: True
- oracle_vs_predicted_gap: 0.0
- integrated_does_not_reintroduce_poison_vs_contract: False

## Finding (honest)

On the V3.3 threat model — appearance-perfect *unsigned* forgeries — admitting unsigned documents by learned authority **cannot** beat the signature gate, and naively doing so reintroduces poison. This is the V3 `spoof_perfect` ceiling reappearing at the pipeline level: an unsigned genuine document and an unsigned impersonation on the same channel are feature-identical to any text/channel classifier — only cryptography separates them. The authority classifier and the signature gate are, for this threat, the *same defense*; stacking them adds no answer accuracy.

What the integrated loop **does** fix is the second failed gate: counterfactual attribution (cite a document iff removing it flips the outcome) identifies decisive evidence directly, rather than by construction. The remaining path to genuinely beating contract-only is a threat model where unsigned-genuine evidence is distinguishable (different channel or a text signal the forger cannot replicate) — or an external naturalistic blind benchmark (Phase 2).
