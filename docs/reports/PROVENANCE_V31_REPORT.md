# Provenance Gate V3.1 - Report

- seed: 0
- trials: 300
- signed genuine share: 0.850

## Milestone

| pipeline | accuracy | poison acceptance | abstain/escalate |
|---|---|---|---|
| classifier-only | 0.000 | 1.000 | 0.000 |
| classifier + provenance gate | 0.850 | 0.000 | 0.150 |

The gate tests authenticity separately from apparent authority. In this synthetic
benchmark, appearance-perfect poison documents look official to the classifier
but lack valid cryptographic provenance, so the deterministic final-trust policy
prevents them from winning resolution.

## Compromised Key Check

- before revocation: verified
- after revocation: revoked
- note: crypto proves key possession, not non-theft; revocation is the mitigation

## Limits

- This is still a synthetic benchmark, not external validation.
- Signature verification proves key possession, not that the key was never stolen.
- Unsigned genuine legacy documents are escalated/restricted, so coverage depends
  on migration, trusted timestamps, revocation, and human review policy.
