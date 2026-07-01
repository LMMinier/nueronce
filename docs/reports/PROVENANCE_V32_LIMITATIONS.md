# Provenance V3.2 Limitations

- This is a deterministic synthetic stratified set, not external human validation.
- Compromised keys are intentionally accepted before revocation; Ed25519 proves key possession, not non-theft.
- Human-authored blind documents and hidden labels are still required before strong claims.
- Claim extraction is not implemented here; V3.2 still uses structured document fixtures.
- The full retrieval/resolution/verifier baseline is represented by the deterministic contract path, not a learned raw-document extractor.
