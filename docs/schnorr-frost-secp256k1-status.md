# Schnorr/FROST secp256k1 Status (Experimental)

## Complete
- Feature gate added:
  - `schnorr_secp256k1_experimental`
- New single-signer algorithm:
  - `schnorr-secp256k1`
- New FROST-compatible algorithm name/adaptor:
  - `frost-secp256k1`
- Algorithm factory registration behind feature flag.
- Domain-separated prehash branches for Schnorr/FROST signing preimages.
- Casper block/deploy verification path includes Schnorr/FROST under feature flag.
- Web API deploy parsing accepts Schnorr/FROST algorithm names under feature flag.
- Unit tests:
  - Schnorr keygen/sign/verify/invalid/domain-separation
  - BIP-340 vector check
  - FROST mock 1-of-1 and t-of-n aggregation checks
  - malformed partial/aggregate negative tests
- Integration-style model test:
  - signed deploy roundtrip verification for Schnorr/FROST algorithm names

## Compile-gated items
- All Schnorr/FROST modules and registrations are behind:
  - `schnorr_secp256k1_experimental`

## Stubbed / mock-only parts
- Threshold orchestration is mock/in-memory only.
- No production DKG protocol implementation.
- No production MPC transport layer.
- No vendor-specific HSM integration.

## Production readiness gaps
1. Replace mock threshold coordinator with audited production FROST secp256k1 implementation.
2. Add authenticated coordination protocol for nonce commitments and partial signatures.
3. Add operational key/share lifecycle management.
4. Add broader cross-language test-vector conformance suite for deployment compatibility.
5. Perform consensus-level rollout planning before enabling network-wide use.

## Compatibility notes
- Existing `secp256k1` and `secp256k1-eth` paths remain unchanged.
- No silent algorithm replacement; new algorithms are additive and explicit.
