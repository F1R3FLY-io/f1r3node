# Schnorr/FROST secp256k1 Design (Experimental)

## Scope and intent
This document describes an **experimental**, compile-gated signing family for:
- `schnorr-secp256k1` (single-signer, BIP-340-style encoding),
- `frost-secp256k1` (FROST-compatible threshold architecture producing final Schnorr signatures).

This is additive to existing `secp256k1` and `secp256k1-eth` and does not replace them.

Feature gate:
- `schnorr_secp256k1_experimental`

## Architecture map
Current ECDSA flow:
1. Deploy payload -> `Signed::signature_hash` (legacy hash branch).
2. `SignaturesAlg` implementation signs/verifies (`secp256k1` or `secp256k1-eth`).
3. `SignaturesAlgFactory` resolves algorithm name during decode/validation.
4. Node/Casper validation checks signature by algorithm name.

Proposed Schnorr flow:
1. Deploy payload -> domain-separated hash for `schnorr-secp256k1`.
2. `SchnorrSecp256k1` signs/verifies 32-byte prehash using secp256k1 Schnorr primitives.
3. Factory resolves `schnorr-secp256k1` under feature gate.
4. Node/Casper validation accepts this algorithm only when feature enabled.

Proposed off-node FROST coordinator flow:
1. Participants and coordinator maintain share/nonce/session state off-node.
2. Coordinator gathers partials and outputs final aggregate Schnorr signature.
3. Node verifies final signature via `frost-secp256k1` adapter (Schnorr verification semantics).
4. Node does not run DKG or MPC orchestration by default.

## Why a separate signing family
- Different key format expectations: x-only public key for Schnorr path.
- Different signature shape: fixed 64-byte Schnorr signature (not DER ECDSA).
- Different preimage hashing branch with explicit domain separation.
- Clear boundary for future MPC/HSM adapters without changing ECDSA behavior.

## Serialization and domain separation rules
Public key:
- `schnorr-secp256k1` / `frost-secp256k1` verifier key is x-only secp256k1 (32 bytes).

Signature:
- fixed 64-byte Schnorr signature.

Account identifier:
- helper derives account identifier from `blake2b256(account_domain || len || x_only_pk)`.
- account-domain tag:
  - `f1r3node/schnorr-secp256k1/account/v1`

Signing preimage:
- `schnorr-secp256k1`: `blake2b256("f1r3node/schnorr-secp256k1/signing/v1" || len || payload)`
- `frost-secp256k1`: `blake2b256("f1r3node/frost-secp256k1/signing/v1" || len || payload)`
- legacy schemes retain existing hash behavior.

## FROST-compatible model and RFC 9591 framing
This implementation follows RFC 9591-style separation of concerns:
- participant/share/session/nonce commitment/partial signature/aggregate signature are explicit types.
- coordinator/provider interface is explicit (`FrostThresholdSignerProvider`).
- node-side requirement is limited to final signature verification.

Important:
- This branch does **not** claim full RFC 9591 ciphersuite compliance for secp256k1 MPC internals.
- The provided threshold coordinator is an in-memory mock for dev/test boundaries.

## Security assumptions and non-goals
Assumptions:
- final aggregate signature verification is equivalent to Schnorr verification security at node boundary.
- deploy prehash domain separation prevents accidental cross-family replay semantics.

Non-goals in this branch:
- production DKG protocol implementation,
- production MPC transport or HSM vendor coupling,
- consensus or schema migrations not explicitly feature-gated.

## Future integration points
- Replace mock coordinator with production FROST coordinator/HSM adapter implementing the provider trait.
- Add transport/auth/session management around nonce commitments and partial signatures.
- Add operational key-rotation and share-lifecycle tooling.
