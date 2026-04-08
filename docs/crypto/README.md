> Last updated: 2026-03-23

# Crate: crypto

**Path**: `crypto/`

Cryptographic primitives for hashing, signing, certificate management, and key serialization.

## Hash Functions

| Type | Output | Notes |
|------|--------|-------|
| `Blake2b256` | 32 bytes | Primary content hash |
| `Blake2b512Block` | 64 bytes | Online tree hashing with configurable fanout/depth |
| `Blake2b512Random` | 32 bytes | Splittable/mergeable PRNG for unforgeable name generation |
| `Keccak256` | 32 bytes | Ethereum-compatible hashing |
| `Sha256Hasher` | 32 bytes | Standard SHA-256 |

**`Blake2b512Random`** is notable -- it's a deterministic PRNG used to generate unique unforgeable names in Rholang. Supports `split_byte(i8)`, `split_short(i16)`, and `merge(Vec<Self>)` for parallel composition.

## Key Types

```rust
pub struct PrivateKey(pub Bytes);  // Wraps prost::bytes::Bytes
pub struct PublicKey(pub Bytes);
```

## Signature Algorithms

**`SignaturesAlg` trait**:
- `verify(data, sig, pub_key) -> bool`
- `sign(data, sec_key) -> Vec<u8>`
- `to_public(PrivateKey) -> PublicKey`
- `new_key_pair() -> (PrivateKey, PublicKey)`

**Implementations**:
- **`Secp256k1`** -- Primary algorithm. DER-encoded signatures, Blake2b256 input hashing. Supports PEM file parsing with OpenSSL encrypted key format via `parse_pem_file(path, password)`.
- **`Secp256k1Eth`** -- Ethereum variant. RS-to-DER conversion, Keccak256 input hashing.
- **`Ed25519`** -- Present but disabled for deploy signing per RCHAIN-3560.

**`Signed<A>`** -- Generic signed wrapper:
- `create(data, algorithm, private_key)` -- Signs and wraps
- `from_signed_data(data, pk, sig, algorithm)` -- Verifies and wraps
- `signature_hash(alg_name, serialized_data)` -- Keccak256 for Eth, Blake2b256 otherwise

## Certificate Operations

**`CertificateHelper`** -- P-256 (secp256r1) TLS certificate management:
- `generate_key_pair(use_non_blocking)` -- P-256 key generation
- `generate_certificate(secret, public)` -- Self-signed X.509
- `public_address(pub_key)` -- Ethereum-style address (Keccak256 of uncompressed key, last 20 bytes)
- `parse_certificate(der_bytes)` / `parse_certificate_pem(pem_str)` -- X.509 parsing

## Additional Utilities

- **`KeyUtil`** -- File I/O for keys: `write_keys()` writes encrypted private key PEM, public key PEM, and public key hex
- **`CertificatePrinter`** -- PEM formatting for certificates and private keys

## Tests

Property-based tests (proptest) for DER encoding roundtrips, certificate generation, and key pair operations in `tests/util/`.

**See also:** [crypto/ crate README](../../crypto/README.md)

[← Back to docs index](../README.md)
