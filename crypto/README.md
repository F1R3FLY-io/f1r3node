# Crypto

Cryptographic primitives for the F1r3fly node: hashing, signing, key management, and TLS certificates.

## Features

| Category | Implementations |
|----------|----------------|
| Hashing | Blake2b256, Blake2b512, Keccak256, SHA-256 |
| Signing | Secp256k1, Secp256k1Eth (Ethereum-compatible), Ed25519, Schnorr (secp256k1), FROST (threshold signatures) |
| Keys | Private/public key types, key generation, Base16 encoding |
| TLS | Certificate generation and validation |

## Building

```bash
cargo build --release -p crypto
cargo build --profile dev -p crypto   # debug mode
```

## Testing

```bash
cargo test -p crypto
cargo test --release -p crypto
```

## Documentation

- [Crypto Module Overview](../docs/crypto/README.md) — Hashing, signing, certificates, Schnorr/FROST
