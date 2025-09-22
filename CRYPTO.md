# Cryptographic Usage Analysis - F1R3FLY

This document summarizes all cryptographic implementations, libraries, and usage patterns found in the F1R3FLY codebase.

## Overview

F1R3FLY implements a comprehensive cryptographic infrastructure supporting blockchain operations, secure communication, and smart contract execution. The system uses multiple hash functions, digital signature schemes, encryption algorithms, and secure random number generation.

## Cryptographic Dependencies

### Primary Libraries
- **Bouncy Castle** (`bcprov-jdk15on` v1.68, `bcpkix-jdk15on` v1.68) - Comprehensive cryptographic provider
- **Kalium** (`com.github.rchain.kalium` v0.8.1) - NaCl/libsodium wrapper for high-performance crypto
- **Bitcoin-S** (`org.bitcoin-s.bitcoin-s-crypto` v1.9.3) - Bitcoin-compatible cryptography
- **Hasher** (`com.roundeights.hasher` v1.2.0) - Utility hashing library

## Hash Functions

### Blake2b
Primary hashing algorithm used throughout the system.

**Blake2b-256** (`crypto/src/main/scala/coop/rchain/crypto/hash/Blake2b256.scala`)
- **Purpose**: Content addressing, block hashing
- **Implementation**: Bouncy Castle
- **Functions**: `hash(input: Array[Byte])`, `hash(inputs: ByteVector*)`
- **Usage**: RSpace state hashing, block identifiers

**Blake2b-512** (`crypto/src/main/scala/coop/rchain/crypto/hash/Blake2b512Random.scala`)
- **Purpose**: Deterministic random number generation for consensus
- **Functions**: `splitByte()`, `splitShort()`, `merge()`, `next()`
- **Usage**: Consensus protocol randomness

### SHA-256
**Implementation** (`crypto/src/main/scala/coop/rchain/crypto/hash/Sha256.scala`)
- **Purpose**: General-purpose hashing, backward compatibility
- **Library**: Java MessageDigest
- **Usage**: Alternative hashing in Rholang system processes

### Keccak-256
**Implementation** (`crypto/src/main/scala/coop/rchain/crypto/hash/Keccak256.scala`)
- **Purpose**: Ethereum compatibility, address derivation
- **Library**: Bouncy Castle (thread-safe implementation)
- **Usage**: Ethereum-style addresses, smart contract compatibility

## Digital Signatures

### Ed25519
**Implementation** (`crypto/src/main/scala/coop/rchain/crypto/signatures/Ed25519.scala`)
- **Library**: Kalium (libsodium)
- **Key Size**: 32 bytes
- **Signature Size**: 64 bytes
- **Functions**: `newKeyPair`, `sign()`, `verify()`, `toPublic()`
- **Usage**: Primary signature scheme for blocks and transactions

### Secp256k1
**Standard Implementation** (`crypto/src/main/scala/coop/rchain/crypto/signatures/Secp256k1.scala`)
- **Library**: Bitcoin-S
- **Features**: PEM file parsing with password support
- **Functions**: `newKeyPair`, `sign()`, `verify()`, `toPublic()`, `secKeyVerify()`
- **Usage**: Bitcoin-compatible signatures

**Ethereum Variant** (`crypto/src/main/scala/coop/rchain/crypto/signatures/Secp256k1Eth.scala`)
- **Purpose**: Ethereum-compatible signature verification
- **Usage**: Address derivation using Keccak-256

## Encryption

### Curve25519
**Implementation** (`crypto/src/main/scala/coop/rchain/crypto/encryption/Curve25519.scala`)
- **Library**: Kalium (NaCl)
- **Algorithm**: Curve25519 + XSalsa20 + Poly1305 (authenticated encryption)
- **Functions**: `newKeyPair`, `encrypt()`, `decrypt()`, `newNonce()`
- **Usage**: Asymmetric encryption for secure node communication

## Certificate Infrastructure

### X.509 Certificate Management
**Implementation** (`crypto/src/main/scala/coop/rchain/crypto/util/CertificateHelper.scala`)
- **Curve**: secp256r1 (P-256) for certificates
- **Functions**:
  - `generateKeyPair()` - Generate certificate key pairs
  - `generate()` - Create X.509 certificates
  - `publicAddress()` - Derive addresses from public keys
  - `readKeyPair()` - Load keys from PEM files
- **Features**: DER signature encoding, certificate validation

### TLS/SSL Security
**Implementation** (`comm/src/main/scala/coop/rchain/comm/transport/HostnameTrustManagerFactory.scala`)
- **Purpose**: Custom TLS certificate validation for node communication
- **Features**: Hostname verification, certificate chain validation
- **Usage**: Secure inter-node communication

## Secure Random Number Generation

### SecureRandomUtil
**Implementation** (`crypto/src/main/scala/coop/rchain/crypto/util/SecureRandomUtil.scala`)
- **Primary Source**: Non-blocking `/dev/urandom`
- **Fallbacks**: Multiple SecureRandom providers
- **Usage**: All cryptographic key generation and nonce creation

## Rholang Crypto System Processes

The following cryptographic operations are exposed to Rholang smart contracts:

**Hash Channels** (`rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala`)
- **SHA256_HASH** (Channel 5) - SHA-256 hashing service
- **KECCAK256_HASH** (Channel 6) - Keccak-256 hashing service
- **BLAKE2B256_HASH** (Channel 7) - Blake2b-256 hashing service

**Signature Verification Channels**
- **ED25519_VERIFY** (Channel 4) - Ed25519 signature verification
- **SECP256K1_VERIFY** (Channel 8) - Secp256k1 signature verification

## Security Properties

### Cryptographic Strength
- **Hash Functions**: All use 256+ bit security levels
- **Signatures**: Ed25519 and Secp256k1 provide ~128-bit security
- **Encryption**: Curve25519 provides ~128-bit security with authenticated encryption
- **Random Generation**: Cryptographically secure, non-blocking implementation

### Implementation Quality
- **Well-Established Libraries**: Uses Bouncy Castle, libsodium, and Bitcoin-S
- **Thread Safety**: Implemented where needed (e.g., Keccak256)
- **Testing**: Comprehensive test coverage including property-based testing
- **Security Best Practices**: Proper key management and secure random generation

## Usage Contexts

### Blockchain Operations
- Block signing and verification using Ed25519/Secp256k1
- Content addressing using Blake2b-256
- Validator consensus with deterministic randomness
- Deploy signature verification for transaction security

### Communication Security
- TLS/SSL for node-to-node communication
- Certificate-based authentication
- Message encryption using Curve25519

### Smart Contract Integration
- Hash functions accessible from Rholang contracts
- Signature verification for multi-party protocols
- Address derivation for cross-chain compatibility

### State Management
- RSpace content addressing with Blake2b-256
- Merkle tree operations
- Deterministic state transitions using cryptographic hashes

## Test Coverage

Comprehensive test suites ensure cryptographic correctness:
- Unit tests for all hash functions, signatures, and encryption
- Property-based testing using ScalaCheck
- Integration tests for Rholang crypto channels
- Cross-compatibility tests with standard implementations

## Conclusion

F1R3FLY implements a robust, multi-algorithm cryptographic infrastructure suitable for enterprise blockchain applications. The system provides multiple options for hashing, signing, and encryption while maintaining security best practices and comprehensive testing coverage.