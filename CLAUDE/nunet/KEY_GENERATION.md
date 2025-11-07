# Key Generation Guide

Complete guide for generating all keys needed for RNode deployment: TLS keys, validator keys, and wallet addresses.

## Table of Contents

1. [Overview](#overview)
2. [TLS Keys (Node Identity)](#tls-keys-node-identity)
3. [Validator Keys (Block Signing)](#validator-keys-block-signing)
4. [Wallet Addresses](#wallet-addresses)
5. [Node ID Calculation](#node-id-calculation)
6. [Complete Example](#complete-example)
7. [Security Considerations](#security-considerations)

## Overview

Three types of keys are needed for RNode deployment:

| Key Type | Purpose | Algorithm | Format | Used In |
|----------|---------|-----------|--------|---------|
| **TLS Keys** | Node identity, secure transport | secp256r1 (EC) | PEM | Peer connections, node-id |
| **Validator Keys** | Block signing | secp256k1 | Hex / PEM | bonds.txt, block signatures |
| **Wallet Keys** | Transaction signing | secp256k1 or ed25519 | Base58 address | wallets.txt, deploys |

**Important**: TLS keys and validator keys are **different** and serve **different purposes**.

## TLS Keys (Node Identity)

### Purpose

- Secure TLS connections between nodes
- Derive the node-id used in bootstrap URIs
- Must be unique per node

### Generation with OpenSSL

```bash
# Generate secp256r1 private key
openssl ecparam -name secp256r1 -genkey -noout -out node.key.pem

# Generate self-signed certificate (valid 365 days)
openssl req -new -x509 -key node.key.pem -out node.certificate.pem -days 365 \
  -subj "/CN=<node-hostname>"

# Example with actual hostname
openssl req -new -x509 -key node.key.pem -out node.certificate.pem -days 365 \
  -subj "/CN=validator1.internal"
```

### Auto-generation

If TLS key files don't exist, RNode generates them automatically on first startup:

```
/var/lib/rnode/
├── node.key.pem (generated)
└── node.certificate.pem (generated)
```

However, for production/DMS deployment, **pre-generate** keys so you know the node-id in advance.

### File Locations

Default: `<data-dir>/node.key.pem` and `<data-dir>/node.certificate.pem`

Docker: `/var/lib/rnode/node.key.pem` and `/var/lib/rnode/node.certificate.pem`

Config override:
```hocon
tls {
  key-path = "/custom/path/node.key.pem"
  certificate-path = "/custom/path/node.certificate.pem"
}
```

### Extract Public Key from TLS Certificate

```bash
# View certificate details
openssl x509 -in node.certificate.pem -text -noout

# Extract public key
openssl x509 -in node.certificate.pem -pubkey -noout > node-tls-public.pem
```

## Validator Keys (Block Signing)

### Purpose

- Sign proposed blocks
- Identify validators in bonds.txt
- Participate in consensus

### Generation with RNode CLI

The recommended method:

```bash
# Generate encrypted key pair
rnode keygen /path/to/output/

# This creates:
# - rnode.key (encrypted PEM private key)
# - rnode.pub.pem (PEM public key)
# - rnode.pub.hex (hex-encoded public key)

# You'll be prompted for password (twice)
```

#### Using the Generated Keys

```bash
# Set password environment variable
export RNODE_VALIDATOR_PASSWORD="your-secure-password"

# Start node with encrypted key
rnode run --validator-private-key-path=/path/to/rnode.key --genesis-validator

# For bonds.txt, use the hex public key
cat /path/to/rnode.pub.hex
```

### Generation with OpenSSL (Manual)

If you need raw hex keys:

```bash
# Generate secp256k1 private key
openssl ecparam -name secp256k1 -genkey -noout -out validator.key.pem

# Extract private key as hex
openssl ec -in validator.key.pem -text -noout | grep priv -A 3 | tail -n +2 | \
  tr -d '\n[:space:]:' > validator.priv.hex

# Extract public key as hex (uncompressed, with 04 prefix)
openssl ec -in validator.key.pem -text -noout | grep pub -A 5 | tail -n +2 | \
  tr -d '\n[:space:]:' > validator.pub.hex

# View keys
echo "Private key (64 hex chars):"
cat validator.priv.hex

echo "Public key (130 hex chars):"
cat validator.pub.hex
```

### Example Keys

**Private key (64 hex chars):**
```
357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9
```

**Public key (130 hex chars, starts with 04):**
```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420
```

### Using Validator Keys

**In bonds.txt:**
```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420 1000
```

**In CLI flags:**
```bash
--validator-private-key=357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9
--validator-public-key=04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420
```

**In config file:**
```hocon
casper {
  validator-private-key = "357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9"
  validator-public-key = "04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420"
}
```

### Deriving Public from Private

If you have the private key, you can derive the public key:

```bash
# Create PEM file from hex private key
echo "-----BEGIN EC PRIVATE KEY-----" > temp.pem
echo "<base64-encoded-key>" >> temp.pem
echo "-----END EC PRIVATE KEY-----" >> temp.pem

# Or use openssl to derive directly
# (requires converting hex to PEM format first)
```

RNode can also derive it automatically if you only provide `--validator-private-key`.

## Wallet Addresses

### Purpose

- Receive and send tokens
- Deploy contracts
- Initial balance allocation in wallets.txt

### REV Address Format

REV addresses are base58-encoded and typically start with `1111`:

```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g
```

### Generation Methods

#### Method 1: Using rust-client

The Firefly rust-client can generate wallet addresses:

```bash
# Generate a new wallet
# TODO: Add rust-client wallet generation command when available

# The tool should output:
# - Private key
# - Public key
# - REV address
```

#### Method 2: Derive from Public Key

REV addresses are derived from secp256k1 or ed25519 public keys through:

1. Hash public key
2. Add version byte
3. Base58check encode

**Manual process** (conceptual):
```bash
# This is a simplified outline - actual implementation requires
# proper base58check encoding with checksums

# 1. Take public key hash
# 2. Add version prefix
# 3. Calculate checksum
# 4. Base58 encode

# Result: 1111<base58-encoded-address>
```

#### Method 3: Use Existing Test Addresses

For testing, use these pre-generated addresses:

```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH
1111LAd2PWaHsw84gxarNx99YVK2aZhCThhrPsWTV7cs1BPcvHftP
1111ocWgUJb5QqnYCvKiPtzcmMyfvD3gS5Eg84NtaLkUtRfw3TDS8
```

**Warning**: These are public test addresses. Never use for real value.

### Using Wallet Addresses

**In wallets.txt:**
```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g,50000000000000000
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA,50000000000000000
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH,50000000000000000
```

**For deploys:**
```bash
rnode deploy \
  --phlo-limit=100000 \
  --phlo-price=1 \
  --private-key=<wallet-private-key-hex> \
  contract.rho
```

## Node ID Calculation

### Purpose

Node ID is used in bootstrap URIs: `rnode://<node-id>@<host>?...`

### Calculation from TLS Key

```bash
# Extract public key from TLS cert/key, remove 04 prefix, hash with Keccak-256
openssl ec -text -in node.key.pem | grep pub -A 5 | tail -n +2 | \
  tr -d '\n[:space:]:' | sed 's/^04//' | keccak-256sum -x -l | \
  tr -d ' -' | tail -c 41

# Result: 40-character hex string
# Example: 1e780e5dfbe0a3d9470a2b414f502d59402e09c2
```

### Requirements

- `keccak-256sum` tool (install: `pip install pysha3` or use standalone binary)
- OpenSSL with EC support

### Alternative: sha3sum

If `keccak-256sum` is not available:

```bash
# Using sha3sum (if compatible)
openssl ec -text -in node.key.pem | grep pub -A 5 | tail -n +2 | \
  tr -d '\n[:space:]:' | sed 's/^04//' | sha3sum -a 256 | cut -c1-40
```

### Pre-calculated Node IDs

For planning/configuration, calculate node-ids before deployment:

```bash
#!/bin/bash
# calculate-node-ids.sh

for node in bootstrap validator1 validator2 validator3; do
  echo "Calculating node-id for $node..."
  NODE_ID=$(openssl ec -text -in certs/$node/node.key.pem 2>/dev/null | \
    grep pub -A 5 | tail -n +2 | tr -d '\n[:space:]:' | sed 's/^04//' | \
    keccak-256sum -x -l | tr -d ' -' | tail -c 41)
  echo "$node: $NODE_ID"
done
```

## Complete Example

### Scenario: Generate all keys for a 3-validator shard

```bash
#!/bin/bash
# generate-shard-keys.sh

# Create directory structure
mkdir -p shard-keys/{bootstrap,validator1,validator2,validator3}
mkdir -p shard-keys/genesis

# Function to generate TLS keys
generate_tls_keys() {
  local node=$1
  echo "Generating TLS keys for $node..."
  openssl ecparam -name secp256r1 -genkey -noout -out shard-keys/$node/node.key.pem
  openssl req -new -x509 -key shard-keys/$node/node.key.pem \
    -out shard-keys/$node/node.certificate.pem -days 365 \
    -subj "/CN=$node.internal"
}

# Function to generate validator keys
generate_validator_keys() {
  local node=$1
  echo "Generating validator keys for $node..."
  openssl ecparam -name secp256k1 -genkey -noout -out shard-keys/$node/validator.key.pem

  # Extract private key hex
  openssl ec -in shard-keys/$node/validator.key.pem -text -noout | \
    grep priv -A 3 | tail -n +2 | tr -d '\n[:space:]:' > shard-keys/$node/validator.priv.hex

  # Extract public key hex
  openssl ec -in shard-keys/$node/validator.key.pem -text -noout | \
    grep pub -A 5 | tail -n +2 | tr -d '\n[:space:]:' > shard-keys/$node/validator.pub.hex

  echo "  Private: $(cat shard-keys/$node/validator.priv.hex)"
  echo "  Public:  $(cat shard-keys/$node/validator.pub.hex)"
}

# Function to calculate node ID
calculate_node_id() {
  local node=$1
  NODE_ID=$(openssl ec -text -in shard-keys/$node/node.key.pem 2>/dev/null | \
    grep pub -A 5 | tail -n +2 | tr -d '\n[:space:]:' | sed 's/^04//' | \
    keccak-256sum -x -l | tr -d ' -' | tail -c 41)
  echo "  Node ID: $NODE_ID"
  echo "$NODE_ID" > shard-keys/$node/node-id.txt
}

# Generate TLS keys for all nodes
for node in bootstrap validator1 validator2 validator3; do
  generate_tls_keys $node
  calculate_node_id $node
done

# Generate validator keys (only for validators, not bootstrap)
for node in validator1 validator2 validator3; do
  generate_validator_keys $node
done

# Create bonds.txt
echo "Creating genesis/bonds.txt..."
{
  cat shard-keys/validator1/validator.pub.hex
  echo " 1000"
  cat shard-keys/validator2/validator.pub.hex
  echo " 1000"
  cat shard-keys/validator3/validator.pub.hex
  echo " 1000"
} | tr -d '\n' | sed 's/ /\n/g' | paste -d' ' - - > shard-keys/genesis/bonds.txt

# Create wallets.txt with test addresses
echo "Creating genesis/wallets.txt..."
cat > shard-keys/genesis/wallets.txt <<EOF
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g,50000000000000000
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA,50000000000000000
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH,50000000000000000
EOF

echo "✓ All keys generated in shard-keys/"
echo "✓ Genesis files created in shard-keys/genesis/"
```

### Running the Script

```bash
chmod +x generate-shard-keys.sh
./generate-shard-keys.sh

# Output:
# Generating TLS keys for bootstrap...
#   Node ID: 1e780e5dfbe0a3d9470a2b414f502d59402e09c2
# Generating TLS keys for validator1...
#   Node ID: a1b2c3d4...
# ...
# Creating genesis/bonds.txt...
# Creating genesis/wallets.txt...
# ✓ All keys generated
```

## Security Considerations

### Production Best Practices

1. **TLS Keys**:
   - Generate unique keys per node
   - Never reuse TLS keys across nodes
   - Store private keys securely (encrypted volumes)

2. **Validator Keys**:
   - Use encrypted PEM format (rnode keygen)
   - Store private keys in HSM or secure key management system
   - Never expose via CLI flags in production (use env vars or encrypted files)
   - Rotate keys according to security policy

3. **Wallet Keys**:
   - Generate fresh addresses for each use case
   - Never reuse test addresses for production
   - Keep private keys offline (cold storage) for high-value accounts
   - Use hardware wallets when available

### Key Storage

**Development:**
- Store in git-ignored directories
- Use `.env` files for local docker-compose

**Production:**
- Use secrets management (HashiCorp Vault, AWS Secrets Manager, etc.)
- Encrypt at rest
- Restrict file permissions: `chmod 600 *.pem`
- Use dedicated key management infrastructure

### Key Backup

- Backup all keys in secure, encrypted storage
- Test recovery procedures
- Document key generation process
- Maintain key inventory (which keys for which nodes)

## Troubleshooting

### keccak-256sum Not Found

```bash
# Install via pip
pip install pysha3

# Or use a standalone binary/tool
# Check if sha3-256sum works as alternative
```

### Wrong Node ID in Bootstrap URI

If nodes can't connect, verify node-id calculation:

```bash
# Recalculate from actual TLS key file
openssl ec -text -in node.key.pem | grep pub -A 5 | tail -n +2 | \
  tr -d '\n[:space:]:' | sed 's/^04//' | keccak-256sum -x -l | \
  tr -d ' -' | tail -c 41
```

### Validator Key Mismatch

If genesis ceremony fails, verify validator public keys:

```bash
# Public key in bonds.txt should match
cat bonds.txt | grep "^04"

# vs. public key used in --validator-public-key flag
# They must be identical
```

## Reference

- RNode keygen: `/node/src/main/scala/coop/rchain/node/Main.scala:173` (Keygen command)
- Key utilities: `/crypto/src/main/scala/coop/rchain/crypto/util/KeyUtil.scala`
- Node ID derivation: Comments in `/node/src/main/resources/defaults.conf:68-69`
- Example keys: `/docker/.env`
