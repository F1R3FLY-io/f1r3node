# 3-Node Shard Configuration Guide

Complete step-by-step configuration for deploying a 3-validator Firefly shard, optimized for DMS deployment.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Configuration Strategy](#configuration-strategy)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [DMS-Specific Configuration](#dms-specific-configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

## Overview

This guide shows how to configure a minimal production-ready shard consisting of:
- **1 Bootstrap/Ceremony Master node** (coordinates genesis, does NOT validate)
- **3 Validator nodes** (participate in consensus)

**Total**: 4 containers, 3 active validators

## Architecture

### Node Roles

```
┌─────────────────┐
│ Bootstrap Node  │ ← Ceremony Master
│ (rnode://...)   │   - Coordinates genesis ceremony
└────────┬────────┘   - Network entry point
         │            - Does NOT validate blocks
         │
    ┌────┴────┬────────────┬────────────┐
    │         │            │            │
┌───▼────┐ ┌─▼──────┐ ┌───▼────┐ ┌────▼────┐
│ Val 1  │ │ Val 2  │ │ Val 3  │ │  Obs N  │
│ (bond) │ │ (bond) │ │ (bond) │ │  ...    │
└────────┘ └────────┘ └────────┘ └─────────┘
  Validates  Validates  Validates   Read-only
```

### Network Topology

In DMS environment with internal DNS:
- `bootstrap.internal` → Bootstrap node
- `validator1.internal` → Validator 1
- `validator2.internal` → Validator 2
- `validator3.internal` → Validator 3

All nodes connect to bootstrap, then discover each other via Kademlia.

## Prerequisites

### Required Files Per Node

1. **TLS Keys** (unique per node):
   - `node.key.pem` - secp256r1 private key
   - `node.certificate.pem` - X.509 certificate

2. **Validator Keys** (for validator nodes only):
   - secp256k1 private key (64 hex chars)
   - secp256k1 public key (130 hex chars, can be derived)

3. **Genesis Files** (identical across all nodes):
   - `bonds.txt` - 3 validator public keys + stakes
   - `wallets.txt` - Initial token distribution

4. **Configuration Files**:
   - `rnode.conf` - Node-specific configuration
   - (Optional) Environment variables

### Tools Needed

- OpenSSL (for TLS keys)
- rnode CLI or rust-client (for validator keys)
- keccak-256 tool (for node-id calculation)

See [KEY_GENERATION.md](./KEY_GENERATION.md) for detailed key generation.

## Configuration Strategy

### Key Design Decisions

**Bootstrap Node:**
- Coordinates genesis ceremony
- Not a validator (not in bonds.txt)
- Permanent network entry point
- Can be recycled/restarted safely after genesis

**Validators:**
- All participate in genesis ceremony
- Must have keys in bonds.txt
- Sign proposed genesis block
- Become active after genesis approval

**Configuration Approach:**
- Use config files for static settings
- Use CLI flags for node-specific values (host, keys)
- Mount identical genesis files to all nodes
- Generate unique TLS keys per node

## Step-by-Step Setup

### Step 1: Generate Keys

#### For Bootstrap Node

```bash
# TLS keys (auto-generated if not present, but pre-generate for node-id)
openssl ecparam -name secp256r1 -genkey -noout -out bootstrap-tls.key.pem
openssl req -new -x509 -key bootstrap-tls.key.pem -out bootstrap-tls.cert.pem -days 365 -subj "/CN=bootstrap"

# Calculate node-id
BOOTSTRAP_NODE_ID=$(openssl ec -text -in bootstrap-tls.key.pem | grep pub -A 5 | tail -n +2 | tr -d '\n[:space:]:' | sed 's/^04//' | keccak-256sum -x -l | tr -d ' -' | tail -c 41)

echo "Bootstrap Node ID: $BOOTSTRAP_NODE_ID"

# No validator key needed (not a validator)
```

#### For Each Validator

```bash
# Example for Validator 1

# TLS keys
openssl ecparam -name secp256r1 -genkey -noout -out validator1-tls.key.pem
openssl req -new -x509 -key validator1-tls.key.pem -out validator1-tls.cert.pem -days 365 -subj "/CN=validator1"

# Validator signing key (secp256k1)
# Using rnode keygen (creates encrypted PEM):
rnode keygen ./validator1-keys/
# This creates:
#   - rnode.key (encrypted private key)
#   - rnode.pub.pem (public key)
#   - rnode.pub.hex (public key as hex)

# OR generate raw keys (example, adapt for your tool):
# validator1_private_key="357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9"
# validator1_public_key="04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420"

# Repeat for validator2 and validator3
```

### Step 2: Create Genesis Files

#### bonds.txt

Create `genesis/bonds.txt` with all 3 validator public keys:

```
04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420 1000
04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f 1000
0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c 1000
```

**Key points:**
- Only validator public keys (NOT TLS keys)
- Bootstrap node NOT included (it's not a validator)
- Equal stakes (1000 each) for simplicity

#### wallets.txt

Create `genesis/wallets.txt` with test wallet addresses:

```
1111AtahZeefej4tvVR6ti9TJtv8yxLebT31SCEVDCKMNikBk5r3g,50000000000000000
111127RX5ZgiAdRaQy4AWy57RdvAAckdELReEBxzvWYVvdnR32PiHA,50000000000000000
111129p33f7vaRrpLqK8Nr35Y2aacAjrR5pd6PCzqcdrMuPHzymczH,50000000000000000
1111LAd2PWaHsw84gxarNx99YVK2aZhCThhrPsWTV7cs1BPcvHftP,50000000000000
1111ocWgUJb5QqnYCvKiPtzcmMyfvD3gS5Eg84NtaLkUtRfw3TDS8,50000000000000000
```

### Step 3: Create Configuration Files

#### Bootstrap Configuration

Create `conf/bootstrap.conf`:

```hocon
standalone = false

protocol-server {
  network-id = "testnet"
  allow-private-addresses = true
  port = 40400
}

protocol-client {
  network-id = "testnet"
}

peers-discovery {
  port = 40404
}

api-server {
  host = "0.0.0.0"
  port-grpc-external = 40401
  port-grpc-internal = 40402
  port-http = 40403
  port-admin-http = 40405
}

storage {
  data-dir = "/var/lib/rnode"
}

tls {
  certificate-path = ${storage.data-dir}/node.certificate.pem
  key-path = ${storage.data-dir}/node.key.pem
}

casper {
  shard-name = "root"
  parent-shard-id = "/"

  genesis-block-data {
    genesis-data-dir = ${storage.data-dir}/genesis
    bonds-file = ${casper.genesis-block-data.genesis-data-dir}/bonds.txt
    wallets-file = ${casper.genesis-block-data.genesis-data-dir}/wallets.txt

    bond-minimum = 1
    bond-maximum = 9223372036854775807
    epoch-length = 10000
    quarantine-length = 50000
    number-of-active-validators = 100
  }

  genesis-ceremony {
    # Bootstrap is ceremony master
    required-signatures = 3
    approve-interval = 10 seconds
    approve-duration = 1 minutes
    genesis-validator-mode = false  # NOT a validator
    ceremony-master-mode = true     # IS the ceremony master
  }

  fault-tolerance-threshold = 0.0
  synchrony-constraint-threshold = 0.33
  finalization-rate = 1
}
```

#### Validator Configuration

Create `conf/validator.conf` (shared by all validators):

```hocon
standalone = false

protocol-server {
  network-id = "testnet"
  allow-private-addresses = true
  port = 40400
}

protocol-client {
  network-id = "testnet"
}

peers-discovery {
  port = 40404
}

api-server {
  host = "0.0.0.0"
  port-grpc-external = 40401
  port-grpc-internal = 40402
  port-http = 40403
  port-admin-http = 40405
}

storage {
  data-dir = "/var/lib/rnode"
}

tls {
  certificate-path = ${storage.data-dir}/node.certificate.pem
  key-path = ${storage.data-dir}/node.key.pem
}

casper {
  shard-name = "root"
  parent-shard-id = "/"

  genesis-block-data {
    genesis-data-dir = ${storage.data-dir}/genesis
    bonds-file = ${casper.genesis-block-data.genesis-data-dir}/bonds.txt
    wallets-file = ${casper.genesis-block-data.genesis-data-dir}/wallets.txt

    bond-minimum = 1
    bond-maximum = 9223372036854775807
    epoch-length = 10000
    quarantine-length = 50000
    number-of-active-validators = 100
  }

  genesis-ceremony {
    # Validators sign genesis
    required-signatures = 3
    approve-interval = 10 seconds
    approve-duration = 1 minutes
    genesis-validator-mode = true   # IS a validator
    ceremony-master-mode = false    # NOT the ceremony master
  }

  fault-tolerance-threshold = 0.0
  synchrony-constraint-threshold = 0.33
  finalization-rate = 1
}
```

### Step 4: Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  bootstrap:
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    container_name: bootstrap.internal
    hostname: bootstrap.internal
    command:
      - "run"
      - "--host=bootstrap.internal"
      - "--bootstrap=rnode://${BOOTSTRAP_NODE_ID}@bootstrap.internal?protocol=40400&discovery=40404"
      - "--allow-private-addresses"
    ports:
      - "40400:40400"
      - "40401:40401"
      - "40402:40402"
      - "40403:40403"
      - "40404:40404"
      - "40405:40405"
    volumes:
      - ./conf/bootstrap.conf:/var/lib/rnode/rnode.conf:ro
      - ./genesis:/var/lib/rnode/genesis:ro
      - ./certs/bootstrap/node.certificate.pem:/var/lib/rnode/node.certificate.pem:ro
      - ./certs/bootstrap/node.key.pem:/var/lib/rnode/node.key.pem:ro
      - ./data/bootstrap:/var/lib/rnode/data
    networks:
      - shard

  validator1:
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    container_name: validator1.internal
    hostname: validator1.internal
    command:
      - "run"
      - "--host=validator1.internal"
      - "--bootstrap=rnode://${BOOTSTRAP_NODE_ID}@bootstrap.internal?protocol=40400&discovery=40404"
      - "--allow-private-addresses"
      - "--validator-private-key=${VALIDATOR1_PRIVATE_KEY}"
      - "--validator-public-key=${VALIDATOR1_PUBLIC_KEY}"
      - "--genesis-validator"
    ports:
      - "40410:40400"
      - "40411:40401"
      - "40412:40402"
      - "40413:40403"
      - "40414:40404"
      - "40415:40405"
    volumes:
      - ./conf/validator.conf:/var/lib/rnode/rnode.conf:ro
      - ./genesis:/var/lib/rnode/genesis:ro
      - ./certs/validator1/node.certificate.pem:/var/lib/rnode/node.certificate.pem:ro
      - ./certs/validator1/node.key.pem:/var/lib/rnode/node.key.pem:ro
      - ./data/validator1:/var/lib/rnode/data
    networks:
      - shard
    depends_on:
      - bootstrap

  validator2:
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    container_name: validator2.internal
    hostname: validator2.internal
    command:
      - "run"
      - "--host=validator2.internal"
      - "--bootstrap=rnode://${BOOTSTRAP_NODE_ID}@bootstrap.internal?protocol=40400&discovery=40404"
      - "--allow-private-addresses"
      - "--validator-private-key=${VALIDATOR2_PRIVATE_KEY}"
      - "--validator-public-key=${VALIDATOR2_PUBLIC_KEY}"
      - "--genesis-validator"
    ports:
      - "40420:40400"
      - "40421:40401"
      - "40422:40402"
      - "40423:40403"
      - "40424:40404"
      - "40425:40405"
    volumes:
      - ./conf/validator.conf:/var/lib/rnode/rnode.conf:ro
      - ./genesis:/var/lib/rnode/genesis:ro
      - ./certs/validator2/node.certificate.pem:/var/lib/rnode/node.certificate.pem:ro
      - ./certs/validator2/node.key.pem:/var/lib/rnode/node.key.pem:ro
      - ./data/validator2:/var/lib/rnode/data
    networks:
      - shard
    depends_on:
      - bootstrap

  validator3:
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    container_name: validator3.internal
    hostname: validator3.internal
    command:
      - "run"
      - "--host=validator3.internal"
      - "--bootstrap=rnode://${BOOTSTRAP_NODE_ID}@bootstrap.internal?protocol=40400&discovery=40404"
      - "--allow-private-addresses"
      - "--validator-private-key=${VALIDATOR3_PRIVATE_KEY}"
      - "--validator-public-key=${VALIDATOR3_PUBLIC_KEY}"
      - "--genesis-validator"
    ports:
      - "40430:40400"
      - "40431:40401"
      - "40432:40402"
      - "40433:40403"
      - "40434:40404"
      - "40435:40405"
    volumes:
      - ./conf/validator.conf:/var/lib/rnode/rnode.conf:ro
      - ./genesis:/var/lib/rnode/genesis:ro
      - ./certs/validator3/node.certificate.pem:/var/lib/rnode/node.certificate.pem:ro
      - ./certs/validator3/node.key.pem:/var/lib/rnode/node.key.pem:ro
      - ./data/validator3:/var/lib/rnode/data
    networks:
      - shard
    depends_on:
      - bootstrap

networks:
  shard:
    driver: bridge
```

### Step 5: Environment Variables

Create `.env`:

```bash
# Bootstrap node
BOOTSTRAP_NODE_ID=1e780e5dfbe0a3d9470a2b414f502d59402e09c2

# Validator 1
VALIDATOR1_PUBLIC_KEY=04fa70d7be5eb750e0915c0f6d19e7085d18bb1c22d030feb2a877ca2cd226d04438aa819359c56c720142fbc66e9da03a5ab960a3d8b75363a226b7c800f60420
VALIDATOR1_PRIVATE_KEY=357cdc4201a5650830e0bc5a03299a30038d9934ba4c7ab73ec164ad82471ff9

# Validator 2
VALIDATOR2_PUBLIC_KEY=04837a4cff833e3157e3135d7b40b8e1f33c6e6b5a4342b9fc784230ca4c4f9d356f258debef56ad4984726d6ab3e7709e1632ef079b4bcd653db00b68b2df065f
VALIDATOR2_PRIVATE_KEY=2c02138097d019d263c1d5383fcaddb1ba6416a0f4e64e3a617fe3af45b7851d

# Validator 3
VALIDATOR3_PUBLIC_KEY=0457febafcc25dd34ca5e5c025cd445f60e5ea6918931a54eb8c3a204f51760248090b0c757c2bdad7b8c4dca757e109f8ef64737d90712724c8216c94b4ae661c
VALIDATOR3_PRIVATE_KEY=b67533f1f99c0ecaedb7d829e430b1c0e605bda10f339f65d5567cb5bd77cbcb
```

### Step 6: Directory Structure

Organize files:

```
shard-deployment/
├── .env
├── docker-compose.yml
├── conf/
│   ├── bootstrap.conf
│   └── validator.conf
├── genesis/
│   ├── bonds.txt
│   └── wallets.txt
├── certs/
│   ├── bootstrap/
│   │   ├── node.certificate.pem
│   │   └── node.key.pem
│   ├── validator1/
│   │   ├── node.certificate.pem
│   │   └── node.key.pem
│   ├── validator2/
│   │   ├── node.certificate.pem
│   │   └── node.key.pem
│   └── validator3/
│       ├── node.certificate.pem
│       └── node.key.pem
└── data/          # Created at runtime
    ├── bootstrap/
    ├── validator1/
    ├── validator2/
    └── validator3/
```

### Step 7: Launch

```bash
# Create data directories
mkdir -p data/{bootstrap,validator1,validator2,validator3}

# Start the shard
docker-compose up -d

# Watch logs
docker-compose logs -f
```

## DMS-Specific Configuration

### DMS Ensemble Definition

For DMS deployment, translate the docker-compose to DMS ensemble format:

```yaml
# ensemble.yml
name: firefly-shard
services:
  - name: bootstrap
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    dns_name: bootstrap.internal
    command: [...]
    ports:
      - container: 40400
        public: true
      - container: 40401
        public: true
      - container: 40404
        public: true
    volumes:
      - source: ./conf/bootstrap.conf
        target: /var/lib/rnode/rnode.conf
      - source: ./genesis
        target: /var/lib/rnode/genesis
      - source: ./certs/bootstrap
        target: /var/lib/rnode

  - name: validator1
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    dns_name: validator1.internal
    command: [...]
    # ... similar structure
```

### Key DMS Considerations

1. **DNS Resolution**: Use `.internal` suffix for inter-container DNS
2. **Port Mapping**: DMS handles external port mapping automatically
3. **Volume Mounting**: Use DMS volume syntax for configs and certs
4. **Environment Variables**: Pass via DMS environment configuration
5. **Service Dependencies**: Use DMS dependency management (like `depends_on`)

### Bootstrap URI in DMS

```bash
--bootstrap=rnode://<node-id>@bootstrap.internal?protocol=40400&discovery=40404
```

The `.internal` DNS name is resolved by DMS's internal DNS server.

## Verification

### Check Genesis Ceremony

Watch bootstrap logs for:

```
Starting genesis ceremony...
Waiting for validator signatures...
Received approval from <validator1-key>
Received approval from <validator2-key>
Received approval from <validator3-key>
Genesis block approved and created!
```

### Check Validator Connections

On each validator, check logs for:

```
Connected to bootstrap node
Discovered peer: validator2.internal
Discovered peer: validator3.internal
Joined the network
```

### Query Node Status

```bash
# Check if node is running
docker exec bootstrap.internal rnode status

# Check connected peers
docker exec validator1.internal rnode status
```

### Test Deploy

```bash
# Deploy a simple Rholang contract
docker exec validator1.internal rnode deploy \
  --phlo-limit=100000 \
  --phlo-price=1 \
  --private-key=<deployer-key> \
  /path/to/contract.rho

# Propose a block (triggers validation)
docker exec validator1.internal rnode propose
```

## Troubleshooting

### Genesis Ceremony Never Completes

**Symptom**: Bootstrap waits forever for validator signatures

**Causes:**
- `required-signatures` config doesn't match actual validator count
- One or more validators not running
- Validator keys in bonds.txt don't match `--validator-private-key` flags
- Network connectivity issue

**Fix:**
```bash
# Check validator count
grep -c "04" genesis/bonds.txt  # Should output: 3

# Verify all validators are running
docker-compose ps

# Check validator logs for errors
docker-compose logs validator1 | grep -i error
```

### Validators Can't Connect to Bootstrap

**Symptom**: Validators timeout connecting to bootstrap

**Causes:**
- Wrong bootstrap node-id in `--bootstrap` URI
- DNS resolution failure
- Port 40400/40404 not accessible

**Fix:**
```bash
# Verify DNS resolution
docker exec validator1.internal ping bootstrap.internal

# Check node-id matches
echo "Expected: $BOOTSTRAP_NODE_ID"
# vs actual in bootstrap logs

# Test port connectivity
docker exec validator1.internal nc -zv bootstrap.internal 40400
docker exec validator1.internal nc -zv bootstrap.internal 40404
```

### Nodes Create Different Genesis Blocks

**Symptom**: Nodes fail to sync, "Unknown block" errors

**Cause**: bonds.txt or wallets.txt differ between nodes

**Fix:**
```bash
# Verify file checksums match across all nodes
docker exec bootstrap.internal sha256sum /var/lib/rnode/genesis/bonds.txt
docker exec validator1.internal sha256sum /var/lib/rnode/genesis/bonds.txt
# All should match

# If different, recreate with identical files
```

### Port Already in Use

**Symptom**: "Port 40400 already in use"

**Fix:**
```bash
# Find process using port
ss -tulpn | grep 40400

# Stop conflicting service or use different external ports
# Edit docker-compose.yml ports mapping
```

## Next Steps

- Add observer nodes (no `--genesis-validator`, no keys in bonds.txt)
- Enable metrics/monitoring (Prometheus endpoints on port 40405)
- Configure autopropose for continuous block production
- Set up log aggregation
- Deploy to DMS production environment

## Reference

- [RNODE_CONFIGURATION_GUIDE.md](./RNODE_CONFIGURATION_GUIDE.md) - Full config reference
- [GENESIS_FILES_FORMAT.md](./GENESIS_FILES_FORMAT.md) - bonds.txt/wallets.txt format
- [KEY_GENERATION.md](./KEY_GENERATION.md) - How to generate all keys
- Example deployment: `/docker/shard-with-autopropose.yml`
