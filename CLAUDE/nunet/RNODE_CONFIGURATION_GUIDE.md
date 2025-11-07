# RNode Configuration Guide

Complete reference for configuring Firefly (RNode) instances for deployment in DMS or other environments.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Sources](#configuration-sources)
3. [Bootstrap URI Format](#bootstrap-uri-format)
4. [Port Configuration](#port-configuration)
5. [Network Configuration](#network-configuration)
6. [Validator Configuration](#validator-configuration)
7. [CLI Flags Reference](#cli-flags-reference)
8. [Configuration File Format](#configuration-file-format)
9. [TLS Keys and Node Identity](#tls-keys-and-node-identity)

## Overview

RNode configuration follows a priority system:
1. **CLI arguments** (highest priority)
2. **Configuration file** (specified via `-c` or `--config-file`)
3. **Default configuration** (`defaults.conf` in resources)

For Docker deployments, the default data directory is `/var/lib/rnode/`.

## Configuration Sources

### Profiles

RNode supports two profiles that set default paths:
- `default`: Data directory at `$HOME/.rnode`
- `docker`: Data directory at `/var/lib/rnode`

Specify via `--profile=docker` or environment-based detection.

### Configuration File Location

Default search path: `<data-dir>/rnode.conf`

Override via: `rnode run -c /path/to/config.conf`

## Bootstrap URI Format

Bootstrap addresses use the format:

```
rnode://<node-id>@<hostname>?protocol=<tcp-port>&discovery=<udp-port>
```

### Components

- **`node-id`**: 40-character hex string derived from the node's TLS public key
- **`hostname`**: DNS name or IP address (can use `<name>.internal` for DMS)
- **`protocol`**: TCP port for RChain Protocol messages (default: 40400)
- **`discovery`**: UDP port for Kademlia peer discovery (default: 40404)

### Example

```
rnode://1e780e5dfbe0a3d9470a2b414f502d59402e09c2@bootstrap.internal?protocol=40400&discovery=40404
```

### Node ID Generation

The `node-id` is the Keccak-256 hash of the TLS public key (minus the `04` prefix).

To manually generate from a TLS key:

```bash
openssl ec -text -in node.key.pem | grep pub -A 5 | tail -n +2 | \
  tr -d '\n[:space:]:' | sed 's/^04//' | keccak-256sum -x -l | \
  tr -d ' -' | tail -c 41
```

**Important**: Each node needs unique TLS keys to have a unique node-id.

## Port Configuration

Each RNode instance uses 6 ports:

| Port  | Purpose | Protocol | Description |
|-------|---------|----------|-------------|
| 40400 | RChain Protocol | TCP | Inter-node protocol messages, block propagation |
| 40401 | gRPC External API | TCP | Public API (deploys, queries) |
| 40402 | gRPC Internal API | TCP | Internal API (propose, admin operations) |
| 40403 | HTTP API | TCP | HTTP services |
| 40404 | Kademlia Discovery | UDP | Peer discovery via Kademlia DHT |
| 40405 | Admin HTTP API | TCP | Admin/monitoring endpoints |

### Port Mapping in Docker

When running multiple nodes on the same host, map external ports to avoid conflicts:

```yaml
ports:
  - "40400:40400"  # Bootstrap node uses default ports
  - "40410:40400"  # Validator1 maps external 40410 to internal 40400
  - "40411:40401"
  # ... etc
```

### Required vs Optional Ports

**Required for validator/bootstrap nodes:**
- 40400 (protocol) - **MUST** be accessible to other validators
- 40404 (discovery) - **MUST** be accessible to other validators

**Required for API access:**
- 40401 (external gRPC) - For external deploy/query operations
- 40402 (internal gRPC) - For propose operations (validators only)

**Optional:**
- 40403 (HTTP API) - Only if using HTTP interface
- 40405 (Admin HTTP) - Only for monitoring/admin features

## Network Configuration

### Private Networks (DMS/Internal)

For private networks, you **MUST** use:

```bash
--allow-private-addresses
```

Or in config file:

```hocon
protocol-server {
  allow-private-addresses = true
}
```

### Network ID

All nodes in a shard must use the same network ID:

```hocon
protocol-server {
  network-id = "testnet"
}
```

CLI: `--network-id=testnet`

### Host Configuration

Specify the hostname/IP that other nodes will use to reach this node:

```bash
--host=bootstrap.internal
```

The node binds to `0.0.0.0` but advertises this host address to peers.

## Validator Configuration

### Validator Keys

Validators need a secp256k1 key pair for block signing:

**CLI (not recommended for production):**
```bash
--validator-private-key=<64-char-hex-string>
--validator-public-key=<130-char-hex-string>  # optional, derived from private
```

**Config file:**
```hocon
casper {
  validator-private-key = "<hex-string>"
  validator-public-key = "<hex-string>"
}
```

**Best practice - encrypted file:**
```bash
--validator-private-key-path=/var/lib/rnode/validator.key
```

Set password via environment: `RNODE_VALIDATOR_PASSWORD=<password>`

### Genesis Ceremony Roles

**Bootstrap/Ceremony Master:**
- `ceremony-master-mode = true`
- `genesis-validator-mode = false` (doesn't validate, only coordinates)
- CLI: No `--genesis-validator` flag
- Waits for validator approvals, then creates genesis block

**Validators:**
- `ceremony-master-mode = false`
- `genesis-validator-mode = true`
- CLI: `--genesis-validator` flag
- Sign the genesis block proposed by ceremony master

### Genesis Files

Required files in `<data-dir>/genesis/`:
- `bonds.txt` - Validator bonds at genesis
- `wallets.txt` - Initial wallet balances

See [GENESIS_FILES_FORMAT.md](./GENESIS_FILES_FORMAT.md) for details.

## CLI Flags Reference

### Essential Flags

```bash
rnode run \
  --host=<hostname>                    # Advertised hostname
  --bootstrap=<rnode-uri>              # Bootstrap node URI
  --allow-private-addresses            # Required for private networks
  --validator-private-key=<hex>        # Validator signing key
  --genesis-validator                  # Mark as genesis validator (not for bootstrap)
```

### Network Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--host` | Advertised hostname | Auto-detected |
| `--bootstrap` | Bootstrap node URI | See defaults.conf |
| `--network-id` | Network identifier | testnet |
| `--allow-private-addresses` | Allow private IPs | false |
| `--protocol-port` | RChain protocol port | 40400 |
| `--discovery-port` | Kademlia port | 40404 |

### Validator Flags

| Flag | Description |
|------|-------------|
| `--validator-private-key` | Private key (hex) for signing |
| `--validator-public-key` | Public key (hex), can be derived |
| `--validator-private-key-path` | Path to encrypted key file |
| `--genesis-validator` | Participate in genesis ceremony |

### API Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--api-port-grpc-external` | External gRPC API | 40401 |
| `--api-port-grpc-internal` | Internal gRPC API | 40402 |
| `--api-port-http` | HTTP API | 40403 |
| `--api-port-admin-http` | Admin HTTP | 40405 |

### Genesis Ceremony Flags

| Flag | Description |
|------|-------------|
| `--required-signatures` | # of validator signatures needed |
| `--bonds-file` | Path to bonds.txt |
| `--wallets-file` | Path to wallets.txt |
| `--shard-name` | Shard name (e.g., "root") |

### Other Flags

| Flag | Description |
|------|-------------|
| `--standalone` | Stand-alone mode (no network) |
| `--autopropose` | Auto-propose blocks on new deploys |
| `-c, --config-file` | Path to config file |
| `--profile` | Profile: default or docker |
| `--data-dir` | Data directory path |

## Configuration File Format

Configuration files use HOCON (Human-Optimized Config Object Notation) format.

### Basic Structure

```hocon
# Protocol server configuration
protocol-server {
  network-id = "testnet"
  port = 40400
  allow-private-addresses = true
}

# Protocol client (bootstrap connection)
protocol-client {
  network-id = "testnet"
  # bootstrap = "rnode://..."
}

# Peer discovery
peers-discovery {
  port = 40404
}

# API servers
api-server {
  host = "0.0.0.0"
  port-grpc-external = 40401
  port-grpc-internal = 40402
  port-http = 40403
  port-admin-http = 40405
}

# Storage
storage {
  data-dir = "/var/lib/rnode"
}

# TLS configuration
tls {
  certificate-path = ${storage.data-dir}/node.certificate.pem
  key-path = ${storage.data-dir}/node.key.pem
}

# Casper consensus
casper {
  # Validator keys (if this node is a validator)
  # validator-private-key = "<hex>"

  # Shard configuration
  shard-name = "root"
  parent-shard-id = "/"

  # Genesis files
  genesis-block-data {
    genesis-data-dir = ${storage.data-dir}/genesis
    bonds-file = ${casper.genesis-block-data.genesis-data-dir}/bonds.txt
    wallets-file = ${casper.genesis-block-data.genesis-data-dir}/wallets.txt

    # PoS parameters
    bond-minimum = 1
    bond-maximum = 9223372036854775807
    epoch-length = 10000
    quarantine-length = 50000
    number-of-active-validators = 100
  }

  # Genesis ceremony
  genesis-ceremony {
    required-signatures = 2
    approve-interval = 5 minutes
    genesis-validator-mode = false
    ceremony-master-mode = false
  }

  # Consensus parameters
  fault-tolerance-threshold = 0.0
  synchrony-constraint-threshold = 0.67
  finalization-rate = 1
}
```

### Variable Substitution

HOCON supports variable substitution:

```hocon
storage {
  data-dir = "/var/lib/rnode"
}

tls {
  # References storage.data-dir
  key-path = ${storage.data-dir}/node.key.pem
}
```

### CLI Override Behavior

CLI flags **override** config file values. For example:

```bash
# Config file has: network-id = "mainnet"
rnode run -c myconfig.conf --network-id=testnet
# Result: network-id = "testnet" (CLI wins)
```

## TLS Keys and Node Identity

### Key Files

Each node needs two TLS files:
- `node.key.pem` - secp256r1 EC private key
- `node.certificate.pem` - X.509 certificate

**Location:** `<data-dir>/node.key.pem` and `<data-dir>/node.certificate.pem`

### Auto-generation

If files don't exist, RNode generates them automatically on first run.

### Manual Generation

```bash
# Generate secp256r1 private key
openssl ecparam -name secp256r1 -genkey -noout -out node.key.pem

# Generate self-signed certificate
openssl req -new -x509 -key node.key.pem -out node.certificate.pem -days 365
```

### Node ID Derivation

The bootstrap URI's `node-id` is derived from the TLS public key:

1. Extract public key from certificate/private key
2. Remove the `04` prefix (uncompressed point indicator)
3. Apply Keccak-256 hash
4. Take last 40 hex characters

**Important**: Different TLS keys = different node-id

### Pre-generating Node IDs

For DMS deployment, you can pre-generate TLS keys and calculate node-ids:

1. Generate TLS key pairs for each node
2. Calculate each node-id using the formula above
3. Bake keys and calculated node-ids into Docker images
4. Use calculated node-id in bootstrap URI configuration

## Configuration Examples

See [3NODE_SHARD_CONFIG.md](./3NODE_SHARD_CONFIG.md) for complete working examples.

## Troubleshooting

### Nodes Can't Connect

- Verify `--allow-private-addresses` is set for private networks
- Check that protocol and discovery ports are accessible
- Ensure `network-id` matches on all nodes
- Verify bootstrap URI node-id matches bootstrap node's TLS key

### Genesis Ceremony Hangs

- Check `required-signatures` matches number of validators with `--genesis-validator` flag
- Verify all validators have their public keys in `bonds.txt`
- Check validator private keys are correct
- Ensure ceremony master has `ceremony-master-mode = true`

### Port Conflicts

- Use `--use-random-ports` for automatic port assignment
- Or manually specify different ports per node
- Check no other services are using the ports: `ss -tulpn | grep 404`

## Reference Links

- Node configuration code: `/node/src/main/scala/coop/rchain/node/configuration/Configuration.scala`
- CLI options: `/node/src/main/scala/coop/rchain/node/configuration/commandline/Options.scala`
- Default configuration: `/node/src/main/resources/defaults.conf`
- PeerNode addressing: `/comm/src/main/scala/coop/rchain/comm/PeerNode.scala`
