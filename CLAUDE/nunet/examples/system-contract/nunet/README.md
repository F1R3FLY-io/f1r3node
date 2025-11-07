# Nunet DMS System Contract Examples

This directory contains Rholang examples demonstrating how to use the Nunet DMS system contracts to orchestrate 3-validator RNode shards.

## Prerequisites

1. **Enable Nunet contracts**:
   ```bash
   export NUNET_ENABLED=true
   ```

2. **Install nunet-dms CLI**:
   ```bash
   # Example installation (adjust for your system)
   curl -sSL https://nunet.io/install-dms.sh | bash
   ```

3. **Verify installation**:
   ```bash
   nunet-dms --version
   ```

4. **Configure RNode** (if using custom CLI path):
   ```bash
   export NUNET_CLI_PATH=/usr/local/bin/nunet-dms
   ```

## Examples

### 1. Basic Shard Deployment

**File**: `deploy-shard.rho`

Deploy a 3-validator shard with predefined validator keys and genesis wallets.

```bash
rnode eval deploy-shard.rho
```

**Output**:
```
=== Shard Deployment Started ===
Job ID: job-a1b2c3d4
Shard ID: shard-x9y8z7w6
Estimated time: 300 seconds
================================
```

**What it does**:
- Validates configuration
- Generates bonds.txt and wallets.txt
- Executes `nunet-dms deploy` command
- Returns job ID and shard ID for tracking

---

### 2. Monitor Shard Status

**File**: `monitor-shard.rho`

Poll shard status until deployment completes or fails.

**Setup**:
1. Deploy a shard using `deploy-shard.rho`
2. Copy the shard ID from output
3. Edit `monitor-shard.rho` and replace `SHARD_ID` with your actual shard ID

```bash
rnode eval monitor-shard.rho
```

**Output**:
```
Checking shard status... (attempt 1 of 60)
Status: deploying - Initializing validators
Waiting 10 seconds before next check...
Checking shard status... (attempt 2 of 60)
Status: deploying - Running genesis ceremony
...
=== Shard is RUNNING ===
Shard ID: shard-x9y8z7w6
Bootstrap URI: rnode://abc123@bootstrap.internal?protocol=40400&discovery=40404
gRPC endpoint: https://shard-x9y8z7w6-grpc.nunet.io:40401
HTTP API: http://shard-x9y8z7w6-api.nunet.io:40403
Genesis timestamp: 2025-10-09T12:34:56Z
========================
Validators:
  Validator 0: validator0.internal:40400 - running
  Validator 1: validator1.internal:40400 - running
  Validator 2: validator2.internal:40400 - running
```

**What it does**:
- Queries `rho:nunet:shard:status` repeatedly
- Prints current deployment phase
- Shows validator information when running
- Displays error if deployment fails

---

### 3. Self-Replicating Shard (Auto-Scaling)

**File**: `self-replicating-shard.rho`

Demonstrates load-based auto-scaling by monitoring transaction rate and spawning child shards.

```bash
rnode eval self-replicating-shard.rho
```

**Output**:
```
=== Starting Load Monitor ===
Threshold: 1000 tx/s
Check interval: 30 seconds
=============================
Current transaction rate: 1500 tx/s
HIGH LOAD DETECTED! Threshold: 1000 tx/s
Available Nunet nodes: 8
Sufficient resources. Spawning child shard...
=== Child Shard Spawned ===
Job ID: job-e5f6g7h8
Shard ID: shard-v5u4t3s2
============================
Next check in 30 seconds...
```

**What it does**:
- Monitors transaction rate (simulated in example)
- Checks if load exceeds threshold
- Queries available Nunet resources
- Deploys child shard if resources available
- Registers child shard for routing

**Note**: This is a conceptual example. Production implementations need:
- Real metrics collection
- Proper delay mechanisms
- Cross-shard message routing
- Resource quotas and limits

---

### 4. List All Shards

**File**: `list-shards.rho`

List all shards deployed by this RNode instance.

```bash
rnode eval list-shards.rho
```

**Output**:
```
=== Listing All Shards ===
Total shards: 3
=============================

Shard 1:
  ID: shard-x9y8z7w6
  Name: rnode-shard-1
  Status: running
  Network: testnet
  Validators: 3
  Created: 2025-10-09T10:15:30Z
  Bootstrap: rnode://abc123@bootstrap.internal?protocol=40400&discovery=40404

Shard 2:
  ID: shard-v5u4t3s2
  Name: rnode-shard-2
  Status: running
  Network: testnet
  Validators: 3
  Created: 2025-10-09T11:22:45Z
  Bootstrap: rnode://def456@bootstrap.internal?protocol=40400&discovery=40404

Shard 3:
  ID: shard-p9o8i7u6
  Name: rnode-shard-3
  Status: deploying
  Network: testnet
  Validators: 3
  Created: 2025-10-09T12:01:00Z
  Bootstrap: rnode://ghi789@bootstrap.internal?protocol=40400&discovery=40404

=============================
```

**What it does**:
- Queries `rho:nunet:shard:list` with no filters
- Prints summary information for each shard
- Shows total shard count

---

## Available Contracts

### Shard Management

| Contract | Purpose | Parameters |
|----------|---------|------------|
| `rho:nunet:shard:deploy` | Deploy new shard | config (Map), ack |
| `rho:nunet:shard:status` | Get shard status | shardId (String), ack |
| `rho:nunet:shard:list` | List all shards | filters (Map), ack |
| `rho:nunet:shard:stop` | Stop shard | shardId (String), ack |
| `rho:nunet:shard:delete` | Delete shard | shardId (String), ack |
| `rho:nunet:shard:logs` | Get shard logs | config (Map), ack |

### Resource Management

| Contract | Purpose | Parameters |
|----------|---------|------------|
| `rho:nunet:resources:list` | List available Nunet nodes | filters (Map), ack |

### Utilities

| Contract | Purpose | Parameters |
|----------|---------|------------|
| `rho:nunet:keys:generate` | Generate validator/wallet keys | config (Map), ack |
| `rho:nunet:config:validate` | Validate shard config | config (Map), ack |

---

## Configuration Schema

### Deploy Config

```rholang
{
  "networkId": "testnet",           // Network identifier
  "validators": 3,                  // Must be 3
  "bonds": [                        // Validator stakes
    {
      "pubkey": "04abc...",          // 130-char hex pubkey (secp256k1)
      "stake": 1000                  // Integer stake amount
    },
    ...
  ],
  "wallets": [                      // Genesis token allocations
    {
      "address": "1111...",          // Base58 REV address
      "balance": 50000000000000000   // Integer balance
    },
    ...
  ],
  "resources": {                    // Optional resource requirements
    "cpuPerNode": 4,
    "memoryPerNode": 8192,           // MB
    "diskPerNode": 100               // GB
  }
}
```

### Status Response

```rholang
{
  "shardId": "shard-abc123",
  "status": "running",               // pending | deploying | running | stopped | failed
  "phase": "operational",            // Current phase (if deploying)
  "validators": [                    // Validator info
    {
      "nodeId": "abc123",
      "endpoint": "validator0.internal:40400",
      "status": "running",
      "blockHeight": 1250
    },
    ...
  ],
  "bootstrapUri": "rnode://...",     // Bootstrap node URI
  "endpoints": {
    "grpcExternal": "https://...:40401",
    "httpApi": "http://...:40403"
  },
  "genesis": {
    "timestamp": "2025-10-09T12:34:56Z",
    "blockHash": "abc123..."
  }
}
```

---

## Troubleshooting

### Error: "Nunet service is not enabled"

**Cause**: NUNET_ENABLED not set or set to false.

**Solution**:
```bash
export NUNET_ENABLED=true
```

Or add to `application.conf`:
```hocon
nunet.enabled = true
```

---

### Error: "Command failed: nunet-dms: command not found"

**Cause**: nunet-dms CLI not in PATH.

**Solution**:
```bash
# Option 1: Add to PATH
export PATH=$PATH:/path/to/nunet-dms

# Option 2: Set CLI path
export NUNET_CLI_PATH=/usr/local/bin/nunet-dms
```

Or in config:
```hocon
nunet.cli-path = "/usr/local/bin/nunet-dms"
```

---

### Error: "Invalid pubkey format"

**Cause**: Validator public key is not 130 hex characters.

**Solution**: Generate keys with `rnode keygen`:
```bash
rnode keygen --algorithm secp256k1
```

Ensure keys are **uncompressed** (130 hex chars, starting with `04`).

---

### Error: "Insufficient resources"

**Cause**: Not enough available Nunet nodes for deployment.

**Solution**: Check available resources:
```rholang
new ack in {
  @"rho:nunet:resources:list"!([{}], *ack) |
  for (@result <- ack) {
    @"rho:io:stdlog"!(result.get("summary"))
  }
}
```

Wait for resources to become available or reduce resource requirements.

---

### Deployment timeout

**Cause**: Shard deployment takes longer than expected.

**Solution**: Increase timeout in config:
```hocon
nunet.timeout = 600  # 10 minutes
```

Or check shard logs for errors:
```rholang
new ack in {
  @"rho:nunet:shard:logs"!([{
    "shardId": "shard-abc123",
    "lines": 100
  }], *ack) |
  for (@result <- ack) {
    @"rho:io:stdlog"!(result.get("logs"))
  }
}
```

---

## Next Steps

1. **Customize genesis configuration**: Edit `deploy-shard.rho` with your own validator keys and wallet addresses

2. **Implement cross-shard routing**: Use bootstrap URIs from deployed shards to configure message routing

3. **Add monitoring**: Set up alerts for shard health and resource usage

4. **Scale horizontally**: Use the self-replicating pattern to auto-scale based on load

5. **Production deployment**: Review security best practices in `NUNET_CONTRACT_DESIGN.md`

---

## Additional Resources

- **Design Documentation**: `NUNET_CONTRACT_DESIGN.md` - Complete API specification
- **Implementation Guide**: `NUNET_IMPLEMENTATION_GUIDE.md` - Step-by-step implementation
- **System Contract Pattern**: `SYSTEM_CONTRACT_PATTERN.md` - General contract architecture
- **RNode Configuration**: `RNODE_CONFIGURATION_GUIDE.md`, `3NODE_SHARD_CONFIG.md` - Shard setup details

---

## Security Notes

1. **Private keys**: Never commit private keys to version control or store them on-chain
2. **Key generation**: `rho:nunet:keys:generate` is for testing only - use secure key management in production
3. **Input validation**: All configs are validated before execution
4. **Resource limits**: Configure quotas to prevent resource exhaustion
5. **Rate limiting**: Deployment rate is limited (default: 1/minute)

---

## License

These examples are part of the Firefly RNode project and are provided for educational and demonstration purposes.
