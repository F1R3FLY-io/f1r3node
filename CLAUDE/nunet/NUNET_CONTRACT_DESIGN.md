# Nunet DMS System Contract Design

**Purpose**: Design specification for Rholang system contracts that enable orchestration of Nunet DMS deployments from within Firefly RNode.

**Related Documents**:
- `SYSTEM_CONTRACT_PATTERN.md` - Implementation patterns
- `RNODE_CONFIGURATION_GUIDE.md` - RNode configuration requirements
- `3NODE_SHARD_CONFIG.md` - 3-validator shard setup
- Task 002 research findings

**Date**: 2025-10-09

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Use Cases](#use-cases)
3. [API Design](#api-design)
4. [Contract Specifications](#contract-specifications)
5. [DMS Integration](#dms-integration)
6. [Security Considerations](#security-considerations)
7. [Example Workflows](#example-workflows)

---

## Executive Summary

The Nunet DMS system contract will allow Rholang smart contracts to:

1. **Deploy** 3-validator RNode shards to Nunet DMS
2. **Monitor** shard health and status
3. **Manage** shard lifecycle (start, stop, upgrade)
4. **Query** shard information (endpoints, validators, genesis state)

This enables **self-replicating shards**: Rholang code running on one shard can spawn child shards on Nunet infrastructure, creating a dynamically scaling blockchain network.

**Key Design Principles**:
- **Shell-based**: Execute `nunet-dms` CLI commands (not HTTP)
- **Asynchronous**: Long-running deployments return immediately with job IDs
- **Deterministic**: Replay mode uses cached results
- **Secure**: Validate all inputs, sanitize commands
- **Observable**: Detailed logging and error reporting

---

## Use Cases

### 1. Shard Spawning

**Scenario**: A Rholang contract detects high transaction load and spawns a child shard.

```rholang
new deployShard, ack in {
  deployShard!({
    "validators": 3,
    "network": "testnet",
    "genesis": {
      "bonds": [...],      // Validator public keys
      "wallets": [...]     // Initial token allocations
    }
  }, *ack) |
  for (@{"jobId": jobId} <- ack) {
    stdout!(["Shard deployment started:", jobId])
  }
}
```

### 2. Shard Monitoring

**Scenario**: Check if a deployed shard is healthy and ready to accept transactions.

```rholang
new getShardStatus, ack in {
  getShardStatus!(["shard-abc123"], *ack) |
  for (@{"status": status, "validators": validators} <- ack) {
    match status {
      "running" => stdout!("Shard is operational")
      "pending" => stdout!("Shard is still deploying")
      "failed" => stdout!("Shard deployment failed")
    }
  }
}
```

### 3. Cross-Shard Communication Setup

**Scenario**: Get a new shard's bootstrap URI to configure cross-shard routing.

```rholang
new getShardInfo, ack in {
  getShardInfo!(["shard-abc123"], *ack) |
  for (@{"bootstrapUri": uri, "endpoints": endpoints} <- ack) {
    // Store bootstrap URI in registry for cross-shard messaging
    @"rho:registry:insert"!(uri, endpoints, Nil)
  }
}
```

### 4. Resource-Based Scaling

**Scenario**: Scale shard count based on available Nunet compute resources.

```rholang
new listResources, deployShard in {
  @"rho:nunet:resources:list"!([], *listResources) |
  for (@{"available": resources} <- listResources) {
    match (resources.cpuCores > 12, resources.memory > 32000) {
      (true, true) => {
        // Enough resources for a new shard
        deployShard!({...}, Nil)
      }
      _ => stdout!("Insufficient resources for new shard")
    }
  }
}
```

---

## API Design

### Core Contracts

#### 1. `rho:nunet:shard:deploy`

**Purpose**: Deploy a new 3-validator RNode shard to Nunet DMS.

**URN**: `rho:nunet:shard:deploy`

**Parameters**:
1. `config` (Map) - Deployment configuration
2. `ack` (UnforgChan) - Acknowledgment channel

**Config Schema**:
```rholang
{
  "shardName": String,        // Optional, generated if not provided
  "networkId": String,        // e.g., "testnet", "mainnet"
  "validators": Int,          // Must be 3 for MVP
  "bonds": [                  // Validator stakes
    {"pubkey": String, "stake": Int},
    {"pubkey": String, "stake": Int},
    {"pubkey": String, "stake": Int}
  ],
  "wallets": [                // Genesis token allocations
    {"address": String, "balance": Int},
    ...
  ],
  "resources": {              // Optional resource requirements
    "cpuPerNode": Int,        // Cores per validator
    "memoryPerNode": Int,     // MB per validator
    "diskPerNode": Int        // GB per validator
  }
}
```

**Returns** (on ack channel):
```rholang
{
  "jobId": String,            // Deployment job ID for tracking
  "shardId": String,          // Unique shard identifier
  "estimatedTime": Int        // Estimated completion time (seconds)
}
```

**Errors**:
- Invalid config (missing fields, wrong types)
- Insufficient Nunet resources
- DMS CLI failure

**Example**:
```rholang
new deployShard, ack in {
  deployShard!({
    "networkId": "testnet",
    "validators": 3,
    "bonds": [
      {"pubkey": "04fa70d7...", "stake": 1000},
      {"pubkey": "04837a4c...", "stake": 1000},
      {"pubkey": "0457feba...", "stake": 1000}
    ],
    "wallets": [
      {"address": "1111AtahZee...", "balance": 50000000000000000}
    ]
  }, *ack) |
  for (@result <- ack) {
    stdout!(["Deployment started:", result.get("jobId")])
  }
}
```

---

#### 2. `rho:nunet:shard:status`

**Purpose**: Get current status of a deployed shard.

**URN**: `rho:nunet:shard:status`

**Parameters**:
1. `shardId` (String) - Shard identifier
2. `ack` (UnforgChan) - Acknowledgment channel

**Returns**:
```rholang
{
  "shardId": String,
  "status": String,           // "pending" | "deploying" | "running" | "stopped" | "failed"
  "phase": String,            // Current deployment phase (if deploying)
  "validators": [             // Validator information
    {
      "nodeId": String,
      "endpoint": String,     // hostname:port
      "status": String,       // "starting" | "running" | "stopped"
      "blockHeight": Int      // Current block height (if running)
    },
    ...
  ],
  "bootstrapUri": String,     // Bootstrap node URI
  "endpoints": {
    "grpcExternal": String,   // External gRPC API endpoint
    "httpApi": String         // HTTP API endpoint
  },
  "genesis": {
    "timestamp": String,      // ISO 8601 timestamp
    "blockHash": String       // Genesis block hash
  },
  "error": String             // Error message (if status = "failed")
}
```

**Example**:
```rholang
new getStatus, ack in {
  @"rho:nunet:shard:status"!(["shard-abc123"], *ack) |
  for (@result <- ack) {
    match result.get("status") {
      "running" => {
        stdout!(["Shard operational at:", result.get("bootstrapUri")])
      }
      "failed" => {
        stdout!(["Deployment failed:", result.get("error")])
      }
      status => stdout!(["Status:", status])
    }
  }
}
```

---

#### 3. `rho:nunet:shard:list`

**Purpose**: List all shards deployed by this RNode instance.

**URN**: `rho:nunet:shard:list`

**Parameters**:
1. `filters` (Map) - Optional filters (empty map for all)
2. `ack` (UnforgChan) - Acknowledgment channel

**Filter Schema**:
```rholang
{
  "status": String,           // Filter by status
  "networkId": String,        // Filter by network
  "minValidators": Int,       // Minimum validator count
  "tags": [String]            // Filter by tags
}
```

**Returns**:
```rholang
{
  "shards": [
    {
      "shardId": String,
      "shardName": String,
      "status": String,
      "networkId": String,
      "validatorCount": Int,
      "createdAt": String,    // ISO 8601 timestamp
      "bootstrapUri": String
    },
    ...
  ],
  "total": Int
}
```

**Example**:
```rholang
new listShards, ack in {
  @"rho:nunet:shard:list"!([{"status": "running"}], *ack) |
  for (@result <- ack) {
    stdout!(["Running shards:", result.get("total")])
  }
}
```

---

#### 4. `rho:nunet:shard:stop`

**Purpose**: Stop a running shard (graceful shutdown).

**URN**: `rho:nunet:shard:stop`

**Parameters**:
1. `shardId` (String) - Shard identifier
2. `ack` (UnforgChan) - Acknowledgment channel

**Returns**:
```rholang
{
  "shardId": String,
  "status": String,           // "stopping" | "stopped"
  "message": String
}
```

**Example**:
```rholang
new stopShard, ack in {
  @"rho:nunet:shard:stop"!(["shard-abc123"], *ack) |
  for (@result <- ack) {
    stdout!(["Shard stopped:", result.get("message")])
  }
}
```

---

#### 5. `rho:nunet:shard:delete`

**Purpose**: Delete a stopped shard (removes all resources).

**URN**: `rho:nunet:shard:delete`

**Parameters**:
1. `shardId` (String) - Shard identifier
2. `ack` (UnforgChan) - Acknowledgment channel

**Returns**:
```rholang
{
  "shardId": String,
  "deleted": Boolean,
  "message": String
}
```

**Example**:
```rholang
new deleteShard, ack in {
  @"rho:nunet:shard:delete"!(["shard-abc123"], *ack) |
  for (@result <- ack) {
    stdout!(["Shard deleted:", result.get("deleted")])
  }
}
```

---

#### 6. `rho:nunet:resources:list`

**Purpose**: List available Nunet compute resources.

**URN**: `rho:nunet:resources:list`

**Parameters**:
1. `filters` (Map) - Optional filters
2. `ack` (UnforgChan) - Acknowledgment channel

**Returns**:
```rholang
{
  "nodes": [
    {
      "nodeId": String,
      "available": Boolean,
      "cpuCores": Int,
      "memory": Int,          // MB
      "disk": Int,            // GB
      "location": String,     // Geographic region
      "price": Int            // Cost per hour (in tokens)
    },
    ...
  ],
  "summary": {
    "totalNodes": Int,
    "availableNodes": Int,
    "totalCpuCores": Int,
    "totalMemory": Int,
    "totalDisk": Int
  }
}
```

**Example**:
```rholang
new listResources, ack in {
  @"rho:nunet:resources:list"!([{}], *ack) |
  for (@result <- ack) {
    stdout!(["Available nodes:", result.get("summary").get("availableNodes")])
  }
}
```

---

#### 7. `rho:nunet:shard:logs`

**Purpose**: Retrieve logs from a shard's validators.

**URN**: `rho:nunet:shard:logs`

**Parameters**:
1. `config` (Map) - Log retrieval config
2. `ack` (UnforgChan) - Acknowledgment channel

**Config Schema**:
```rholang
{
  "shardId": String,
  "validatorIndex": Int,      // Optional, all validators if not specified
  "lines": Int,               // Number of recent lines (default 100)
  "follow": Boolean           // Stream logs (not supported in MVP)
}
```

**Returns**:
```rholang
{
  "shardId": String,
  "logs": [
    {
      "validatorIndex": Int,
      "lines": [String]       // Array of log lines
    },
    ...
  ]
}
```

**Example**:
```rholang
new getLogs, ack in {
  @"rho:nunet:shard:logs"!([{"shardId": "shard-abc123", "lines": 50}], *ack) |
  for (@result <- ack) {
    // Print logs from all validators
    stdout!(result.get("logs"))
  }
}
```

---

### Supporting Contracts

#### 8. `rho:nunet:keys:generate`

**Purpose**: Generate validator and wallet keys for genesis configuration.

**URN**: `rho:nunet:keys:generate`

**Parameters**:
1. `config` (Map) - Key generation config
2. `ack` (UnforgChan) - Acknowledgment channel

**Config Schema**:
```rholang
{
  "count": Int,               // Number of key pairs to generate
  "type": String              // "validator" | "wallet"
}
```

**Returns**:
```rholang
{
  "keys": [
    {
      "publicKey": String,    // Hex (validator) or Base58 (wallet)
      "privateKey": String,   // WARNING: Handle securely!
      "address": String       // For wallets only
    },
    ...
  ]
}
```

**Security Warning**: Private keys should never be stored on-chain. This contract is for testing/demo only.

**Example**:
```rholang
new generateKeys, ack in {
  @"rho:nunet:keys:generate"!([{"count": 3, "type": "validator"}], *ack) |
  for (@result <- ack) {
    // Use public keys for bonds.txt
    stdout!(result.get("keys"))
  }
}
```

---

#### 9. `rho:nunet:config:validate`

**Purpose**: Validate shard configuration before deployment.

**URN**: `rho:nunet:config:validate`

**Parameters**:
1. `config` (Map) - Shard configuration to validate
2. `ack` (UnforgChan) - Acknowledgment channel

**Returns**:
```rholang
{
  "valid": Boolean,
  "errors": [String],         // Validation errors (if any)
  "warnings": [String]        // Non-fatal warnings
}
```

**Example**:
```rholang
new validateConfig, ack in {
  @"rho:nunet:config:validate"!([{
    "validators": 3,
    "bonds": [...]
  }], *ack) |
  for (@result <- ack) {
    match result.get("valid") {
      true => stdout!("Config is valid")
      false => stdout!(["Errors:", result.get("errors")])
    }
  }
}
```

---

## Contract Specifications

### Fixed Channels and Body Refs

Following the pattern from `SYSTEM_CONTRACT_PATTERN.md`:

| Contract | Fixed Channel | Body Ref | URN |
|----------|---------------|----------|-----|
| `shard:deploy` | `byteName(32)` | `30L` | `rho:nunet:shard:deploy` |
| `shard:status` | `byteName(33)` | `31L` | `rho:nunet:shard:status` |
| `shard:list` | `byteName(34)` | `32L` | `rho:nunet:shard:list` |
| `shard:stop` | `byteName(35)` | `33L` | `rho:nunet:shard:stop` |
| `shard:delete` | `byteName(36)` | `34L` | `rho:nunet:shard:delete` |
| `resources:list` | `byteName(37)` | `35L` | `rho:nunet:resources:list` |
| `shard:logs` | `byteName(38)` | `36L` | `rho:nunet:shard:logs` |
| `keys:generate` | `byteName(39)` | `37L` | `rho:nunet:keys:generate` |
| `config:validate` | `byteName(40)` | `38L` | `rho:nunet:config:validate` |

**Note**: Assumes bytes 0-31 are allocated to existing contracts. Adjust if necessary.

---

### Service Architecture

Following the pattern from `SYSTEM_CONTRACT_PATTERN.md`:

```scala
// Service trait
trait NunetService {
  def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
  def getShardStatus[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]]
  def listShards[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
  def stopShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]]
  def deleteShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]]
  def listResources[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
  def getShardLogs[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
  def generateKeys[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
  def validateConfig[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
}

// Real implementation (executes nunet-dms CLI)
class NunetServiceImpl extends NunetService {
  // Implementation uses scala.sys.process._ to execute shell commands
}

// Disabled implementation (raises errors)
class DisabledNunetService extends NunetService {
  // All methods raise UnsupportedOperationException
}

// Singleton with enable/disable
object NunetServiceImpl {
  lazy val instance: NunetService = {
    if (isNunetEnabled) new NunetServiceImpl
    else new DisabledNunetService
  }
}
```

---

### Enable/Disable Configuration

**Environment Variable**: `NUNET_ENABLED`

**Config File** (`defaults.conf`):
```hocon
nunet {
  enabled = false           # Default: disabled
  cli-path = "nunet-dms"    # Path to nunet-dms executable
  timeout = 300             # Command timeout (seconds)

  # Optional: DMS endpoint configuration
  dms {
    endpoint = "https://dms.nunet.io"
    api-key = ""            # If authentication required
  }
}
```

**Enable Logic** (in `package.scala`):
```scala
private[rholang] def isNunetEnabled: Boolean = {
  val config = ConfigFactory.load()

  val envEnabled = Option(System.getenv("NUNET_ENABLED")).flatMap { value =>
    value.toLowerCase(Locale.ENGLISH) match {
      case "true" | "1" | "yes" | "on"  => Some(true)
      case "false" | "0" | "no" | "off" => Some(false)
      case _ => None
    }
  }

  val configEnabled = if (config.hasPath("nunet.enabled")) {
    Some(config.getBoolean("nunet.enabled"))
  } else None

  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

---

## DMS Integration

### Shell Command Execution

The Nunet service will execute `nunet-dms` CLI commands using `scala.sys.process._`:

```scala
import scala.sys.process._
import scala.concurrent.{ExecutionContext, Future}

class NunetServiceImpl(implicit ec: ExecutionContext) extends NunetService {
  private val cliPath = sys.env.getOrElse("NUNET_CLI_PATH", "nunet-dms")

  def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        // 1. Generate bonds.txt and wallets.txt from config
        val bondsFile = createBondsFile(config("bonds"))
        val walletsFile = createWalletsFile(config("wallets"))

        // 2. Execute nunet-dms deploy command
        val cmd = Seq(
          cliPath,
          "deploy",
          "--ensemble", "rnode-shard",
          "--bonds", bondsFile,
          "--wallets", walletsFile,
          "--validators", config("validators").toString,
          "--network-id", config("networkId").toString
        )

        val output = cmd.!!  // Execute and capture stdout

        // 3. Parse output for job ID and shard ID
        val jobId = extractJobId(output)
        val shardId = extractShardId(output)

        cb(Right(Map(
          "jobId" -> jobId,
          "shardId" -> shardId,
          "estimatedTime" -> 300
        )))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def getShardStatus[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val cmd = Seq(cliPath, "status", "--shard-id", shardId, "--format", "json")
        val output = cmd.!!
        val status = parseJson(output)  // Parse JSON output
        cb(Right(status))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  // ... other methods follow same pattern
}
```

### DMS Command Mapping

| Contract | DMS Command | Example |
|----------|-------------|---------|
| `shard:deploy` | `nunet-dms deploy` | `nunet-dms deploy --ensemble rnode-shard --validators 3` |
| `shard:status` | `nunet-dms status` | `nunet-dms status --shard-id abc123 --format json` |
| `shard:list` | `nunet-dms list` | `nunet-dms list --filter status=running --format json` |
| `shard:stop` | `nunet-dms stop` | `nunet-dms stop --shard-id abc123` |
| `shard:delete` | `nunet-dms delete` | `nunet-dms delete --shard-id abc123 --confirm` |
| `resources:list` | `nunet-dms resources` | `nunet-dms resources list --format json` |
| `shard:logs` | `nunet-dms logs` | `nunet-dms logs --shard-id abc123 --lines 100` |
| `keys:generate` | `rnode keygen` | `rnode keygen --algorithm secp256k1` |
| `config:validate` | `nunet-dms validate` | `nunet-dms validate --config config.json` |

**Assumptions**:
- `nunet-dms` CLI is available on the host system
- Commands support `--format json` for machine-readable output
- Authentication (if needed) uses environment variables or config file

**Alternative**: If DMS provides an HTTP API, use Akka HTTP (like Ollama) instead of shell commands.

---

### Genesis File Generation

The service must generate `bonds.txt` and `wallets.txt` from Rholang config:

```scala
private def createBondsFile(bonds: List[Map[String, Any]]): String = {
  val tempFile = File.createTempFile("bonds-", ".txt")
  val writer = new PrintWriter(tempFile)

  bonds.foreach { bond =>
    val pubkey = bond("pubkey").toString
    val stake = bond("stake").toString
    writer.println(s"$pubkey $stake")
  }

  writer.close()
  tempFile.getAbsolutePath
}

private def createWalletsFile(wallets: List[Map[String, Any]]): String = {
  val tempFile = File.createTempFile("wallets-", ".txt")
  val writer = new PrintWriter(tempFile)

  wallets.foreach { wallet =>
    val address = wallet("address").toString
    val balance = wallet("balance").toString
    writer.println(s"$address,$balance")
  }

  writer.close()
  tempFile.getAbsolutePath
}
```

**File Formats** (from Task 002 research):
- **bonds.txt**: `<130-char-hex-pubkey> <stake>` (space-separated)
- **wallets.txt**: `<rev-address>,<balance>` (comma-separated)

---

### RNode Configuration Generation

For complete shard deployments, the service must also generate RNode config files:

```scala
private def generateRNodeConfigs(shardConfig: Map[String, Any]): Map[String, String] = {
  val networkId = shardConfig("networkId").toString
  val validators = shardConfig("validators").asInstanceOf[Int]

  // Bootstrap node config
  val bootstrapConfig = s"""
protocol-server {
  network-id = "$networkId"
  allow-private-addresses = true
  port = 40400
}

casper {
  genesis-ceremony {
    ceremony-master-mode = true
    genesis-validator-mode = false
  }
}
"""

  // Validator configs (x3)
  val validatorConfigs = (0 until validators).map { i =>
    s"""
protocol-server {
  network-id = "$networkId"
  allow-private-addresses = true
  port = 40400
}

casper {
  genesis-ceremony {
    ceremony-master-mode = false
    genesis-validator-mode = true
    required-signatures = $validators
  }
}
"""
  }

  Map(
    "bootstrap.conf" -> bootstrapConfig,
    "validator0.conf" -> validatorConfigs(0),
    "validator1.conf" -> validatorConfigs(1),
    "validator2.conf" -> validatorConfigs(2)
  )
}
```

**Reference**: See `3NODE_SHARD_CONFIG.md` for complete configuration examples.

---

## Security Considerations

### 1. Input Validation

**Critical**: All Rholang inputs must be validated before shell execution.

```scala
private def validateShardConfig(config: Map[String, Any]): Either[String, Unit] = {
  // Validate required fields
  if (!config.contains("validators")) {
    return Left("Missing 'validators' field")
  }

  // Validate types
  config("validators") match {
    case v: Int if v == 3 => // OK
    case _ => return Left("'validators' must be 3")
  }

  // Validate bonds
  config.get("bonds") match {
    case Some(bonds: List[_]) =>
      bonds.foreach {
        case m: Map[_, _] =>
          if (!m.contains("pubkey") || !m.contains("stake")) {
            return Left("Each bond must have 'pubkey' and 'stake'")
          }
          // Validate pubkey format (130 hex chars)
          val pubkey = m("pubkey").toString
          if (!pubkey.matches("^[0-9a-fA-F]{130}$")) {
            return Left(s"Invalid pubkey format: $pubkey")
          }
          // Validate stake is positive integer
          m("stake") match {
            case s: Int if s > 0 => // OK
            case _ => return Left(s"Invalid stake: ${m("stake")}")
          }
        case _ => return Left("Each bond must be a map")
      }
    case _ => return Left("'bonds' must be a list")
  }

  Right(())
}
```

### 2. Command Injection Prevention

**Never** concatenate user input directly into shell commands:

```scala
// BAD - vulnerable to injection
val cmd = s"nunet-dms deploy --shard-id ${shardId}"

// GOOD - use Seq with separate arguments
val cmd = Seq("nunet-dms", "deploy", "--shard-id", shardId)
```

### 3. Privilege Separation

The `nunet-dms` CLI should run with minimal privileges:
- No root access
- Restricted file system access
- Network access only to DMS endpoints
- Resource limits (CPU, memory, time)

### 4. Private Key Handling

**NEVER** store private keys on-chain or in contract state:
- `rho:nunet:keys:generate` should only be used for testing/demos
- Production deployments must use external key management
- Consider removing key generation from production builds

### 5. Rate Limiting

Prevent abuse by rate-limiting contract calls:
```scala
private val deploymentRateLimit = RateLimiter.create(1.0)  // 1 deployment per second

def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
  if (!deploymentRateLimit.tryAcquire()) {
    return F.raiseError(new Exception("Rate limit exceeded. Try again later."))
  }
  // ... proceed with deployment
}
```

### 6. Resource Quotas

Limit the number of shards a single RNode can deploy:
```scala
private val maxShardsPerNode = 10

def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
  val currentShardCount = getActiveShardCount()
  if (currentShardCount >= maxShardsPerNode) {
    return F.raiseError(new Exception(s"Maximum shard limit reached: $maxShardsPerNode"))
  }
  // ... proceed with deployment
}
```

---

## Example Workflows

### Complete Shard Deployment

```rholang
// 1. Generate validator keys
new generateKeys, keysAck in {
  @"rho:nunet:keys:generate"!([{"count": 3, "type": "validator"}], *keysAck) |
  for (@{"keys": validatorKeys} <- keysAck) {

    // 2. Generate wallet keys
    new walletKeysAck in {
      @"rho:nunet:keys:generate"!([{"count": 5, "type": "wallet"}], *walletKeysAck) |
      for (@{"keys": walletKeys} <- walletKeysAck) {

        // 3. Prepare shard config
        new deployAck in {
          @"rho:nunet:shard:deploy"!([{
            "networkId": "testnet",
            "validators": 3,
            "bonds": [
              {"pubkey": validatorKeys.nth(0).get("publicKey"), "stake": 1000},
              {"pubkey": validatorKeys.nth(1).get("publicKey"), "stake": 1000},
              {"pubkey": validatorKeys.nth(2).get("publicKey"), "stake": 1000}
            ],
            "wallets": [
              {"address": walletKeys.nth(0).get("address"), "balance": 50000000000000000},
              {"address": walletKeys.nth(1).get("address"), "balance": 50000000000000000},
              {"address": walletKeys.nth(2).get("address"), "balance": 50000000000000000},
              {"address": walletKeys.nth(3).get("address"), "balance": 50000000000000},
              {"address": walletKeys.nth(4).get("address"), "balance": 50000000000000000}
            ]
          }], *deployAck) |
          for (@{"jobId": jobId, "shardId": shardId} <- deployAck) {

            // 4. Poll for deployment completion
            new pollStatus in {
              contract pollStatus(@count) = {
                new statusAck in {
                  @"rho:nunet:shard:status"!([shardId], *statusAck) |
                  for (@{"status": status} <- statusAck) {
                    match status {
                      "running" => {
                        stdout!(["Shard deployed successfully:", shardId])
                      }
                      "failed" => {
                        stdout!(["Deployment failed for:", shardId])
                      }
                      _ => {
                        // Still deploying, poll again in 10 seconds
                        if (count < 60) {  // Max 10 minutes
                          @"rho:io:stdlog"!(["Still deploying... status:", status]) |
                          // Wait and poll again
                          pollStatus!(count + 1)
                        } else {
                          stdout!("Deployment timeout exceeded")
                        }
                      }
                    }
                  }
                }
              } |
              pollStatus!(0)
            }
          }
        }
      }
    }
  }
}
```

### Self-Replicating Shard

```rholang
// Contract that monitors load and spawns child shards
new loadMonitor in {
  contract loadMonitor(@threshold) = {
    // 1. Check current transaction rate
    new getRateAck in {
      @"rho:metrics:transaction-rate"!([], *getRateAck) |
      for (@{"rate": txRate} <- getRateAck) {

        // 2. If above threshold, spawn child shard
        if (txRate > threshold) {
          stdout!(["High load detected:", txRate, "tx/s"]) |

          // 3. Check available resources
          new resourcesAck in {
            @"rho:nunet:resources:list"!([{}], *resourcesAck) |
            for (@{"summary": summary} <- resourcesAck) {

              if (summary.get("availableNodes") >= 4) {  // 1 bootstrap + 3 validators
                // 4. Deploy child shard
                new deployAck in {
                  @"rho:nunet:shard:deploy"!([{
                    "networkId": "testnet",
                    "validators": 3,
                    "bonds": [...],  // Reuse existing validator keys
                    "wallets": [...]
                  }], *deployAck) |
                  for (@{"shardId": childShardId} <- deployAck) {
                    stdout!(["Child shard spawned:", childShardId]) |

                    // 5. Register child shard in routing table
                    @"rho:registry:insert"!(["child-shards", childShardId], Nil)
                  }
                }
              } else {
                stdout!("Insufficient resources for child shard")
              }
            }
          }
        } else {
          stdout!(["Load normal:", txRate, "tx/s"])
        }
      }
    } |

    // 6. Continue monitoring (recursive call after delay)
    // In production, use a timer mechanism
    loadMonitor!(threshold)
  } |

  // Start monitoring with threshold of 1000 tx/s
  loadMonitor!(1000)
}
```

### Cross-Shard Message Routing

```rholang
// Route messages to specific shards based on address space
new routeMessage in {
  contract routeMessage(@targetAddress, @message) = {
    // 1. Determine target shard from address
    new getShardAck in {
      @"rho:registry:lookup"!(["address-to-shard", targetAddress], *getShardAck) |
      for (@{"shardId": shardId} <- getShardAck) {

        // 2. Get shard connection info
        new shardInfoAck in {
          @"rho:nunet:shard:status"!([shardId], *shardInfoAck) |
          for (@{"endpoints": endpoints} <- shardInfoAck) {

            // 3. Send message via gRPC
            new sendAck in {
              @"rho:grpc:send"!([
                endpoints.get("grpcExternal"),
                "deploy",
                message
              ], *sendAck) |
              for (@result <- sendAck) {
                stdout!(["Message routed to shard:", shardId, "result:", result])
              }
            }
          }
        }
      }
    }
  }
}
```

---

## Summary

This design provides a comprehensive system contract API for orchestrating Nunet DMS deployments from Rholang:

**Core Capabilities**:
1. Deploy 3-validator shards with custom genesis configuration
2. Monitor shard status and health
3. List, stop, and delete shards
4. Query available Nunet resources
5. Retrieve shard logs
6. Generate keys and validate configs (testing only)

**Key Features**:
- Shell-based execution via `nunet-dms` CLI
- Asynchronous operations with job tracking
- Replay-safe for deterministic blockchain execution
- Comprehensive input validation and security controls
- Enable/disable via environment and config

**Use Cases**:
- Self-replicating shards for dynamic scaling
- Cross-shard message routing
- Resource-based shard spawning
- Automated shard management

**Next Steps**:
1. Finalize `nunet-dms` CLI specification with Nunet team
2. Implement service layer (`NunetService.scala`)
3. Implement system process handlers (`SystemProcesses.scala`)
4. Add registration in `RhoRuntime.scala`
5. Create comprehensive tests
6. Document deployment procedures

See `IMPLEMENTATION_GUIDE.md` for step-by-step implementation instructions.
