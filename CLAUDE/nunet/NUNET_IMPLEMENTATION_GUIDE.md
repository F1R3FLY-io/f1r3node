# Nunet DMS System Contract - Implementation Guide

**Purpose**: Step-by-step instructions for implementing the Nunet DMS system contracts in the Firefly RNode codebase.

**Prerequisites**:
- Read `SYSTEM_CONTRACT_PATTERN.md` (understanding of system contract architecture)
- Read `NUNET_CONTRACT_DESIGN.md` (API specifications)
- Familiarity with Scala, Cats Effect, and Rholang
- Access to Firefly RNode source code

**Estimated Time**: 2-3 weeks for complete implementation

**Date**: 2025-10-09

---

## Table of Contents

1. [Implementation Roadmap](#implementation-roadmap)
2. [Phase 1: Service Layer](#phase-1-service-layer)
3. [Phase 2: System Processes](#phase-2-system-processes)
4. [Phase 3: Registration](#phase-3-registration)
5. [Phase 4: Configuration](#phase-4-configuration)
6. [Phase 5: Testing](#phase-5-testing)
7. [Phase 6: Documentation](#phase-6-documentation)
8. [Troubleshooting](#troubleshooting)

---

## Implementation Roadmap

### Overview

```
Phase 1: Service Layer (3-5 days)
  ├─ Create NunetService.scala (trait + implementations)
  ├─ Add to ExternalServices.scala
  └─ Implement enable/disable logic

Phase 2: System Processes (5-7 days)
  ├─ Add fixed channels and body refs
  ├─ Implement contract handlers (9 contracts)
  └─ Add type conversions

Phase 3: Registration (1 day)
  ├─ Create stdRhoNunetProcesses
  └─ Add to dispatch table

Phase 4: Configuration (1 day)
  ├─ Add HOCON config
  └─ Update defaults.conf

Phase 5: Testing (3-5 days)
  ├─ Unit tests for service layer
  ├─ Integration tests for contracts
  └─ End-to-end shard deployment test

Phase 6: Documentation (1-2 days)
  ├─ Update CHANGELOG.md
  ├─ Add Rholang examples
  └─ Write deployment guide
```

---

## Phase 1: Service Layer

### Step 1.1: Create NunetService.scala

**Location**: `rholang/src/main/scala/coop/rchain/rholang/externalservices/NunetService.scala`

**File structure**:

```scala
package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.syntax.all._
import scala.sys.process._
import scala.concurrent.{ExecutionContext, Future}
import java.io.{File, PrintWriter}
import spray.json._

/**
 * Service for orchestrating Nunet DMS deployments.
 *
 * Allows Rholang contracts to deploy, manage, and monitor
 * 3-validator RNode shards on Nunet infrastructure.
 */
trait NunetService {
  /**
   * Deploy a new 3-validator shard to Nunet DMS.
   *
   * @param config Deployment configuration including validators, bonds, wallets
   * @return Job ID and shard ID for tracking deployment
   */
  def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * Get current status of a deployed shard.
   *
   * @param shardId Unique shard identifier
   * @return Status info including validators, endpoints, genesis info
   */
  def getShardStatus[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * List all shards deployed by this RNode instance.
   *
   * @param filters Optional filters (status, networkId, etc.)
   * @return List of shards with summary information
   */
  def listShards[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * Stop a running shard (graceful shutdown).
   *
   * @param shardId Unique shard identifier
   * @return Stop confirmation with status
   */
  def stopShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * Delete a stopped shard (removes all resources).
   *
   * @param shardId Unique shard identifier
   * @return Deletion confirmation
   */
  def deleteShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * List available Nunet compute resources.
   *
   * @param filters Optional filters (location, minCpu, etc.)
   * @return List of available nodes with resource info
   */
  def listResources[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * Retrieve logs from a shard's validators.
   *
   * @param config Log retrieval config (shardId, lines, validatorIndex)
   * @return Log lines from specified validators
   */
  def getShardLogs[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * Generate validator or wallet keys.
   *
   * WARNING: For testing/demo only. Do not use in production.
   *
   * @param config Key generation config (count, type)
   * @return Generated key pairs with public/private keys
   */
  def generateKeys[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]

  /**
   * Validate shard configuration before deployment.
   *
   * @param config Shard configuration to validate
   * @return Validation result with errors/warnings
   */
  def validateConfig[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]]
}

/**
 * Disabled implementation - raises errors when service is not enabled.
 */
class DisabledNunetService extends NunetService {
  private def disabled[F[_]]()(implicit F: Concurrent[F]): F[Nothing] = {
    F.raiseError(new UnsupportedOperationException(
      "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true"
    ))
  }

  def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def getShardStatus[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def listShards[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def stopShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def deleteShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def listResources[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def getShardLogs[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def generateKeys[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
  def validateConfig[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = disabled()
}

/**
 * Real implementation - executes nunet-dms CLI commands.
 */
class NunetServiceImpl(implicit ec: ExecutionContext) extends NunetService {
  import NunetServiceImpl._

  private val cliPath = sys.env.getOrElse("NUNET_CLI_PATH", "nunet-dms")
  private val timeout = sys.env.getOrElse("NUNET_TIMEOUT", "300").toInt

  def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        // Validate config
        validateShardConfigOrThrow(config)

        // Generate genesis files
        val bondsFile = createBondsFile(config("bonds").asInstanceOf[List[Map[String, Any]]])
        val walletsFile = createWalletsFile(config("wallets").asInstanceOf[List[Map[String, Any]]])

        try {
          // Execute deployment command
          val cmd = Seq(
            cliPath,
            "deploy",
            "--ensemble", "rnode-shard",
            "--bonds", bondsFile,
            "--wallets", walletsFile,
            "--validators", config("validators").toString,
            "--network-id", config.getOrElse("networkId", "testnet").toString,
            "--format", "json"
          )

          val output = executeCommand(cmd, timeout)
          val result = parseJsonOutput(output)

          cb(Right(result))
        } finally {
          // Clean up temp files
          new File(bondsFile).delete()
          new File(walletsFile).delete()
        }
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def getShardStatus[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val cmd = Seq(
          cliPath,
          "status",
          "--shard-id", shardId,
          "--format", "json"
        )

        val output = executeCommand(cmd, timeout)
        val result = parseJsonOutput(output)

        cb(Right(result))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def listShards[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val filterArgs = filters.flatMap {
          case (key, value) => Seq(s"--filter", s"$key=$value")
        }.toSeq

        val cmd = Seq(cliPath, "list", "--format", "json") ++ filterArgs

        val output = executeCommand(cmd, timeout)
        val result = parseJsonOutput(output)

        cb(Right(result))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def stopShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val cmd = Seq(
          cliPath,
          "stop",
          "--shard-id", shardId,
          "--format", "json"
        )

        val output = executeCommand(cmd, timeout)
        val result = parseJsonOutput(output)

        cb(Right(result))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def deleteShard[F[_]](shardId: String)(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val cmd = Seq(
          cliPath,
          "delete",
          "--shard-id", shardId,
          "--confirm",
          "--format", "json"
        )

        val output = executeCommand(cmd, timeout)
        val result = parseJsonOutput(output)

        cb(Right(result))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def listResources[F[_]](filters: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val filterArgs = filters.flatMap {
          case (key, value) => Seq(s"--filter", s"$key=$value")
        }.toSeq

        val cmd = Seq(cliPath, "resources", "list", "--format", "json") ++ filterArgs

        val output = executeCommand(cmd, timeout)
        val result = parseJsonOutput(output)

        cb(Right(result))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def getShardLogs[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val shardId = config("shardId").toString
        val lines = config.getOrElse("lines", 100).toString

        val validatorArgs = config.get("validatorIndex") match {
          case Some(idx) => Seq("--validator", idx.toString)
          case None => Seq()
        }

        val cmd = Seq(
          cliPath,
          "logs",
          "--shard-id", shardId,
          "--lines", lines,
          "--format", "json"
        ) ++ validatorArgs

        val output = executeCommand(cmd, timeout)
        val result = parseJsonOutput(output)

        cb(Right(result))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def generateKeys[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        val count = config("count").asInstanceOf[Int]
        val keyType = config("type").toString

        val keys = keyType match {
          case "validator" => generateValidatorKeys(count)
          case "wallet" => generateWalletKeys(count)
          case _ => throw new IllegalArgumentException(s"Unknown key type: $keyType")
        }

        cb(Right(Map("keys" -> keys)))
      }.recover {
        case e: Exception => cb(Left(e))
      }
    }
  }

  def validateConfig[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
    F.async { cb =>
      Future {
        try {
          validateShardConfigOrThrow(config)
          cb(Right(Map(
            "valid" -> true,
            "errors" -> List(),
            "warnings" -> List()
          )))
        } catch {
          case e: Exception =>
            cb(Right(Map(
              "valid" -> false,
              "errors" -> List(e.getMessage),
              "warnings" -> List()
            )))
        }
      }
    }
  }
}

/**
 * Companion object with helper methods.
 */
object NunetServiceImpl {
  /**
   * Singleton instance with enable/disable logic.
   */
  lazy val instance: NunetService = {
    import scala.concurrent.ExecutionContext.Implicits.global
    if (isNunetEnabled) new NunetServiceImpl
    else new DisabledNunetService
  }

  /**
   * Execute shell command with timeout.
   */
  private def executeCommand(cmd: Seq[String], timeoutSeconds: Int): String = {
    import scala.concurrent.duration._

    val process = Process(cmd).run()
    val exitCode = process.exitValue()

    if (exitCode != 0) {
      throw new RuntimeException(s"Command failed with exit code $exitCode: ${cmd.mkString(" ")}")
    }

    scala.io.Source.fromInputStream(process.inputStream).mkString
  }

  /**
   * Parse JSON output from DMS CLI.
   */
  private def parseJsonOutput(json: String): Map[String, Any] = {
    import spray.json._

    val parsed = json.parseJson.asJsObject
    parsed.fields.map {
      case (key, JsString(value)) => key -> value
      case (key, JsNumber(value)) => key -> value.toInt
      case (key, JsBoolean(value)) => key -> value
      case (key, JsArray(elements)) => key -> elements.map(_.toString)
      case (key, JsObject(fields)) => key -> fields
      case (key, _) => key -> null
    }
  }

  /**
   * Create bonds.txt from config.
   */
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

  /**
   * Create wallets.txt from config.
   */
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

  /**
   * Validate shard configuration.
   */
  private def validateShardConfigOrThrow(config: Map[String, Any]): Unit = {
    // Validate validators
    require(config.contains("validators"), "Missing 'validators' field")
    require(config("validators") == 3, "'validators' must be 3")

    // Validate bonds
    require(config.contains("bonds"), "Missing 'bonds' field")
    val bonds = config("bonds").asInstanceOf[List[Map[String, Any]]]
    require(bonds.length == 3, "Must have exactly 3 bonds")

    bonds.foreach { bond =>
      require(bond.contains("pubkey"), "Each bond must have 'pubkey'")
      require(bond.contains("stake"), "Each bond must have 'stake'")

      val pubkey = bond("pubkey").toString
      require(pubkey.matches("^[0-9a-fA-F]{130}$"), s"Invalid pubkey format: $pubkey")

      val stake = bond("stake").asInstanceOf[Int]
      require(stake > 0, s"Stake must be positive: $stake")
    }

    // Validate wallets
    require(config.contains("wallets"), "Missing 'wallets' field")
    val wallets = config("wallets").asInstanceOf[List[Map[String, Any]]]
    require(wallets.nonEmpty, "Must have at least one wallet")

    wallets.foreach { wallet =>
      require(wallet.contains("address"), "Each wallet must have 'address'")
      require(wallet.contains("balance"), "Each wallet must have 'balance'")

      val address = wallet("address").toString
      require(address.startsWith("1111"), s"Invalid REV address: $address")

      val balance = wallet("balance").asInstanceOf[Int]
      require(balance > 0, s"Balance must be positive: $balance")
    }
  }

  /**
   * Generate validator keys (placeholder - use rnode keygen in production).
   */
  private def generateValidatorKeys(count: Int): List[Map[String, String]] = {
    (1 to count).map { _ =>
      // TODO: Call rnode keygen command
      Map(
        "publicKey" -> "04" + ("0" * 128),  // Placeholder
        "privateKey" -> "0" * 64             // Placeholder
      )
    }.toList
  }

  /**
   * Generate wallet keys (placeholder - use rnode keygen in production).
   */
  private def generateWalletKeys(count: Int): List[Map[String, String]] = {
    (1 to count).map { _ =>
      // TODO: Call rnode keygen command
      Map(
        "address" -> "1111" + ("A" * 44),    // Placeholder
        "publicKey" -> "04" + ("0" * 128),   // Placeholder
        "privateKey" -> "0" * 64              // Placeholder
      )
    }.toList
  }
}
```

**Implementation Notes**:
- Uses `scala.sys.process._` for shell command execution
- Parses JSON output from `nunet-dms` CLI
- Creates temporary files for bonds.txt and wallets.txt
- Validates all inputs before execution
- Cleans up temp files after use

**TODO**:
- Replace placeholder key generation with actual `rnode keygen` calls
- Add more robust JSON parsing (handle nested structures)
- Add timeout handling for long-running commands
- Add retry logic for transient failures

---

### Step 1.2: Add to ExternalServices.scala

**Location**: `rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala`

**Changes**:

```scala
trait ExternalServices {
  def openAIService: OpenAIService
  def grpcClient: GrpcClientService
  def ollamaService: OllamaService
  def nunetService: NunetService  // ADD THIS
}

class RealExternalServices extends ExternalServices {
  val openAIService: OpenAIService = OpenAIServiceImpl.instance
  val grpcClient: GrpcClientService = GrpcClientServiceImpl.instance
  val ollamaService: OllamaService = OllamaServiceImpl.instance
  val nunetService: NunetService = NunetServiceImpl.instance  // ADD THIS
}

class NoOpExternalServices extends ExternalServices {
  val openAIService: OpenAIService = new DisabledOpenAIService
  val grpcClient: GrpcClientService = new DisabledGrpcClientService
  val ollamaService: OllamaService = new DisabledOllamaService
  val nunetService: NunetService = new DisabledNunetService  // ADD THIS
}

class ObserverExternalServices(openAI: OpenAIService, grpc: GrpcClientService, ollama: OllamaService, nunet: NunetService) extends ExternalServices {
  val openAIService: OpenAIService = openAI
  val grpcClient: GrpcClientService = grpc
  val ollamaService: OllamaService = ollama
  val nunetService: NunetService = nunet  // ADD THIS
}
```

---

### Step 1.3: Add Enable/Disable Logic

**Location**: `rholang/src/main/scala/coop/rchain/rholang/externalservices/package.scala`

**Add function**:

```scala
private[rholang] def isNunetEnabled: Boolean = {
  val config = ConfigFactory.load()

  // Check environment variable (highest priority)
  val envEnabled = Option(System.getenv("NUNET_ENABLED")).flatMap { value =>
    value.toLowerCase(Locale.ENGLISH) match {
      case "true" | "1" | "yes" | "on"  => Some(true)
      case "false" | "0" | "no" | "off" => Some(false)
      case _ => None
    }
  }

  // Check configuration file (fallback)
  val configEnabled = if (config.hasPath("nunet.enabled")) {
    Some(config.getBoolean("nunet.enabled"))
  } else None

  // Priority: env > config > default false
  envEnabled.getOrElse(configEnabled.getOrElse(false))
}
```

---

## Phase 2: System Processes

### Step 2.1: Add Fixed Channels and Body Refs

**Location**: `rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala`

**Add to FixedChannels object** (around line 61-92):

```scala
object FixedChannels {
  // ... existing channels (0-30)

  // Nunet DMS contracts (31-40)
  val NUNET_SHARD_DEPLOY: Par = byteName(32)
  val NUNET_SHARD_STATUS: Par = byteName(33)
  val NUNET_SHARD_LIST: Par = byteName(34)
  val NUNET_SHARD_STOP: Par = byteName(35)
  val NUNET_SHARD_DELETE: Par = byteName(36)
  val NUNET_RESOURCES_LIST: Par = byteName(37)
  val NUNET_SHARD_LOGS: Par = byteName(38)
  val NUNET_KEYS_GENERATE: Par = byteName(39)
  val NUNET_CONFIG_VALIDATE: Par = byteName(40)
}
```

**Add to BodyRefs object** (around line 94-124):

```scala
object BodyRefs {
  // ... existing refs (0-28)

  // Nunet DMS contracts (29-38)
  val NUNET_SHARD_DEPLOY: Long = 30L
  val NUNET_SHARD_STATUS: Long = 31L
  val NUNET_SHARD_LIST: Long = 32L
  val NUNET_SHARD_STOP: Long = 33L
  val NUNET_SHARD_DELETE: Long = 34L
  val NUNET_RESOURCES_LIST: Long = 35L
  val NUNET_SHARD_LOGS: Long = 36L
  val NUNET_KEYS_GENERATE: Long = 37L
  val NUNET_CONFIG_VALIDATE: Long = 38L
}
```

**Note**: Adjust byte/long values if existing contracts use different ranges.

---

### Step 2.2: Implement Contract Handlers

**Location**: `rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala`

**Add handlers** (around line 521-600, after existing contract implementations):

```scala
// ============================================================
// Nunet DMS Contracts
// ============================================================

def nunetShardDeploy: Contract[F] = {
  // Replay mode
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  // Real mode
  case isContractCall(produce, _, _, Seq(RhoType.RhoMap(config), ack)) => {
    (for {
      // Convert Rholang map to Scala map
      scalaConfig <- convertRhoMapToScala(config)

      // Call service
      result <- externalServices.nunetService.deployShard(scalaConfig)

      // Convert result back to Rholang
      output = Seq(convertScalaMapToRho(result))

      // Produce result
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  // Invalid arguments
  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetShardStatus: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.String(shardId), ack)) => {
    (for {
      result <- externalServices.nunetService.getShardStatus(shardId)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetShardList: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.RhoMap(filters), ack)) => {
    (for {
      scalaFilters <- convertRhoMapToScala(filters)
      result <- externalServices.nunetService.listShards(scalaFilters)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetShardStop: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.String(shardId), ack)) => {
    (for {
      result <- externalServices.nunetService.stopShard(shardId)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetShardDelete: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.String(shardId), ack)) => {
    (for {
      result <- externalServices.nunetService.deleteShard(shardId)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetResourcesList: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.RhoMap(filters), ack)) => {
    (for {
      scalaFilters <- convertRhoMapToScala(filters)
      result <- externalServices.nunetService.listResources(scalaFilters)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetShardLogs: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.RhoMap(config), ack)) => {
    (for {
      scalaConfig <- convertRhoMapToScala(config)
      result <- externalServices.nunetService.getShardLogs(scalaConfig)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetKeysGenerate: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.RhoMap(config), ack)) => {
    (for {
      scalaConfig <- convertRhoMapToScala(config)
      result <- externalServices.nunetService.generateKeys(scalaConfig)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}

def nunetConfigValidate: Contract[F] = {
  case isContractCall(produce, true, previousOutput, Seq(_, ack)) => {
    produce(previousOutput, ack).map(_ => previousOutput)
  }

  case isContractCall(produce, _, _, Seq(RhoType.RhoMap(config), ack)) => {
    (for {
      scalaConfig <- convertRhoMapToScala(config)
      result <- externalServices.nunetService.validateConfig(scalaConfig)
      output = Seq(convertScalaMapToRho(result))
      _ <- produce(output, ack)
    } yield output).recoverWith {
      case e => NonDeterministicProcessFailure(
        outputNotProduced = Seq.empty,
        cause = e
      ).raiseError
    }
  }

  case _ => Unit.pure[F].map(_ => Seq.empty)
}
```

---

### Step 2.3: Add Type Conversion Helpers

**Location**: Same file, add helper methods for Rholang ↔ Scala conversions:

```scala
/**
 * Convert Rholang map to Scala map.
 */
private def convertRhoMapToScala(rhoMap: Map[Par, Par]): F[Map[String, Any]] = {
  F.delay {
    rhoMap.map {
      case (RhoType.String(key), RhoType.String(value)) => key -> value
      case (RhoType.String(key), RhoType.Number(value)) => key -> value.toInt
      case (RhoType.String(key), RhoType.Boolean(value)) => key -> value
      case (RhoType.String(key), RhoType.RhoList(items)) => key -> items.map(convertRhoToScala)
      case (RhoType.String(key), RhoType.RhoMap(nested)) => key -> convertRhoMapToScala(nested)
      case (key, value) => throw new IllegalArgumentException(s"Unsupported map entry: $key -> $value")
    }
  }
}

/**
 * Convert Scala map to Rholang map.
 */
private def convertScalaMapToRho(scalaMap: Map[String, Any]): Par = {
  val entries = scalaMap.map {
    case (key, value: String) => RhoType.String(key) -> RhoType.String(value)
    case (key, value: Int) => RhoType.String(key) -> RhoType.Number(value)
    case (key, value: Boolean) => RhoType.String(key) -> RhoType.Boolean(value)
    case (key, value: List[_]) => RhoType.String(key) -> RhoType.RhoList(value.map(convertScalaToRho))
    case (key, value: Map[_, _]) => RhoType.String(key) -> convertScalaMapToRho(value.asInstanceOf[Map[String, Any]])
    case (key, value) => throw new IllegalArgumentException(s"Unsupported value type: $key -> $value")
  }.toMap

  RhoType.RhoMap(entries)
}

/**
 * Convert individual Rholang value to Scala.
 */
private def convertRhoToScala(par: Par): Any = par match {
  case RhoType.String(value) => value
  case RhoType.Number(value) => value.toInt
  case RhoType.Boolean(value) => value
  case RhoType.RhoList(items) => items.map(convertRhoToScala)
  case RhoType.RhoMap(entries) => convertRhoMapToScala(entries)
  case _ => throw new IllegalArgumentException(s"Unsupported Rholang value: $par")
}

/**
 * Convert individual Scala value to Rholang.
 */
private def convertScalaToRho(value: Any): Par = value match {
  case s: String => RhoType.String(s)
  case i: Int => RhoType.Number(i)
  case b: Boolean => RhoType.Boolean(b)
  case l: List[_] => RhoType.RhoList(l.map(convertScalaToRho))
  case m: Map[_, _] => convertScalaMapToRho(m.asInstanceOf[Map[String, Any]])
  case _ => throw new IllegalArgumentException(s"Unsupported Scala value: $value")
}
```

**Note**: These are simplified helpers. Production code should handle more types and edge cases.

---

## Phase 3: Registration

### Step 3.1: Create Process Group

**Location**: `rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoRuntime.scala`

**Add function** (around line 475-554):

```scala
def stdRhoNunetProcesses[F[_]]: Seq[Definition[F]] = Seq(
  Definition[F](
    "rho:nunet:shard:deploy",
    FixedChannels.NUNET_SHARD_DEPLOY,
    2,  // config + ack
    BodyRefs.NUNET_SHARD_DEPLOY,
    { ctx => ctx.systemProcesses.nunetShardDeploy }
  ),
  Definition[F](
    "rho:nunet:shard:status",
    FixedChannels.NUNET_SHARD_STATUS,
    2,  // shardId + ack
    BodyRefs.NUNET_SHARD_STATUS,
    { ctx => ctx.systemProcesses.nunetShardStatus }
  ),
  Definition[F](
    "rho:nunet:shard:list",
    FixedChannels.NUNET_SHARD_LIST,
    2,  // filters + ack
    BodyRefs.NUNET_SHARD_LIST,
    { ctx => ctx.systemProcesses.nunetShardList }
  ),
  Definition[F](
    "rho:nunet:shard:stop",
    FixedChannels.NUNET_SHARD_STOP,
    2,  // shardId + ack
    BodyRefs.NUNET_SHARD_STOP,
    { ctx => ctx.systemProcesses.nunetShardStop }
  ),
  Definition[F](
    "rho:nunet:shard:delete",
    FixedChannels.NUNET_SHARD_DELETE,
    2,  // shardId + ack
    BodyRefs.NUNET_SHARD_DELETE,
    { ctx => ctx.systemProcesses.nunetShardDelete }
  ),
  Definition[F](
    "rho:nunet:resources:list",
    FixedChannels.NUNET_RESOURCES_LIST,
    2,  // filters + ack
    BodyRefs.NUNET_RESOURCES_LIST,
    { ctx => ctx.systemProcesses.nunetResourcesList }
  ),
  Definition[F](
    "rho:nunet:shard:logs",
    FixedChannels.NUNET_SHARD_LOGS,
    2,  // config + ack
    BodyRefs.NUNET_SHARD_LOGS,
    { ctx => ctx.systemProcesses.nunetShardLogs }
  ),
  Definition[F](
    "rho:nunet:keys:generate",
    FixedChannels.NUNET_KEYS_GENERATE,
    2,  // config + ack
    BodyRefs.NUNET_KEYS_GENERATE,
    { ctx => ctx.systemProcesses.nunetKeysGenerate }
  ),
  Definition[F](
    "rho:nunet:config:validate",
    FixedChannels.NUNET_CONFIG_VALIDATE,
    2,  // config + ack
    BodyRefs.NUNET_CONFIG_VALIDATE,
    { ctx => ctx.systemProcesses.nunetConfigValidate }
  )
)
```

---

### Step 3.2: Add to Dispatch Table

**Location**: Same file, around line 542

**Change**:
```scala
// Before:
(stdSystemProcesses[F] ++
 stdRhoCryptoProcesses[F] ++
 stdRhoAIProcesses[F] ++
 stdRhoOllamaProcesses[F] ++
 extraSystemProcesses)

// After:
(stdSystemProcesses[F] ++
 stdRhoCryptoProcesses[F] ++
 stdRhoAIProcesses[F] ++
 stdRhoOllamaProcesses[F] ++
 stdRhoNunetProcesses[F] ++  // ADD THIS
 extraSystemProcesses)
```

---

## Phase 4: Configuration

### Step 4.1: Add HOCON Config

**Location**: `node/src/main/resources/defaults.conf`

**Add section** (around line 400+):

```hocon
# Nunet DMS Integration
nunet {
  # Enable/disable Nunet system contracts
  enabled = false

  # Path to nunet-dms CLI executable
  cli-path = "nunet-dms"

  # Command timeout (seconds)
  timeout = 300

  # DMS endpoint configuration (optional)
  dms {
    endpoint = "https://dms.nunet.io"
    api-key = ""
  }

  # Resource limits
  limits {
    max-shards-per-node = 10
    max-deployments-per-minute = 1
  }
}
```

---

### Step 4.2: Update Application Config

**Location**: Your deployment's `application.conf`

**Example**:
```hocon
# Enable Nunet in production
nunet {
  enabled = true
  cli-path = "/usr/local/bin/nunet-dms"
  timeout = 600

  dms {
    endpoint = "https://prod.nunet.io"
    api-key = ${?NUNET_API_KEY}  # From environment
  }

  limits {
    max-shards-per-node = 20
    max-deployments-per-minute = 5
  }
}
```

---

## Phase 5: Testing

### Step 5.1: Unit Tests for Service Layer

**Location**: `rholang/src/test/scala/coop/rchain/rholang/externalservices/NunetServiceSpec.scala`

**Test structure**:

```scala
package coop.rchain.rholang.externalservices

import cats.effect.IO
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NunetServiceSpec extends AnyFlatSpec with Matchers {
  "DisabledNunetService" should "raise error on all methods" in {
    val service = new DisabledNunetService

    assertThrows[UnsupportedOperationException] {
      service.deployShard[IO](Map()).unsafeRunSync()
    }
  }

  "NunetServiceImpl.validateShardConfigOrThrow" should "accept valid config" in {
    val config = Map(
      "validators" -> 3,
      "bonds" -> List(
        Map("pubkey" -> ("04" + "a" * 128), "stake" -> 1000),
        Map("pubkey" -> ("04" + "b" * 128), "stake" -> 1000),
        Map("pubkey" -> ("04" + "c" * 128), "stake" -> 1000)
      ),
      "wallets" -> List(
        Map("address" -> "1111AtahZee...", "balance" -> 50000000)
      )
    )

    noException should be thrownBy {
      NunetServiceImpl.validateShardConfigOrThrow(config)
    }
  }

  it should "reject config with wrong validator count" in {
    val config = Map("validators" -> 2, "bonds" -> List(), "wallets" -> List())

    assertThrows[IllegalArgumentException] {
      NunetServiceImpl.validateShardConfigOrThrow(config)
    }
  }

  // Add more tests for each validation rule
}
```

---

### Step 5.2: Integration Tests for Contracts

**Location**: `rholang/src/test/scala/coop/rchain/rholang/interpreter/NunetContractsSpec.scala`

**Test structure**:

```scala
package coop.rchain.rholang.interpreter

import cats.effect.IO
import coop.rchain.rholang.externalservices.NunetService
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NunetContractsSpec extends AnyFlatSpec with Matchers {
  // Mock Nunet service for testing
  class MockNunetService extends NunetService {
    def deployShard[F[_]](config: Map[String, Any])(implicit F: Concurrent[F]): F[Map[String, Any]] = {
      F.pure(Map(
        "jobId" -> "job-123",
        "shardId" -> "shard-abc",
        "estimatedTime" -> 300
      ))
    }

    // Implement other methods with mock responses
  }

  "rho:nunet:shard:deploy" should "deploy shard with valid config" in {
    // TODO: Set up RhoRuntime with mock service
    // TODO: Deploy Rholang contract
    // TODO: Verify result on ack channel
  }

  // Add more integration tests
}
```

---

### Step 5.3: End-to-End Test

**Location**: `integration-tests/`

**Test scenario**:
1. Enable Nunet service
2. Deploy Rholang contract that calls `rho:nunet:shard:deploy`
3. Wait for deployment (poll `rho:nunet:shard:status`)
4. Verify shard is running
5. Query shard info (`rho:nunet:shard:list`)
6. Stop shard (`rho:nunet:shard:stop`)
7. Delete shard (`rho:nunet:shard:delete`)

**Implementation**: Create shell script or Scala test that orchestrates the entire flow.

---

## Phase 6: Documentation

### Step 6.1: Update CHANGELOG

**Location**: `CHANGELOG.md`

**Add entry**:
```markdown
## [Unreleased]

### Added
- Nunet DMS system contracts for orchestrating 3-validator shards from Rholang
- 9 new system contracts: `rho:nunet:shard:*`, `rho:nunet:resources:*`, `rho:nunet:keys:*`, `rho:nunet:config:*`
- Configuration options for Nunet integration (NUNET_ENABLED, nunet.cli-path, etc.)
- Comprehensive validation for shard deployment configurations

### Changed
- Extended ExternalServices trait with NunetService

### Documentation
- Added SYSTEM_CONTRACT_PATTERN.md
- Added NUNET_CONTRACT_DESIGN.md
- Added NUNET_IMPLEMENTATION_GUIDE.md
- Added Rholang examples for shard deployment and management
```

---

### Step 6.2: Create Rholang Examples

**Location**: `examples/rholang/nunet/`

**Files to create**:
- `deploy-shard.rho` - Basic shard deployment
- `monitor-shard.rho` - Poll shard status until running
- `self-replicating-shard.rho` - Load-based auto-scaling
- `cross-shard-routing.rho` - Message routing between shards
- `README.md` - Usage instructions

**Example** (`deploy-shard.rho`):
```rholang
// Deploy a 3-validator shard to Nunet DMS
new deployShard, ack in {
  @"rho:nunet:shard:deploy"!([{
    "networkId": "testnet",
    "validators": 3,
    "bonds": [
      {"pubkey": "04fa70d7be5eb750...", "stake": 1000},
      {"pubkey": "04837a4cff833e31...", "stake": 1000},
      {"pubkey": "0457febafcc25dd3...", "stake": 1000}
    ],
    "wallets": [
      {"address": "1111AtahZeefej4t...", "balance": 50000000000000000}
    ]
  }], *ack) |
  for (@result <- ack) {
    @"rho:io:stdlog"!(["Deployment started:", result.get("jobId")]) |
    @"rho:io:stdlog"!(["Shard ID:", result.get("shardId")])
  }
}
```

---

### Step 6.3: Write Deployment Guide

**Location**: `docs/NUNET_DEPLOYMENT.md`

**Contents**:
1. Prerequisites (nunet-dms CLI installation)
2. Configuration steps (environment variables, config file)
3. Enabling Nunet contracts
4. Testing deployment with example contracts
5. Troubleshooting common issues
6. Security best practices

---

## Troubleshooting

### Common Issues

#### 1. "Nunet service is not enabled"

**Cause**: NUNET_ENABLED not set or set to false.

**Solution**:
```bash
export NUNET_ENABLED=true
```

Or in `application.conf`:
```hocon
nunet.enabled = true
```

---

#### 2. "Command failed: nunet-dms: command not found"

**Cause**: nunet-dms CLI not in PATH.

**Solution**:
```bash
export NUNET_CLI_PATH=/path/to/nunet-dms
```

Or:
```hocon
nunet.cli-path = "/usr/local/bin/nunet-dms"
```

---

#### 3. "Invalid pubkey format"

**Cause**: Validator public key is not 130 hex characters.

**Solution**: Ensure keys are generated with `rnode keygen --algorithm secp256k1` and are uncompressed (130 chars).

---

#### 4. "Contract call timeout"

**Cause**: DMS deployment takes longer than default timeout.

**Solution**: Increase timeout:
```hocon
nunet.timeout = 600  # 10 minutes
```

---

#### 5. Replay mode not working

**Cause**: Body ref or fixed channel collision with existing contracts.

**Solution**: Check that byte/long allocations don't overlap with existing contracts. Adjust values in FixedChannels and BodyRefs.

---

## Next Steps

After completing all phases:

1. **Code review**: Submit PR for team review
2. **Security audit**: Have security team review input validation, command execution, privilege separation
3. **Performance testing**: Test with multiple concurrent deployments
4. **Production deployment**: Deploy to staging, then production
5. **Monitor**: Set up alerts for errors, timeouts, resource exhaustion
6. **Iterate**: Gather feedback and improve based on real-world usage

---

## Summary

This guide provides step-by-step instructions for implementing the Nunet DMS system contract:

**Phase 1**: Service layer (trait, real/disabled impls, enable/disable)
**Phase 2**: System processes (fixed channels, handlers, type conversions)
**Phase 3**: Registration (process group, dispatch table)
**Phase 4**: Configuration (HOCON config, defaults)
**Phase 5**: Testing (unit, integration, end-to-end)
**Phase 6**: Documentation (CHANGELOG, examples, deployment guide)

Follow each step carefully, test thoroughly, and coordinate with the Nunet team for CLI specification finalization.

**Estimated Timeline**: 2-3 weeks for complete implementation, testing, and documentation.

**Questions?** Refer to `SYSTEM_CONTRACT_PATTERN.md` and `NUNET_CONTRACT_DESIGN.md` for additional context.
