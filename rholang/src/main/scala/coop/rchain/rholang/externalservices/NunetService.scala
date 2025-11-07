package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.syntax.all._
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import scala.sys.process._
import scala.concurrent.{ExecutionContext, Future}
import java.io.{File, PrintWriter}

/**
  * Service for orchestrating Nunet DMS deployments from Rholang contracts.
  *
  * Provides deployment operations for managing 3-validator RNode shards
  * on Nunet infrastructure using the DMS actor behavior model.
  *
  * All methods execute shell commands via the nunet CLI using the pattern:
  * `nunet actor cmd --context user /dms/node/<behavior>/<action> [flags]`
  */
trait NunetService {

  /**
    * Deploy a new ensemble to Nunet DMS.
    *
    * @param ensembleYaml YAML content defining the ensemble
    * @param timeoutMinutes Deployment timeout in minutes
    * @return Ensemble ID for tracking the deployment
    */
  def deployEnsemble[F[_]](ensembleYaml: String, timeoutMinutes: Int)(
      implicit
      F: Concurrent[F]
  ): F[String]

  /**
    * Get current status of a deployed ensemble.
    *
    * @param ensembleId Unique ensemble identifier
    * @return Deployment status information
    */
  def getDeploymentStatus[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentStatus]

  /**
    * List all deployments managed by this DMS instance.
    *
    * @return List of deployment summaries
    */
  def listDeployments[F[_]]()(implicit F: Concurrent[F]): F[List[DeploymentSummary]]

  /**
    * Retrieve logs from all containers in a deployment.
    *
    * @param ensembleId Unique ensemble identifier
    * @return Combined logs from all allocations
    */
  def getDeploymentLogs[F[_]](ensembleId: String)(implicit F: Concurrent[F]): F[String]

  /**
    * Get complete deployment details including IP addresses and ports.
    *
    * @param ensembleId Unique ensemble identifier
    * @return Full deployment manifest with network details
    */
  def getDeploymentManifest[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentManifest]

  /**
    * Generate ensemble YAML for a 3-validator Firefly shard.
    *
    * @param config Shard configuration including validators, bonds, wallets
    * @return YAML content ready for deployment
    */
  def generateFireflyEnsemble[F[_]](config: ShardConfig)(implicit F: Concurrent[F]): F[String]

  /**
    * Validate ensemble YAML before deployment.
    *
    * @param ensembleYaml YAML content to validate
    * @return Validation result with any errors or warnings
    */
  def validateEnsemble[F[_]](ensembleYaml: String)(
      implicit
      F: Concurrent[F]
  ): F[ValidationResult]
}

/**
  * Configuration for a 3-validator Firefly shard.
  */
case class ShardConfig(
    // Number of validators (must be 3)
    validators: Int = 3,
    // Validator bonds (pubkey -> stake)
    bonds: List[Bond],
    // Genesis wallets (address -> balance)
    wallets: List[Wallet],
    // Network ID for the shard
    networkId: String = "testnet",
    // Resource allocations per validator
    cpuCores: Int = 2,
    ramGb: Int = 4,
    diskGb: Int = 20,
    // Validator private keys
    bootstrapKey: String,
    validator1Key: String,
    validator2Key: String,
    // Corresponding public keys for bootstrap string
    bootstrapPubkey: String,
    validator1Pubkey: String,
    validator2Pubkey: String
)

case class Bond(pubkey: String, stake: Int)
case class Wallet(address: String, balance: Int)

/**
  * Deployment status information.
  */
case class DeploymentStatus(
    ensembleId: String,
    status: String, // Pending, Running, Completed, Failed
    allocations: List[AllocationStatus]
)

case class AllocationStatus(
    name: String,
    status: String,
    peerId: Option[String]
)

/**
  * Summary of a deployment.
  */
case class DeploymentSummary(
    ensembleId: String,
    status: String,
    allocationCount: Int
)

/**
  * Complete deployment manifest with network details.
  */
case class DeploymentManifest(
    ensembleId: String,
    peers: List[String],
    ipAddresses: Map[String, String],
    portMappings: Map[String, List[PortMapping]]
)

case class PortMapping(
    publicPort: Int,
    privatePort: Int,
    allocation: String
)

/**
  * Validation result.
  */
case class ValidationResult(
    valid: Boolean,
    errors: List[String],
    warnings: List[String]
)

/**
  * Disabled implementation - raises errors when service is not enabled.
  */
class DisabledNunetService extends NunetService {

  def deployEnsemble[F[_]](ensembleYaml: String, timeoutMinutes: Int)(
      implicit
      F: Concurrent[F]
  ): F[String] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )

  def getDeploymentStatus[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentStatus] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )

  def listDeployments[F[_]]()(implicit F: Concurrent[F]): F[List[DeploymentSummary]] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )

  def getDeploymentLogs[F[_]](ensembleId: String)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )

  def getDeploymentManifest[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentManifest] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )

  def generateFireflyEnsemble[F[_]](config: ShardConfig)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )

  def validateEnsemble[F[_]](ensembleYaml: String)(
      implicit
      F: Concurrent[F]
  ): F[ValidationResult] =
    F.raiseError(
      new UnsupportedOperationException(
        "Nunet service is not enabled. Set NUNET_ENABLED=true or configure nunet.enabled=true in application.conf"
      )
    )
}

/**
  * Real implementation - executes nunet CLI commands using actor behavior model.
  */
class NunetServiceImpl(config: NunetConf)(implicit ec: ExecutionContext) extends NunetService {
  import NunetServiceImpl._

  private[this] val logger: Logger = Logger[this.type]

  // Use configuration provided via constructor
  private val cliPath    = config.cliPath
  private val context    = config.context
  private val timeout    = config.timeout
  private val passphrase = config.passphrase

  def deployEnsemble[F[_]](ensembleYaml: String, timeoutMinutes: Int)(
      implicit
      F: Concurrent[F]
  ): F[String] =
    F.async { cb =>
      Future {
        logger.error(s"deployEnsemble called with timeout=$timeoutMinutes minutes")
        require(ensembleYaml.nonEmpty, "Ensemble YAML cannot be empty")
        require(timeoutMinutes > 0, "Timeout must be positive")

        // Write YAML to temp file
        val tempFile = File.createTempFile("ensemble-", ".yml")
        val writer   = new PrintWriter(tempFile)

        try {
          writer.write(ensembleYaml)
          writer.close()
          logger.error(s"Wrote ensemble YAML to temp file: ${tempFile.getAbsolutePath}")

          // Execute deployment command
          val cmd = Seq(
            cliPath,
            "actor",
            "cmd",
            "--context",
            context,
            "/dms/node/deployment/new",
            "-f",
            tempFile.getAbsolutePath,
            "-t",
            s"${timeoutMinutes}m"
          )

          logger.error(s"Executing command: ${cmd.mkString(" ")}")
          val output = executeCommand(cmd, timeout, passphrase)
          logger.error(s"Command output: $output")
          val ensembleId = parseEnsembleId(output)
          logger.error(s"Parsed ensemble ID: $ensembleId")

          cb(Right(ensembleId))
        } catch {
          case e: Exception =>
            logger.error(s"ERROR in deployEnsemble: ${e.getMessage}", e)
            throw e
        } finally {
          tempFile.delete()
        }
      }.recover {
        case e: Exception =>
          logger.error(s"Future.recover caught exception: ${e.getMessage}", e)
          cb(Left(e))
      }
    }

  def getDeploymentStatus[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentStatus] =
    F.async { cb =>
      Future {
        validateEnsembleId(ensembleId)

        val cmd = Seq(
          cliPath,
          "actor",
          "cmd",
          "--context",
          context,
          "/dms/node/deployment/status",
          "-i",
          ensembleId
        )

        val output = executeCommand(cmd, timeout, passphrase)
        val status = parseDeploymentStatus(output)

        cb(Right(status))
      }.recover {
        case e: Exception =>
          cb(Left(e))
      }
    }

  def listDeployments[F[_]]()(implicit F: Concurrent[F]): F[List[DeploymentSummary]] =
    F.async { cb =>
      Future {
        val cmd = Seq(
          cliPath,
          "actor",
          "cmd",
          "--context",
          context,
          "/dms/node/deployment/list"
        )

        val output      = executeCommand(cmd, timeout, passphrase)
        val deployments = parseDeploymentList(output)

        cb(Right(deployments))
      }.recover {
        case e: Exception =>
          cb(Left(e))
      }
    }

  def getDeploymentLogs[F[_]](ensembleId: String)(implicit F: Concurrent[F]): F[String] =
    F.async { cb =>
      Future {
        validateEnsembleId(ensembleId)

        val cmd = Seq(
          cliPath,
          "actor",
          "cmd",
          "--context",
          context,
          "/dms/node/deployment/logs",
          "-i",
          ensembleId
        )

        val output = executeCommand(cmd, timeout, passphrase)

        cb(Right(output))
      }.recover {
        case e: Exception =>
          cb(Left(e))
      }
    }

  def getDeploymentManifest[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentManifest] =
    F.async { cb =>
      Future {
        validateEnsembleId(ensembleId)

        val cmd = Seq(
          cliPath,
          "actor",
          "cmd",
          "--context",
          context,
          "/dms/node/deployment/manifest",
          "-i",
          ensembleId
        )

        val output   = executeCommand(cmd, timeout, passphrase)
        val manifest = parseDeploymentManifest(output)

        cb(Right(manifest))
      }.recover {
        case e: Exception =>
          cb(Left(e))
      }
    }

  def generateFireflyEnsemble[F[_]](config: ShardConfig)(implicit F: Concurrent[F]): F[String] =
    F.async { cb =>
      Future {
        validateShardConfig(config)

        // Get peer ID for pinned deployment
        val peerID = getPeerID()

        // Generate YAML using EnsembleGenerator
        val yaml = EnsembleGenerator.generateFireflyEnsemble(config, peerID)

        cb(Right(yaml))
      }.recover {
        case e: Exception =>
          cb(Left(e))
      }
    }

  def validateEnsemble[F[_]](ensembleYaml: String)(
      implicit
      F: Concurrent[F]
  ): F[ValidationResult] =
    F.async { cb =>
      Future {
        val errors   = scala.collection.mutable.ListBuffer[String]()
        val warnings = scala.collection.mutable.ListBuffer[String]()

        // Basic validation
        if (ensembleYaml.isEmpty) {
          errors += "Ensemble YAML is empty"
        }

        if (!ensembleYaml.contains("version:")) {
          errors += "Missing 'version' field"
        }

        if (!ensembleYaml.contains("allocations:")) {
          errors += "Missing 'allocations' section"
        }

        if (!ensembleYaml.contains("nodes:")) {
          errors += "Missing 'nodes' section"
        }

        if (!ensembleYaml.contains("version: \"V1\"")) {
          warnings += "Version should be \"V1\" (quoted)"
        }

        val result = ValidationResult(
          valid = errors.isEmpty,
          errors = errors.toList,
          warnings = warnings.toList
        )

        cb(Right(result))
      }.recover {
        case e: Exception =>
          cb(Left(e))
      }
    }

  /**
    * Get self peer ID for pinned deployments (private helper).
    */
  private def getPeerID(): String = {
    val cmd = Seq(
      cliPath,
      "actor",
      "cmd",
      "--context",
      context,
      "/dms/node/peers/self"
    )

    executeCommand(cmd, timeout, passphrase).trim
  }
}

/**
  * Companion object with helper methods.
  */
object NunetServiceImpl {

  private[this] val logger: Logger = Logger[this.type]

  /**
    * Singleton instance with enable/disable logic.
    *
    * @deprecated Use new NunetServiceImpl(nunetConf) with configuration injection instead
    */
  @deprecated("Use new NunetServiceImpl(nunetConf) instead", "1.0.0")
  lazy val instance: NunetService = {
    import scala.concurrent.ExecutionContext.Implicits.global
    if (isNunetEnabled) {
      // Fallback to loading config directly (old behavior)
      val config = ConfigFactory.load()
      val cliPath =
        if (config.hasPath("nunet.cli-path")) config.getString("nunet.cli-path") else "nunet"
      val context =
        if (config.hasPath("nunet.context")) config.getString("nunet.context") else "user"
      val timeout = if (config.hasPath("nunet.timeout")) config.getInt("nunet.timeout") else 600
      val passphrase =
        if (config.hasPath("nunet.passphrase")) config.getString("nunet.passphrase") else ""

      val nunetConf = NunetConf(
        enabled = true,
        cliPath = cliPath,
        context = context,
        timeout = timeout,
        passphrase = passphrase
      )

      new NunetServiceImpl(nunetConf)
    } else new DisabledNunetService
  }

  /**
    * Execute shell command with timeout and environment variables.
    *
    * @param cmd Command and arguments to execute
    * @param timeoutSeconds Timeout in seconds
    * @param passphrase DMS passphrase to set in DMS_PASSPHRASE environment variable
    */
  private def executeCommand(cmd: Seq[String], timeoutSeconds: Int, passphrase: String): String = {
    logger.error(s"executeCommand - Starting command execution: ${cmd.mkString(" ")}")
    logger.error(s"executeCommand - Timeout: $timeoutSeconds seconds")

    val stdout = new StringBuilder
    val stderr = new StringBuilder

    val processLogger = ProcessLogger(
      line => {
        logger.error(s"executeCommand - STDOUT: $line")
        stdout.append(line + "\n")
      },
      line => {
        logger.error(s"executeCommand - STDERR: $line")
        stderr.append(line + "\n")
      }
    )

    try {
      // Create process with DMS_PASSPHRASE environment variable
      val processBuilder = Process(cmd, None, "DMS_PASSPHRASE" -> passphrase)
      val exitCode       = processBuilder.!(processLogger)

      logger.error(s"executeCommand - Process exit code: $exitCode")

      if (exitCode == 0) {
        val result = stdout.toString.trim
        logger.error(s"executeCommand - SUCCESS - Returning stdout: $result")
        result
      } else {
        val errorMsg = s"Command failed with exit code $exitCode: ${stderr.toString}"
        logger.error(s"executeCommand - ERROR - $errorMsg")
        throw new RuntimeException(errorMsg)
      }
    } catch {
      case e: Exception =>
        logger.error(s"executeCommand - EXCEPTION during command execution: ${e.getMessage}", e)
        throw e
    }
  }

  /**
    * Parse ensemble ID from deployment response.
    */
  private def parseEnsembleId(json: String): String = {
    val pattern = """"ensemble_id"\s*:\s*"([^"]+)"""".r
    pattern
      .findFirstMatchIn(json)
      .map(_.group(1))
      .getOrElse(
        throw new RuntimeException(s"Failed to parse ensemble_id from: $json")
      )
  }

  /**
    * Parse deployment status from JSON.
    *
    * Expected format (simplified, actual format may vary):
    * {
    *   "ensemble_id": "...",
    *   "status": "Running",
    *   "allocations": [
    *     { "name": "bootstrap", "status": "Running", "peer_id": "..." },
    *     ...
    *   ]
    * }
    */
  private def parseDeploymentStatus(json: String): DeploymentStatus = {
    val ensembleIdPattern = """"ensemble_id"\s*:\s*"([^"]+)"""".r
    val statusPattern     = """"status"\s*:\s*"([^"]+)"""".r
    val allocationPattern =
      """"name"\s*:\s*"([^"]+)"[^}]*"status"\s*:\s*"([^"]+)"[^}]*(?:"peer_id"\s*:\s*"([^"]+)")?""".r

    val ensembleId = ensembleIdPattern
      .findFirstMatchIn(json)
      .map(_.group(1))
      .getOrElse("unknown")

    val status = statusPattern
      .findFirstMatchIn(json)
      .map(_.group(1))
      .getOrElse("unknown")

    val allocations = allocationPattern
      .findAllMatchIn(json)
      .map { m =>
        val name        = m.group(1)
        val allocStatus = m.group(2)
        val peerId      = Option(m.group(3))
        AllocationStatus(name, allocStatus, peerId)
      }
      .toList

    DeploymentStatus(ensembleId, status, allocations)
  }

  /**
    * Parse deployment list from JSON.
    *
    * Expected format:
    * {
    *   "Deployments": {
    *     "ensemble_id_1": "Status1",
    *     "ensemble_id_2": "Status2",
    *     ...
    *   }
    * }
    */
  private def parseDeploymentList(json: String): List[DeploymentSummary] = {
    // Pattern to match ensemble ID and status pairs within the Deployments object
    val deploymentPattern = """"([a-f0-9]{64})"\s*:\s*"([^"]+)"""".r

    deploymentPattern
      .findAllMatchIn(json)
      .map { m =>
        val ensembleId = m.group(1)
        val status     = m.group(2)
        // We don't have allocation count in the list response, so default to 0
        DeploymentSummary(ensembleId, status, allocationCount = 0)
      }
      .toList
  }

  /**
    * Parse deployment manifest from JSON.
    *
    * Expected format (simplified):
    * {
    *   "ensemble_id": "...",
    *   "peers": ["peer_id_1", "peer_id_2", ...],
    *   "ip_addresses": { "allocation_name": "ip_address", ... },
    *   "port_mappings": {
    *     "allocation_name": [
    *       { "public_port": 40400, "private_port": 40400, "allocation": "bootstrap" },
    *       ...
    *     ]
    *   }
    * }
    */
  private def parseDeploymentManifest(json: String): DeploymentManifest = {
    val ensembleIdPattern = """"ensemble_id"\s*:\s*"([^"]+)"""".r
    val peerPattern       = """"peers"\s*:\s*\[((?:"[^"]+",?\s*)*)\]""".r
    val peerItemPattern   = """"([^"]+)"""".r
    val ipPattern         = """"([^"]+)"\s*:\s*"([^"]+)"""".r
    val portMappingPattern =
      """"public_port"\s*:\s*(\d+)[^}]*"private_port"\s*:\s*(\d+)[^}]*"allocation"\s*:\s*"([^"]+)"""".r

    val ensembleId = ensembleIdPattern
      .findFirstMatchIn(json)
      .map(_.group(1))
      .getOrElse("unknown")

    // Parse peers array
    val peers = peerPattern
      .findFirstMatchIn(json)
      .map { m =>
        peerItemPattern
          .findAllMatchIn(m.group(1))
          .map(_.group(1))
          .toList
      }
      .getOrElse(List.empty)

    // Parse IP addresses - look for patterns in ip_addresses section
    val ipAddresses = {
      val ipSectionPattern = """"ip_addresses"\s*:\s*\{([^}]+)\}""".r
      ipSectionPattern
        .findFirstMatchIn(json)
        .map { m =>
          ipPattern
            .findAllMatchIn(m.group(1))
            .map { ipMatch =>
              (ipMatch.group(1), ipMatch.group(2))
            }
            .toMap
        }
        .getOrElse(Map.empty)
    }

    // Parse port mappings - simplified, assumes flat structure
    val portMappings = portMappingPattern
      .findAllMatchIn(json)
      .map { m =>
        val publicPort  = m.group(1).toInt
        val privatePort = m.group(2).toInt
        val allocation  = m.group(3)
        (allocation, PortMapping(publicPort, privatePort, allocation))
      }
      .toList
      .groupBy(_._1)
      .map { case (alloc, ports) => (alloc, ports.map(_._2).toList) }

    DeploymentManifest(ensembleId, peers, ipAddresses, portMappings)
  }

  /**
    * Validate ensemble ID format.
    */
  private def validateEnsembleId(ensembleId: String): Unit = {
    require(ensembleId.nonEmpty, "Ensemble ID cannot be empty")
    require(
      ensembleId.matches("^[a-zA-Z0-9]+$"),
      s"Invalid ensemble ID format: $ensembleId"
    )
  }

  /**
    * Validate shard configuration.
    */
  private def validateShardConfig(config: ShardConfig): Unit = {
    require(config.validators == 3, s"Must have exactly 3 validators, got ${config.validators}")
    require(config.bonds.length == 3, s"Must have exactly 3 bonds, got ${config.bonds.length}")
    require(config.wallets.nonEmpty, "Must have at least one wallet")

    require(config.cpuCores > 0, s"CPU cores must be positive: ${config.cpuCores}")
    require(config.ramGb > 0, s"RAM must be positive: ${config.ramGb}")
    require(config.diskGb > 0, s"Disk must be positive: ${config.diskGb}")

    require(config.bootstrapKey.nonEmpty, "Bootstrap key cannot be empty")
    require(config.validator1Key.nonEmpty, "Validator1 key cannot be empty")
    require(config.validator2Key.nonEmpty, "Validator2 key cannot be empty")

    require(config.bootstrapPubkey.nonEmpty, "Bootstrap pubkey cannot be empty")
    require(config.validator1Pubkey.nonEmpty, "Validator1 pubkey cannot be empty")
    require(config.validator2Pubkey.nonEmpty, "Validator2 pubkey cannot be empty")

    // Validate bonds
    config.bonds.foreach { bond =>
      require(
        bond.pubkey.matches("^[0-9a-fA-F]{130}$"),
        s"Invalid bond pubkey format: ${bond.pubkey}"
      )
      require(bond.stake > 0, s"Bond stake must be positive: ${bond.stake}")
    }

    // Validate wallets
    config.wallets.foreach { wallet =>
      require(
        wallet.address.startsWith("1111"),
        s"Invalid REV address: ${wallet.address}"
      )
      require(wallet.balance > 0, s"Wallet balance must be positive: ${wallet.balance}")
    }
  }
}

/**
  * Helper object for generating ensemble YAML files.
  */
object EnsembleGenerator {

  /**
    * Generate ensemble YAML for a 3-validator Firefly shard.
    */
  def generateFireflyEnsemble(config: ShardConfig, peerID: String): String = {
    val javaOpts = s"-Xmx${config.ramGb - 1}g"

    s"""version: "V1"

allocations:
  bootstrap:
    type: service
    executor: docker
    resources:
      cpu:
        cores: ${config.cpuCores}
      ram:
        size: ${config.ramGb}
      disk:
        size: ${config.diskGb}
      gpus: []
    execution:
      type: docker
      image: f1r3flyindustries/f1r3fly-scala-node
      working_directory: /var/lib/rnode
      environment:
        - JAVA_OPTS=$javaOpts
      cmd:
        - "-Dlogback.configurationFile=/var/lib/rnode/logback.xml"
        - "run"
        - "--host=bootstrap.internal"
        - "--bootstrap=rnode://${config.bootstrapPubkey}@bootstrap.internal?protocol=40400&discovery=40404"
        - "--allow-private-addresses"
        - "--validator-private-key=${config.bootstrapKey}"
    dnsname: bootstrap
    failure_recovery: restart

  validator1:
    type: service
    executor: docker
    resources:
      cpu:
        cores: ${config.cpuCores}
      ram:
        size: ${config.ramGb}
      disk:
        size: ${config.diskGb}
      gpus: []
    execution:
      type: docker
      image: f1r3flyindustries/f1r3fly-scala-node
      working_directory: /var/lib/rnode
      environment:
        - JAVA_OPTS=$javaOpts
      cmd:
        - "-Dlogback.configurationFile=/var/lib/rnode/logback.xml"
        - "run"
        - "--host=validator1.internal"
        - "--allow-private-addresses"
        - "--bootstrap=rnode://${config.bootstrapPubkey}@bootstrap.internal?protocol=40400&discovery=40404"
        - "--validator-public-key=${config.validator1Pubkey}"
        - "--validator-private-key=${config.validator1Key}"
        - "--genesis-validator"
    dnsname: validator1
    failure_recovery: restart
    depends_on: ["bootstrap"]

  validator2:
    type: service
    executor: docker
    resources:
      cpu:
        cores: ${config.cpuCores}
      ram:
        size: ${config.ramGb}
      disk:
        size: ${config.diskGb}
      gpus: []
    execution:
      type: docker
      image: f1r3flyindustries/f1r3fly-scala-node
      working_directory: /var/lib/rnode
      environment:
        - JAVA_OPTS=$javaOpts
      cmd:
        - "-Dlogback.configurationFile=/var/lib/rnode/logback.xml"
        - "run"
        - "--host=validator2.internal"
        - "--allow-private-addresses"
        - "--bootstrap=rnode://${config.bootstrapPubkey}@bootstrap.internal?protocol=40400&discovery=40404"
        - "--validator-public-key=${config.validator2Pubkey}"
        - "--validator-private-key=${config.validator2Key}"
        - "--genesis-validator"
    dnsname: validator2
    failure_recovery: restart
    depends_on: ["bootstrap"]

nodes:
  node-bootstrap:
    allocations:
      - bootstrap
    failure_recovery: restart
    ports:
      - public: 40400
        private: 40400
        allocation: bootstrap
      - public: 40401
        private: 40401
        allocation: bootstrap
      - public: 40402
        private: 40402
        allocation: bootstrap
      - public: 40403
        private: 40403
        allocation: bootstrap
      - public: 40404
        private: 40404
        allocation: bootstrap
      - public: 40405
        private: 40405
        allocation: bootstrap
    peer: "$peerID"

  node-validator1:
    allocations:
      - validator1
    failure_recovery: restart
    ports:
      - public: 40410
        private: 40400
        allocation: validator1
      - public: 40411
        private: 40401
        allocation: validator1
      - public: 40412
        private: 40402
        allocation: validator1
      - public: 40413
        private: 40403
        allocation: validator1
      - public: 40414
        private: 40404
        allocation: validator1
      - public: 40415
        private: 40405
        allocation: validator1
    peer: "$peerID"

  node-validator2:
    allocations:
      - validator2
    failure_recovery: restart
    ports:
      - public: 40420
        private: 40400
        allocation: validator2
      - public: 40421
        private: 40401
        allocation: validator2
      - public: 40422
        private: 40402
        allocation: validator2
      - public: 40423
        private: 40403
        allocation: validator2
      - public: 40424
        private: 40404
        allocation: validator2
      - public: 40425
        private: 40405
        allocation: validator2
    peer: "$peerID"
"""
  }
}
