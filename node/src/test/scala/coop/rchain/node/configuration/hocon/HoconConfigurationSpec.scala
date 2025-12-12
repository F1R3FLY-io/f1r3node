package coop.rchain.node.configuration.hocon

import com.typesafe.config.ConfigFactory
import coop.rchain.casper.util.GenesisBuilder
import coop.rchain.casper.{
  CasperConf,
  GenesisBlockData,
  GenesisCeremonyConf,
  HeartbeatConf,
  RoundRobinDispatcher
}
import coop.rchain.comm.transport.TlsConf
import coop.rchain.comm.{CommError, PeerNode}
import coop.rchain.node.configuration._
import coop.rchain.node.configuration.{
  ApiServer,
  DevConf,
  Metrics,
  NodeConf,
  PeersDiscovery,
  ProtocolClient,
  ProtocolServer,
  Storage
}
import org.scalatest.{FunSuite, Matchers}
import pureconfig.{ConfigReader, ConfigSource, ConvertHelpers}
import pureconfig.generic.auto._

import java.nio.file.Paths
import scala.concurrent.duration._

class HoconConfigurationSpec extends FunSuite with Matchers {

  test("Parse default config") {
    val defaultConfig = ConfigSource
      .resources("defaults.conf")
      .withFallback(
        ConfigSource.string(
          s"default-data-dir = /var/lib/rnode"
        )
      )

    // Custom reader for PeerNode type is required
    def commErrToThrow(commErr: CommError) =
      new Exception(CommError.errorMessage(commErr))

    implicit val peerNodeReader = ConfigReader.fromStringTry[PeerNode](
      PeerNode.fromAddress(_).left.map(commErrToThrow).toTry
    )

    // Make Int values support size-in-bytes format, e.g. 16MB
    implicit val myIntReader = ConfigReader.fromString[Long](
      ConvertHelpers.catchReadError(s => ConfigFactory.parseString(s"v = $s").getBytes("v"))
    )
    val config = defaultConfig.load[NodeConf].right.get

    val expectedConfig = NodeConf(
      defaultDataDir = "/var/lib/rnode",
      standalone = false,
      autopropose = false,
      devMode = false,
      protocolServer = ProtocolServer(
        networkId = "testnet",
        host = None,
        allowPrivateAddresses = false,
        useRandomPorts = false,
        dynamicIp = false,
        noUpnp = false,
        port = 40400,
        grpcMaxRecvMessageSize = 262144,
        grpcMaxRecvStreamMessageSize = 268435456,
        maxMessageConsumers = 400,
        disableStateExporter = false
      ),
      openai = Some(
        OpenAIConf(apiKey = "", enabled = false, validateApiKey = true, validationTimeoutSec = 15)
      ), // defaults from config
      ollama = Some(
        OllamaConf(
          enabled = false,
          baseUrl = "http://localhost:11434",
          defaultModel = "llama4:latest",
          validateConnection = true,
          timeoutSec = 30
        )
      ), // defaults from config
      protocolClient = ProtocolClient(
        networkId = "testnet",
        bootstrap = PeerNode
          .fromAddress(
            "rnode://0000000000000000000000000000000000000000@127.0.0.1?protocol=40400&discovery=40404"
          )
          .right
          .get,
        disableLfs = false,
        batchMaxConnections = 20,
        networkTimeout = 5.seconds,
        grpcMaxRecvMessageSize = 262144,
        grpcStreamChunkSize = 262144
      ),
      peersDiscovery = PeersDiscovery(
        port = 40404,
        lookupInterval = 20.seconds,
        cleanupInterval = 20.minutes,
        heartbeatBatchSize = 100,
        initWaitLoopInterval = 1.second
      ),
      apiServer = ApiServer(
        host = "0.0.0.0",
        portGrpcExternal = 40401,
        portGrpcInternal = 40402,
        grpcMaxRecvMessageSize = 16777216,
        portHttp = 40403,
        portAdminHttp = 40405,
        maxBlocksLimit = 50,
        enableReporting = false,
        keepAliveTime = 2.hours,
        keepAliveTimeout = 20.seconds,
        permitKeepAliveTime = 5.minutes,
        maxConnectionAge = 1.hours,
        maxConnectionIdle = 1.hours,
        maxConnectionAgeGrace = 1.hours
      ),
      storage = Storage(
        dataDir = Paths.get("/var/lib/rnode")
      ),
      tls = TlsConf(
        certificatePath = Paths.get("/var/lib/rnode/node.certificate.pem"),
        keyPath = Paths.get("/var/lib/rnode/node.key.pem"),
        secureRandomNonBlocking = false,
        customCertificateLocation = false,
        customKeyLocation = false
      ),
      casper = CasperConf(
        faultToleranceThreshold = 0,
        validatorPublicKey = None,
        validatorPrivateKey = None,
        validatorPrivateKeyPath = None,
        shardName = "root",
        parentShardId = "/",
        casperLoopInterval = 30.seconds,
        requestedBlocksTimeout = 240.seconds,
        finalizationRate = 1,
        maxNumberOfParents = 2147483647,
        maxParentDepth = Some(2147483647),
        forkChoiceStaleThreshold = 10.minutes,
        forkChoiceCheckIfStaleInterval = 11.minutes,
        synchronyConstraintThreshold = 0.67,
        heightConstraintThreshold = 1000,
        roundRobinDispatcher = RoundRobinDispatcher(
          maxPeerQueueSize = 100,
          giveUpAfterSkipped = 0,
          dropPeerAfterRetries = 0
        ),
        genesisBlockData = GenesisBlockData(
          genesisDataDir = Paths.get("/var/lib/rnode/genesis"),
          bondsFile = "/var/lib/rnode/genesis/bonds.txt",
          walletsFile = "/var/lib/rnode/genesis/wallets.txt",
          bondMaximum = 9223372036854775807L,
          bondMinimum = 1,
          epochLength = 10000,
          quarantineLength = 50000,
          numberOfActiveValidators = 100,
          deployTimestamp = None,
          genesisBlockNumber = 0,
          posMultiSigPublicKeys = GenesisBuilder.defaultPosMultiSigPublicKeys,
          posMultiSigQuorum = GenesisBuilder.defaultPosMultiSigPublicKeys.length - 1
        ),
        genesisCeremony = GenesisCeremonyConf(
          requiredSignatures = 0,
          approveDuration = 5.minutes,
          approveInterval = 5.minute,
          autogenShardSize = 5,
          genesisValidatorMode = false,
          ceremonyMasterMode = false
        ),
        minPhloPrice = 1,
        enableMergeableChannelGC = false,
        mergeableChannelsGCInterval = 5.minutes,
        mergeableChannelsGCDepthBuffer = 10,
        heartbeat = HeartbeatConf(
          enabled = false,
          checkInterval = 30.seconds,
          maxLfbAge = 60.seconds
        ),
        disableLateBlockFiltering = false
      ),
      metrics = Metrics(
        prometheus = false,
        influxdb = false,
        influxdbUdp = false,
        zipkin = false,
        sigar = false
      ),
      dev = DevConf(deployerPrivateKey = None),
      rspacePlusPlus = false
    )
    config shouldEqual expectedConfig
  }
}
