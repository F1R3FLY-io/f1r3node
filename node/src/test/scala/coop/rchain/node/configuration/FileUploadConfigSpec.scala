package coop.rchain.node.configuration

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
import coop.rchain.node.configuration.commandline.{ConfigMapper, Options}
import org.scalatest.{FunSuite, Matchers}
import pureconfig._
import pureconfig.generic.auto._

import java.nio.file.{Files, Paths}
import scala.concurrent.duration._

/**
  * Tests for file-upload configuration: default values parsed from defaults.conf,
  * CLI overrides for all four flags, and directory auto-creation on startup.
  */
class FileUploadConfigSpec extends FunSuite with Matchers {

  // Custom readers shared by all test helpers
  private def commErrToThrow(commErr: CommError) =
    new Exception(CommError.errorMessage(commErr))

  implicit private val peerNodeReader: ConfigReader[PeerNode] =
    ConfigReader.fromStringTry[PeerNode](
      PeerNode.fromAddress(_).left.map(commErrToThrow).toTry
    )

  implicit private val myLongReader: ConfigReader[Long] =
    ConfigReader.fromString[Long](
      ConvertHelpers.catchReadError(s => ConfigFactory.parseString(s"v = $s").getBytes("v"))
    )

  /** Load NodeConf from defaults, optionally merged with a CLI args config. */
  private def loadConf(cliArgs: Seq[String] = Seq("run")): NodeConf = {
    val options = Options(cliArgs)
    val defaultConfig = ConfigSource
      .resources("defaults.conf")
      .withFallback(ConfigSource.string("default-data-dir = /var/lib/rnode"))
    val optionsConfig = ConfigSource.fromConfig(ConfigMapper.fromOptions(options))

    optionsConfig
      .withFallback(defaultConfig)
      .load[NodeConf]
      .fold(
        errs => fail(s"Config load failed: ${errs.toList.mkString("\n")}"),
        identity
      )
  }

  // ---------------------------------------------------------------------------
  // Default values
  // ---------------------------------------------------------------------------

  test("Parse defaults.conf: file-upload defaults are present") {
    val conf = loadConf()
    conf.fileUpload.chunkSize shouldEqual 4194304L // 4 MB
    conf.fileUpload.replicationDir shouldEqual "file-replication"
    conf.fileUpload.phloPerStorageByte shouldEqual 1L
    conf.fileUpload.baseRegisterPhlo shouldEqual 300L
    conf.fileUpload.maxConcurrentDownloadsPerIp shouldEqual 4
  }

  // ---------------------------------------------------------------------------
  // CLI overrides
  // ---------------------------------------------------------------------------

  test("CLI --file-replication-dir overrides default") {
    val conf = loadConf(Seq("run", "--file-replication-dir", "/tmp/custom-replication"))
    conf.fileUpload.replicationDir shouldEqual "/tmp/custom-replication"
  }

  test("CLI --file-upload-phlo-per-storage-byte overrides default") {
    val conf = loadConf(Seq("run", "--file-upload-phlo-per-storage-byte", "2"))
    conf.fileUpload.phloPerStorageByte shouldEqual 2L
  }

  test("CLI --file-upload-chunk-size overrides default") {
    val conf = loadConf(Seq("run", "--file-upload-chunk-size", "8388608"))
    conf.fileUpload.chunkSize shouldEqual 8388608L // 8 MB
  }

  test("CLI --max-concurrent-downloads-per-ip overrides default") {
    val conf = loadConf(Seq("run", "--max-concurrent-downloads-per-ip", "8"))
    conf.fileUpload.maxConcurrentDownloadsPerIp shouldEqual 8
  }

  // ---------------------------------------------------------------------------
  // Directory creation
  // ---------------------------------------------------------------------------

  test("replicationDir path resolves correctly under dataDir") {
    val conf     = loadConf()
    val dataDir  = conf.storage.dataDir
    val resolved = dataDir.resolve(conf.fileUpload.replicationDir)
    // Verify the path is a child of dataDir
    resolved.startsWith(dataDir) shouldBe true
    resolved.getFileName.toString shouldEqual "file-replication"
  }

  test("replicationDir is created when it doesn't exist") {
    val tmpDir            = Files.createTempDirectory("file-upload-config-spec")
    val replicationSubDir = tmpDir.resolve("file-replication")
    replicationSubDir.toFile.exists() shouldBe false
    Files.createDirectories(replicationSubDir)
    replicationSubDir.toFile.exists() shouldBe true
    // Cleanup
    replicationSubDir.toFile.delete()
    tmpDir.toFile.delete()
  }

  test("createDirectories is idempotent for existing replicationDir") {
    val tmpDir            = Files.createTempDirectory("file-upload-config-spec-existing")
    val replicationSubDir = tmpDir.resolve("file-replication")
    Files.createDirectories(replicationSubDir)
    // Calling again must not throw
    noException should be thrownBy Files.createDirectories(replicationSubDir)
    // Cleanup
    replicationSubDir.toFile.delete()
    tmpDir.toFile.delete()
  }
}
