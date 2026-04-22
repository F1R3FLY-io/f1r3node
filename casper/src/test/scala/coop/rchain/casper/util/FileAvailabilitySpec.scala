package coop.rchain.casper.util

import cats.syntax.all._
import coop.rchain.casper.protocol.{BlockMessage, DeployData, ProcessedDeploy}
import coop.rchain.casper.{BlockStatus, InvalidBlock}
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import coop.rchain.casper.util.GenesisBuilder.defaultValidatorSks
import coop.rchain.models.PCost
import coop.rchain.models.blockImplicits.getRandomBlock
import coop.rchain.p2p.EffectsTestInstances.LogStub
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import java.nio.file.{Files, Path}

/**
  * Unit tests for FileAvailability.
  *
  * Tests block deploy scanning for file-registration terms and
  * local file existence checks.
  */
class FileAvailabilitySpec extends FlatSpec with Matchers {

  implicit private val logStub: Log[Task] = new LogStub[Task]()

  private val validatorSk = defaultValidatorSks.head
  private val testHash    = "a" * 64 // 64-char hex hash
  private val testHash2   = "b" * 64

  /** Helper: file-registration deploy term. */
  private def fileRegTerm(hash: String): String =
    s"""new ret, file(`rho:io:file`) in { file!("register", "$hash", 1048576, "test.bin", *ret) }"""

  /** Helper: create a signed deploy with the given term. */
  private def mkSignedDeploy(term: String): Signed[DeployData] = {
    val dd = DeployData(
      term = term,
      timestamp = System.currentTimeMillis(),
      phloPrice = 1,
      phloLimit = 1000,
      validAfterBlockNumber = 0L,
      shardId = "test-shard"
    )
    Signed(dd, Secp256k1, validatorSk)
  }

  /** Helper: wrap a signed deploy into a ProcessedDeploy. */
  private def mkProcessedDeploy(term: String): ProcessedDeploy =
    ProcessedDeploy(
      deploy = mkSignedDeploy(term),
      cost = PCost(0L),
      deployLog = List.empty,
      isFailed = false
    )

  /** Helper: create a block with specified deploy terms. */
  private def mkBlock(terms: String*): BlockMessage =
    getRandomBlock(
      setDeploys = Some(terms.map(mkProcessedDeploy))
    )

  // ---- extractFileHashes ----

  "extractFileHashes" should "return file hashes from file-registration deploys" in {
    val block = mkBlock(fileRegTerm(testHash), fileRegTerm(testHash2))
    FileAvailability.extractFileHashes(block) should contain theSameElementsAs List(
      testHash,
      testHash2
    )
  }

  it should "return empty list for block with no file deploys" in {
    val block = mkBlock("new x in { x!(42) }", "Nil")
    FileAvailability.extractFileHashes(block) shouldBe List.empty
  }

  it should "return empty list for block with no deploys" in {
    val block = mkBlock()
    FileAvailability.extractFileHashes(block) shouldBe List.empty
  }

  it should "deduplicate file hashes" in {
    val block = mkBlock(fileRegTerm(testHash), fileRegTerm(testHash))
    FileAvailability.extractFileHashes(block) shouldBe List(testHash)
  }

  it should "skip non-file-registration deploys while extracting file ones" in {
    val block = mkBlock("new x in { x!(42) }", fileRegTerm(testHash))
    FileAvailability.extractFileHashes(block) shouldBe List(testHash)
  }

  // ---- checkFileAvailability ----

  "checkFileAvailability" should "return Valid when all files are present" in {
    val dir = Files.createTempDirectory("file-avail-test-")
    try {
      Files.write(dir.resolve(testHash), Array[Byte](1, 2, 3))
      val block = mkBlock(fileRegTerm(testHash))
      val result =
        FileAvailability.checkFileAvailability[Task](block, dir).runSyncUnsafe()
      result shouldBe Right(BlockStatus.valid)
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }

  it should "return MissingFileData when a file is missing" in {
    val dir = Files.createTempDirectory("file-avail-test-")
    try {
      val block = mkBlock(fileRegTerm(testHash))
      val result =
        FileAvailability.checkFileAvailability[Task](block, dir).runSyncUnsafe()
      result shouldBe Left(InvalidBlock.MissingFileData)
    } finally {
      dir.toFile.delete()
    }
  }

  it should "return Valid when block has no file deploys" in {
    val dir = Files.createTempDirectory("file-avail-test-")
    try {
      val block  = mkBlock("new x in { x!(42) }")
      val result = FileAvailability.checkFileAvailability[Task](block, dir).runSyncUnsafe()
      result shouldBe Right(BlockStatus.valid)
    } finally {
      dir.toFile.delete()
    }
  }

  it should "return MissingFileData when only some files are present" in {
    val dir = Files.createTempDirectory("file-avail-test-")
    try {
      Files.write(dir.resolve(testHash), Array[Byte](1, 2, 3))
      // testHash2 is NOT on disk
      val block = mkBlock(fileRegTerm(testHash), fileRegTerm(testHash2))
      val result =
        FileAvailability.checkFileAvailability[Task](block, dir).runSyncUnsafe()
      result shouldBe Left(InvalidBlock.MissingFileData)
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }

  // ---- findMissingFiles ----

  "findMissingFiles" should "return only missing file hashes" in {
    val dir = Files.createTempDirectory("file-avail-test-")
    try {
      Files.write(dir.resolve(testHash), Array[Byte](1, 2, 3))
      val block   = mkBlock(fileRegTerm(testHash), fileRegTerm(testHash2))
      val missing = FileAvailability.findMissingFiles[Task](block, dir).runSyncUnsafe()
      missing shouldBe List(testHash2)
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }
}
