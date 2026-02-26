package coop.rchain.casper.engine

import cats.syntax.all._
import coop.rchain.casper.{BlockStatus, InvalidBlock, ValidBlock}
import coop.rchain.casper.protocol.{BlockMessage, DeployData, ProcessedDeploy}
import coop.rchain.casper.util.FileAvailability
import coop.rchain.casper.util.GenesisBuilder.defaultValidatorSks
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import coop.rchain.models.PCost
import coop.rchain.models.blockImplicits.getRandomBlock
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import java.nio.file.{Files, Path}

/**
  * Unit tests for DA-gated block validation logic.
  *
  * These tests verify the suspend/fetch/await pattern used in
  * `MultiParentCasperImpl.validate` for data availability checks.
  */
class DAGateSpec extends FlatSpec with Matchers {

  private val validatorSk = defaultValidatorSks.head
  private val testHash    = "a" * 64
  private val testHash2   = "b" * 64

  private def fileRegTerm(hash: String): String =
    s"""new ret, file(`rho:io:file`) in { file!("register", "$hash", 1048576, "test.bin", *ret) }"""

  private def mkSignedDeploy(term: String): Signed[DeployData] = {
    val dd = DeployData(
      term = term,
      timestamp = System.currentTimeMillis(),
      phloPrice = 1,
      phloLimit = 1000,
      validAfterBlockNumber = 0L,
      shardId = "root"
    )
    Signed(dd, Secp256k1, validatorSk)
  }

  private def mkProcessedDeploy(term: String): ProcessedDeploy =
    ProcessedDeploy(
      deploy = mkSignedDeploy(term),
      cost = PCost(0L),
      deployLog = List.empty,
      isFailed = false
    )

  private def mkBlock(terms: String*): BlockMessage =
    getRandomBlock(setDeploys = Some(terms.map(mkProcessedDeploy)))

  /**
    * Simulates the DA-gate logic from MultiParentCasperImpl.validate:
    *   1. Find missing files
    *   2. If missing, call daFetchFiles callback
    *   3. Re-check what's still missing after callback
    *   4. Return Valid or MissingFileData
    */
  private def runDAGate(
      block: BlockMessage,
      dir: Path,
      daFetchFiles: (BlockMessage, List[String]) => Task[List[String]]
  ): Task[Either[InvalidBlock, ValidBlock]] =
    Task
      .delay {
        FileAvailability.findMissingFiles(block, dir)
      }
      .flatMap { missing =>
        if (missing.isEmpty)
          Task.pure(Right(BlockStatus.valid))
        else
          daFetchFiles(block, missing).map { stillMissing =>
            if (stillMissing.isEmpty) Right(BlockStatus.valid)
            else Left(InvalidBlock.MissingFileData)
          }
      }

  // ---- DA-gate tests ----

  "DA-gate" should "pass immediately when all files are already present" in {
    val dir = Files.createTempDirectory("da-gate-test-")
    try {
      Files.write(dir.resolve(testHash), Array[Byte](1, 2, 3))
      val block = mkBlock(fileRegTerm(testHash))

      // Callback should NOT be called
      val callbackCalled = new java.util.concurrent.atomic.AtomicBoolean(false)
      val noopFetch: (BlockMessage, List[String]) => Task[List[String]] =
        (_, _) => Task.delay { callbackCalled.set(true); List.empty }

      val result = runDAGate(block, dir, noopFetch).runSyncUnsafe()
      result shouldBe Right(BlockStatus.valid)
      callbackCalled.get() shouldBe false
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }

  it should "pass when file is missing but arrives via DA fetch" in {
    val dir = Files.createTempDirectory("da-gate-test-")
    try {
      val block = mkBlock(fileRegTerm(testHash))

      // Simulate successful P2P fetch: write the file and return empty list
      val fetchAndDeliver: (BlockMessage, List[String]) => Task[List[String]] =
        (_, hashes) =>
          Task.delay {
            hashes.foreach(h => Files.write(dir.resolve(h), Array[Byte](1, 2, 3)))
            List.empty[String] // all delivered
          }

      val result = runDAGate(block, dir, fetchAndDeliver).runSyncUnsafe()
      result shouldBe Right(BlockStatus.valid)
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }

  it should "fail with MissingFileData when files remain missing after timeout" in {
    val dir = Files.createTempDirectory("da-gate-test-")
    try {
      val block = mkBlock(fileRegTerm(testHash))

      // Simulate timeout: return the same missing hashes
      val timeoutFetch: (BlockMessage, List[String]) => Task[List[String]] =
        (_, hashes) => Task.pure(hashes)

      val result = runDAGate(block, dir, timeoutFetch).runSyncUnsafe()
      result shouldBe Left(InvalidBlock.MissingFileData)
    } finally {
      dir.toFile.delete()
    }
  }

  it should "pass immediately for blocks with no file deploys" in {
    val dir = Files.createTempDirectory("da-gate-test-")
    try {
      val block = mkBlock("new x in { x!(42) }")

      val callbackCalled = new java.util.concurrent.atomic.AtomicBoolean(false)
      val noopFetch: (BlockMessage, List[String]) => Task[List[String]] =
        (_, _) => Task.delay { callbackCalled.set(true); List.empty }

      val result = runDAGate(block, dir, noopFetch).runSyncUnsafe()
      result shouldBe Right(BlockStatus.valid)
      callbackCalled.get() shouldBe false
    } finally {
      dir.toFile.delete()
    }
  }

  it should "fail when only some files arrive via DA fetch" in {
    val dir = Files.createTempDirectory("da-gate-test-")
    try {
      val block = mkBlock(fileRegTerm(testHash), fileRegTerm(testHash2))

      // Simulate partial fetch: only deliver testHash, testHash2 remains missing
      val partialFetch: (BlockMessage, List[String]) => Task[List[String]] =
        (_, hashes) =>
          Task.delay {
            hashes.filter(_ == testHash).foreach { h =>
              Files.write(dir.resolve(h), Array[Byte](1, 2, 3))
            }
            hashes.filterNot(_ == testHash)
          }

      val result = runDAGate(block, dir, partialFetch).runSyncUnsafe()
      result shouldBe Left(InvalidBlock.MissingFileData)
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }

  it should "pass the correct missing hashes to the callback" in {
    val dir = Files.createTempDirectory("da-gate-test-")
    try {
      // testHash is present, testHash2 is missing
      Files.write(dir.resolve(testHash), Array[Byte](1, 2, 3))
      val block = mkBlock(fileRegTerm(testHash), fileRegTerm(testHash2))

      var receivedHashes: List[String] = Nil
      val captureFetch: (BlockMessage, List[String]) => Task[List[String]] =
        (_, hashes) =>
          Task.delay {
            receivedHashes = hashes
            List.empty[String] // pretend all delivered
          }

      runDAGate(block, dir, captureFetch).runSyncUnsafe()
      receivedHashes shouldBe List(testHash2)
    } finally {
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }
}
