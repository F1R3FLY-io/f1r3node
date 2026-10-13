package coop.rchain.casper.util

import cats.syntax.all._
import coop.rchain.casper.protocol.DeployData
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import coop.rchain.casper.util.GenesisBuilder.defaultValidatorSks
import coop.rchain.p2p.EffectsTestInstances.LogStub
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import java.nio.file.{Files, Path}

/**
  * Unit tests for OrphanFileCleanup.
  *
  * Tests file-registration deploy detection, hash extraction,
  * cross-reference checking, and physical file deletion.
  */
class OrphanFileCleanupSpec extends FlatSpec with Matchers {

  implicit private val logStub: Log[Task] = new LogStub[Task]()

  private val validatorSk = defaultValidatorSks.head
  private val testHash    = "a" * 64 // 64-char hex hash

  /** Helper to create a signed deploy with the given term. */
  private def mkDeploy(term: String): Signed[DeployData] = {
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

  /** Helper to create a file-registration deploy term for the given hash. */
  private def fileRegTerm(hash: String): String =
    s"""new ret, file(`rho:io:file`) in { file!("register", "$hash", 1048576, "test.bin", *ret) }"""

  /** Helper to create a temp dir with a file and its meta.json. */
  private def withFileOnDisk(hash: String)(test: Path => Task[Unit]): Unit = {
    val dir = Files.createTempDirectory("orphan-cleanup-test-")
    try {
      Files.write(dir.resolve(hash), Array[Byte](1, 2, 3))
      Files.write(dir.resolve(s"$hash.meta.json"), """{"fileName":"test.bin"}""".getBytes("UTF-8"))
      test(dir).runSyncUnsafe()
    } finally {
      // Cleanup temp dir
      dir.toFile.listFiles().foreach(_.delete())
      dir.toFile.delete()
    }
  }

  // --- Test 1: File deploy term detected ---
  "isFileRegistrationDeploy" should "return true for a file registration term" in {
    val term = fileRegTerm(testHash)
    OrphanFileCleanup.isFileRegistrationDeploy(DeployData(term, 0, 0, 0, 0, "")) shouldBe true
  }

  // --- Test 2: Non-file deploy term ---
  it should "return false for a regular Rholang term" in {
    OrphanFileCleanup.isFileRegistrationDeploy(
      DeployData("new x in { x!(42) }", 0, 0, 0, 0, "")
    ) shouldBe false
  }

  it should "return false for a term with register and hash but no rho:io:file" in {
    val fakeHash = "b" * 64
    val term     = s"""new x in { x!("register", "$fakeHash") }"""
    OrphanFileCleanup.isFileRegistrationDeploy(
      DeployData(term, 0, 0, 0, 0, "")
    ) shouldBe false
  }

  // --- Test 3: Hash extraction ---
  "extractFileHash" should "return the 64-char hex hash from a file registration term" in {
    val term = fileRegTerm(testHash)
    OrphanFileCleanup.extractFileHash(DeployData(term, 0, 0, 0, 0, "")) shouldBe Some(testHash)
  }

  it should "return None for a term without a valid hash" in {
    OrphanFileCleanup.extractFileHash(
      DeployData("new x in { x!(42) }", 0, 0, 0, 0, "")
    ) shouldBe None
  }

  // --- Test 4: Single expired file deploy → files deleted ---
  "cleanupOrphanedFiles" should "delete physical file and meta.json when deploy is removed" in {
    withFileOnDisk(testHash) { dir =>
      val removed   = List(mkDeploy(fileRegTerm(testHash)))
      val remaining = List.empty[Signed[DeployData]]

      OrphanFileCleanup.cleanupOrphanedFiles[Task](removed, remaining, dir).map { _ =>
        Files.exists(dir.resolve(testHash)) shouldBe false
        Files.exists(dir.resolve(s"$testHash.meta.json")) shouldBe false
      }
    }
  }

  // --- Test 5: Two deploys same hash → expire one → file NOT deleted ---
  it should "NOT delete file when another deploy references the same hash" in {
    withFileOnDisk(testHash) { dir =>
      val removed   = List(mkDeploy(fileRegTerm(testHash)))
      val remaining = List(mkDeploy(fileRegTerm(testHash))) // another deploy with same hash

      OrphanFileCleanup.cleanupOrphanedFiles[Task](removed, remaining, dir).map { _ =>
        Files.exists(dir.resolve(testHash)) shouldBe true
        Files.exists(dir.resolve(s"$testHash.meta.json")) shouldBe true
      }
    }
  }

  // --- Test 6: Expire second (last) deploy for same hash → file IS deleted ---
  it should "delete file when the last deploy referencing the hash is removed" in {
    withFileOnDisk(testHash) { dir =>
      val removed   = List(mkDeploy(fileRegTerm(testHash)))
      val remaining = List.empty[Signed[DeployData]] // no more references

      OrphanFileCleanup.cleanupOrphanedFiles[Task](removed, remaining, dir).map { _ =>
        Files.exists(dir.resolve(testHash)) shouldBe false
        Files.exists(dir.resolve(s"$testHash.meta.json")) shouldBe false
      }
    }
  }

  // --- Test 7: Non-file deploy expired → no filesystem side-effects ---
  it should "not touch files when expired deploy is not a file registration deploy" in {
    withFileOnDisk(testHash) { dir =>
      val removed   = List(mkDeploy("new x in { x!(42) }")) // regular deploy
      val remaining = List.empty[Signed[DeployData]]

      OrphanFileCleanup.cleanupOrphanedFiles[Task](removed, remaining, dir).map { _ =>
        // Files should still exist — no cleanup triggered
        Files.exists(dir.resolve(testHash)) shouldBe true
        Files.exists(dir.resolve(s"$testHash.meta.json")) shouldBe true
      }
    }
  }

  // --- Test: deleteFileAndMeta is safe when files don't exist ---
  "deleteFileAndMeta" should "not throw when files do not exist" in {
    val dir = Files.createTempDirectory("orphan-cleanup-test-")
    try {
      val test = OrphanFileCleanup.deleteFileAndMeta[Task](dir, "nonexistent" * 4)
      test.runSyncUnsafe() // should not throw
    } finally {
      dir.toFile.delete()
    }
  }
}
