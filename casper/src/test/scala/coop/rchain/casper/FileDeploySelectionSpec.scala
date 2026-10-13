package coop.rchain.casper

import cats.syntax.all._
import coop.rchain.casper.protocol.DeployData
import coop.rchain.casper.util.OrphanFileCleanup
import coop.rchain.casper.util.GenesisBuilder.defaultValidatorSks
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Unit tests for file-deploy selection backpressure.
  *
  * These test the helpers and partition/limit logic used in
  * `BlockCreator.prepareUserDeploys`.
  */
class FileDeploySelectionSpec extends FlatSpec with Matchers {

  private val validatorSk = defaultValidatorSks.head

  /** Helper: file-registration deploy with configurable size. */
  private def fileRegTerm(hash: String, fileSize: Long = 1048576): String =
    s"""new ret, file(`rho:io:file`) in { file!("register", "$hash", $fileSize, "test.bin", *ret) }"""

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

  private def mkNonFileDeploy(i: Int): Signed[DeployData] =
    mkSignedDeploy(s"new x in { x!($i) }")

  private def mkFileDeploy(hashSuffix: String, fileSize: Long = 1048576): Signed[DeployData] =
    mkSignedDeploy(fileRegTerm(hashSuffix.padTo(64, '0'), fileSize))

  /** Applies the same backpressure logic from BlockCreator.prepareUserDeploys */
  private def applyBackpressure(
      deploys: Set[Signed[DeployData]],
      maxFileDeploys: Int,
      maxFileDataSize: Long
  ): Set[Signed[DeployData]] = {
    val (fileDeploys, nonFileDeploys) = deploys.partition(
      d => OrphanFileCleanup.isFileRegistrationDeploy(d.data)
    )
    val limitedFileDeploys = fileDeploys.toList
      .sortBy(_.data.timestamp)
      .take(maxFileDeploys)
      .foldLeft((0L, List.empty[Signed[DeployData]])) {
        case ((usedSize, accepted), d) =>
          val size = OrphanFileCleanup.extractFileSize(d.data).getOrElse(Long.MaxValue)
          if (usedSize + size <= maxFileDataSize) (usedSize + size, d :: accepted)
          else (usedSize, accepted)
      }
      ._2
      .toSet
    nonFileDeploys ++ limitedFileDeploys
  }

  // ---- extractFileSize tests ----

  "extractFileSize" should "extract file size from a file-registration deploy" in {
    val deploy = mkSignedDeploy(fileRegTerm("a" * 64, 5242880))
    OrphanFileCleanup.extractFileSize(deploy.data) shouldBe Some(5242880L)
  }

  it should "return None for non-file-registration deploys" in {
    val deploy = mkSignedDeploy("new x in { x!(42) }")
    OrphanFileCleanup.extractFileSize(deploy.data) shouldBe None
  }

  it should "handle large file sizes (multi-GB)" in {
    val deploy = mkSignedDeploy(fileRegTerm("a" * 64, 53687091200L))
    OrphanFileCleanup.extractFileSize(deploy.data) shouldBe Some(53687091200L)
  }

  // ---- count limit tests ----

  "file deploy count limit" should "keep all when under limit" in {
    val deploys = (1 to 5).map(i => mkFileDeploy(i.toString)).toSet
    val result  = applyBackpressure(deploys, maxFileDeploys = 10, maxFileDataSize = Long.MaxValue)
    result.size shouldBe 5
  }

  it should "drop excess file deploys when over count limit" in {
    val deploys = (1 to 15).map(i => mkFileDeploy(i.toString)).toSet
    val result  = applyBackpressure(deploys, maxFileDeploys = 10, maxFileDataSize = Long.MaxValue)
    result.size shouldBe 10
    // All results should be file-registration deploys
    result.foreach(d => OrphanFileCleanup.isFileRegistrationDeploy(d.data) shouldBe true)
  }

  it should "keep zero file deploys when limit is zero" in {
    val deploys = (1 to 5).map(i => mkFileDeploy(i.toString)).toSet
    val result  = applyBackpressure(deploys, maxFileDeploys = 0, maxFileDataSize = Long.MaxValue)
    result.size shouldBe 0
  }

  // ---- size limit tests ----

  "file deploy size limit" should "keep all when under size limit" in {
    // 3 deploys × 1MB each = 3MB, limit = 50GB
    val deploys = (1 to 3).map(i => mkFileDeploy(i.toString, 1048576L)).toSet
    val result =
      applyBackpressure(deploys, maxFileDeploys = 100, maxFileDataSize = 50L * 1024 * 1024 * 1024)
    result.size shouldBe 3
  }

  it should "stop accepting file deploys when size limit is reached" in {
    // 6 deploys × 10GB each = 60GB, limit = 50GB → should accept at most 5
    val tenGB   = 10L * 1024 * 1024 * 1024
    val deploys = (1 to 6).map(i => mkFileDeploy(i.toString, tenGB)).toSet
    val result =
      applyBackpressure(deploys, maxFileDeploys = 100, maxFileDataSize = 50L * 1024 * 1024 * 1024)
    result.size shouldBe 5
  }

  it should "reject a single deploy that exceeds the size limit" in {
    val hugeSize = 100L * 1024 * 1024 * 1024 // 100GB
    val deploys  = Set(mkFileDeploy("1", hugeSize))
    val result =
      applyBackpressure(deploys, maxFileDeploys = 100, maxFileDataSize = 50L * 1024 * 1024 * 1024)
    result.size shouldBe 0
  }

  // ---- mixed deploy tests ----

  "mixed file and non-file deploys" should "leave non-file deploys unaffected by limits" in {
    val fileDeploys    = (1 to 15).map(i => mkFileDeploy(i.toString)).toSet
    val nonFileDeploys = (1 to 10).map(i => mkNonFileDeploy(i)).toSet
    val allDeploys     = fileDeploys ++ nonFileDeploys
    val result =
      applyBackpressure(allDeploys, maxFileDeploys = 5, maxFileDataSize = Long.MaxValue)

    // All 10 non-file deploys should survive
    val resultNonFile = result.filterNot(d => OrphanFileCleanup.isFileRegistrationDeploy(d.data))
    resultNonFile.size shouldBe 10

    // Only 5 file deploys should survive
    val resultFile = result.filter(d => OrphanFileCleanup.isFileRegistrationDeploy(d.data))
    resultFile.size shouldBe 5
  }

  it should "return all deploys when there are no file deploys" in {
    val nonFileDeploys = (1 to 20).map(i => mkNonFileDeploy(i)).toSet
    val result =
      applyBackpressure(nonFileDeploys, maxFileDeploys = 5, maxFileDataSize = 10L)
    result.size shouldBe 20
  }

  it should "apply both count AND size limits" in {
    val tenGB = 10L * 1024 * 1024 * 1024
    // 8 file deploys × 10GB = 80GB; count limit=10, size limit=25GB
    // Count keeps all 8 (under 10), but size only allows 2 (20GB ≤ 25GB, 30GB > 25GB)
    val fileDeploys = (1 to 8).map(i => mkFileDeploy(i.toString, tenGB)).toSet
    val nonFile     = Set(mkNonFileDeploy(1))
    val result =
      applyBackpressure(
        fileDeploys ++ nonFile,
        maxFileDeploys = 10,
        maxFileDataSize = 25L * 1024 * 1024 * 1024
      )

    val resultFile = result.filter(d => OrphanFileCleanup.isFileRegistrationDeploy(d.data))
    resultFile.size shouldBe 2

    val resultNonFile = result.filterNot(d => OrphanFileCleanup.isFileRegistrationDeploy(d.data))
    resultNonFile.size shouldBe 1
  }
}
