package coop.rchain.node.api

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import monix.reactive.Observable
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import java.nio.file.{Files, Path}
import scala.concurrent.duration._

class FileUploadCostSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  private val uploadDir: Path = Files.createTempDirectory("file-upload-cost-test-")

  override def afterAll(): Unit = {
    def deleteTree(p: Path): Unit =
      if (Files.isDirectory(p)) {
        import scala.jdk.CollectionConverters._
        Files.list(p).iterator().asScala.foreach(deleteTree)
        Files.deleteIfExists(p)
        ()
      } else {
        Files.deleteIfExists(p)
        ()
      }
    deleteTree(uploadDir)
  }

  // ---- helpers -------------------------------------------------------

  private val chunkSize  = 1024
  private val numChunks  = 1
  private val totalBytes = chunkSize * numChunks

  private def makeChunkData: Array[Byte] = Array.fill(chunkSize)(0xAB.toByte)

  private def metadataChunk(
      fileSize: Long = totalBytes.toLong,
      phloLimit: Long = 10000000L,
      phloPrice: Long = 1L
  ): FileUploadChunk =
    FileUploadChunk(
      FileUploadChunk.Chunk.Metadata(
        FileUploadMetadata(
          deployer = ByteString.copyFrom(Array[Byte](1, 2, 3)),
          timestamp = 1700000000L,
          shardId = "root",
          fileName = "test.bin",
          fileSize = fileSize,
          fileHash = "",
          phloPrice = phloPrice,
          phloLimit = phloLimit,
          validAfterBlockNumber = 0L,
          sigAlgorithm = "ed25519",
          sig = ByteString.EMPTY,
          term =
            s"""new return, file(`rho:io:file`) in { file!("register", "${"0" * 64}", $fileSize, "test.bin", *return) }"""
        )
      )
    )

  private def dataChunks: Observable[FileUploadChunk] =
    Observable.pure(
      FileUploadChunk(FileUploadChunk.Chunk.Data(ByteString.copyFrom(makeChunkData)))
    )

  private def fullStream(
      fileSize: Long = totalBytes.toLong,
      phloLimit: Long = 10000000L,
      phloPrice: Long = 1L
  ): Observable[FileUploadChunk] =
    Observable.cons(metadataChunk(fileSize, phloLimit, phloPrice), dataChunks)

  private def run[A](task: Task[A]): A = task.runSyncUnsafe(30.seconds)

  private def newDir(): Path = {
    val d = uploadDir.resolve(java.util.UUID.randomUUID().toString)
    Files.createDirectories(d)
    d
  }

  // ---- FileUploadCosts pure tests ------------------------------------

  "FileUploadCosts.totalRequired" should "return base only for zero-byte file" in {
    FileUploadCosts.totalRequired(0L, 1L) shouldBe 300L
  }

  it should "return base + fileSize for phloPerStorageByte=1" in {
    FileUploadCosts.totalRequired(1024L, 1L) shouldBe 1324L
  }

  it should "compute correctly for 10 GB file" in {
    val tenGB = 10L * 1024L * 1024L * 1024L // 10,737,418,240
    FileUploadCosts.totalRequired(tenGB, 1L) shouldBe (300L + tenGB)
  }

  it should "scale with phloPerStorageByte" in {
    FileUploadCosts.totalRequired(500L, 5L) shouldBe (300L + 2500L)
  }

  it should "throw ArithmeticException on overflow" in {
    intercept[ArithmeticException] {
      FileUploadCosts.totalRequired(Long.MaxValue, 2L)
    }
  }

  // ---- Upload-time rejection -----------------------------------------

  "FileUploadAPI" should "reject upload when phloLimit < totalRequired" in {
    val dir = newDir()
    // totalRequired = 300 + 1024 * 1 = 1324
    val ex = intercept[Exception] {
      run(FileUploadAPI.processFileUpload(fullStream(phloLimit = 100L), "root", 1L, false, dir))
    }
    ex.getMessage should include("Insufficient phlo")
    ex.getMessage should include("phloLimit=100")
    ex.getMessage should include("required=1324")
  }

  it should "accept upload when phloLimit >= totalRequired" in {
    val dir = newDir()
    // totalRequired = 300 + 1024 * 1 = 1324
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(phloLimit = 2000L), "root", 1L, false, dir)
    )
    output.result.fileHash should not be empty
  }

  it should "accept upload when phloLimit equals totalRequired exactly" in {
    val dir = newDir()
    // totalRequired = 300 + 1024 * 1 = 1324
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(phloLimit = 1324L), "root", 1L, false, dir)
    )
    output.result.fileHash should not be empty
  }

  // ---- Cost fields in FileUploadResult --------------------------------

  it should "set storagePhloCost = fileSize * phloPerStorageByte in result" in {
    val dir = newDir()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(phloLimit = 10000L), "root", 1L, false, dir)
    )
    output.result.storagePhloCost shouldBe totalBytes.toLong * FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE
  }

  it should "set totalPhloCharged = BASE_REGISTER_PHLO + storagePhloCost in result" in {
    val dir = newDir()
    val output = run(
      FileUploadAPI.processFileUpload(fullStream(phloLimit = 10000L), "root", 1L, false, dir)
    )
    val expectedStorage = totalBytes.toLong * FileUploadCosts.DEFAULT_PHLO_PER_STORAGE_BYTE
    output.result.totalPhloCharged shouldBe FileUploadCosts.BASE_REGISTER_PHLO + expectedStorage
  }

  // ---- Custom phloPerStorageByte --------------------------------------

  it should "reject when phloPerStorageByte=2 makes cost exceed phloLimit" in {
    val dir = newDir()
    // totalRequired = 300 + 1024 * 2 = 2348
    val ex = intercept[Exception] {
      run(
        FileUploadAPI.processFileUpload(
          fullStream(phloLimit = 2000L),
          "root",
          1L,
          false,
          dir,
          phloPerStorageByte = 2L
        )
      )
    }
    ex.getMessage should include("Insufficient phlo")
  }

  it should "compute storagePhloCost with custom phloPerStorageByte=3" in {
    val dir = newDir()
    // totalRequired = 300 + 1024 * 3 = 3372
    val output = run(
      FileUploadAPI.processFileUpload(
        fullStream(phloLimit = 10000L),
        "root",
        1L,
        false,
        dir,
        phloPerStorageByte = 3L
      )
    )
    output.result.storagePhloCost shouldBe 1024L * 3L
    output.result.totalPhloCharged shouldBe 300L + 1024L * 3L
  }
}
