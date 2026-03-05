package coop.rchain.node.api

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.deploy.v1._
import coop.rchain.crypto.hash.Blake2b256
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import monix.reactive.Observable
import org.bouncycastle.crypto.digests.Blake2bDigest
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import java.nio.file.{Files, Path}
import scala.concurrent.duration._

class FileUploadAPISpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  private val uploadDir: Path = Files.createTempDirectory("file-upload-test-")

  override def afterAll(): Unit = {
    // Clean up temp directory tree
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

  private val chunkSize  = 64 * 1024 // 64 KB
  private val numChunks  = 16 // 16 × 64 KB = 1 MB
  private val totalBytes = chunkSize * numChunks

  /** Build a repeatable chunk payload for chunk index `i`. */
  private def makeChunkData(i: Int): Array[Byte] = Array.fill(chunkSize)(i.toByte)

  private def allBytes: Array[Byte] = (0 until numChunks).flatMap(i => makeChunkData(i)).toArray

  private def referenceHash: String = Blake2b256.hash(allBytes).map("%02x".format(_)).mkString

  private def metadataChunk(
      fileHash: String = "",
      phloPrice: Long = 1L,
      shardId: String = "root",
      fileSize: Long = totalBytes.toLong
  ): FileUploadChunk =
    FileUploadChunk(
      FileUploadChunk.Chunk.Metadata(
        FileUploadMetadata(
          deployer = ByteString.copyFrom(Array[Byte](1, 2, 3)),
          timestamp = 1700000000L,
          shardId = shardId,
          fileName = "test.bin",
          fileSize = fileSize,
          fileHash = fileHash,
          phloPrice = phloPrice,
          phloLimit = 10000000L,
          validAfterBlockNumber = 0L,
          sigAlgorithm = "ed25519",
          sig = ByteString.EMPTY,
          term = "new x in { x!(42) }" // stub term for validation
        )
      )
    )

  private def dataChunks(n: Int = numChunks): Observable[FileUploadChunk] =
    Observable.fromIterable(
      (0 until n).map(
        i => FileUploadChunk(FileUploadChunk.Chunk.Data(ByteString.copyFrom(makeChunkData(i))))
      )
    )

  private def fullStream(
      fileHash: String = "",
      fileSize: Long = totalBytes.toLong
  ): Observable[FileUploadChunk] =
    Observable.cons(metadataChunk(fileHash, fileSize = fileSize), dataChunks())

  private def run[A](task: Task[A]): A = task.runSyncUnsafe(30.seconds)

  private def newDir(): Path = {
    val d = uploadDir.resolve(java.util.UUID.randomUUID().toString)
    Files.createDirectories(d)
    d
  }

  // ---- tests ---------------------------------------------------------

  "FileUploadAPI" should "write the correct bytes to disk for a 1 MB stream" in {
    val dir    = newDir()
    val output = run(FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir))

    val finalFile = dir.resolve(output.result.fileHash)
    finalFile.toFile.exists() shouldBe true
    Files.readAllBytes(finalFile) shouldBe allBytes
  }

  it should "successfully upload a small file with 3 chunks" in {
    val dir        = newDir()
    val smallChunk = 128
    val chunks = List(
      Array.fill(smallChunk)(0xAA.toByte),
      Array.fill(smallChunk)(0xBB.toByte),
      Array.fill(smallChunk)(0xCC.toByte)
    )
    val expected = chunks.reduce(_ ++ _)
    val size     = expected.length.toLong

    val meta = metadataChunk(fileSize = size)
    val data = Observable.fromIterable(
      chunks.map(b => FileUploadChunk(FileUploadChunk.Chunk.Data(ByteString.copyFrom(b))))
    )
    val stream = Observable.cons(meta, data)

    val output = run(FileUploadAPI.processFileUpload(stream, "root", 1L, false, dir))

    // Verify file content matches all bytes
    val finalFile = dir.resolve(output.result.fileHash)
    val written   = Files.readAllBytes(finalFile)
    written shouldBe expected

    // Verify each chunk is intact and in correct order
    for ((chunkData, idx) <- chunks.zipWithIndex) {
      val offset = idx * smallChunk
      val slice  = written.slice(offset, offset + smallChunk)
      slice shouldBe chunkData
    }

    // Verify hash matches reference
    val refHash = Blake2b256.hash(expected).map("%02x".format(_)).mkString
    output.result.fileHash shouldBe refHash

    // Verify .meta.json was written
    val metaFile = dir.resolve(s"${output.result.fileHash}.meta.json")
    metaFile.toFile.exists() shouldBe true
  }

  it should "compute the correct Blake2b-256 hash for a 1 MB stream" in {
    val dir    = newDir()
    val output = run(FileUploadAPI.processFileUpload(fullStream(), "root", 1L, false, dir))

    output.result.fileHash shouldBe referenceHash
  }

  it should "delete the .tmp file and leave no orphans when the stream is interrupted midway" in {
    val dir = newDir()

    // Interrupt after 8 chunks by appending an error
    val interrupted: Observable[FileUploadChunk] =
      Observable.cons(
        metadataChunk(),
        dataChunks(numChunks / 2) ++ Observable.raiseError(new RuntimeException("gRPC cancel"))
      )

    intercept[RuntimeException] {
      run(FileUploadAPI.processFileUpload(interrupted, "root", 1L, false, dir))
    }

    // No .tmp files must remain
    import scala.jdk.CollectionConverters._
    val remaining = Files.list(dir).iterator().asScala.toList
    remaining.filter(_.toString.endsWith(".tmp")) shouldBe empty
  }

  it should "abort and clean up when bytesReceived > fileSize" in {
    val dir = newDir()

    // Declare fileSize = half of what we'll actually send
    val stream = Observable.cons(
      metadataChunk(fileSize = (totalBytes / 2).toLong),
      dataChunks()
    )

    val ex = intercept[Exception] {
      run(FileUploadAPI.processFileUpload(stream, "root", 1L, false, dir))
    }
    ex.getMessage should include("exceeds declared fileSize")

    import scala.jdk.CollectionConverters._
    Files.list(dir).iterator().asScala.filter(_.toString.endsWith(".tmp")) shouldBe empty
  }

  it should "reject and clean up when computedHash != fileHash" in {
    val dir    = newDir()
    val stream = fullStream(fileHash = "cafecafe" * 8) // 64 hex chars, wrong hash

    val ex = intercept[Exception] {
      run(FileUploadAPI.processFileUpload(stream, "root", 1L, false, dir))
    }
    ex.getMessage should include("Hash mismatch")

    import scala.jdk.CollectionConverters._
    Files.list(dir).iterator().asScala.filter(_.toString.endsWith(".tmp")) shouldBe empty
  }

  it should "skip disk write and return existing hash on duplicate upload" in {
    val dir  = newDir()
    val hash = referenceHash

    // Pre-seed the file
    val existingFile = dir.resolve(hash)
    Files.write(existingFile, allBytes)

    val output = run(
      FileUploadAPI.processFileUpload(fullStream(fileHash = hash), "root", 1L, false, dir)
    )

    output.result.fileHash shouldBe hash
    // Only the pre-seeded file; no .tmp written
    import scala.jdk.CollectionConverters._
    val files = Files.list(dir).iterator().asScala.toList
    // Dedup path now builds deploy but doesn't write .meta.json for dups
    files.map(_.getFileName.toString) should contain(hash)
  }

  "FileMetadata" should "round-trip through JSON serialization" in {
    val meta = FileMetadata(
      fileName = "ubuntu.iso",
      fileSize = 1073741824L,
      uploaderPubKey = "deadbeef",
      timestamp = 1700000000L,
      hash = "aabbccdd"
    )
    FileMetadata.fromJson(FileMetadata.toJson(meta)) shouldBe Right(meta)
  }

  it should "escape special characters in JSON output" in {
    val meta   = FileMetadata("file\"with\\quotes", 0L, "pub", 0L, "hash")
    val json   = FileMetadata.toJson(meta)
    val parsed = FileMetadata.fromJson(json)
    parsed.map(_.fileName) shouldBe Right(meta.fileName)
  }

  "FileUploadAPI" should "reject when declared fileSize > actual bytes received" in {
    val dir = newDir()

    // Declare fileSize = 2× what we'll actually send
    val stream = Observable.cons(
      metadataChunk(fileSize = (totalBytes * 2).toLong),
      dataChunks()
    )

    val ex = intercept[Exception] {
      run(FileUploadAPI.processFileUpload(stream, "root", 1L, false, dir))
    }
    ex.getMessage should include("Size mismatch")

    import scala.jdk.CollectionConverters._
    Files.list(dir).iterator().asScala.filter(_.toString.endsWith(".tmp")) shouldBe empty
  }
}
