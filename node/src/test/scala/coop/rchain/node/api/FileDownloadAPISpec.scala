package coop.rchain.node.api

import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, FlatSpec, Matchers}

import java.nio.file.{Files, Path}
import scala.concurrent.duration._

class FileDownloadAPISpec
    extends FlatSpec
    with Matchers
    with BeforeAndAfterAll
    with BeforeAndAfterEach {

  private val baseDir: Path = Files.createTempDirectory("file-download-test-")

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
    deleteTree(baseDir)
  }

  override def beforeEach(): Unit =
    FileDownloadAPI.resetRateLimiter()

  // ---- helpers -------------------------------------------------------

  /** Default chunk size from configuration. */
  private val DefaultChunkSize = 4 * 1024 * 1024

  /** Default max concurrent downloads per IP from configuration. */
  private val DefaultMaxConcurrentPerIp = 4

  private def newDir(): Path = {
    val d = baseDir.resolve(java.util.UUID.randomUUID().toString)
    Files.createDirectories(d)
    d
  }

  /** Create a file with deterministic content and return (dir, hash, content). */
  private def seedFile(size: Int = 1024 * 1024): (Path, String, Array[Byte]) = {
    val dir   = newDir()
    val bytes = Array.tabulate[Byte](size)(i => (i % 251).toByte)
    val hash  = coop.rchain.crypto.hash.Blake2b256.hash(bytes).map("%02x".format(_)).mkString
    Files.write(dir.resolve(hash), bytes)
    (dir, hash, bytes)
  }

  private def run[A](task: Task[A]): A = task.runSyncUnsafe(30.seconds)

  private def collectChunks(
      request: FileDownloadRequest,
      isReadOnly: Boolean,
      dir: Path,
      ip: String = "127.0.0.1",
      maxPerIp: Int = DefaultMaxConcurrentPerIp,
      devMode: Boolean = false
  ): List[FileDownloadChunk] =
    run(
      FileDownloadAPI
        .streamFile(request, isReadOnly, dir, ip, DefaultChunkSize, maxPerIp, devMode)
        .toListL
    )

  // ---- tests ---------------------------------------------------------

  "FileDownloadAPI" should "stream the full file on a read-only node" in {
    val (dir, hash, expected) = seedFile()
    val request               = FileDownloadRequest(fileHash = hash)
    val chunks                = collectChunks(request, isReadOnly = true, dir)

    // First chunk should be metadata
    chunks.head.chunk.isMetadata shouldBe true
    val meta = chunks.head.getMetadata
    meta.fileHash shouldBe hash
    meta.fileSize shouldBe expected.length.toLong

    // Remaining chunks are data
    val dataBytes = chunks.tail.flatMap(_.getData.toByteArray).toArray
    dataBytes shouldBe expected
  }

  it should "reject download on a non-read-only node when devMode is false" in {
    val (dir, hash, _) = seedFile()
    val request        = FileDownloadRequest(fileHash = hash)

    val ex = intercept[IllegalArgumentException] {
      collectChunks(request, isReadOnly = false, dir)
    }
    ex.getMessage should include("read-only f1r3node")
  }

  it should "allow download on a non-read-only node when devMode is true" in {
    val (dir, hash, expected) = seedFile()
    val request               = FileDownloadRequest(fileHash = hash)
    val chunks                = collectChunks(request, isReadOnly = false, dir, devMode = true)

    chunks.head.chunk.isMetadata shouldBe true
    val dataBytes = chunks.tail.flatMap(_.getData.toByteArray).toArray
    dataBytes shouldBe expected
  }

  it should "reject path traversal fileHash ../../etc/passwd" in {
    val dir     = newDir()
    val request = FileDownloadRequest(fileHash = "../../etc/passwd")

    val ex = intercept[IllegalArgumentException] {
      collectChunks(request, isReadOnly = true, dir)
    }
    ex.getMessage should include("INVALID_ARGUMENT")
  }

  it should "reject fileHash that is too short" in {
    val dir     = newDir()
    val request = FileDownloadRequest(fileHash = "abc123")

    val ex = intercept[IllegalArgumentException] {
      collectChunks(request, isReadOnly = true, dir)
    }
    ex.getMessage should include("INVALID_ARGUMENT")
  }

  it should "return NOT_FOUND when file does not exist on disk" in {
    val dir = newDir()
    val validHash =
      "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    val request = FileDownloadRequest(fileHash = validHash)

    val ex = intercept[IllegalArgumentException] {
      collectChunks(request, isReadOnly = true, dir)
    }
    ex.getMessage should include("NOT_FOUND")
  }

  it should "support resume via offset" in {
    val (dir, hash, expected) = seedFile()
    val offset                = 500L
    val request               = FileDownloadRequest(fileHash = hash, offset = offset)
    val chunks                = collectChunks(request, isReadOnly = true, dir)

    // Metadata is always first
    chunks.head.chunk.isMetadata shouldBe true

    // Data should start from offset
    val dataBytes = chunks.tail.flatMap(_.getData.toByteArray).toArray
    dataBytes shouldBe expected.drop(offset.toInt)
  }

  it should "reject the 5th concurrent download from the same IP when limit is 4" in {
    val (dir, hash, _) = seedFile()
    val request        = FileDownloadRequest(fileHash = hash)
    val ip             = "10.0.0.1"

    // streamFile acquires semaphore permits eagerly at call time.
    // Permits are released only when the observable is subscribed and terminates.
    // Calling streamFile without subscribing holds the permits indefinitely.
    val _held = (1 to 4).map { _ =>
      FileDownloadAPI
        .streamFile(request, isNodeReadOnly = true, dir, ip, maxConcurrentPerIp = 4)
    }

    // 5th download from same IP should be rejected
    val ex = intercept[IllegalArgumentException] {
      collectChunks(request, isReadOnly = true, dir, ip = ip, maxPerIp = 4)
    }
    ex.getMessage should include("RESOURCE_EXHAUSTED")
  }

  it should "allow concurrent downloads from different IPs" in {
    val (dir, hash, expected) = seedFile()
    val request               = FileDownloadRequest(fileHash = hash)

    // 5 sequential downloads from 5 different IPs — all should succeed (maxPerIp = 1)
    val results = (1 to 5).map { i =>
      collectChunks(request, isReadOnly = true, dir, ip = s"192.168.1.$i", maxPerIp = 1)
    }

    results.foreach { chunks =>
      chunks.head.chunk.isMetadata shouldBe true
      val dataBytes = chunks.tail.flatMap(_.getData.toByteArray).toArray
      dataBytes shouldBe expected
    }
  }
}
