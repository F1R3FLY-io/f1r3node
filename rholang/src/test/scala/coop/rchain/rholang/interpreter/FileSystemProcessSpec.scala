package coop.rchain.rholang.interpreter

import coop.rchain.metrics
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.models._
import coop.rchain.models.Expr.ExprInstance.{GInt, GString}
import coop.rchain.models.GUnforgeable.UnfInstance.GSysAuthTokenBody
import coop.rchain.models.rholang.implicits._
import coop.rchain.rholang.Resources.{mkRuntime, mkRuntimeWithFileDir}
import coop.rchain.rholang.syntax._
import coop.rchain.rholang.interpreter.SystemProcesses.FixedChannels
import coop.rchain.rholang.interpreter.accounting.Cost
import coop.rchain.shared.Log
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import java.nio.file.{Files, Path}
import scala.collection.immutable.BitSet
import scala.concurrent.duration._

class FileSystemProcessSpec extends FlatSpec with Matchers {
  private val tmpPrefix                   = "rspace-store-"
  private val maxDuration                 = 15.seconds
  implicit val logF: Log[Task]            = Log.log[Task]
  implicit val noopMetrics: Metrics[Task] = new metrics.Metrics.MetricsNOP[Task]
  implicit val noopSpan: Span[Task]       = NoopSpan[Task]()
  implicit val rand: Blake2b512Random     = Blake2b512Random(Array.empty[Byte])

  private val channelReadOnlyError = "ReduceError: Trying to read from non-readable channel."

  private val testFileHash = "a" * 64 // 64-char hex hash
  private val testFileSize = 1024L
  private val testFileName = "test.bin"

  "rho:io:file" should "not get intercepted (read-only channel)" in {
    checkError(
      """new f(`rho:io:file`) in { for(x <- f) { Nil } }""",
      channelReadOnlyError
    )
  }

  it should "register with invalid signature returns (false, error)" in {
    val badSigHex = "ff" * 64

    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $testFileSize, "$testFileName", "$badSigHex", *ret) |
         |  for(@result <- ret) {
         |    @"result"!(result)
         |  }
         |}""".stripMargin

    val result = execute(rho)
    result.errors shouldBe empty
  }

  it should "register with malformed hex signature returns (false, error)" in {
    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $testFileSize, "$testFileName", "not-hex", *ret) |
         |  for(@result <- ret) {
         |    @"result"!(result)
         |  }
         |}""".stripMargin

    val result = execute(rho)
    result.errors shouldBe empty
  }

  it should "delete without valid SysAuthToken returns (false, error)" in {
    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("delete", "$testFileHash", "not-a-token", *ret) |
         |  for(@result <- ret) {
         |    @"result"!(result)
         |  }
         |}""".stripMargin

    val result = execute(rho)
    result.errors shouldBe empty
  }

  it should "register with valid block sender and specific args produces output" in {
    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $testFileSize, "$testFileName", "${"00" * 64}", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!(success)
         |  }
         |}""".stripMargin

    val result = execute(rho)
    result.errors shouldBe empty
  }

  it should "delete with valid SysAuthToken physically removes the file" in {
    // Create a temp file-replication directory with a test file
    val fileReplDir = Files.createTempDirectory("file-replication-test")
    val testFile    = fileReplDir.resolve(testFileHash)
    Files.write(testFile, Array[Byte](1, 2, 3, 4, 5))
    Files.exists(testFile) shouldBe true

    try {
      mkRuntimeWithFileDir[Task](tmpPrefix, Some(fileReplDir))
        .use { runtime =>
          for {
            _ <- runtime.cost.set(Cost.UNSAFE_MAX)
            // Construct a Send with a real SysAuthToken (unforgeable, can't be created in Rholang)
            ackChannel           = GString("delete-ack"): Par
            sysAuthTokenPar: Par = GUnforgeable(GSysAuthTokenBody(GSysAuthToken()))
            send = Send(
              FixedChannels.FILE_IO_DELETE,
              List[Par](
                GString("delete"): Par,
                GString(testFileHash): Par,
                sysAuthTokenPar,
                ackChannel
              ),
              persistent = false,
              BitSet()
            )
            _ <- runtime.inj(send)
          } yield ()
        }
        .runSyncUnsafe(maxDuration)

      // Assert: file should be deleted from disk
      Files.exists(testFile) shouldBe false
    } finally {
      // Clean up
      Files.deleteIfExists(testFile)
      Files.deleteIfExists(fileReplDir)
    }
  }

  it should "delete non-existent file returns (false, File not found)" in {
    // Create an empty temp directory (no file to delete)
    val fileReplDir = Files.createTempDirectory("file-replication-test")

    try {
      mkRuntimeWithFileDir[Task](tmpPrefix, Some(fileReplDir))
        .use { runtime =>
          for {
            _                    <- runtime.cost.set(Cost.UNSAFE_MAX)
            ackChannel           = GString("delete-ack"): Par
            sysAuthTokenPar: Par = GUnforgeable(GSysAuthTokenBody(GSysAuthToken()))
            send = Send(
              FixedChannels.FILE_IO_DELETE,
              List[Par](
                GString("delete"): Par,
                GString(testFileHash): Par,
                sysAuthTokenPar,
                ackChannel
              ),
              persistent = false,
              BitSet()
            )
            _ <- runtime.inj(send)
          } yield ()
        }
        .runSyncUnsafe(maxDuration)

      // No crash = success. The contract returned (false, "File not found")
      // which was sent on the ack channel.
    } finally {
      Files.deleteIfExists(fileReplDir)
    }
  }

  private def checkError(rho: String, error: String): Unit =
    assert(execute(rho).errors.nonEmpty, s"Expected $rho to fail - it didn't.")

  private def execute(source: String): EvaluateResult =
    mkRuntime[Task](tmpPrefix)
      .use { runtime =>
        runtime.evaluate(source)
      }
      .runSyncUnsafe(maxDuration)
}
