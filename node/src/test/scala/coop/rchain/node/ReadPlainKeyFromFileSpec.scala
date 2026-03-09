package coop.rchain.node

import cats.effect.Sync
import cats.effect.laws.util.TestContext
import coop.rchain.crypto.PrivateKey
import coop.rchain.shared.Base16
import monix.eval.Task
import monix.execution.Scheduler
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

import java.nio.file.{Files, Path, Paths}

class ReadPlainKeyFromFileSpec extends FunSuite with Matchers with BeforeAndAfterAll {
  implicit val scheduler: Scheduler = Scheduler.Implicits.global

  // Sample private key for testing (64 hex characters = 32 bytes)
  val validPrivateKeyHex   = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
  val validPrivateKeyBytes = Base16.decode(validPrivateKeyHex).get

  var tempDir: Path = _

  override def beforeAll(): Unit =
    tempDir = Files.createTempDirectory("rnode-test-keys")

  override def afterAll(): Unit = {
    // Clean up temp directory
    import scala.collection.JavaConverters._
    Files
      .walk(tempDir)
      .iterator()
      .asScala
      .toSeq
      .reverse
      .foreach(Files.deleteIfExists)
  }

  // Helper to create a test key file
  def createKeyFile(content: String, filename: String = "test.key"): Path = {
    val keyFile = tempDir.resolve(filename)
    Files.write(keyFile, content.getBytes)
    keyFile
  }

  // Access to the private method via reflection
  def readPlainKeyFromFile[F[_]: Sync](keyPath: Path): F[PrivateKey] = {
    val mainClass  = Class.forName("coop.rchain.node.Main$")
    val mainObject = mainClass.getField("MODULE$").get(null)
    val method = mainClass.getDeclaredMethod(
      "readPlainKeyFromFile",
      classOf[Path],
      classOf[Sync[F]]
    )
    method.setAccessible(true)
    method.invoke(mainObject, keyPath, implicitly[Sync[F]]).asInstanceOf[F[PrivateKey]]
  }

  test("readPlainKeyFromFile should read a valid hex private key") {
    val keyFile = createKeyFile(validPrivateKeyHex)

    val result = readPlainKeyFromFile[Task](keyFile).runSyncUnsafe()

    result.bytes shouldEqual validPrivateKeyBytes
  }

  test("readPlainKeyFromFile should handle hex key with whitespace") {
    val keyWithWhitespace = s"  \n  $validPrivateKeyHex  \n  "
    val keyFile           = createKeyFile(keyWithWhitespace)

    val result = readPlainKeyFromFile[Task](keyFile).runSyncUnsafe()

    result.bytes shouldEqual validPrivateKeyBytes
  }

  test("readPlainKeyFromFile should handle hex key with newlines") {
    val keyWithNewlines = validPrivateKeyHex.grouped(16).mkString("\n")
    val keyFile         = createKeyFile(keyWithNewlines)

    val result = readPlainKeyFromFile[Task](keyFile).runSyncUnsafe()

    result.bytes shouldEqual validPrivateKeyBytes
  }

  test("readPlainKeyFromFile should fail on invalid hex") {
    val invalidHex = "not a hex string!!!"
    val keyFile    = createKeyFile(invalidHex)

    val exception = intercept[Exception] {
      readPlainKeyFromFile[Task](keyFile).runSyncUnsafe()
    }

    exception.getMessage should include("Invalid private key file format")
    exception.getMessage should include("Expected base16-encoded (hex) private key")
  }

  test("readPlainKeyFromFile should fail on non-existent file") {
    val nonExistentFile = tempDir.resolve("does-not-exist.key")

    val exception = intercept[Exception] {
      readPlainKeyFromFile[Task](nonExistentFile).runSyncUnsafe()
    }

    // Should be a file not found exception
    exception.getMessage should include("does-not-exist.key")
  }

  test("readPlainKeyFromFile should fail on empty file") {
    val emptyFile = createKeyFile("")

    val exception = intercept[Exception] {
      readPlainKeyFromFile[Task](emptyFile).runSyncUnsafe()
    }

    exception.getMessage should include("Invalid private key file format")
  }

  test("readPlainKeyFromFile should handle uppercase hex") {
    val uppercaseHex = validPrivateKeyHex.toUpperCase
    val keyFile      = createKeyFile(uppercaseHex)

    val result = readPlainKeyFromFile[Task](keyFile).runSyncUnsafe()

    result.bytes shouldEqual validPrivateKeyBytes
  }

  test("readPlainKeyFromFile should handle mixed case hex") {
    val mixedCaseHex  = "1234567890AbCdEf1234567890aBcDeF1234567890abCDEF1234567890ABCDEF"
    val expectedBytes = Base16.decode(mixedCaseHex).get
    val keyFile       = createKeyFile(mixedCaseHex)

    val result = readPlainKeyFromFile[Task](keyFile).runSyncUnsafe()

    result.bytes shouldEqual expectedBytes
  }
}
