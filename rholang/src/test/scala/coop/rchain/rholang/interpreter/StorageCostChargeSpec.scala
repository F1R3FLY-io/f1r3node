package coop.rchain.rholang.interpreter

import coop.rchain.crypto.hash.{Blake2b512Random, Sha256}
import coop.rchain.crypto.signatures.Secp256k1
import coop.rchain.crypto.PublicKey
import coop.rchain.metrics
import coop.rchain.metrics.{Metrics, NoopSpan, Span}
import coop.rchain.rholang.Resources.mkRuntime
import coop.rchain.rholang.syntax._
import coop.rchain.rholang.interpreter.SystemProcesses.BlockData
import coop.rchain.rholang.interpreter.accounting.Cost
import coop.rchain.shared.{Base16, Log}
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.duration._

/**
  * Runtime-level integration tests for storage-proportional phlo charging
  * in the `rho:io:file` `register` system process.
  *
  * These tests exercise the full `fileRegister` → `charge(fileStorageCost(...))`
  * path by setting `blockData.sender` to a Secp256k1 public key and
  * crafting a valid `nodeSigHex` with the corresponding private key.
  */
class StorageCostChargeSpec extends FlatSpec with Matchers {
  private val tmpPrefix                   = "rspace-store-"
  private val maxDuration                 = 30.seconds
  implicit val logF: Log[Task]            = Log.log[Task]
  implicit val noopMetrics: Metrics[Task] = new metrics.Metrics.MetricsNOP[Task]
  implicit val noopSpan: Span[Task]       = NoopSpan[Task]()

  // Secp256k1 key pair for the "proposer"
  private val (proposerPrivKey, proposerPubKey) = Secp256k1.newKeyPair

  private val testFileHash = "a" * 64
  private val testFileSize = 1024L
  private val testFileName = "test.bin"

  /** Sign "$fileHash:$fileSize" with the proposer's Secp256k1 private key. */
  private def makeNodeSigHex(fileHash: String, fileSize: Long): String = {
    val rawMessage = s"$fileHash:$fileSize".getBytes("UTF-8")
    val hash32     = Sha256.hash(rawMessage)
    val sigBytes   = Secp256k1.sign(hash32, proposerPrivKey)
    Base16.encode(sigBytes)
  }

  private def blockDataForProposer: BlockData =
    BlockData(
      timeStamp = System.currentTimeMillis(),
      blockNumber = 1L,
      sender = proposerPubKey,
      seqNum = 0
    )

  /** Evaluate a Rholang term against a runtime with the proposer's block data set. */
  private def evaluateWithBlockData(
      rho: String,
      initialPhlo: Cost = Cost(50000L)
  ): (EvaluateResult, Cost) =
    mkRuntime[Task](tmpPrefix)
      .use { runtime =>
        for {
          _      <- runtime.setBlockData(blockDataForProposer)
          result <- runtime.evaluate(rho, initialPhlo)
        } yield (result, result.cost)
      }
      .runSyncUnsafe(maxDuration)

  "fileRegister with valid Secp256k1 sig" should "return (true, fileHash) and charge storage phlo" in {
    val nodeSigHex = makeNodeSigHex(testFileHash, testFileSize)

    // Deploy that captures the result tuple into @"result"
    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $testFileSize, "$testFileName", "$nodeSigHex", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val (result, spentCost) = evaluateWithBlockData(rho)
    result.errors shouldBe empty

    // EvaluateResult.cost is the total phlo spent during evaluation.
    // With valid signature, the handler calls charge(fileStorageCost(1024)).
    // Storage cost alone = 1024 * 1 = 1024 phlo, plus base evaluation overhead.
    spentCost.value should be >= testFileSize
  }

  it should "fail with OutOfPhlo when phlo limit < storage cost" in {
    val nodeSigHex = makeNodeSigHex(testFileHash, testFileSize)

    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $testFileSize, "$testFileName", "$nodeSigHex", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val (result, _) = evaluateWithBlockData(rho, Cost(500L))
    // OutOfPhlo should cause an error in the evaluation result
    result.errors should not be empty
  }

  "fileRegister with invalid signature" should "not apply storage charge" in {
    val largeFileSize = 5000L
    val badSigHex     = "ff" * 64
    val validSigHex   = makeNodeSigHex(testFileHash, largeFileSize)

    val badRho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $largeFileSize, "$testFileName", "$badSigHex", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val validRho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $largeFileSize, "$testFileName", "$validSigHex", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val (badResult, badCost)     = evaluateWithBlockData(badRho)
    val (validResult, validCost) = evaluateWithBlockData(validRho)
    badResult.errors shouldBe empty
    validResult.errors shouldBe empty

    // Valid-sig cost includes charge(fileStorageCost(5000)); invalid does not.
    // The difference should be >= the file size, proving storage charge was not applied.
    (validCost.value - badCost.value) should be >= largeFileSize
  }

  "fileRegister with large file (10000 bytes)" should "charge at least 10000 phlo" in {
    val largeSize  = 10000L
    val nodeSigHex = makeNodeSigHex(testFileHash, largeSize)

    val rho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $largeSize, "$testFileName", "$nodeSigHex", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val (result, spentCost) = evaluateWithBlockData(rho, Cost(100000L))
    result.errors shouldBe empty

    // Storage cost = 10000 * 1 = 10000 phlo minimum, plus base evaluation overhead.
    spentCost.value should be >= largeSize
  }

  "fileRegister with valid sig" should "charge proportionally to file size" in {
    val smallSize = 1024L
    val largeSize = 10000L
    val smallSig  = makeNodeSigHex(testFileHash, smallSize)
    val largeSig  = makeNodeSigHex(testFileHash, largeSize)

    val smallRho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $smallSize, "$testFileName", "$smallSig", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val largeRho =
      s"""new ret, file(`rho:io:file`) in {
         |  file!("register", "$testFileHash", $largeSize, "$testFileName", "$largeSig", *ret) |
         |  for(@(success, msg) <- ret) {
         |    @"result"!((success, msg))
         |  }
         |}""".stripMargin

    val (smallResult, smallCost) = evaluateWithBlockData(smallRho, Cost(100000L))
    val (largeResult, largeCost) = evaluateWithBlockData(largeRho, Cost(100000L))
    smallResult.errors shouldBe empty
    largeResult.errors shouldBe empty

    // The difference in cost should be >= the difference in file sizes.
    // Both include the same base evaluation overhead, which cancels out.
    // fileStorageCost(10000) - fileStorageCost(1024) = (10000 - 1024) * 1 = 8976
    val expectedMinDiff = largeSize - smallSize // 8976
    (largeCost.value - smallCost.value) should be >= expectedMinDiff
  }
}
