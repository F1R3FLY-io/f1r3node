package coop.rchain.casper

import coop.rchain.casper.blocks.proposer.Created
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode.Effect
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.casper.util.GenesisBuilder._
import coop.rchain.crypto.hash.Sha256
import coop.rchain.crypto.signatures.Secp256k1
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.shared.Base16
import coop.rchain.shared.scalatestcontrib._
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

/**
  * Casper-level integration test for storage-proportional phlo pricing
  * during block execution.
  *
  * Uses `TestNode.standaloneEff` to create standalone validator nodes,
  * then deploys Rholang terms invoking `rho:io:file` `register`.
  *
  * The test extracts the genesis validator's Secp256k1 private key from
  * `GenesisContext.validatorKeyPairs` to construct a **valid** `nodeSigHex`,
  * enabling the `fileRegister` handler's storage charge path to execute.
  * This proves that `charge(fileStorageCost(...))` is consensus-enforced:
  * the deducted phlo appears in the block's processed deploy cost.
  *
  * IMPORTANT: Each comparison test uses a FRESH `TestNode` instance to
  * avoid state accumulation between blocks.
  */
class StorageCostEnforcementSpec extends FlatSpec with Matchers {

  implicit val timeEff: LogicalTime[Effect] = new LogicalTime[Effect]

  val genesis          = buildGenesis()
  private val SHARD_ID = genesis.genesisBlock.shardId

  // Extract the first validator's key pair — standaloneEff uses validatorSks.head
  private val (validatorSk, _) = genesis.validatorKeyPairs.head

  private val testFileHash = "a" * 64
  private val testFileName = "test.bin"

  /** Sign "fileHash:fileSize" with the validator's key (matching fileRegister verification). */
  private def makeNodeSigHex(fileHash: String, fileSize: Long): String = {
    val rawMessage = s"$fileHash:$fileSize".getBytes("UTF-8")
    val hash32     = Sha256.hash(rawMessage)
    val sigBytes   = Secp256k1.sign(hash32, validatorSk)
    Base16.encode(sigBytes)
  }

  private def fileRegisterTerm(fileSize: Long, nodeSigHex: String): String =
    s"""new ret, file(`rho:io:file`) in {
       |  file!("register", "$testFileHash", $fileSize, "$testFileName", "$nodeSigHex", *ret) |
       |  for(@result <- ret) {
       |    @"result"!(result)
       |  }
       |}""".stripMargin

  // --- Core tests: valid sig, storage charge fires ---

  "A deploy invoking rho:io:file register with valid validator sig" should
    "succeed and deduct storage-proportional phlo" in effectTest {
    val fileSize   = 1024L
    val nodeSigHex = makeNodeSigHex(testFileHash, fileSize)

    TestNode.standaloneEff(genesis).use { node =>
      for {
        deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                   fileRegisterTerm(fileSize, nodeSigHex),
                   phloLimit = 100000,
                   shardId = SHARD_ID
                 )
        r              <- node.createBlock(deploy)
        Created(block) = r
      } yield {
        block.body.deploys should not be empty
        val processedDeploy = block.body.deploys.head
        processedDeploy.isFailed shouldBe false

        // The cost recorded in the block must include the storage charge.
        // charge(fileStorageCost(1024)) = 1024 phlo, plus evaluation overhead.
        processedDeploy.cost.cost should be >= fileSize
      }
    }
  }

  it should "fail with OutOfPhlo when phlo limit is below storage cost" in effectTest {
    val fileSize   = 10000L
    val nodeSigHex = makeNodeSigHex(testFileHash, fileSize)

    TestNode.standaloneEff(genesis).use { node =>
      for {
        deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                   fileRegisterTerm(fileSize, nodeSigHex),
                   phloLimit = 500, // Not enough for 10000 storage + eval overhead
                   shardId = SHARD_ID
                 )
        r              <- node.createBlock(deploy)
        Created(block) = r
      } yield {
        block.body.deploys should not be empty
        block.body.deploys.head.isFailed shouldBe true
      }
    }
  }

  // --- Proportional test: uses separate nodes to avoid state accumulation ---

  it should "charge proportionally: larger file costs more" in effectTest {
    val smallFileSize = 1024L
    val largeFileSize = 10000L
    val smallSigHex   = makeNodeSigHex(testFileHash, smallFileSize)
    val largeSigHex   = makeNodeSigHex(testFileHash, largeFileSize)

    for {
      smallCost <- TestNode.standaloneEff(genesis).use { node =>
                    for {
                      deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                                 fileRegisterTerm(smallFileSize, smallSigHex),
                                 phloLimit = 100000,
                                 shardId = SHARD_ID
                               )
                      r              <- node.createBlock(deploy)
                      Created(block) = r
                    } yield {
                      block.body.deploys.head.isFailed shouldBe false
                      block.body.deploys.head.cost.cost
                    }
                  }

      largeCost <- TestNode.standaloneEff(genesis).use { node =>
                    for {
                      deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                                 fileRegisterTerm(largeFileSize, largeSigHex),
                                 phloLimit = 100000,
                                 shardId = SHARD_ID
                               )
                      r              <- node.createBlock(deploy)
                      Created(block) = r
                    } yield {
                      block.body.deploys.head.isFailed shouldBe false
                      block.body.deploys.head.cost.cost
                    }
                  }
    } yield {
      // Larger file should cost strictly more.
      largeCost should be > smallCost
      // The difference should be close to the file size difference (8976 phlo).
      // Allow small tolerance for evaluation overhead noise.
      val expectedDiff = largeFileSize - smallFileSize // 8976
      (largeCost - smallCost).toDouble should be >= (expectedDiff * 0.99)
    }
  }

  // --- Invalid signature: no storage charge ---

  "A deploy with invalid signature" should "succeed but cost less (no storage charge)" in effectTest {
    val fileSize    = 5000L
    val validSigHex = makeNodeSigHex(testFileHash, fileSize)
    val badSigHex   = "ff" * 64

    for {
      validCost <- TestNode.standaloneEff(genesis).use { node =>
                    for {
                      deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                                 fileRegisterTerm(fileSize, validSigHex),
                                 phloLimit = 100000,
                                 shardId = SHARD_ID
                               )
                      r              <- node.createBlock(deploy)
                      Created(block) = r
                    } yield {
                      block.body.deploys.head.isFailed shouldBe false
                      block.body.deploys.head.cost.cost
                    }
                  }

      badCost <- TestNode.standaloneEff(genesis).use { node =>
                  for {
                    deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                               fileRegisterTerm(fileSize, badSigHex),
                               phloLimit = 100000,
                               shardId = SHARD_ID
                             )
                    r              <- node.createBlock(deploy)
                    Created(block) = r
                  } yield {
                    block.body.deploys.head.isFailed shouldBe false
                    block.body.deploys.head.cost.cost
                  }
                }
    } yield {
      // Valid sig includes storage charge; bad sig does not.
      // The difference should be >= fileSize (the storage charge amount).
      (validCost - badCost) should be >= fileSize
    }
  }

  it should "fail when phlo limit is extremely low (1 phlo)" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      for {
        deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                   fileRegisterTerm(1024L, "ff" * 64),
                   phloLimit = 1,
                   shardId = SHARD_ID
                 )
        r              <- node.createBlock(deploy)
        Created(block) = r
      } yield {
        block.body.deploys should not be empty
        block.body.deploys.head.isFailed shouldBe true
      }
    }
  }

  // --- Baseline: file register costs more than Nil ---

  "A plain Nil deploy" should "cost less than a file-register deploy" in effectTest {
    val fileSize   = 1024L
    val nodeSigHex = makeNodeSigHex(testFileHash, fileSize)

    for {
      nilCost <- TestNode.standaloneEff(genesis).use { node =>
                  for {
                    deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                               "Nil",
                               phloLimit = 100000,
                               shardId = SHARD_ID
                             )
                    r                 <- node.createBlock(deploy)
                    Created(nilBlock) = r
                  } yield nilBlock.body.deploys.head.cost.cost
                }

      fileCost <- TestNode.standaloneEff(genesis).use { node =>
                   for {
                     deploy <- ConstructDeploy.sourceDeployNowF[Effect](
                                fileRegisterTerm(fileSize, nodeSigHex),
                                phloLimit = 100000,
                                shardId = SHARD_ID
                              )
                     r                  <- node.createBlock(deploy)
                     Created(fileBlock) = r
                   } yield fileBlock.body.deploys.head.cost.cost
                 }
    } yield {
      fileCost should be > nilCost
    }
  }
}
