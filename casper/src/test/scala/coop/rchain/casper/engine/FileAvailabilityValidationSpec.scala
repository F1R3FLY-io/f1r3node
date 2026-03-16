package coop.rchain.casper.engine

import cats.syntax.all._
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode._
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.casper.util.GenesisBuilder._
import coop.rchain.casper.{InvalidBlock, ValidBlock}
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.shared.scalatestcontrib._
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import java.nio.file.Files

/**
  * Integration test for P2P file replication during block validation.
  *
  * Uses `TestNode.networkEff` (2 nodes) to verify that the DA-gate in
  * `MultiParentCasperImpl.validate` correctly checks file availability
  * for blocks containing `rho:io:file` registration deploys.
  *
  * File replication is simulated via TestNode's shared `peerFileDirsRef` —
  * when Node B validates a block from Node A, the `daFetchFiles` callback
  * copies files directly from A's `file-replication/` directory to B's,
  * simulating P2P transfer within TestNode's synchronous model.
  */
class FileAvailabilityValidationSpec extends FlatSpec with Matchers {

  implicit val timeEff = new LogicalTime[Effect]

  private val genesis = buildGenesis()

  // A 64-char hex hash for test files
  private val testFileHash = "a" * 64

  // Content for the test file
  private val testFileContent = new Array[Byte](1024)
  java.util.Arrays.fill(testFileContent, 42.toByte)

  /** Rholang term for a file-registration deploy. */
  private def fileRegTerm(hash: String): String =
    s"""new ret, file(`rho:io:file`) in { file!("register", "$hash", 1024, "test.bin", *ret) }"""

  "Two-node DA-gate" should "replicate file from proposer to validator during block validation" in effectTest {
    TestNode.networkEff(genesis, networkSize = 2).use {
      case n1 +: n2 +: Seq() =>
        for {
          // File exists ONLY on n1 — n2 does NOT have it yet
          _ <- monix.eval.Task.delay {
                val dir1 = n1.dataDir.resolve("file-replication")
                Files.write(dir1.resolve(testFileHash), testFileContent)
              }

          // Verify file is NOT on n2 before block processing
          _ <- monix.eval.Task.delay {
                val fileOnN2 = n2.dataDir.resolve("file-replication").resolve(testFileHash)
                assert(!Files.exists(fileOnN2), "File should NOT exist on n2 before replication")
              }

          // n1 creates a block with a file-registration deploy
          deploy <- ConstructDeploy.sourceDeployNowF(
                     fileRegTerm(testFileHash),
                     shardId = genesis.genesisBlock.shardId
                   )
          block <- n1.addBlock(deploy)

          // n2 processes the block — DA-gate detects missing file,
          // daFetchFiles copies it from n1's dir, block is accepted
          result <- n2.processBlock(block)
          _      = result shouldBe Right(ValidBlock.Valid)

          // Verify the file was replicated to n2
          _ <- monix.eval.Task.delay {
                val fileOnN2 = n2.dataDir.resolve("file-replication").resolve(testFileHash)
                assert(
                  Files.exists(fileOnN2),
                  "File should be replicated to n2 after block validation"
                )

                // Verify content matches
                val contentOnN2 = Files.readAllBytes(fileOnN2)
                contentOnN2 shouldBe testFileContent
              }
        } yield ()
    }
  }

  it should "reject block when file is missing from BOTH nodes (no source to replicate from)" in effectTest {
    TestNode.networkEff(genesis, networkSize = 2).use {
      case n1 +: n2 +: Seq() =>
        for {
          // File does NOT exist on either node — impossible to replicate
          deploy <- ConstructDeploy.sourceDeployNowF(
                     fileRegTerm(testFileHash),
                     shardId = genesis.genesisBlock.shardId
                   )
          // Use createBlockUnsafe — it skips local validation, so n1 won't
          // reject the block for missing files (file isn't on n1 either)
          block <- n1.createBlockUnsafe(deploy)

          // n2 processes the block — DA-gate can't find the file anywhere, rejects
          result <- n2.processBlock(block)
          _      = result shouldBe Left(InvalidBlock.MissingFileData)
        } yield ()
    }
  }

  it should "accept block with normal (non-file) deploys without any DA overhead" in effectTest {
    TestNode.networkEff(genesis, networkSize = 2).use {
      case n1 +: n2 +: Seq() =>
        for {
          deploy <- ConstructDeploy.sourceDeployNowF(
                     "new x in { x!(42) }",
                     shardId = genesis.genesisBlock.shardId
                   )
          block  <- n1.addBlock(deploy)
          result <- n2.processBlock(block)
          _      = result shouldBe Right(ValidBlock.Valid)
        } yield ()
    }
  }

  it should "replicate file and propagate block via full publish flow" in effectTest {
    TestNode.networkEff(genesis, networkSize = 2).use {
      case n1 +: n2 +: Seq() =>
        for {
          // File exists ONLY on n1
          _ <- monix.eval.Task.delay {
                val dir1 = n1.dataDir.resolve("file-replication")
                Files.write(dir1.resolve(testFileHash), testFileContent)
              }

          deploy <- ConstructDeploy.sourceDeployNowF(
                     fileRegTerm(testFileHash),
                     shardId = genesis.genesisBlock.shardId
                   )

          // propagateBlock creates block on n1 and sends to n2 for processing
          block <- n1.propagateBlock(deploy)(n1, n2)

          // Verify n2 accepted the block into its DAG
          contained <- n2.casperEff.dagContains(block.blockHash)
          _         = contained shouldBe true

          // Verify the file was replicated to n2
          _ <- monix.eval.Task.delay {
                val fileOnN2 = n2.dataDir.resolve("file-replication").resolve(testFileHash)
                assert(Files.exists(fileOnN2), "File should be replicated to n2 via propagation")
                Files.readAllBytes(fileOnN2) shouldBe testFileContent
              }
        } yield ()
    }
  }
}
