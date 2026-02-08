package coop.rchain.casper.batch1

import cats.syntax.all._
import coop.rchain.casper.ValidBlock
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode._
import coop.rchain.casper.util.{ConstructDeploy, RSpaceUtil}
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.shared.scalatestcontrib._
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Inspectors, Matchers}

class MultiParentCasperMergeSpec extends FlatSpec with Matchers with Inspectors {

  import RSpaceUtil._
  import coop.rchain.casper.util.GenesisBuilder._

  implicit val timeEff = new LogicalTime[Effect]

  val genesisParams = buildGenesisParameters(validatorsNum = 3)
  val genesis       = buildGenesis(genesisParams)

  "HashSetCasper" should "handle multi-parent blocks correctly" in effectTest {
    TestNode.networkEff(genesis, networkSize = 3).use { nodes =>
      implicit val rm = nodes(1).runtimeManager
      val shardId     = genesis.genesisBlock.shardId
      for {
        deployData0 <- ConstructDeploy.basicDeployData[Effect](
                        0,
                        sec = ConstructDeploy.defaultSec2,
                        shardId = shardId
                      )
        deployData1 <- ConstructDeploy
                        .sourceDeployNowF("@1!(1) | for(@x <- @1){ @1!(x) }", shardId = shardId)
        deployData2 <- ConstructDeploy.basicDeployData[Effect](2, shardId = shardId)
        deploys = Vector(
          deployData0,
          deployData1,
          deployData2
        )
        block0 <- nodes(0).addBlock(deploys(0))
        block1 <- nodes(1).addBlock(deploys(1))
        _      <- TestNode.propagate(nodes)

        _ <- nodes(0).blockDagStorage.getRepresentation
              .flatMap(_.isFinalized(genesis.genesisBlock.blockHash)) shouldBeF true
        _ <- nodes(0).blockDagStorage.getRepresentation
              .flatMap(_.isFinalized(block0.blockHash)) shouldBeF false
        _ <- nodes(0).blockDagStorage.getRepresentation
              .flatMap(_.isFinalized(block1.blockHash)) shouldBeF false

        //multiparent block joining block0 and block1 since they do not conflict
        multiparentBlock <- nodes(0).propagateBlock(deploys(2))(nodes: _*)

        _ = block0.header.parentsHashList shouldBe Seq(genesis.genesisBlock.blockHash)
        _ = block1.header.parentsHashList shouldBe Seq(genesis.genesisBlock.blockHash)
        // With multi-parent merging, all validators' latest blocks are included as parents
        // (block0 from node0, block1 from node1, genesis from node2 who hasn't created a block yet)
        _ = multiparentBlock.header.parentsHashList.size shouldBe 3
        _ <- nodes(0).contains(multiparentBlock.blockHash) shouldBeF true
        _ <- nodes(1).contains(multiparentBlock.blockHash) shouldBeF true
        _ = multiparentBlock.body.rejectedDeploys.size shouldBe 0
        _ <- getDataAtPublicChannel[Effect](multiparentBlock, 0).map(_ shouldBe Seq("0"))
        _ <- getDataAtPublicChannel[Effect](multiparentBlock, 1).map(_ shouldBe Seq("1"))
        _ <- getDataAtPublicChannel[Effect](multiparentBlock, 2).map(_ shouldBe Seq("2"))
      } yield ()
    }
  }

  it should "not produce UnusedCommEvent while merging non conflicting blocks in the presence of conflicting ones" in effectTest {

    val registryRho =
      """
        |// Expected output
        |//
        |// "REGISTRY_SIMPLE_INSERT_TEST: create arbitrary process X to store in the registry"
        |// Unforgeable(0xd3f4cbdcc634e7d6f8edb05689395fef7e190f68fe3a2712e2a9bbe21eb6dd10)
        |// "REGISTRY_SIMPLE_INSERT_TEST: adding X to the registry and getting back a new identifier"
        |// `rho:id:pnrunpy1yntnsi63hm9pmbg8m1h1h9spyn7zrbh1mcf6pcsdunxcci`
        |// "REGISTRY_SIMPLE_INSERT_TEST: got an identifier for X from the registry"
        |// "REGISTRY_SIMPLE_LOOKUP_TEST: looking up X in the registry using identifier"
        |// "REGISTRY_SIMPLE_LOOKUP_TEST: got X from the registry using identifier"
        |// Unforgeable(0xd3f4cbdcc634e7d6f8edb05689395fef7e190f68fe3a2712e2a9bbe21eb6dd10)
        |
        |new simpleInsertTest, simpleInsertTestReturnID, simpleLookupTest,
        |    signedInsertTest, signedInsertTestReturnID, signedLookupTest,
        |    ri(`rho:registry:insertArbitrary`),
        |    rl(`rho:registry:lookup`),
        |    stdout(`rho:io:stdout`),
        |    stdoutAck(`rho:io:stdoutAck`), ack in {
        |        simpleInsertTest!(*simpleInsertTestReturnID) |
        |        for(@idFromTest1 <- simpleInsertTestReturnID) {
        |            simpleLookupTest!(idFromTest1, *ack)
        |        } |
        |
        |        contract simpleInsertTest(registryIdentifier) = {
        |            stdout!("REGISTRY_SIMPLE_INSERT_TEST: create arbitrary process X to store in the registry") |
        |            new X, Y, innerAck in {
        |                stdoutAck!(*X, *innerAck) |
        |                for(_ <- innerAck){
        |                    stdout!("REGISTRY_SIMPLE_INSERT_TEST: adding X to the registry and getting back a new identifier") |
        |                    ri!(*X, *Y) |
        |                    for(@uri <- Y) {
        |                        stdout!("REGISTRY_SIMPLE_INSERT_TEST: got an identifier for X from the registry") |
        |                        stdout!(uri) |
        |                        registryIdentifier!(uri)
        |                    }
        |                }
        |            }
        |        } |
        |
        |        contract simpleLookupTest(@uri, result) = {
        |            stdout!("REGISTRY_SIMPLE_LOOKUP_TEST: looking up X in the registry using identifier") |
        |            new lookupResponse in {
        |                rl!(uri, *lookupResponse) |
        |                for(@val <- lookupResponse) {
        |                    stdout!("REGISTRY_SIMPLE_LOOKUP_TEST: got X from the registry using identifier") |
        |                    stdoutAck!(val, *result)
        |                }
        |            }
        |        }
        |    }
      """.stripMargin

    val tuplesRho =
      """
        |// tuples only support random access
        |new stdout(`rho:io:stdout`) in {
        |
        |  // prints 2 because tuples are 0-indexed
        |  stdout!((1,2,3).nth(1))
        |}
      """.stripMargin
    val timeRho =
      """
        |new getBlockData(`rho:block:data`), stdout(`rho:io:stdout`), tCh in {
        |  getBlockData!(*tCh) |
        |  for(@_, @t, @_ <- tCh) {
        |    match t {
        |      Nil => { stdout!("no block time; no blocks yet? Not connected to Casper network?") }
        |      _ => { stdout!({"block time": t}) }
        |    }
        |  }
        |}
      """.stripMargin

    TestNode.networkEff(genesis, networkSize = 3).use { nodes =>
      val shardId = genesis.genesisBlock.shardId
      val n1      = nodes(0)
      val n2      = nodes(1)
      val n3      = nodes(2)
      val short   = ConstructDeploy.sourceDeploy("new x in { x!(0) }", 1L, shardId = shardId)
      val time    = ConstructDeploy.sourceDeploy(timeRho, 3L, shardId = shardId)
      val tuples  = ConstructDeploy.sourceDeploy(tuplesRho, 2L, shardId = shardId)
      val reg     = ConstructDeploy.sourceDeploy(registryRho, 4L, shardId = shardId)
      for {
        b1n3 <- n3.addBlock(short)
        b1n2 <- n2.addBlock(time)
        b1n1 <- n1.addBlock(tuples)
        _    <- n2.handleReceive()
        b2n2 <- n2.createBlock(reg)
      } yield ()
    }
  }

  it should "compute identical post-states across validators for merge blocks" in effectTest {
    // This test verifies the determinism fix for LCA computation and merge ordering.
    // Before the fix, validators could compute different post-states for the same
    // merge block due to non-deterministic ancestor traversal (isFinalized boundary)
    // and ordering (hashCode, Set.head). This test creates a multi-round scenario
    // where each round forces a multi-parent merge and verifies all nodes agree.
    TestNode.networkEff(genesis, networkSize = 3).use { nodes =>
      val shardId = genesis.genesisBlock.shardId
      for {
        // Round 1: Create divergent blocks on all three validators
        d0 <- ConstructDeploy.sourceDeployNowF("@10!(1)", shardId = shardId)
        d1 <- ConstructDeploy.sourceDeployNowF(
               "@20!(2)",
               sec = ConstructDeploy.defaultSec2,
               shardId = shardId
             )
        b0 <- nodes(0).addBlock(d0)
        b1 <- nodes(1).addBlock(d1)
        _  <- TestNode.propagate(nodes)

        // Merge block from node2 -- must have both b0 and b1 as parents
        d2         <- ConstructDeploy.sourceDeployNowF("@30!(3)", shardId = shardId)
        mergeBlock <- nodes(2).propagateBlock(d2)(nodes: _*)

        // All validators must have the same post-state for the merge block
        _ <- nodes(0).contains(mergeBlock.blockHash) shouldBeF true
        _ <- nodes(1).contains(mergeBlock.blockHash) shouldBeF true
        _ <- nodes(2).contains(mergeBlock.blockHash) shouldBeF true

        // Round 2: Another round of divergent blocks + merge
        d3 <- ConstructDeploy.sourceDeployNowF("@40!(4)", shardId = shardId)
        d4 <- ConstructDeploy.sourceDeployNowF(
               "@50!(5)",
               sec = ConstructDeploy.defaultSec2,
               shardId = shardId
             )
        b3 <- nodes(0).addBlock(d3)
        b4 <- nodes(1).addBlock(d4)
        _  <- TestNode.propagate(nodes)

        d5          <- ConstructDeploy.sourceDeployNowF("@60!(6)", shardId = shardId)
        mergeBlock2 <- nodes(2).propagateBlock(d5)(nodes: _*)

        _ <- nodes(0).contains(mergeBlock2.blockHash) shouldBeF true
        _ <- nodes(1).contains(mergeBlock2.blockHash) shouldBeF true
        _ <- nodes(2).contains(mergeBlock2.blockHash) shouldBeF true

        // Verify no deploys were rejected (non-conflicting channels)
        _ = mergeBlock.body.rejectedDeploys.size shouldBe 0
        _ = mergeBlock2.body.rejectedDeploys.size shouldBe 0
      } yield ()
    }
  }

  it should "produce identical merge results regardless of finalization state divergence" in effectTest {
    // Regression test for the InvalidBondsCache bug.
    //
    // Scenario: Two validators have the same DAG structure but different finalization
    // states. With the old code (isFinalized-bounded ancestor traversal), they would
    // compute different ancestor sets, different LCAs, and different post-state hashes
    // for the same block -- causing the receiving validator to reject the block with
    // InvalidBondsCache. With the Phase 1 fix (allAncestors), finalization state is
    // irrelevant to the merge computation, so both validators accept the block.
    TestNode.networkEff(genesis, networkSize = 3).use { nodes =>
      val shardId = genesis.genesisBlock.shardId
      for {
        // Create divergent blocks on two validators
        d0 <- ConstructDeploy.sourceDeployNowF("@100!(1)", shardId = shardId)
        d1 <- ConstructDeploy.sourceDeployNowF(
               "@200!(2)",
               sec = ConstructDeploy.defaultSec2,
               shardId = shardId
             )
        block0 <- nodes(0).addBlock(d0)
        block1 <- nodes(1).addBlock(d1)
        _      <- TestNode.propagate(nodes)

        // All nodes have the same DAG: genesis -> {block0, block1}
        _ <- nodes(0).contains(block0.blockHash) shouldBeF true
        _ <- nodes(0).contains(block1.blockHash) shouldBeF true
        _ <- nodes(1).contains(block0.blockHash) shouldBeF true
        _ <- nodes(1).contains(block1.blockHash) shouldBeF true

        // Advance finalization on node0 to block0 (node1 does NOT finalize block0)
        _ <- nodes(0).blockDagStorage
              .recordDirectlyFinalized(block0.blockHash, _ => ().pure[Effect])

        // Verify divergent finalization state
        _ <- nodes(0).blockDagStorage.getRepresentation
              .flatMap(_.isFinalized(block0.blockHash)) shouldBeF true
        _ <- nodes(1).blockDagStorage.getRepresentation
              .flatMap(_.isFinalized(block0.blockHash)) shouldBeF false

        // Node2 creates a merge block (node2 has NOT finalized block0 either)
        d2         <- ConstructDeploy.sourceDeployNowF("@300!(3)", shardId = shardId)
        mergeBlock <- nodes(2).createBlockUnsafe(d2)

        // Process merge block on node2 (self-validate, no finalization advance)
        status2 <- nodes(2).processBlock(mergeBlock)
        _       = status2 shouldBe ValidBlock.Valid.asRight

        // Process the same merge block on node0 (HAS finalized block0)
        status0 <- nodes(0).processBlock(mergeBlock)
        // With the old code, this would fail with InvalidBondsCache because node0's
        // finalization-bounded ancestor traversal would produce a different LCA.
        _ = status0 shouldBe ValidBlock.Valid.asRight

        // Process the same merge block on node1 (has NOT finalized block0)
        status1 <- nodes(1).processBlock(mergeBlock)
        _       = status1 shouldBe ValidBlock.Valid.asRight
      } yield ()
    }
  }

  it should "not merge blocks that touch the same channel involving joins" ignore effectTest {
    TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      for {
        current0 <- timeEff.currentMillis
        deploy0 = ConstructDeploy.sourceDeploy(
          "@1!(47)",
          current0,
          sec = ConstructDeploy.defaultSec2
        )
        current1 <- timeEff.currentMillis
        deploy1 = ConstructDeploy.sourceDeploy(
          "for(@x <- @1 & @y <- @2){ @1!(x) }",
          current1
        )
        deploy2 <- ConstructDeploy.basicDeployData[Effect](2)
        deploys = Vector(
          deploy0,
          deploy1,
          deploy2
        )

        block0 <- nodes(0).addBlock(deploys(0))
        block1 <- nodes(1).addBlock(deploys(1))
        _      <- TestNode.propagate(nodes)

        singleParentBlock <- nodes(0).addBlock(deploys(2))
        _                 <- nodes(1).handleReceive()

        _ = singleParentBlock.header.parentsHashList.size shouldBe 1
        _ <- nodes(0).contains(singleParentBlock.blockHash) shouldBeF true
        _ <- nodes(1).knowsAbout(singleParentBlock.blockHash) shouldBeF true
      } yield ()
    }
  }
}
