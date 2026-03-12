package coop.rchain.casper.batch1

import cats.syntax.all._
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode._
import coop.rchain.casper.protocol.BlockMessage
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.shared.scalatestcontrib._
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Inspectors, Matchers}

class MultiParentCasperFinalizationSpec extends FlatSpec with Matchers with Inspectors {

  import coop.rchain.casper.util.GenesisBuilder._

  implicit val timeEff = new LogicalTime[Effect]

  val genesis = buildGenesis(
    buildGenesisParameters(bondsFunction = _.map(pk => pk -> 10L).toMap)
  )

  "MultiParentCasper" should "advance finalization monotonically in round robin" in effectTest {
    TestNode.networkEff(genesis, networkSize = 3).use { nodes =>
      for {
        deployDatas <- (0 to 7).toList.traverse(
                        i =>
                          ConstructDeploy
                            .basicDeployData[Effect](i, shardId = genesis.genesisBlock.shardId)
                      )

        // Round-robin block production across 3 validators with propagation.
        // With multi-parent merging, each block may have multiple parents
        // (one from each validator's latest message).
        block1 <- nodes(0).propagateBlock(deployDatas(0))(nodes: _*)
        block2 <- nodes(1).propagateBlock(deployDatas(1))(nodes: _*)
        block3 <- nodes(2).propagateBlock(deployDatas(2))(nodes: _*)
        block4 <- nodes(0).propagateBlock(deployDatas(3))(nodes: _*)
        block5 <- nodes(1).propagateBlock(deployDatas(4))(nodes: _*)

        // After 5 blocks in round-robin with 3 validators (equal bonds),
        // finalization should have advanced past genesis.
        lfbAfter5 <- nodes(0).casperEff.lastFinalizedBlock

        block6 <- nodes(2).propagateBlock(deployDatas(5))(nodes: _*)

        lfbAfter6 <- nodes(0).casperEff.lastFinalizedBlock

        block7 <- nodes(0).propagateBlock(deployDatas(6))(nodes: _*)

        lfbAfter7 <- nodes(0).casperEff.lastFinalizedBlock

        block8 <- nodes(1).propagateBlock(deployDatas(7))(nodes: _*)

        lfbAfter8 <- nodes(0).casperEff.lastFinalizedBlock

        // Verify finalization has advanced past genesis
        _ = lfbAfter5 should not be genesis.genesisBlock

        // Verify finalization advances monotonically (block number never decreases)
        _ = lfbAfter6.body.state.blockNumber should be >= lfbAfter5.body.state.blockNumber
        _ = lfbAfter7.body.state.blockNumber should be >= lfbAfter6.body.state.blockNumber
        _ = lfbAfter8.body.state.blockNumber should be >= lfbAfter7.body.state.blockNumber

        // Verify finalization has advanced meaningfully by the end
        // With 8 blocks from 3 validators, LFB should be well past block 1
        _ = lfbAfter8.body.state.blockNumber should be >= 2L

        // Verify all validators agree on finalization
        lfbNode1 <- nodes(1).casperEff.lastFinalizedBlock
        lfbNode2 <- nodes(2).casperEff.lastFinalizedBlock
        _        = lfbNode1 shouldBe lfbAfter8
        _        = lfbNode2 shouldBe lfbAfter8
      } yield ()
    }
  }
}
