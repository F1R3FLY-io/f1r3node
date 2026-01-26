package coop.rchain.casper.batch2

import cats.instances.list._
import cats.syntax.traverse._
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.util.ConstructDeploy.basicDeployData
import coop.rchain.casper.util.GenesisBuilder.buildGenesis
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import monix.eval.Task
import monix.execution.Scheduler
import org.scalatest.{FlatSpec, Matchers}

class LimitedParentDepthSpec extends FlatSpec with Matchers {
  implicit val scheduler = Scheduler.fixedPool("limited-parent-depth-scheduler", 2)
  implicit val timeEff   = new LogicalTime[Task]

  val genesisContext = buildGenesis()

  "Estimator" should "obey present parent depth limitation" in {
    val t = TestNode.networkEff(genesisContext, networkSize = 2, maxParentDepth = Some(2)).use {
      case nodes @ n1 +: n2 +: Seq() =>
        for {
          produceDeploys <- (0 until 6).toList
                             .traverse(i => basicDeployData[Task](i, shardId = "root"))

          b1 <- n1.propagateBlock(produceDeploys(0))()
          b2 <- n2.propagateBlock(produceDeploys(1))(nodes: _*)
          b4 <- n2.propagateBlock(produceDeploys(2))(nodes: _*)
          b4 <- n2.propagateBlock(produceDeploys(3))(nodes: _*)
          b5 <- n2.propagateBlock(produceDeploys(4))(nodes: _*)
          b6 <- n1.propagateBlock(produceDeploys(5))(nodes: _*)
        } yield b6.header.parentsHashList shouldBe List(b5.blockHash)
    }
    t.runSyncUnsafe()
  }

  it should "obey absent parent depth limitation" in {
    val t = TestNode.networkEff(genesisContext, networkSize = 2, maxParentDepth = None).use {
      case nodes @ n1 +: n2 +: Seq() =>
        for {
          produceDeploys <- (0 until 6).toList
                             .traverse(i => basicDeployData[Task](i, shardId = "root"))

          b1 <- n1.propagateBlock(produceDeploys(0))()
          b2 <- n2.propagateBlock(produceDeploys(1))(nodes: _*)
          b4 <- n2.propagateBlock(produceDeploys(2))(nodes: _*)
          b4 <- n2.propagateBlock(produceDeploys(3))(nodes: _*)
          b5 <- n2.propagateBlock(produceDeploys(4))(nodes: _*)
          b6 <- n1.propagateBlock(produceDeploys(5))(nodes: _*)
        } yield {
          // With multi-parent merging, all validators' latest blocks are included as parents.
          // Genesis has 4 validators, only 2 nodes create blocks, so validators 2 and 3 still
          // have genesis as their latest message. This results in 3 unique parents:
          // - b1 (from validator 0)
          // - b5 (from validator 1)
          // - genesis (from validators 2 and 3 who haven't created blocks)
          b6.header.parentsHashList should have size 3
          b6.header.parentsHashList should contain(b1.blockHash)
          b6.header.parentsHashList should contain(b5.blockHash)
        }
    }
    t.runSyncUnsafe()
  }
}
