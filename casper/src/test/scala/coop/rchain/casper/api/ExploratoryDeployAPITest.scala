package coop.rchain.casper.api

import cats.effect.{Concurrent, Sync}
import cats.syntax.all._
import coop.rchain.blockstorage.BlockStore
import coop.rchain.casper.helper.{BlockDagStorageFixture, BlockGenerator, TestNode}
import coop.rchain.casper.util.GenesisBuilder.{buildGenesis, buildGenesisParameters}
import coop.rchain.shared.scalatestcontrib.effectTest
import coop.rchain.casper.engine.Engine
import coop.rchain.casper.SafetyOracle
import coop.rchain.casper.helper.BlockGenerator._
import coop.rchain.casper.util.ConstructDeploy.{basicDeployData, sourceDeployNowF}
import coop.rchain.metrics.{Metrics, Span}
import coop.rchain.shared.{Cell, Log}
import coop.rchain.models._
import coop.rchain.models.Expr.ExprInstance.GString
import coop.rchain.casper.PrettyPrinter
import coop.rchain.casper.batch2.EngineWithCasper
import monix.eval.Task
import org.scalatest.{EitherValues, FlatSpec, Matchers}
import monix.execution.Scheduler.Implicits.global

class ExploratoryDeployAPITest
    extends FlatSpec
    with Matchers
    with EitherValues
    with BlockGenerator
    with BlockDagStorageFixture {
  implicit val metricsEff = new Metrics.MetricsNOP[Task]
  implicit val spanEff    = Span.noop[Task]
  val genesisParameters   = buildGenesisParameters(bondsFunction = _.zip(List(10L, 10L, 10L)).toMap)
  val genesisContext      = buildGenesis(genesisParameters)

  def exploratoryDeploy(term: String)(engineCell: Cell[Task, Engine[Task]])(
      implicit blockStore: BlockStore[Task],
      safetyOracle: SafetyOracle[Task],
      log: Log[Task]
  ) = {
    implicit val ec: Cell[Task, Engine[Task]] = engineCell
    BlockAPI.exploratoryDeploy[Task](term)
  }

  /*
   * DAG structure for finalization:
   * With 3 validators at 10 stake each (total 30), finalization requires >15 stake.
   * For b2 to be finalized, at least 2 validators must build on top of it.
   *
   *     n1: genesis -> b1 -> b2
   *     n2: genesis ---------> b3 (parent: b2)
   *     n3: genesis ---------> b4 (parent: b3, which includes b2)
   *
   * After b3 and b4, b2 has 20 stake (n2 + n3) building on it -> finalized
   */
  it should "exploratoryDeploy get data from the read only node" in effectTest {
    TestNode.networkEff(genesisContext, networkSize = 3, withReadOnlySize = 1).use {
      case nodes @ n1 +: n2 +: n3 +: readOnly +: Seq() =>
        import readOnly.{blockStore, cliqueOracleEffect, logEff}
        val engine     = new EngineWithCasper[Task](readOnly.casperEff)
        val storedData = "data"
        for {
          produceDeploys <- (0 until 3).toList.traverse(
                             i =>
                               basicDeployData[Task](
                                 i,
                                 shardId = genesisContext.genesisBlock.shardId
                               )
                           )
          putDataDeploy <- sourceDeployNowF[Task](
                            s"""@"store"!("$storedData")""",
                            shardId = genesisContext.genesisBlock.shardId
                          )
          _  <- n1.propagateBlock(putDataDeploy)(nodes: _*)     // b1
          b2 <- n1.propagateBlock(produceDeploys(0))(nodes: _*) // b2
          _  <- n2.propagateBlock(produceDeploys(1))(nodes: _*) // b3 - builds on b2
          _  <- n3.propagateBlock(produceDeploys(2))(nodes: _*) // b4 - builds on b3, finalizes b2

          engineCell <- Cell.mvarCell[Task, Engine[Task]](engine)
          result <- exploratoryDeploy(
                     "new return in { for (@data <- @\"store\") {return!(data)}}"
                   )(
                     engineCell
                   ).map(_.right.value)
          (par, lastFinalizedBlock) = result
          _                         = lastFinalizedBlock.blockHash shouldBe PrettyPrinter.buildStringNoLimit(b2.blockHash)
          _ = par match {
            case Seq(Par(_, _, _, Seq(expr), _, _, _, _, _, _)) =>
              expr match {
                case Expr(GString(data)) => data shouldBe storedData
                case _                   => fail("Could not get data from exploretory api")
              }
          }

        } yield ()
    }
  }

  it should "exploratoryDeploy return error on bonded validator" in effectTest {
    TestNode.networkEff(genesisContext, networkSize = 1).use {
      case nodes @ n1 +: Seq() =>
        import n1.{blockStore, cliqueOracleEffect, logEff}
        val engine = new EngineWithCasper[Task](n1.casperEff)
        for {
          produceDeploys <- (0 until 1).toList.traverse(
                             i =>
                               basicDeployData[Task](
                                 i,
                                 shardId = genesisContext.genesisBlock.shardId
                               )
                           )
          _ <- n1.propagateBlock(produceDeploys(0))(nodes: _*)

          engineCell <- Cell.mvarCell[Task, Engine[Task]](engine)
          result <- exploratoryDeploy("new return in { return!(1) }")(
                     engineCell
                   )
          _ = result.left.value shouldBe "Exploratory deploy can only be executed on read-only RNode."

        } yield ()
    }
  }
}
