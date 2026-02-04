package coop.rchain.casper.batch1

import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode.Effect
import coop.rchain.casper.protocol.CommEvent
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.casper.ReportingCasper
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.shared.scalatestcontrib.effectTest
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Inspectors, Matchers}

// NOTE: This spec depends on ReportingRuntime.createReportingRuntime, which is currently
// not implemented (see coop.rchain.casper.ReportingRuntime.createReportingRuntime with `???`).
// As a result, running the reporter ends with NotImplementedError. Once the reporting
// runtime is fully implemented on top of the new RSpace++ backend, this @Ignore should
// be removed and the test re-enabled.
@org.scalatest.Ignore
class MultiParentCasperReportingSpec extends FlatSpec with Matchers with Inspectors {

  import coop.rchain.casper.util.GenesisBuilder._

  implicit val timeEff: LogicalTime[Effect] = new LogicalTime[Effect]

  val genesis: GenesisContext = buildGenesis()

  "ReportingCasper" should "behave the same way as MultiParentCasper" in effectTest {
    val correctRholang =
      """ for(@a <- @"1"){ Nil } | @"1"!("x") """
    TestNode.standaloneEff(genesis).use { node =>
      import node._

      val reportingCasper =
        ReportingCasper.rhoReporter[Effect](node.dataDir.toString())
      val deploy = ConstructDeploy
        .sourceDeployNow(
          correctRholang,
          shardId = this.genesis.genesisBlock.shardId
        )

      for {
        signedBlock <- node.addBlock(deploy)
        _           = logEff.warns.isEmpty should be(true)
        trace       <- reportingCasper.trace(signedBlock)
        // only the comm events should be equal
        // it is possible that there are additional produce or consume in persistent mode
        reportingCommEventsNum = trace.deployReportResult.head.processedDeploy.deployLog.collect {
          case CommEvent(_, _, _) => 1
        }.sum
        deployCommEventsNum = signedBlock.body.deploys.head.deployLog.count {
          case CommEvent(_, _, _) => true
          case _                  => false
        }
        reportingReplayPostStateHash = trace.postStateHash
        _                            = reportingReplayPostStateHash shouldBe signedBlock.body.state.postStateHash
        _                            = deployCommEventsNum shouldBe reportingCommEventsNum
      } yield ()
    }
  }
}
