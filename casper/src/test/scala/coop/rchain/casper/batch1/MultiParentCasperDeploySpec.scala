package coop.rchain.casper.batch1

import coop.rchain.casper.blocks.proposer.{Created, NoNewDeploys}
import coop.rchain.casper.MultiParentCasper
import coop.rchain.casper.api.BlockAPI
import coop.rchain.casper.batch2.EngineWithCasper
import coop.rchain.casper.engine.Engine
import coop.rchain.casper.helper.TestNode
import coop.rchain.casper.helper.TestNode._
import coop.rchain.casper.PrettyPrinter
import coop.rchain.casper.protocol.{DeployParameterData, RholangValueData}
import coop.rchain.casper.util.ConstructDeploy
import coop.rchain.metrics.{NoopSpan, Span}
import coop.rchain.p2p.EffectsTestInstances.LogicalTime
import coop.rchain.shared.Cell
import coop.rchain.shared.scalatestcontrib._
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Inspectors, Matchers}

class MultiParentCasperDeploySpec extends FlatSpec with Matchers with Inspectors {

  import coop.rchain.casper.util.GenesisBuilder._

  implicit val timeEff = new LogicalTime[Effect]

  val genesis          = buildGenesis()
  private val SHARD_ID = genesis.genesisBlock.shardId

  "MultiParentCasper" should "accept a deploy and return it's id" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      import node._
      implicit val timeEff = new LogicalTime[Effect]

      for {
        deploy   <- ConstructDeploy.basicDeployData[Effect](0)
        res      <- MultiParentCasper[Effect].deploy(deploy)
        deployId = res.right.get
      } yield deployId shouldBe deploy.sig
    }
  }

  it should "not create a block with a repeated deploy" in effectTest {
    implicit val timeEff = new LogicalTime[Effect]
    TestNode.networkEff(genesis, networkSize = 2).use { nodes =>
      val List(node0, node1) = nodes.toList
      for {
        deploy             <- ConstructDeploy.basicDeployData[Effect](0, shardId = genesis.genesisBlock.shardId)
        _                  <- node0.propagateBlock(deploy)(node1)
        createBlockResult2 <- node1.createBlock(deploy)
      } yield (createBlockResult2 should be(NoNewDeploys))
    }
  }

  it should "fail when deploying with insufficient phlos" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      implicit val timeEff = new LogicalTime[Effect]

      for {
        deployData     <- ConstructDeploy.sourceDeployNowF[Effect]("Nil", phloLimit = 1)
        r              <- node.createBlock(deployData)
        Created(block) = r
      } yield assert(block.body.deploys.head.isFailed)
    }
  }

  it should "succeed if given enough phlos for deploy" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      implicit val timeEff = new LogicalTime[Effect]

      for {
        deployData     <- ConstructDeploy.sourceDeployNowF[Effect]("Nil", phloLimit = 100)
        r              <- node.createBlock(deployData)
        Created(block) = r
      } yield assert(!block.body.deploys.head.isFailed)
    }
  }

  it should "reject deploy with phloPrice lower than minPhloPrice" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      implicit val noopSpan: Span[Effect] = NoopSpan[Effect]()
      val engine                          = new EngineWithCasper[Effect](node.casperEff)
      Cell.mvarCell[Effect, Engine[Effect]](engine).flatMap { implicit engineCell =>
        val minPhloPrice   = 10.toLong
        val phloPrice      = 1.toLong
        val isNodeReadOnly = false
        for {
          deployData <- ConstructDeploy
                         .sourceDeployNowF[Effect](
                           "Nil",
                           phloPrice = phloPrice,
                           shardId = genesis.genesisBlock.shardId
                         )
          err <- BlockAPI
                  .deploy[Effect](
                    deployData,
                    None,
                    minPhloPrice = minPhloPrice,
                    isNodeReadOnly,
                    shardId = SHARD_ID
                  )
                  .attempt
        } yield {
          err.isLeft shouldBe true
          val ex = err.left.get
          ex shouldBe a[RuntimeException]
          ex.getMessage shouldBe s"Phlo price $phloPrice is less than minimum price $minPhloPrice."
        }
      }
    }
  }

  // Tests for deploy with parameters (DeployData now supports parameters natively)
  it should "accept a deploy with parameters and return it's id" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      implicit val noopSpan: Span[Effect] = NoopSpan[Effect]()
      val engine                          = new EngineWithCasper[Effect](node.casperEff)
      Cell.mvarCell[Effect, Engine[Effect]](engine).flatMap { implicit engineCell =>
        val isNodeReadOnly = false
        val minPhloPrice   = 1.toLong
        for {
          deployData <- ConstructDeploy.sourceDeployNowF[Effect](
                         "Nil",
                         parameters = Seq.empty,
                         shardId = genesis.genesisBlock.shardId
                       )
          result <- BlockAPI
                     .deploy[Effect](
                       deployData,
                       None,
                       minPhloPrice = minPhloPrice,
                       isNodeReadOnly,
                       shardId = SHARD_ID
                     )
        } yield {
          result.isRight shouldBe true
          val deployId = result.right.get
          deployId should include("Success! DeployId is:")
          deployId should include(PrettyPrinter.buildStringNoLimit(deployData.sig))
        }
      }
    }
  }

  it should "reject deploy with parameters when phloPrice lower than minPhloPrice" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      implicit val noopSpan: Span[Effect] = NoopSpan[Effect]()
      val engine                          = new EngineWithCasper[Effect](node.casperEff)
      Cell.mvarCell[Effect, Engine[Effect]](engine).flatMap { implicit engineCell =>
        val minPhloPrice   = 10.toLong
        val phloPrice      = 1.toLong
        val isNodeReadOnly = false
        for {
          deployData <- ConstructDeploy.sourceDeployNowF[Effect](
                         "Nil",
                         parameters = Seq.empty,
                         phloPrice = phloPrice,
                         shardId = genesis.genesisBlock.shardId
                       )
          err <- BlockAPI
                  .deploy[Effect](
                    deployData,
                    None,
                    minPhloPrice = minPhloPrice,
                    isNodeReadOnly,
                    shardId = SHARD_ID
                  )
                  .attempt
        } yield {
          err.isLeft shouldBe true
          val ex = err.left.get
          ex shouldBe a[RuntimeException]
          ex.getMessage shouldBe s"Phlo price $phloPrice is less than minimum price $minPhloPrice."
        }
      }
    }
  }

  it should "reject deploy with parameters when term has invalid syntax" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      implicit val noopSpan: Span[Effect] = NoopSpan[Effect]()
      val engine                          = new EngineWithCasper[Effect](node.casperEff)
      Cell.mvarCell[Effect, Engine[Effect]](engine).flatMap { implicit engineCell =>
        val isNodeReadOnly = false
        val minPhloPrice   = 1.toLong
        for {
          deployData <- ConstructDeploy.sourceDeployNowF[Effect](
                         "invalid syntax {{{",
                         parameters = Seq.empty,
                         shardId = genesis.genesisBlock.shardId
                       )
          result <- BlockAPI
                     .deploy[Effect](
                       deployData,
                       None,
                       minPhloPrice = minPhloPrice,
                       isNodeReadOnly,
                       shardId = SHARD_ID
                     )
        } yield {
          // Parsing errors are returned as Left(errorString), not thrown as exceptions
          result.isLeft shouldBe true
          val errorMsg = result.left.get
          errorMsg should include("Error in parsing term")
        }
      }
    }
  }

  it should "accept deploy with valid typed parameters" in effectTest {
    TestNode.standaloneEff(genesis).use { node =>
      import node.logEff
      implicit val noopSpan: Span[Effect] = NoopSpan[Effect]()
      val engine                          = new EngineWithCasper[Effect](node.casperEff)
      Cell.mvarCell[Effect, Engine[Effect]](engine).flatMap { implicit engineCell =>
        val isNodeReadOnly = false
        val minPhloPrice   = 1.toLong
        val parameters = Seq(
          DeployParameterData("myInt", RholangValueData.IntValue(42L)),
          DeployParameterData("myString", RholangValueData.StringValue("hello")),
          DeployParameterData("myBool", RholangValueData.BoolValue(true)),
          DeployParameterData(
            "myBytes",
            RholangValueData.BytesValue(com.google.protobuf.ByteString.copyFromUtf8("test"))
          )
        )
        for {
          deployData <- ConstructDeploy.sourceDeployNowF[Effect](
                         "Nil",
                         parameters = parameters,
                         shardId = genesis.genesisBlock.shardId
                       )
          result <- BlockAPI
                     .deploy[Effect](
                       deployData,
                       None,
                       minPhloPrice = minPhloPrice,
                       isNodeReadOnly,
                       shardId = SHARD_ID
                     )
        } yield {
          result.isRight shouldBe true
          val deployId = result.right.get
          deployId should include("Success! DeployId is:")
          deployId should include(PrettyPrinter.buildStringNoLimit(deployData.sig))
        }
      }
    }
  }
}
