package coop.rchain.node.instances

import cats.effect.Timer
import cats.effect.concurrent.Ref
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.casper._
import coop.rchain.casper.blocks.proposer._
import coop.rchain.casper.engine.EngineCell._
import coop.rchain.casper.helper.{
  BlockDagStorageFixture,
  NoOpsCasperEffect,
  TestCasperSnapshot,
  TestEngine
}
import coop.rchain.casper.protocol.{BlockMessage, DeployData}
import coop.rchain.casper.util.GenesisBuilder.defaultValidatorSks
import coop.rchain.casper.util.rholang.Resources.mkRuntimeManager
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import coop.rchain.models.Validator.Validator
import coop.rchain.models.blockImplicits.getRandomBlock
import coop.rchain.p2p.EffectsTestInstances.LogStub
import coop.rchain.shared.{Cell, Log, Time}
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.duration._

/**
  * Unit tests for HeartbeatProposer.
  *
  * Tests call checkAndMaybePropose directly (like ProposerSpec tests Proposer.propose)
  * rather than running streams, making tests deterministic and fast.
  */
class HeartbeatProposerSpec extends FlatSpec with Matchers with BlockDagStorageFixture {

  implicit private val classLog: Log[Task] = new LogStub[Task]()
  implicit val timer: Timer[Task]          = Task.timer
  implicit val timeEff: Time[Task] = new Time[Task] {
    override def currentMillis: Task[Long]                   = Task.delay(System.currentTimeMillis())
    override def nanoTime: Task[Long]                        = Task.delay(System.nanoTime())
    override def sleep(duration: FiniteDuration): Task[Unit] = Task.sleep(duration)
  }

  private val runtimeManagerResource = mkRuntimeManager[Task]("heartbeat-proposer-test")
  private val validatorSk            = defaultValidatorSks.head
  private val validatorIdentity      = ValidatorIdentity(validatorSk)
  private val validatorId: Validator = ByteString.copyFrom(validatorIdentity.publicKey.bytes)

  private def createProposeTracker(): Task[(Ref[Task, Int], ProposeFunction[Task])] =
    Ref[Task].of(0).map { proposeCount =>
      val proposeFunc: ProposeFunction[Task] = (_: Casper[Task], _: Boolean) =>
        proposeCount.update(_ + 1) >> Task.pure[ProposerResult](
          ProposerSuccess(ProposeSuccess(ValidBlock.Valid), getRandomBlock())
        )
      (proposeCount, proposeFunc)
    }

  private def createMockDeploy(): Signed[DeployData] = {
    val deployData = DeployData(
      term = "new x in { x!(0) }",
      timestamp = System.currentTimeMillis(),
      phloPrice = 1,
      phloLimit = 1000,
      validAfterBlockNumber = 0,
      shardId = "test-shard"
    )
    Signed(deployData, Secp256k1, validatorSk)
  }

  private def createLfbWithAge(ageMs: Long): BlockMessage = {
    val block = getRandomBlock()
    block.copy(header = block.header.copy(timestamp = System.currentTimeMillis() - ageMs))
  }

  private def createEngineCell(casper: NoOpsCasperEffect[Task]): Task[EngineCell[Task]] = {
    val engine = TestEngine[Task](casper)
    Cell.mvarCell[Task, coop.rchain.casper.engine.Engine[Task]](engine)
  }

  // ==================== Configuration Tests ====================

  "HeartbeatProposer.create" should "return empty stream when config is disabled" in withStorage {
    implicit blockStore => implicit blockDagStorage =>
      runtimeManagerResource.use { implicit rm =>
        val logStub  = new LogStub[Task]()
        val snapshot = TestCasperSnapshot.bondedNoDeploys[Task](validatorId)

        createProposeTracker().flatMap {
          case (proposeCount, proposeFunc) =>
            for {
              casper     <- NoOpsCasperEffect.withSnapshot[Task](snapshot, getRandomBlock())
              engineCell <- createEngineCell(casper)
              config = HeartbeatConf(
                enabled = false,
                checkInterval = 1.second,
                maxLfbAge = 1.second
              )
              result <- HeartbeatProposer.create[Task](proposeFunc, validatorIdentity, config, 10)(
                         implicitly,
                         implicitly,
                         implicitly,
                         logStub,
                         engineCell
                       )
              (stream, _) = result
              elements    <- stream.compile.toList
              _           = elements shouldBe empty
              count       <- proposeCount.get
              _           = count shouldBe 0
            } yield ()
        }
      }
  }

  it should "return empty stream and log error when max-number-of-parents is 1" in withStorage {
    implicit blockStore => implicit blockDagStorage =>
      runtimeManagerResource.use { implicit rm =>
        val logStub  = new LogStub[Task]()
        val snapshot = TestCasperSnapshot.bondedNoDeploys[Task](validatorId)

        createProposeTracker().flatMap {
          case (proposeCount, proposeFunc) =>
            for {
              casper     <- NoOpsCasperEffect.withSnapshot[Task](snapshot, getRandomBlock())
              engineCell <- createEngineCell(casper)
              config     = HeartbeatConf(enabled = true, checkInterval = 1.second, maxLfbAge = 1.second)
              result <- HeartbeatProposer.create[Task](proposeFunc, validatorIdentity, config, 1)(
                         implicitly,
                         implicitly,
                         implicitly,
                         logStub,
                         engineCell
                       )
              (stream, _) = result
              _           = logStub.errors.exists(_.contains("CONFIGURATION ERROR")) shouldBe true
              elements    <- stream.compile.toList
              _           = elements shouldBe empty
              count       <- proposeCount.get
              _           = count shouldBe 0
            } yield ()
        }
      }
  }

  // ==================== Decision Logic Tests (Direct Method Calls) ====================

  "HeartbeatProposer.checkAndMaybePropose" should "trigger propose when pending deploys exist" in withStorage {
    implicit blockStore => implicit blockDagStorage =>
      runtimeManagerResource.use { implicit rm =>
        val logStub        = new LogStub[Task]()
        val pendingDeploys = Set(createMockDeploy())
        val snapshot       = TestCasperSnapshot.withPendingDeploys[Task](validatorId, pendingDeploys)
        val lfb            = createLfbWithAge(100) // Fresh LFB

        createProposeTracker().flatMap {
          case (proposeCount, proposeFunc) =>
            for {
              casper     <- NoOpsCasperEffect.withSnapshot[Task](snapshot, lfb)
              engineCell <- createEngineCell(casper)
              config = HeartbeatConf(
                enabled = true,
                checkInterval = 1.second,
                maxLfbAge = 10.seconds
              )
              _ <- HeartbeatProposer
                    .checkAndMaybePropose[Task](proposeFunc, validatorIdentity, config)(
                      implicitly,
                      implicitly,
                      logStub,
                      engineCell
                    )
              count <- proposeCount.get
              _     = count shouldBe 1
              _     = logStub.infos.exists(_.contains("pending user deploys")) shouldBe true
            } yield ()
        }
      }
  }

  it should "trigger propose when LFB is stale and new parents exist" in withStorage {
    implicit blockStore => implicit blockDagStorage =>
      runtimeManagerResource.use { implicit rm =>
        val logStub  = new LogStub[Task]()
        val snapshot = TestCasperSnapshot.bondedNoDeploys[Task](validatorId)
        val lfb      = createLfbWithAge(60000) // Stale LFB (60 seconds)

        createProposeTracker().flatMap {
          case (proposeCount, proposeFunc) =>
            for {
              casper     <- NoOpsCasperEffect.withSnapshot[Task](snapshot, lfb)
              engineCell <- createEngineCell(casper)
              config     = HeartbeatConf(enabled = true, checkInterval = 1.second, maxLfbAge = 1.second)
              _ <- HeartbeatProposer
                    .checkAndMaybePropose[Task](proposeFunc, validatorIdentity, config)(
                      implicitly,
                      implicitly,
                      logStub,
                      engineCell
                    )
              count <- proposeCount.get
              _     = count shouldBe 1
              _     = logStub.infos.exists(_.contains("LFB is stale")) shouldBe true
            } yield ()
        }
      }
  }

  it should "skip propose when validator is not bonded" in withStorage {
    implicit blockStore => implicit blockDagStorage =>
      runtimeManagerResource.use { implicit rm =>
        val logStub  = new LogStub[Task]()
        val snapshot = TestCasperSnapshot.notBonded[Task]
        val lfb      = createLfbWithAge(60000) // Stale LFB

        createProposeTracker().flatMap {
          case (proposeCount, proposeFunc) =>
            for {
              casper     <- NoOpsCasperEffect.withSnapshot[Task](snapshot, lfb)
              engineCell <- createEngineCell(casper)
              config     = HeartbeatConf(enabled = true, checkInterval = 1.second, maxLfbAge = 1.second)
              _ <- HeartbeatProposer
                    .checkAndMaybePropose[Task](proposeFunc, validatorIdentity, config)(
                      implicitly,
                      implicitly,
                      logStub,
                      engineCell
                    )
              count <- proposeCount.get
              _     = count shouldBe 0
              _     = logStub.infos.exists(_.contains("not bonded")) shouldBe true
            } yield ()
        }
      }
  }

  it should "skip propose when LFB is fresh and no pending deploys" in withStorage {
    implicit blockStore => implicit blockDagStorage =>
      runtimeManagerResource.use { implicit rm =>
        val logStub  = new LogStub[Task]()
        val snapshot = TestCasperSnapshot.bondedNoDeploys[Task](validatorId)
        val lfb      = createLfbWithAge(100) // Fresh LFB (100ms)

        createProposeTracker().flatMap {
          case (proposeCount, proposeFunc) =>
            for {
              casper     <- NoOpsCasperEffect.withSnapshot[Task](snapshot, lfb)
              engineCell <- createEngineCell(casper)
              config = HeartbeatConf(
                enabled = true,
                checkInterval = 1.second,
                maxLfbAge = 10.seconds
              )
              _ <- HeartbeatProposer
                    .checkAndMaybePropose[Task](proposeFunc, validatorIdentity, config)(
                      implicitly,
                      implicitly,
                      logStub,
                      engineCell
                    )
              count <- proposeCount.get
              _     = count shouldBe 0
              _     = logStub.debugs.exists(_.contains("No action needed")) shouldBe true
            } yield ()
        }
      }
  }
}
