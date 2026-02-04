package coop.rchain.casper.blocks.proposer

import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.{BlockStore, KeyValueBlockStore}
import coop.rchain.blockstorage.dag.BlockDagKeyValueStorage
import coop.rchain.blockstorage.deploy.{DeployStorage, KeyValueDeployStorage}
import coop.rchain.casper.{CasperShardConf, CasperSnapshot, OnChainCasperState, ValidatorIdentity}
import coop.rchain.casper.helper.TestBlockDagRepresentation
import coop.rchain.casper.protocol.DeployData
import coop.rchain.casper.util.GenesisBuilder.defaultValidatorSks
import coop.rchain.casper.util.rholang.{Resources, RuntimeManager}
import coop.rchain.crypto.signatures.{Secp256k1, Signed}
import coop.rchain.metrics.{Metrics, Span}
import coop.rchain.metrics.Metrics.MetricsNOP
import coop.rchain.p2p.EffectsTestInstances.LogStub
import coop.rchain.rholang
import coop.rchain.shared.{Log, Time}
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.duration._

/**
  * Unit tests for BlockCreator.
  *
  * Tests the deploy preparation and cleanup logic.
  */
class BlockCreatorSpec extends FlatSpec with Matchers {

  implicit private val logStub: Log[Task]     = new LogStub[Task]()
  implicit private val metrics: Metrics[Task] = new MetricsNOP[Task]()
  implicit private val span: Span[Task]       = Span.noop[Task]
  implicit private val timeEff: Time[Task] = new Time[Task] {
    override def currentMillis: Task[Long]                   = Task.delay(System.currentTimeMillis())
    override def nanoTime: Task[Long]                        = Task.delay(System.nanoTime())
    override def sleep(duration: FiniteDuration): Task[Unit] = Task.sleep(duration)
  }

  private val validatorSk       = defaultValidatorSks.head
  private val validatorIdentity = ValidatorIdentity(validatorSk)
  private val validatorId       = ByteString.copyFrom(validatorIdentity.publicKey.bytes)

  private val deployLifespan = 50

  private def createDeploy(validAfterBlockNumber: Long): Signed[DeployData] = {
    val deployData = DeployData(
      term = s"new x in { x!($validAfterBlockNumber) }",
      timestamp = System.currentTimeMillis(),
      phloPrice = 1,
      phloLimit = 1000,
      validAfterBlockNumber = validAfterBlockNumber,
      shardId = "test-shard"
    )
    Signed(deployData, Secp256k1, validatorSk)
  }

  private def createSnapshot(maxBlockNum: Long): CasperSnapshot[Task] = {
    val dag = TestBlockDagRepresentation[Task]()

    val onChainState = OnChainCasperState(
      shardConf = CasperShardConf(
        faultToleranceThreshold = 0f,
        shardName = "test-shard",
        parentShardId = "",
        finalizationRate = 0,
        maxNumberOfParents = 10,
        maxParentDepth = 0,
        synchronyConstraintThreshold = 0f,
        heightConstraintThreshold = 0L,
        deployLifespan = deployLifespan,
        casperVersion = 1L,
        configVersion = 1L,
        bondMinimum = 0L,
        bondMaximum = Long.MaxValue,
        epochLength = 0,
        quarantineLength = 0,
        minPhloPrice = 0L,
        enableMergeableChannelGC = false,
        mergeableChannelsGCDepthBuffer = 10,
        disableLateBlockFiltering = false,
        disableValidatorProgressCheck = false
      ),
      bondsMap = Map(validatorId -> 100L),
      activeValidators = Seq(validatorId)
    )

    CasperSnapshot[Task](
      dag = dag,
      lastFinalizedBlock = ByteString.EMPTY,
      lca = ByteString.EMPTY,
      tips = IndexedSeq.empty,
      parents = List.empty,
      justifications = Set.empty,
      invalidBlocks = Map.empty,
      deploysInScope = Set.empty,
      maxBlockNum = maxBlockNum,
      maxSeqNums = Map(validatorId -> 0),
      onChainState = onChainState
    )
  }

  "BlockCreator.create" should "remove expired deploys from storage during block creation" in {
    val test = rholang.Resources
      .mkTempDir[Task]("block-creator-test-")
      .evalMap(Resources.mkTestRNodeStoreManager[Task])
      .use { kvm =>
        for {
          blockStore            <- KeyValueBlockStore[Task](kvm)
          _                     <- BlockDagKeyValueStorage.create[Task](kvm)
          deployStorageInstance <- KeyValueDeployStorage[Task](kvm)
          runtimeManager        <- Resources.mkRuntimeManagerAt[Task](kvm)

          result <- {
            implicit val ds: DeployStorage[Task]  = deployStorageInstance
            implicit val bs: BlockStore[Task]     = blockStore
            implicit val rm: RuntimeManager[Task] = runtimeManager

            for {
              // Create an expired deploy: validAfterBlockNumber = 0
              // With deployLifespan = 50 and currentBlock = 101,
              // earliestBlockNumber = 101 - 50 = 51
              // This deploy expires because validAfterBlockNumber (0) <= earliestBlockNumber (51)
              expiredDeploy <- Task.delay(createDeploy(validAfterBlockNumber = 0L))

              // Add the expired deploy to storage
              _ <- ds.add(List(expiredDeploy))

              // Verify deploy is in storage
              deploysBeforeCreate <- ds.readAll
              _                   = deploysBeforeCreate.size shouldBe 1
              _                   = deploysBeforeCreate.head.sig shouldBe expiredDeploy.sig

              // Create a snapshot where currentBlock would be 101 (maxBlockNum = 100)
              snapshot = createSnapshot(maxBlockNum = 100L)

              // Call BlockCreator.create
              // This should filter out the expired deploy AND remove it from storage
              // Note: The block creation may fail due to empty parents, but the cleanup
              // happens before that in prepareUserDeploys
              _ <- BlockCreator.create(snapshot, validatorIdentity).attempt

              // Verify the expired deploy was removed from storage
              deploysAfterCreate <- ds.readAll
              _ = deploysAfterCreate.size shouldBe 0
            } yield ()
          }
        } yield result
      }

    test.runSyncUnsafe()
  }

  it should "remove only expired deploys while keeping valid ones in storage" in {
    val test = rholang.Resources
      .mkTempDir[Task]("block-creator-test-")
      .evalMap(Resources.mkTestRNodeStoreManager[Task])
      .use { kvm =>
        for {
          blockStore            <- KeyValueBlockStore[Task](kvm)
          _                     <- BlockDagKeyValueStorage.create[Task](kvm)
          deployStorageInstance <- KeyValueDeployStorage[Task](kvm)
          runtimeManager        <- Resources.mkRuntimeManagerAt[Task](kvm)

          result <- {
            implicit val ds: DeployStorage[Task]  = deployStorageInstance
            implicit val bs: BlockStore[Task]     = blockStore
            implicit val rm: RuntimeManager[Task] = runtimeManager

            for {
              // With deployLifespan = 50 and currentBlock = 101 (maxBlockNum = 100),
              // earliestBlockNumber = 101 - 50 = 51
              //
              // Expired deploy: validAfterBlockNumber = 0 (<= 51, expired)
              // Valid deploy: validAfterBlockNumber = 60 (> 51 and < 101, valid)
              expiredDeploy <- Task.delay(createDeploy(validAfterBlockNumber = 0L))
              validDeploy   <- Task.delay(createDeploy(validAfterBlockNumber = 60L))

              // Add both deploys to storage
              _ <- ds.add(List(expiredDeploy, validDeploy))

              // Verify both deploys are in storage
              deploysBeforeCreate <- ds.readAll
              _                   = deploysBeforeCreate.size shouldBe 2

              snapshot = createSnapshot(maxBlockNum = 100L)

              // Call BlockCreator.create
              // The cleanup happens in prepareUserDeploys before block creation
              // Block creation may fail due to empty parents, but that's after cleanup
              _ <- BlockCreator.create(snapshot, validatorIdentity).attempt

              // Verify: expired deploy removed, valid deploy kept
              deploysAfterCreate <- ds.readAll
              _                  = deploysAfterCreate.size shouldBe 1
              _                  = deploysAfterCreate.head.sig shouldBe validDeploy.sig
            } yield ()
          }
        } yield result
      }

    test.runSyncUnsafe()
  }
}
