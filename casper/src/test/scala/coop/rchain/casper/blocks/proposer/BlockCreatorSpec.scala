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

  private def createDeploy(
      validAfterBlockNumber: Long,
      expirationTimestamp: Option[Long] = None
  ): Signed[DeployData] = {
    val deployData = DeployData(
      term = s"new x in { x!($validAfterBlockNumber) }",
      timestamp = System.currentTimeMillis(),
      phloPrice = 1,
      phloLimit = 1000,
      validAfterBlockNumber = validAfterBlockNumber,
      shardId = "test-shard",
      expirationTimestamp = expirationTimestamp
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

  "BlockCreator.create" should "remove block-expired deploys while keeping valid ones in storage" in {
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

  it should "remove both block-expired and time-expired deploys while keeping valid ones" in {
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
              pastTimestamp <- Task.delay(System.currentTimeMillis() - 60000L) // 1 minute ago

              // Block-expired deploy (validAfterBlockNumber = 0 is expired)
              blockExpiredDeploy <- Task.delay(createDeploy(validAfterBlockNumber = 0L))

              // Time-expired deploy (validAfterBlockNumber = 60 is valid, but expirationTimestamp is past)
              timeExpiredDeploy <- Task.delay(
                                    createDeploy(
                                      validAfterBlockNumber = 60L,
                                      expirationTimestamp = Some(pastTimestamp)
                                    )
                                  )

              // Valid deploy (validAfterBlockNumber = 60 is valid, no expiration timestamp)
              validDeploy <- Task.delay(createDeploy(validAfterBlockNumber = 60L))

              // Add all deploys to storage
              _ <- ds.add(List(blockExpiredDeploy, timeExpiredDeploy, validDeploy))

              // Verify all deploys are in storage
              deploysBeforeCreate <- ds.readAll
              _                   = deploysBeforeCreate.size shouldBe 3

              snapshot = createSnapshot(maxBlockNum = 100L)

              _ <- BlockCreator.create(snapshot, validatorIdentity).attempt

              // Verify: both expired deploys removed, valid deploy kept
              deploysAfterCreate <- ds.readAll
              _                  = deploysAfterCreate.size shouldBe 1
              _                  = deploysAfterCreate.head.sig shouldBe validDeploy.sig
            } yield ()
          }
        } yield result
      }

    test.runSyncUnsafe()
  }

  it should "filter out phantom file registrations while keeping valid file registrations" in {
    val test = rholang.Resources
      .mkTempDir[Task]("block-creator-test-")
      .evalMap(Resources.mkTestRNodeStoreManager[Task])
      .use { kvm =>
        for {
          blockStore            <- KeyValueBlockStore[Task](kvm)
          _                     <- BlockDagKeyValueStorage.create[Task](kvm)
          deployStorageInstance <- KeyValueDeployStorage[Task](kvm)
          runtimeManager        <- Resources.mkRuntimeManagerAt[Task](kvm)

          // Create a temp directory for fileReplicationDir
          fileReplDir <- Task.delay(java.nio.file.Files.createTempDirectory("file-repl-test"))

          result <- {
            implicit val ds: DeployStorage[Task]  = deployStorageInstance
            implicit val bs: BlockStore[Task]     = blockStore
            implicit val rm: RuntimeManager[Task] = runtimeManager

            val validFileHash   = "a" * 64
            val phantomFileHash = "b" * 64
            val validFileDeployTerm =
              s"""new ret, file(`rho:io:file`) in { file!("register", "$validFileHash", 3, "test.bin", *ret) }"""
            val phantomFileDeployTerm =
              s"""new ret, file(`rho:io:file`) in { file!("register", "$phantomFileHash", 3, "test.bin", *ret) }"""

            val validFileDeployData = DeployData(
              term = validFileDeployTerm,
              timestamp = System.currentTimeMillis(),
              phloPrice = 1,
              phloLimit = 1000,
              validAfterBlockNumber = 60L,
              shardId = "test-shard"
            )
            val phantomFileDeployData = DeployData(
              term = phantomFileDeployTerm,
              timestamp = System.currentTimeMillis(),
              phloPrice = 1,
              phloLimit = 1000,
              validAfterBlockNumber = 60L,
              shardId = "test-shard"
            )

            val validFileDeploy   = Signed(validFileDeployData, Secp256k1, validatorSk)
            val phantomFileDeploy = Signed(phantomFileDeployData, Secp256k1, validatorSk)
            val normalDeploy      = createDeploy(validAfterBlockNumber = 60L)

            for {
              // Valid file on disk
              _ <- Task.delay(
                    java.nio.file.Files
                      .write(fileReplDir.resolve(validFileHash), Array[Byte](1, 2, 3))
                  )

              _ <- ds.add(List(validFileDeploy, phantomFileDeploy, normalDeploy))

              // Verify all 3 are in storage
              deploysBeforeCreate <- ds.readAll
              _                   = deploysBeforeCreate.size shouldBe 3

              // Create snapshot with fileReplicationDir configured
              baseSnapshot = createSnapshot(maxBlockNum = 100L)
              snapshot = baseSnapshot.copy(
                onChainState = baseSnapshot.onChainState.copy(
                  shardConf = baseSnapshot.onChainState.shardConf.copy(
                    fileConf = coop.rchain.casper.FileConf(fileReplicationDir = Some(fileReplDir))
                  )
                )
              )

              val stub = logStub.asInstanceOf[coop.rchain.p2p.EffectsTestInstances.LogStub[Task]]
              _        <- Task.delay(stub.reset())

              // The create will fail because there are no parents, but prepareUserDeploys
              // runs before that and performs the filtering, logging a warning.
              _ <- BlockCreator.create(snapshot, validatorIdentity).attempt

              _ <- Task.delay {
                    val phantomSigPrefix =
                      coop.rchain.shared.Base16.encode(phantomFileDeploy.sig.toByteArray.take(8))
                    val validSigPrefix =
                      coop.rchain.shared.Base16.encode(validFileDeploy.sig.toByteArray.take(8))

                    stub.warns.exists(
                      _.contains(s"Deploy $phantomSigPrefix... FILTERED (missing file)")
                    ) shouldBe true
                    stub.warns.exists(
                      _.contains(s"Deploy $validSigPrefix... FILTERED (missing file)")
                    ) shouldBe false
                  }

              // Deploys are still in storage (filtered, not expired)
              deploysAfterCreate <- ds.readAll
              _                  = deploysAfterCreate.size shouldBe 3
            } yield ()
          }
          _ <- Task.delay {
                java.nio.file.Files.deleteIfExists(fileReplDir.resolve("a" * 64))
                java.nio.file.Files.deleteIfExists(fileReplDir)
              }
        } yield result
      }

    test.runSyncUnsafe()
  }
}
