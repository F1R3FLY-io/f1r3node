package coop.rchain.casper.helper

import cats.Applicative
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.dag.BlockDagRepresentation
import coop.rchain.blockstorage.dag.BlockDagStorage.DeployId
import coop.rchain.casper.{CasperShardConf, CasperSnapshot, MultiParentCasper, OnChainCasperState}
import coop.rchain.casper.engine.Engine
import coop.rchain.casper.protocol.{CasperMessage, DeployData}
import coop.rchain.comm.PeerNode
import coop.rchain.crypto.signatures.Signed
import coop.rchain.models.BlockHash.BlockHash
import coop.rchain.models.BlockMetadata
import coop.rchain.models.Validator.Validator

/**
  * Controllable BlockDagRepresentation for unit testing.
  *
  * Allows tests to specify exactly what the DAG methods return,
  * enabling precise testing of logic that depends on DAG state.
  */
class TestBlockDagRepresentation[F[_]: Applicative](
    latestMessagesMap: Map[Validator, BlockHash] = Map.empty,
    blockMetadataMap: Map[BlockHash, BlockMetadata] = Map.empty,
    lfbHash: BlockHash = ByteString.EMPTY
) extends BlockDagRepresentation[F] {

  override def latestMessageHash(validator: Validator): F[Option[BlockHash]] =
    latestMessagesMap.get(validator).pure[F]

  override def latestMessageHashes: F[Map[Validator, BlockHash]] =
    latestMessagesMap.pure[F]

  override def lookup(blockHash: BlockHash): F[Option[BlockMetadata]] =
    blockMetadataMap.get(blockHash).pure[F]

  override def contains(blockHash: BlockHash): F[Boolean] =
    blockMetadataMap.contains(blockHash).pure[F]

  override def children(blockHash: BlockHash): F[Option[Set[BlockHash]]] =
    none[Set[BlockHash]].pure[F]

  override def invalidBlocks: F[Set[BlockMetadata]] =
    Set.empty[BlockMetadata].pure[F]

  override def latestBlockNumber: F[Long] =
    0L.pure[F]

  override def lookupByDeployId(deployId: DeployId): F[Option[BlockHash]] =
    none[BlockHash].pure[F]

  override def topoSort(
      startBlockNumber: Long,
      maybeEndBlockNumber: Option[Long]
  ): F[Vector[Vector[BlockHash]]] =
    Vector.empty[Vector[BlockHash]].pure[F]

  override def lastFinalizedBlock: BlockHash = lfbHash

  override def isFinalized(blockHash: BlockHash): F[Boolean] =
    false.pure[F]

  override def find(truncatedHash: String): F[Option[BlockHash]] =
    none[BlockHash].pure[F]
}

object TestBlockDagRepresentation {
  def apply[F[_]: Applicative](
      latestMessages: Map[Validator, BlockHash] = Map.empty,
      blockMetadata: Map[BlockHash, BlockMetadata] = Map.empty,
      lfbHash: BlockHash = ByteString.EMPTY
  ): TestBlockDagRepresentation[F] =
    new TestBlockDagRepresentation[F](latestMessages, blockMetadata, lfbHash)
}

/**
  * Helper to create controllable CasperSnapshot instances for testing.
  *
  * Default values are empty/minimal, allowing tests to override only
  * the fields they care about.
  */
object TestCasperSnapshot {

  /**
    * Create a CasperSnapshot with controllable fields for testing.
    *
    * @param deploysInScope Set of pending deploys (for testing pending deploy detection)
    * @param activeValidators Validators that are bonded (for testing bond check)
    * @param latestMessages Map of validator -> their latest block hash (for new parents check)
    * @param blockMetadata Map of block hash -> metadata (for DAG traversal)
    * @param lfbHash Last finalized block hash
    * @return Configured CasperSnapshot
    */
  def apply[F[_]: Applicative](
      deploysInScope: Set[Signed[DeployData]] = Set.empty,
      activeValidators: Seq[Validator] = Seq.empty,
      latestMessages: Map[Validator, BlockHash] = Map.empty,
      blockMetadata: Map[BlockHash, BlockMetadata] = Map.empty,
      lfbHash: BlockHash = ByteString.EMPTY,
      bondsMap: Map[Validator, Long] = Map.empty
  ): CasperSnapshot[F] = {
    val dag = TestBlockDagRepresentation[F](latestMessages, blockMetadata, lfbHash)

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
        deployLifespan = 100,
        casperVersion = 1L,
        configVersion = 1L,
        bondMinimum = 0L,
        bondMaximum = Long.MaxValue,
        epochLength = 0,
        quarantineLength = 0,
        minPhloPrice = 0L,
        enableMergeableChannelGC = false,
        mergeableChannelsGCDepthBuffer = 10,
        disableLateBlockFiltering = false
      ),
      bondsMap = bondsMap,
      activeValidators = activeValidators
    )

    CasperSnapshot[F](
      dag = dag,
      lastFinalizedBlock = lfbHash,
      lca = ByteString.EMPTY,
      tips = IndexedSeq.empty,
      parents = List.empty,
      justifications = Set.empty,
      invalidBlocks = Map.empty,
      deploysInScope = deploysInScope,
      maxBlockNum = 0,
      maxSeqNums = Map.empty,
      onChainState = onChainState
    )
  }

  /**
    * Create a snapshot where the validator is bonded and has pending deploys.
    */
  def withPendingDeploys[F[_]: Applicative](
      validatorId: Validator,
      deploys: Set[Signed[DeployData]]
  ): CasperSnapshot[F] =
    apply[F](
      deploysInScope = deploys,
      activeValidators = Seq(validatorId)
    )

  /**
    * Create a snapshot where the validator is bonded but has no pending deploys.
    */
  def bondedNoDeploys[F[_]: Applicative](validatorId: Validator): CasperSnapshot[F] =
    apply[F](activeValidators = Seq(validatorId))

  /**
    * Create a snapshot where the validator is NOT bonded.
    */
  def notBonded[F[_]: Applicative]: CasperSnapshot[F] =
    apply[F](activeValidators = Seq.empty)
}

/**
  * Test Engine that wraps a Casper instance, making it available via withCasper.
  *
  * Unlike Engine.noop which always returns the default, this engine
  * executes the provided function with the mock Casper.
  */
class TestEngine[F[_]: Applicative](casper: MultiParentCasper[F]) extends Engine[F] {
  override def init: F[Unit]                                       = Applicative[F].unit
  override def handle(peer: PeerNode, msg: CasperMessage): F[Unit] = Applicative[F].unit
  override def withCasper[A](f: MultiParentCasper[F] => F[A], default: F[A]): F[A] =
    f(casper)
}

object TestEngine {
  def apply[F[_]: Applicative](casper: MultiParentCasper[F]): TestEngine[F] =
    new TestEngine[F](casper)
}
