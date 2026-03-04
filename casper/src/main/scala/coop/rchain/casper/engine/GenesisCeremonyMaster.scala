package coop.rchain.casper.engine

import cats.Applicative
import cats.effect.concurrent.Ref
import cats.syntax.all._
import cats.effect.{Concurrent, Sync, Timer}
import coop.rchain.blockstorage.BlockStore
import coop.rchain.blockstorage.casperbuffer.CasperBufferStorage
import coop.rchain.blockstorage.dag.BlockDagStorage
import coop.rchain.blockstorage.deploy.DeployStorage
import coop.rchain.casper.LastApprovedBlock.LastApprovedBlock
import coop.rchain.casper._
import coop.rchain.casper.engine.EngineCell._
import coop.rchain.casper.protocol._
import coop.rchain.casper.syntax._
import coop.rchain.casper.util.comm.CommUtil
import coop.rchain.casper.util.rholang.RuntimeManager
import coop.rchain.comm.PeerNode
import coop.rchain.comm.rp.Connect.{ConnectionsCell, RPConfAsk}
import coop.rchain.comm.transport.TransportLayer
import coop.rchain.metrics.{Metrics, Span}
import coop.rchain.models.BlockHash.BlockHash
import coop.rchain.rspace.state.RSpaceStateManager
import coop.rchain.shared._
import fs2.concurrent.Queue

import scala.concurrent.duration._

class GenesisCeremonyMaster[F[_]: Sync: BlockStore: CommUtil: TransportLayer: RPConfAsk: Log: Time: SafetyOracle: LastApprovedBlock](
    approveProtocol: ApproveBlockProtocol[F]
) extends Engine[F] {
  import Engine._
  private val F    = Applicative[F]
  private val noop = F.unit

  override def init: F[Unit] = approveProtocol.run()

  override def handle(peer: PeerNode, msg: CasperMessage): F[Unit] = msg match {
    case br: ApprovedBlockRequest     => sendNoApprovedBlockAvailable(peer, br.identifier)
    case ba: BlockApproval            => approveProtocol.addApproval(ba)
    case na: NoApprovedBlockAvailable => logNoApprovedBlockAvailable[F](na.nodeIdentifer)
    case _                            => noop
  }
}

// format: off
object GenesisCeremonyMaster {
  import Engine._
  def waitingForApprovedBlockLoop[F[_]
    /* Execution */   : Concurrent: Time: Timer
    /* Transport */   : TransportLayer: CommUtil: BlockRetriever: EventPublisher
    /* State */       : EngineCell: RPConfAsk: ConnectionsCell: LastApprovedBlock
    /* Rholang */     : RuntimeManager
    /* Casper */      : Estimator: SafetyOracle: LastFinalizedHeightConstraintChecker: SynchronyConstraintChecker
    /* Storage */     : BlockStore: BlockDagStorage: DeployStorage: CasperBufferStorage: RSpaceStateManager
    /* Diagnostics */ : Log: EventLog: Metrics: Span] // format: on
  (
      blockProcessingQueue: Queue[F, (Casper[F], BlockMessage)],
      blocksInProcessing: Ref[F, Set[BlockHash]],
      casperShardConf: CasperShardConf,
      validatorId: Option[ValidatorIdentity],
      disableStateExporter: Boolean,
      onBlockFinalized: String => F[Unit]
  ): F[Unit] =
    for {
      // This loop sleep can be short as it does not do anything except checking if there is last approved block available
      _                  <- Time[F].sleep(2.seconds)
      lastApprovedBlockO <- LastApprovedBlock[F].get
      cont <- lastApprovedBlockO match {
               case None =>
                 waitingForApprovedBlockLoop[F](
                   blockProcessingQueue: Queue[F, (Casper[F], BlockMessage)],
                   blocksInProcessing: Ref[F, Set[BlockHash]],
                   casperShardConf,
                   validatorId,
                   disableStateExporter,
                   onBlockFinalized
                 )
               case Some(approvedBlock) =>
                 val ab = approvedBlock.candidate.block
                 for {
                   _ <- insertIntoBlockAndDagStore[F](ab, approvedBlock)
                   // Create FileRequester for P2P file transfers
                   dataDir <- cats.effect.Sync[F].delay {
                               val dir = casperShardConf.fileReplicationDir.getOrElse(
                                 java.nio.file.Paths.get("file-replication")
                               )
                               if (!dir.toFile.exists()) dir.toFile.mkdirs()
                               dir
                             }
                   fileRequester = new FileRequester[F](
                     dataDir,
                     casperShardConf.fileChunkSize,
                     casperShardConf.fileSyncTimeout
                   )
                   // DA-gating callback: request missing files from all peers and await.
                   // Broadcasts HasFile queries to all connected peers, then polls until
                   // the files arrive (via the FilePacket handler) or the timeout expires.
                   daCallback = (block: BlockMessage, missingHashes: List[String]) =>
                     for {
                       _ <- Log[F].info(
                             s"[GenesisCeremonyMaster] daCallback fired: " +
                               s"missingHashes=${missingHashes.size}: ${missingHashes.map(_.take(16)).mkString("[", ", ", "]")}"
                           )
                       peers <- ConnectionsCell[F].read
                       _ <- Log[F].info(
                             s"[GenesisCeremonyMaster] daCallback: requesting files from ${peers.size} peers"
                           )
                       // Broadcast file requests to ALL peers
                       _ <- peers.toList.traverse_(
                             peer => fileRequester.requestFiles(peer, missingHashes)
                           )
                       // Wait for files to arrive via P2P (with bounded timeout)
                       stillMissing <- fileRequester.awaitFiles(
                                        missingHashes,
                                        casperShardConf.fileSyncTimeout
                                      )
                       _ <- Log[F].info(
                             s"[GenesisCeremonyMaster] daCallback: awaitFiles returned, stillMissing=${stillMissing.size}"
                           )
                     } yield stillMissing
                   // Create heartbeat signal ref for triggering fast proposals on deploy submission
                   heartbeatSignalRef <- Ref[F].of(Option.empty[HeartbeatSignal[F]])
                   casper <- MultiParentCasper
                              .hashSetCasper[F](
                                validatorId,
                                casperShardConf: CasperShardConf,
                                ab,
                                heartbeatSignalRef,
                                onBlockFinalized,
                                daCallback
                              )
                   _ <- Engine
                         .transitionToRunning[F](
                           blockProcessingQueue,
                           blocksInProcessing,
                           casper,
                           approvedBlock,
                           validatorId,
                           ().pure[F],
                           disableStateExporter,
                           fileRequester
                         )
                   _ <- CommUtil[F].sendForkChoiceTipRequest
                 } yield ()
             }
    } yield cont
}
