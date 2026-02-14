package coop.rchain.casper.engine

import cats.Applicative
import cats.effect.concurrent.Ref
import cats.effect.{Concurrent, Timer}
import cats.syntax.all._
import coop.rchain.blockstorage.BlockStore
import coop.rchain.blockstorage.dag.BlockDagStorage
import coop.rchain.blockstorage.deploy.DeployStorage
import coop.rchain.casper.LastApprovedBlock.LastApprovedBlock
import coop.rchain.casper._
import coop.rchain.blockstorage.casperbuffer.CasperBufferStorage
import coop.rchain.casper.engine.EngineCell.EngineCell
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

// format: off
class GenesisValidator[F[_]
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
    validatorId: ValidatorIdentity,
    blockApprover: BlockApproverProtocol,
    approveInterval: FiniteDuration,
    onBlockFinalized: String => F[Unit]
) extends Engine[F] {
  import Engine._
  private val F    = Applicative[F]
  private val noop = F.unit

  private val seenCandidates                          = Cell.unsafe[F, Map[BlockHash, Boolean]](Map.empty)
  private def isRepeated(hash: BlockHash): F[Boolean] = seenCandidates.reads(_.contains(hash))
  private def ack(hash: BlockHash): F[Unit]           = seenCandidates.modify(_ + (hash -> true))

  // Guard against concurrent transitions from both UnapprovedBlock and ApprovedBlock
  // handlers (messages are dispatched concurrently — fair dispatcher is disabled).
  private val transitioned = Ref.unsafe[F, Boolean](false)

  // Start a background fiber that periodically requests ApprovedBlock from bootstrap.
  // This handles the case where the genesis ceremony completed before this node
  // connected, so no UnapprovedBlock will ever arrive.
  override val init: F[Unit] = Concurrent[F].start(approvedBlockFallback).void

  private def approvedBlockFallback: F[Unit] = {
    def loop: F[Unit] =
      transitioned.get.flatMap {
        case true =>
          Log[F].info(
            "GenesisValidator: engine transitioned, stopping ApprovedBlock request loop."
          )
        case false =>
          Log[F]
            .info(
              "GenesisValidator: requesting ApprovedBlock from bootstrap (ceremony may already be complete)."
            )
            .>>({
              CommUtil[F].requestApprovedBlock(true)
            }.handleErrorWith { err =>
              Log[F].warn(
                s"GenesisValidator: failed to request ApprovedBlock: ${err.getMessage}. Will retry."
              )
            }) >>
            Time[F].sleep(approveInterval) >>
            loop
      }

    // Initial delay: give the normal UnapprovedBlock flow a chance before
    // starting to poll for ApprovedBlock from bootstrap.
    Time[F].sleep(approveInterval) >> loop
  }

  override def handle(peer: PeerNode, msg: CasperMessage): F[Unit] = msg match {
    case br: ApprovedBlockRequest => sendNoApprovedBlockAvailable(peer, br.identifier)
    case _: ApprovedBlock         =>
      // Ceremony already completed — transition to Initializing to sync approved state.
      transitioned
        .modify {
          case false => (true, true)
          case _     => (true, false)
        }
        .flatMap {
          case true =>
            Log[F].info(
              "GenesisValidator: received ApprovedBlock — ceremony already complete. " +
                "Transitioning to Initializing to sync approved state."
            ) >>
              Engine.transitionToInitializing(
                blockProcessingQueue,
                blocksInProcessing,
                casperShardConf,
                Some(validatorId),
                init = noop,
                trimState = true,
                disableStateExporter = false,
                onBlockFinalized = onBlockFinalized
              )
          case false =>
            Log[F].info(
              "GenesisValidator: already transitioning via UnapprovedBlock, ignoring ApprovedBlock."
            )
        }
    case ub: UnapprovedBlock =>
      isRepeated(ub.candidate.block.blockHash)
        .ifM(
          Log[F].warn(
            s"UnapprovedBlock ${PrettyPrinter.buildString(ub.candidate.block.blockHash)} is already being verified. " +
              s"Dropping repeated message."
          ),
          ack(ub.candidate.block.blockHash) >> blockApprover
            .unapprovedBlockPacketHandler(peer, ub, casperShardConf.shardName) >>
            // Guard: if ApprovedBlock path already transitioned, skip this transition
            // but still send the BlockApproval above (harmless and correct).
            transitioned
              .modify {
                case false => (true, true)
                case _     => (true, false)
              }
              .flatMap {
                case true =>
                  Engine.transitionToInitializing(
                    blockProcessingQueue,
                    blocksInProcessing,
                    casperShardConf,
                    Some(validatorId),
                    init = noop,
                    trimState = true,
                    disableStateExporter = false,
                    onBlockFinalized = onBlockFinalized
                  )
                case false =>
                  Log[F].info(
                    "GenesisValidator: already transitioned via ApprovedBlock, " +
                      "skipping post-verification transition."
                  )
              }
        )
    case na: NoApprovedBlockAvailable => logNoApprovedBlockAvailable[F](na.nodeIdentifer)
    case _                            => noop
  }
}
