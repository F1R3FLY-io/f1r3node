package coop.rchain.casper.util.rholang

import cats.effect._
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.BlockStore
import coop.rchain.blockstorage.dag.BlockDagRepresentation
import coop.rchain.casper.InvalidBlock.InvalidRejectedDeploy
import coop.rchain.casper._
import coop.rchain.casper.merging.{BlockIndex, DagMerger}
import coop.rchain.casper.protocol._
import coop.rchain.casper.syntax._
import coop.rchain.casper.util.ProtoUtil
import coop.rchain.casper.util.rholang.RuntimeManager._
import coop.rchain.crypto.signatures.Signed
import coop.rchain.metrics.{Metrics, Span}
import coop.rchain.models.BlockHash.BlockHash
import coop.rchain.models.NormalizerEnv.ToEnvMap
import coop.rchain.models.Validator.Validator
import coop.rchain.models.syntax.modelsSyntaxByteString
import coop.rchain.models.{NormalizerEnv, Par}
import coop.rchain.rholang.interpreter.SystemProcesses.BlockData
import coop.rchain.rholang.interpreter.compiler.Compiler
import coop.rchain.rholang.interpreter.errors.InterpreterError
import coop.rchain.rholang.interpreter.merging.RholangMergingLogic
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.shared.{Log, LogSource}
import monix.eval.Coeval
import retry._

import scala.collection.Seq

object InterpreterUtil {

  implicit private val logSource: LogSource = LogSource(this.getClass)

  private[this] val ComputeDeploysCheckpointMetricsSource =
    Metrics.Source(CasperMetricsSource, "compute-deploys-checkpoint")

  private[this] val ComputeParentPostStateMetricsSource =
    Metrics.Source(CasperMetricsSource, "compute-parents-post-state")

  private[this] val ReplayBlockMetricsSource =
    Metrics.Source(CasperMetricsSource, "replay-block")

  def mkTerm[Env](rho: String, normalizerEnv: NormalizerEnv[Env])(
      implicit ev: ToEnvMap[Env]
  ): Either[Throwable, Par] =
    Compiler[Coeval].sourceToADT(rho, normalizerEnv.toEnv).runAttempt

  //Returns (None, checkpoints) if the block's tuplespace hash
  //does not match the computed hash based on the deploys
  def validateBlockCheckpoint[F[_]: Concurrent: Log: BlockStore: Span: Metrics: Timer](
      block: BlockMessage,
      s: CasperSnapshot[F],
      runtimeManager: RuntimeManager[F]
  ): F[BlockProcessing[Option[StateHash]]] = {
    val incomingPreStateHash = ProtoUtil.preStateHash(block)
    for {
      _                   <- Span[F].mark("before-unsafe-get-parents")
      parents             <- ProtoUtil.getParents(block)
      _                   <- Span[F].mark("before-compute-parents-post-state")
      disableLateFilter   = s.onChainState.shardConf.disableLateBlockFiltering
      computedParentsInfo <- computeParentsPostState(parents, s, runtimeManager, disableLateFilter).attempt
      _                   <- Log[F].info(s"Computed parents post state for ${PrettyPrinter.buildString(block)}.")
      result <- computedParentsInfo match {
                 case Left(ex) =>
                   BlockStatus.exception(ex).asLeft[Option[StateHash]].pure
                 case Right((computedPreStateHash, rejectedDeploys @ _)) =>
                   val rejectedDeployIds = rejectedDeploys.toSet
                   if (incomingPreStateHash != computedPreStateHash) {
                     //TODO at this point we may just as well terminate the replay, there's no way it will succeed.
                     Log[F]
                       .error(
                         s"CRITICAL: Computed pre-state hash ${PrettyPrinter.buildString(computedPreStateHash)} does not equal block's pre-state hash ${PrettyPrinter
                           .buildString(incomingPreStateHash)}. Refusing to equivocate, dropping block #${block.body.state.blockNumber} (${PrettyPrinter
                           .buildString(block.blockHash)})."
                       )
                       .as(none[StateHash].asRight[BlockError])
                   } else if (rejectedDeployIds != block.body.rejectedDeploys.map(_.sig).toSet) {
                     // Detailed logging for InvalidRejectedDeploy mismatch
                     val blockRejectedIds  = block.body.rejectedDeploys.map(_.sig).toSet
                     val extraInComputed   = rejectedDeployIds.diff(blockRejectedIds)
                     val missingInComputed = blockRejectedIds.diff(rejectedDeployIds)

                     // Get all deploy signatures in the block for duplicate detection
                     val allBlockDeploys = block.body.deploys.map(_.deploy.sig)
                     val allDeploySigs   = allBlockDeploys ++ block.body.rejectedDeploys.map(_.sig)
                     val duplicates =
                       allDeploySigs.groupBy(identity).filter(_._2.size > 1).keys.toSet

                     // Try to correlate rejected deploy sigs with actual deploy data
                     val deployDataMap =
                       block.body.deploys.map(pd => pd.deploy.sig -> pd.deploy).toMap

                     def analyzeDeploySig(sig: ByteString): String = {
                       val sigStr      = PrettyPrinter.buildString(sig)
                       val isDuplicate = if (duplicates.contains(sig)) " [DUPLICATE]" else ""
                       val deployInfo = deployDataMap.get(sig) match {
                         case Some(deploy) =>
                           s" (term=${deploy.data.term.take(50)}..., timestamp=${deploy.data.timestamp}, phloLimit=${deploy.data.phloLimit})"
                         case None =>
                           " (deploy data not found in block)"
                       }
                       s"$sigStr$isDuplicate$deployInfo"
                     }

                     Log[F]
                       .error(
                         s"""
                       |=== InvalidRejectedDeploy Analysis ===
                       |Block #${block.body.state.blockNumber} (${PrettyPrinter.buildString(
                              block.blockHash
                            )})
                       |Sender: ${PrettyPrinter.buildString(block.sender)}
                       |Parents: ${parents
                              .map(p => PrettyPrinter.buildString(p.blockHash))
                              .mkString(", ")}
                       |
                       |Rejected deploy mismatch:
                       |  Validator computed: ${rejectedDeployIds.size} rejected deploys
                       |  Block contains:     ${blockRejectedIds.size} rejected deploys
                       |
                       |Extra in computed (validator wants to reject, but block creator didn't):
                       |  Count: ${extraInComputed.size}
                       |${if (extraInComputed.nonEmpty)
                              extraInComputed.map(analyzeDeploySig).mkString("  ", "\n  ", "")
                            else "  None"}
                       |
                       |Missing in computed (block creator rejected, but validator doesn't think should be):
                       |  Count: ${missingInComputed.size}
                       |${if (missingInComputed.nonEmpty)
                              missingInComputed.map(analyzeDeploySig).mkString("  ", "\n  ", "")
                            else "  None"}
                       |
                       |Duplicates found in block: ${duplicates.size}
                       |${if (duplicates.nonEmpty)
                              duplicates
                                .map(PrettyPrinter.buildString)
                                .mkString("  ", "\n  ", "")
                            else "  None"}
                       |
                       |All deploys in block: ${allBlockDeploys.size}
                       |All rejected in block: ${blockRejectedIds.size}
                       |========================================
                       |""".stripMargin
                       )
                       .as(InvalidRejectedDeploy.asLeft)
                   } else {
                     for {
                       replayResult <- replayBlock(
                                        incomingPreStateHash,
                                        block,
                                        s.dag,
                                        runtimeManager
                                      )
                       result <- handleErrors(ProtoUtil.postStateHash(block), replayResult)
                     } yield result
                   }
               }
    } yield result
  }

  private def replayBlock[F[_]: Sync: Log: BlockStore: Timer](
      initialStateHash: StateHash,
      block: BlockMessage,
      dag: BlockDagRepresentation[F],
      runtimeManager: RuntimeManager[F]
  )(implicit spanF: Span[F]): F[Either[ReplayFailure, StateHash]] =
    spanF.trace(ReplayBlockMetricsSource) {
      val internalDeploys       = ProtoUtil.deploys(block)
      val internalSystemDeploys = ProtoUtil.systemDeploys(block)

      // Check for duplicate deploys in the block before replay
      val allDeploySigs    = internalDeploys.map(_.deploy.sig) ++ block.body.rejectedDeploys.map(_.sig)
      val deployDuplicates = allDeploySigs.groupBy(identity).filter(_._2.size > 1)
      val hasDuplicates    = deployDuplicates.nonEmpty

      for {
        _ <- if (hasDuplicates) {
              Log[F].warn(
                s"""
                |=== Duplicate Deploys Detected in Block ===
                |Block #${block.body.state.blockNumber} (${PrettyPrinter.buildString(
                     block.blockHash
                   )})
                |Found ${deployDuplicates.size} duplicate deploy signatures:
                |${deployDuplicates
                     .map {
                       case (sig, occurrences) =>
                         s"  ${PrettyPrinter.buildString(sig)} (appears ${occurrences.size} times)"
                     }
                     .mkString("\n")}
                |Total deploys: ${internalDeploys.size}
                |Total rejected: ${block.body.rejectedDeploys.size}
                |============================================
                |""".stripMargin
              )
            } else {
              Log[F].debug(
                s"Block #${block.body.state.blockNumber}: replaying ${internalDeploys.size} deploys, ${block.body.rejectedDeploys.size} rejected"
              )
            }
        invalidBlocksSet <- dag.invalidBlocks
        unseenBlocksSet  <- ProtoUtil.unseenBlockHashes(dag, block)
        seenInvalidBlocksSet = invalidBlocksSet.filterNot(
          block => unseenBlocksSet.contains(block.blockHash)
        ) // TODO: Write test in which switching this to .filter makes it fail
        invalidBlocks = seenInvalidBlocksSet
          .map(block => (block.blockHash, block.sender))
          .toMap
        _         <- Span[F].mark("before-process-pre-state-hash")
        blockData = BlockData.fromBlock(block)
        isGenesis = block.header.parentsHashList.isEmpty
        replayResultF = runtimeManager.replayComputeState(initialStateHash)(
          internalDeploys,
          internalSystemDeploys,
          blockData,
          invalidBlocks,
          isGenesis
        )
        replayResult <- retryingOnFailures[Either[ReplayFailure, StateHash]](
                         RetryPolicies.limitRetries(3), {
                           case Right(stateHash) => stateHash == block.body.state.postStateHash
                           case _                => false
                         },
                         (e, retryDetails) =>
                           e match {
                             case Right(stateHash) =>
                               Log[F].error(
                                 s"Replay block ${PrettyPrinter.buildStringNoLimit(block.blockHash)} with " +
                                   s"${PrettyPrinter.buildStringNoLimit(block.body.state.postStateHash)} " +
                                   s"got tuple space mismatch error with error hash ${PrettyPrinter
                                     .buildStringNoLimit(stateHash)}, retries details: ${retryDetails}"
                               )
                             case Left(replayError) =>
                               Log[F].error(
                                 s"Replay block ${PrettyPrinter.buildStringNoLimit(block.blockHash)} got " +
                                   s"error ${replayError}, retries details: ${retryDetails}"
                               )
                           }
                       )(replayResultF)
      } yield replayResult
    }

  private def handleErrors[F[_]: Sync: Log](
      tsHash: ByteString,
      result: Either[ReplayFailure, StateHash]
  ): F[BlockProcessing[Option[StateHash]]] =
    result.pure.flatMap {
      case Left(status) =>
        status match {
          case InternalError(throwable) =>
            BlockStatus
              .exception(
                new Exception(
                  s"Internal errors encountered while processing deploy: ${throwable.getMessage}"
                )
              )
              .asLeft[Option[StateHash]]
              .pure
          case ReplayStatusMismatch(replayFailed, initialFailed) =>
            Log[F]
              .warn(
                s"Found replay status mismatch; replay failure is $replayFailed and orig failure is $initialFailed"
              )
              .as(none[StateHash].asRight[BlockError])
          case UnusedCOMMEvent(replayException) =>
            Log[F]
              .warn(
                s"Found replay exception: ${replayException.getMessage}"
              )
              .as(none[StateHash].asRight[BlockError])
          case ReplayCostMismatch(initialCost, replayCost) =>
            Log[F]
              .warn(
                s"Found replay cost mismatch: initial deploy cost = $initialCost, replay deploy cost = $replayCost"
              )
              .as(none[StateHash].asRight[BlockError])
          // Restructure errors so that this case is unnecessary
          case SystemDeployErrorMismatch(playMsg, replayMsg) =>
            Log[F]
              .warn(
                s"Found system deploy error mismatch: initial deploy error message = $playMsg, replay deploy error message = $replayMsg"
              )
              .as(none[StateHash].asRight[BlockError])
        }
      case Right(computedStateHash) =>
        if (tsHash == computedStateHash) {
          // state hash in block matches computed hash!
          computedStateHash.some.asRight[BlockError].pure
        } else {
          // state hash in block does not match computed hash -- invalid!
          // return no state hash, do not update the state hash set
          Log[F]
            .warn(
              s"Tuplespace hash ${PrettyPrinter.buildString(tsHash)} does not match computed hash ${PrettyPrinter
                .buildString(computedStateHash)}."
            )
            .as(none[StateHash].asRight[BlockError])
        }
    }

  /**
    * Temporary solution to print user deploy errors to Log so we can have
    * at least some way to debug user errors.
    */
  def printDeployErrors[F[_]: Sync: Log](
      deploySig: ByteString,
      errors: Seq[InterpreterError]
  ): F[Unit] = Sync[F].defer {
    val deployInfo = PrettyPrinter.buildStringSig(deploySig)
    Log[F].info(s"Deploy ($deployInfo) errors: ${errors.mkString(", ")}")
  }

  def computeDeploysCheckpoint[F[_]: Concurrent: BlockStore: Log: Metrics](
      parents: Seq[BlockMessage],
      deploys: Seq[Signed[DeployData]],
      systemDeploys: Seq[SystemDeploy],
      s: CasperSnapshot[F],
      runtimeManager: RuntimeManager[F],
      blockData: BlockData,
      invalidBlocks: Map[BlockHash, Validator]
  )(
      implicit spanF: Span[F]
  ): F[
    (StateHash, StateHash, Seq[ProcessedDeploy], Seq[ByteString], Seq[ProcessedSystemDeploy])
  ] =
    spanF.trace(ComputeDeploysCheckpointMetricsSource) {
      for {
        nonEmptyParents <- parents.pure
                            .ensure(new IllegalArgumentException("Parents must not be empty"))(
                              _.nonEmpty
                            )
        disableLateFilter = s.onChainState.shardConf.disableLateBlockFiltering
        computedParentsInfo <- computeParentsPostState(
                                nonEmptyParents,
                                s,
                                runtimeManager,
                                disableLateFilter
                              )
        (preStateHash, rejectedDeploys) = computedParentsInfo
        result <- runtimeManager.computeState(preStateHash)(
                   deploys,
                   systemDeploys,
                   blockData,
                   invalidBlocks
                 )
        (postStateHash, processedDeploys, processedSystemDeploys) = result
      } yield (
        preStateHash,
        postStateHash,
        processedDeploys,
        rejectedDeploys,
        processedSystemDeploys
      )
    }

  def computeParentsPostState[F[_]: Concurrent: BlockStore: Log: Metrics](
      parents: Seq[BlockMessage],
      s: CasperSnapshot[F],
      runtimeManager: RuntimeManager[F],
      disableLateBlockFiltering: Boolean = false
  )(implicit spanF: Span[F]): F[(StateHash, Seq[ByteString])] =
    spanF.trace(ComputeParentPostStateMetricsSource) {
      parents match {
        // For genesis, use empty trie's root hash
        case Seq() =>
          (RuntimeManager.emptyStateHashFixed, Seq.empty[ByteString]).pure[F]

        // For single parent, get its post state hash
        case Seq(parent) =>
          (ProtoUtil.postStateHash(parent), Seq.empty[ByteString]).pure[F]

        // we might want to take some data from the parent with the most stake,
        // e.g. bonds map, slashing deploys, bonding deploys.
        // such system deploys are not mergeable, so take them from one of the parents.
        case _ => {
          val blockIndexF = (v: BlockHash) => {
            val cached = BlockIndex.cache.get(v).map(_.pure)
            cached.getOrElse {
              for {
                b         <- BlockStore[F].getUnsafe(v)
                preState  = b.body.state.preStateHash
                postState = b.body.state.postStateHash
                sender    = b.sender.toByteArray
                seqNum    = b.seqNum

                mergeableChs <- runtimeManager.loadMergeableChannels(postState, sender, seqNum)

                // KEEP system deploys in the index, but they will be filtered out during conflict
                // detection in DagMerger. System deploys (CloseBlockDeploy, SlashDeploy, etc.) are
                // block-specific and have IDs that include the block hash. They should NOT participate
                // in multi-parent merge conflict resolution (only user deploys can conflict), but they
                // need to be in the index so mergeable channels count matches deploy count.
                blockIndex <- BlockIndex(
                               b.blockHash,
                               b.body.deploys,
                               b.body.systemDeploys,
                               preState.toBlake2b256Hash,
                               postState.toBlake2b256Hash,
                               runtimeManager.getHistoryRepo,
                               mergeableChs
                             )
              } yield blockIndex
            }
          }

          // Compute scope: all ancestors of parents (blocks visible from these parents)
          val parentHashes = parents.map(_.blockHash)

          for {
            // Get all ancestors of all parents (including the parents themselves)
            // Use bounded traversal that stops at finalized blocks to prevent O(chain_length) growth
            ancestorSets <- parentHashes.toList.traverse(
                             h => s.dag.withAncestors(h, bh => s.dag.isFinalized(bh).map(!_))
                           )
            // Each set includes the parent itself, so intersect to find common ancestors
            ancestorSetsWithParents = parentHashes.toList.zip(ancestorSets).map {
              case (parent, ancestors) => ancestors + parent
            }
            visibleBlocks = ancestorSetsWithParents.flatten.toSet

            // Find the lowest common ancestor of all parents.
            // This is the highest block that is an ancestor of ALL parents.
            // This is deterministic because it depends only on DAG structure, not finalization state.
            commonAncestors = ancestorSetsWithParents.reduce(_ intersect _)
            commonAncestorsWithHeight <- commonAncestors.toList.traverse { h =>
                                          s.dag.lookupUnsafe(h).map(m => (h, m.blockNum))
                                        }
            // The LCA is the common ancestor with the highest block number
            // Fall back to genesis if no common ancestor found (shouldn't happen with valid parents)
            lcaOpt = if (commonAncestorsWithHeight.nonEmpty)
              Some(commonAncestorsWithHeight.maxBy(_._2)._1)
            else
              None
            // Use LCA as the LFB for computing descendants, fall back to snapshot LFB
            lfbForDescendants: BlockHash = lcaOpt.getOrElse(s.lastFinalizedBlock)

            // Get the LFB block to use its post-state as the merge base
            lfbBlock <- BlockStore[F].getUnsafe(lfbForDescendants)
            lfbState = Blake2b256Hash.fromByteString(lfbBlock.body.state.postStateHash)

            parentHashStr  = parentHashes.map(h => PrettyPrinter.buildString(h)).mkString(", ")
            lcaStr         = PrettyPrinter.buildString(lfbForDescendants)
            lcaStateStr    = PrettyPrinter.buildString(lfbBlock.body.state.postStateHash)
            snapshotLfbStr = PrettyPrinter.buildString(s.lastFinalizedBlock)
            _ <- Log[F].info(
                  s"computeParentsPostState: parents=[$parentHashStr], " +
                    s"commonAncestors=${commonAncestors.size}, LCA=$lcaStr (block ${lfbBlock.body.state.blockNumber}), " +
                    s"LCA state=$lcaStateStr, visibleBlocks=${visibleBlocks.size}, snapshotLFB=$snapshotLfbStr"
                )

            r <- DagMerger.merge[F](
                  s.dag,
                  lfbForDescendants,
                  lfbState,
                  blockIndexF(_).map(_.deployChains),
                  runtimeManager.getHistoryRepo,
                  DagMerger.costOptimalRejectionAlg,
                  Some(visibleBlocks),
                  disableLateBlockFiltering
                )
            (state, rejected) = r
          } yield (ByteString.copyFrom(state.bytes.toArray), rejected)

        }
      }
    }
}
