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
import scala.collection.immutable.SortedMap

object InterpreterUtil {

  implicit private val logSource: LogSource = LogSource(this.getClass)

  private[this] val ComputeDeploysCheckpointMetricsSource =
    Metrics.Source(CasperMetricsSource, "compute-deploys-checkpoint")

  private[this] val ComputeParentPostStateMetricsSource =
    Metrics.Source(CasperMetricsSource, "compute-parents-post-state")

  private[this] val ReplayBlockMetricsSource =
    Metrics.Source(CasperMetricsSource, "replay-block")

  /**
    * Bounded convergent LCA (Lowest Common Ancestor) algorithm.
    *
    * Walks back from all parents simultaneously in descending blockNum order,
    * tracking which original parents can reach each visited block. The first
    * block reachable from ALL parents (highest blockNum) is the LCA.
    *
    * Cost: O(blocks_between_parents_and_LCA) DAG lookups, which is typically
    * O(num_validators * few_levels) for a healthy network.
    *
    * Uses purely structural traversal (parent links only, no finalization state)
    * to guarantee determinism across all validators with the same DAG.
    *
    * NOTE: The returned visibleBlocks only contains blocks at or above the LCA.
    * It does NOT include "sibling branches below the LCA" -- blocks below the
    * LCA that are ancestors of some parents but not of the LCA. Use this function
    * only for LCA identification; the full merge scope requires a separate
    * boundedScopeWalk with allAncestors(LCA) as the stop set.
    *
    * @return (lca, visibleBlocks) where lca is the highest common ancestor and
    *         visibleBlocks contains all blocks visited during the walk (at or above LCA).
    */
  private[casper] def boundedLcaWalk[F[_]: Concurrent](
      dag: BlockDagRepresentation[F],
      parentHashes: List[BlockHash]
  ): F[(BlockHash, Set[BlockHash])] = {
    import coop.rchain.blockstorage.syntax._

    case class LcaWalkState(
        frontier: SortedMap[Long, Set[BlockHash]],
        reachableFrom: Map[BlockHash, Set[Int]],
        visited: Set[BlockHash],
        numParents: Int
    )

    type LcaResult = (BlockHash, Set[BlockHash])

    def step(state: LcaWalkState): F[Either[LcaWalkState, LcaResult]] =
      if (state.frontier.isEmpty) {
        // Should not happen with valid parents (all paths converge at genesis).
        // Fall back: return the first parent as LCA.
        val result: LcaResult = (parentHashes.head, state.visited)
        result.asRight[LcaWalkState].pure[F]
      } else {
        val (_, blocksAtHeight) = state.frontier.last
        val newFrontier         = state.frontier - state.frontier.lastKey

        blocksAtHeight.toList.traverse(dag.lookupUnsafe(_)).flatMap { metas =>
          val processResult = metas.foldLeft(
            (
              newFrontier,
              state.reachableFrom,
              state.visited,
              Option.empty[LcaResult]
            )
          ) {
            case ((fr, reach, vis, Some((lcaHash, _))), meta) =>
              // LCA already found at this height layer. Continue processing remaining
              // blocks to include partial ancestors in visibleBlocks.
              if (vis.contains(meta.blockHash)) {
                (fr, reach, vis, Some((lcaHash, vis)))
              } else {
                val newVis = vis + meta.blockHash
                (fr, reach, newVis, Some((lcaHash, newVis)))
              }
            case ((fr, reach, vis, None), meta) =>
              if (vis.contains(meta.blockHash)) {
                (fr, reach, vis, None)
              } else {
                val newVis     = vis + meta.blockHash
                val blockReach = reach.getOrElse(meta.blockHash, Set.empty)
                if (blockReach.size == state.numParents) {
                  val result: LcaResult = (meta.blockHash, newVis)
                  (fr, reach, newVis, Some(result))
                } else {
                  val updatedReach =
                    meta.parents.foldLeft(reach) { (r, parentHash) =>
                      val parentReach = r.getOrElse(parentHash, Set.empty) ++ blockReach
                      r.updated(parentHash, parentReach)
                    }
                  (fr, updatedReach, newVis, None)
                }
              }
          }

          val (updatedFrontier, updatedReach, updatedVisited, lcaFound) = processResult

          lcaFound match {
            case Some(found) =>
              found.asRight[LcaWalkState].pure[F]
            case None =>
              val alreadyInFrontier = updatedFrontier.values.flatten.toSet
              val newParents = metas
                .filter(m => updatedVisited.contains(m.blockHash))
                .flatMap(_.parents)
                .distinct
                .filterNot(h => updatedVisited.contains(h) || alreadyInFrontier.contains(h))

              newParents
                .traverse(h => dag.lookupUnsafe(h).map(m => (h, m.blockNum)))
                .map { parentNums =>
                  val finalFrontier = parentNums.foldLeft(updatedFrontier) {
                    case (f, (h, bNum)) =>
                      val existing = f.getOrElse(bNum, Set.empty)
                      f.updated(bNum, existing + h)
                  }
                  Left(
                    LcaWalkState(finalFrontier, updatedReach, updatedVisited, state.numParents)
                  )
                }
          }
        }
      }

    for {
      parentMetas <- parentHashes.traverse(dag.lookupUnsafe(_))
      initialFrontier = parentMetas.foldLeft(SortedMap.empty[Long, Set[BlockHash]]) {
        case (fm, meta) =>
          val existing = fm.getOrElse(meta.blockNum, Set.empty)
          fm.updated(meta.blockNum, existing + meta.blockHash)
      }
      initialReach = parentHashes.zipWithIndex.map {
        case (h, i) => h -> Set(i)
      }.toMap
      initialState = LcaWalkState(initialFrontier, initialReach, Set.empty, parentHashes.size)
      result       <- Concurrent[F].tailRecM(initialState)(step)
    } yield result
  }

  /**
    * Bounded scope walk: discovers blocks reachable from startHashes that are
    * NOT in the given stopSet. Walks backwards via parent links, stopping each
    * path when it reaches a block in stopSet.
    *
    * Used to find blocks that are ancestors of the parents but not ancestors of
    * the LCA ("sibling branches below the LCA"). Combined with the stopSet
    * (which is allAncestors(LCA)), this produces the complete merge scope.
    *
    * Cost: O(|result|) DAG lookups -- proportional to the number of blocks
    * discovered, NOT the full chain length.
    *
    * @param startHashes blocks to walk backwards from (typically the parent hashes)
    * @param stopSet     blocks to treat as boundary (typically allAncestors(LCA));
    *                    paths stop when they hit a block in this set
    * @return set of blocks reachable from startHashes that are not in stopSet
    */
  private def boundedScopeWalk[F[_]: Concurrent](
      dag: BlockDagRepresentation[F],
      startHashes: List[BlockHash],
      stopSet: Set[BlockHash]
  ): F[Set[BlockHash]] = {
    import coop.rchain.blockstorage.syntax._

    case class WalkState(queue: List[BlockHash], result: Set[BlockHash])

    def step(state: WalkState): F[Either[WalkState, Set[BlockHash]]] =
      state.queue match {
        case Nil => state.result.asRight[WalkState].pure[F]
        case head :: tail =>
          if (state.result.contains(head) || stopSet.contains(head)) {
            WalkState(tail, state.result).asLeft[Set[BlockHash]].pure[F]
          } else {
            dag.lookupUnsafe(head).map { meta =>
              val newResult = state.result + head
              val newParents =
                meta.parents.filterNot(p => newResult.contains(p) || stopSet.contains(p))
              WalkState(tail ++ newParents, newResult).asLeft[Set[BlockHash]]
            }
          }
      }

    Concurrent[F].tailRecM(WalkState(startHashes, Set.empty))(step)
  }

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

          // Compute scope: all ancestors of parents (blocks visible from these parents).
          //
          // Two-phase algorithm to reduce cost from O(N * chain_length) to
          // O(chain_length + |actualBlocks|):
          //
          // Phase 1: Find the LCA using a bounded convergent walk from parents.
          //   Cost: O(blocks between parents and LCA), typically small.
          //
          // Phase 2: Compute allAncestors(LCA) -- single O(chain_length) traversal.
          //   Then discover blocks reachable from parents but NOT in allAncestors(LCA)
          //   using a bounded walk that stops at the allAncestors(LCA) boundary.
          //   This correctly captures "sibling branches below the LCA" -- blocks
          //   below the LCA height that are ancestors of some parents but not of the
          //   LCA itself. These carry state changes not in the LCA's post-state.
          //
          // Result: visibleBlocks = allAncestors(LCA) + scopeWalkResult
          //       = allAncestors(all_parents), identical to N separate allAncestors calls.
          //
          // CRITICAL: Uses purely structural traversal (parent links only, no
          // finalization state) to guarantee determinism across all validators.
          val parentHashes = parents.map(_.blockHash)

          for {
            // Phase 1: Bounded LCA walk -- find the highest common ancestor.
            // Only used for the LCA hash; the walk's visibleBlocks are NOT used
            // for the merge scope (they miss sibling branches below the LCA).
            lcaResult <- boundedLcaWalk(s.dag, parentHashes.toList)
            (lca, _)  = lcaResult

            // Phase 2a: Full traversal of LCA's ancestry -- O(chain_length).
            // This is the single expensive call. It gives us all blocks whose
            // state is already baked into the LCA's post-state.
            lcaAncestors <- s.dag.allAncestors(lca)

            // Phase 2b: Bounded scope walk from parents, stopping at lcaAncestors.
            // Discovers blocks reachable from parents that are NOT ancestors of the
            // LCA. This includes blocks above the LCA AND sibling branches below it.
            // Cost: O(|result|), proportional to blocks being merged.
            nonLcaBlocks <- boundedScopeWalk(s.dag, parentHashes.toList, lcaAncestors)

            // Combine: equivalent to allAncestors(all_parents)
            visibleBlocks = lcaAncestors ++ nonLcaBlocks

            // Get the LCA block to use its post-state as the merge base
            lcaBlock <- BlockStore[F].getUnsafe(lca)
            lcaState = Blake2b256Hash.fromByteString(lcaBlock.body.state.postStateHash)

            parentHashStr = parentHashes.map(h => PrettyPrinter.buildString(h)).mkString(", ")
            lcaStr        = PrettyPrinter.buildString(lca)
            lcaStateStr   = PrettyPrinter.buildString(lcaBlock.body.state.postStateHash)
            _ <- Log[F].info(
                  s"computeParentsPostState: parents=[$parentHashStr], " +
                    s"LCA=$lcaStr (block ${lcaBlock.body.state.blockNumber}), " +
                    s"LCA state=$lcaStateStr, visibleBlocks=${visibleBlocks.size}, " +
                    s"nonLcaBlocks=${nonLcaBlocks.size}"
                )

            // Pass preComputedLfbAncestors = lcaAncestors (the FULL allAncestors(lca)
            // set) to DagMerger to avoid a redundant O(chain_length) traversal.
            r <- DagMerger.merge[F](
                  s.dag,
                  lca,
                  lcaState,
                  blockIndexF(_).map(_.deployChains),
                  runtimeManager.getHistoryRepo,
                  DagMerger.costOptimalRejectionAlg,
                  Some(visibleBlocks),
                  disableLateBlockFiltering,
                  preComputedLfbAncestors = Some(lcaAncestors)
                )
            (state, rejected) = r
          } yield (ByteString.copyFrom(state.bytes.toArray), rejected)

        }
      }
    }
}
