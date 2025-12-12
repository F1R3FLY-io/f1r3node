package coop.rchain.casper.merging

import cats.effect.Concurrent
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.blockstorage.dag.BlockDagRepresentation
import coop.rchain.blockstorage.syntax._
import coop.rchain.models.BlockHash.BlockHash
import coop.rchain.rholang.interpreter.RhoRuntime.RhoHistoryRepository
import coop.rchain.rholang.interpreter.merging.RholangMergingLogic
import coop.rchain.rspace.HotStoreTrieAction
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.merger.MergingLogic.NumberChannelsDiff
import coop.rchain.rspace.merger.{ChannelChange, MergingLogic, StateChange, StateChangeMerger}
import coop.rchain.rspace.syntax._
import coop.rchain.shared.Log
import scodec.bits.ByteVector

object DagMerger {

  def costOptimalRejectionAlg: DeployChainIndex => Long =
    (r: DeployChainIndex) => r.deploysWithCost.map(_.cost).sum

  // Helper function to detect system deploy IDs
  // System deploy IDs are 33 bytes: [32-byte blockHash][1-byte marker]
  // Markers: 0x01 (slash), 0x02 (close block), 0x03 (empty/heartbeat)
  private def isSystemDeployId(id: ByteString): Boolean =
    id.size == 33 && {
      val lastByte = id.byteAt(32)
      lastByte == 1 || lastByte == 2 || lastByte == 3
    }

  def merge[F[_]: Concurrent: Log](
      dag: BlockDagRepresentation[F],
      lfb: BlockHash,
      lfbPostState: Blake2b256Hash,
      index: BlockHash => F[Vector[DeployChainIndex]],
      historyRepository: RhoHistoryRepository[F],
      rejectionCostF: DeployChainIndex => Long,
      scope: Option[Set[BlockHash]] = None,
      disableLateBlockFiltering: Boolean = false
  ): F[(Blake2b256Hash, Seq[ByteString])] =
    for {
      // Get ancestors of LFB (blocks whose state is already included in LFB's post-state)
      lfbAncestors <- dag.allAncestors(lfb)

      // Blocks to merge are all blocks in scope that are NOT the LFB or its ancestors.
      // This includes:
      // 1. Descendants of LFB (blocks built on top of LFB)
      // 2. Siblings of LFB (blocks at same height but different branch) that are ancestors of the tips
      // Previously we only included descendants, which missed deploy effects from sibling branches.
      // Note: lfbAncestors includes the LFB itself (via allAncestors)
      actualBlocks <- scope match {
                       case Some(scopeBlocks) =>
                         // Include all scope blocks except LFB and its ancestors
                         (scopeBlocks -- lfbAncestors).pure[F]
                       case None =>
                         // Legacy behavior: use descendants of LFB
                         dag.descendants(lfb)
                     }

      // Late blocks: With the new actualBlocks definition that includes sibling branches,
      // there are no "late" blocks when scope is provided - all non-ancestor blocks are in actualBlocks.
      // Late block filtering is now only relevant for legacy code paths without scope.
      lateBlocks <- if (disableLateBlockFiltering || scope.isDefined) {
                     // No late blocks when scope is provided (all relevant blocks are in actualBlocks)
                     Set.empty[BlockHash].pure[F]
                   } else {
                     // Legacy: query nonFinalizedBlocks (non-deterministic, but no scope means
                     // this is not a multi-parent merge validation)
                     dag.nonFinalizedBlocks.map(_ diff actualBlocks)
                   }

      // Log the block sets for debugging
      _ <- Log[F].info(
            s"DagMerger.merge: LFB=${ByteVector.view(lfb.toByteArray).toHex.take(16)}..., " +
              s"scope=${scope.fold("ALL")(s => s"${s.size} blocks")}, " +
              s"actualBlocks (above LFB)=${actualBlocks.size}, " +
              s"lfbAncestors=${lfbAncestors.size}, " +
              s"lateBlocks=${lateBlocks.size}"
          )

      // Convert to sorted lists to ensure deterministic iteration order
      actualSet       <- actualBlocks.toList.traverse(index).map(_.flatten.toSet)
      actualSetSorted = actualSet.toList.sorted
      // TODO reject only late units conflicting with finalised body
      lateSet       <- lateBlocks.toList.traverse(index).map(_.flatten.toSet)
      lateSetSorted = lateSet.toList.sorted

      branchesAreConflicting = (as: Set[DeployChainIndex], bs: Set[DeployChainIndex]) => {
        // Filter out system deploy IDs - they should never be treated as conflicts
        // System deploys are deterministic and execute the same way in all branches
        val asUserDeploys = as.flatMap(_.deploysWithCost.map(_.id)).filterNot(isSystemDeployId)
        val bsUserDeploys = bs.flatMap(_.deploysWithCost.map(_.id)).filterNot(isSystemDeployId)

        // Also filter system deploys from event log comparison
        val asUserIndices =
          as.filter(idx => idx.deploysWithCost.forall(d => !isSystemDeployId(d.id)))
        val bsUserIndices =
          bs.filter(idx => idx.deploysWithCost.forall(d => !isSystemDeployId(d.id)))

        val sameDeployInBoth = (asUserDeploys intersect bsUserDeploys).nonEmpty
        val aEventLog        = asUserIndices.toList.sorted.map(_.eventLogIndex).combineAll
        val bEventLog        = bsUserIndices.toList.sorted.map(_.eventLogIndex).combineAll
        val eventLogConflict = MergingLogic.areConflicting(aEventLog, bEventLog)

        // Debug: log conflict reason when conflict is detected
        if (sameDeployInBoth || eventLogConflict) {
          val asDeployIds =
            asUserDeploys.map(id => ByteVector.view(id.toByteArray).toHex.take(16)).mkString(",")
          val bsDeployIds =
            bsUserDeploys.map(id => ByteVector.view(id.toByteArray).toHex.take(16)).mkString(",")
          val conflictReasonStr = if (sameDeployInBoth) {
            s"sameDeployInBoth: ${(asUserDeploys intersect bsUserDeploys)
              .map(id => ByteVector.view(id.toByteArray).toHex.take(16))
              .mkString(",")}"
          } else {
            MergingLogic.conflictReason(aEventLog, bEventLog).getOrElse("unknown")
          }
          println(
            s"[DEBUG] CONFLICT DETECTED: [$asDeployIds] vs [$bsDeployIds] - reason: $conflictReasonStr"
          )
        }

        sameDeployInBoth || eventLogConflict
      }
      historyReader <- historyRepository.getHistoryReader(lfbPostState)
      baseReader    = historyReader.readerBinary
      baseGetData   = historyReader.getData _
      overrideTrieAction = (
          hash: Blake2b256Hash,
          changes: ChannelChange[ByteVector],
          numberChs: NumberChannelsDiff
      ) =>
        numberChs.get(hash).traverse {
          RholangMergingLogic.calculateNumberChannelMerge(hash, _, changes, baseGetData)
        }
      computeTrieActions = (changes: StateChange, mergeableChs: NumberChannelsDiff) =>
        StateChangeMerger.computeTrieActions(
          changes,
          baseReader,
          mergeableChs,
          overrideTrieAction
        )

      applyTrieActions = (actions: Seq[HotStoreTrieAction]) =>
        historyRepository.reset(lfbPostState).flatMap(_.doCheckpoint(actions).map(_.root))

      r <- ConflictSetMerger.merge[F, DeployChainIndex](
            actualSeq = actualSetSorted,
            lateSeq = lateSetSorted,
            depends =
              (target, source) => MergingLogic.depends(target.eventLogIndex, source.eventLogIndex),
            conflicts = branchesAreConflicting,
            cost = rejectionCostF,
            stateChanges = _.stateChanges.pure,
            mergeableChannels = _.eventLogIndex.numberChannelsData,
            computeTrieActions = computeTrieActions,
            applyTrieActions = applyTrieActions,
            getData = historyReader.getData
          )

      (newState, rejected) = r
      rejectedDeploys      = rejected.flatMap(_.deploysWithCost.map(_.id))

      // Filter out system deploy IDs - they should not appear in block's rejected deploys
      // System deploys are deterministic and excluded from conflict detection
      rejectedUserDeploys = rejectedDeploys.filterNot(isSystemDeployId)

      // Sort rejected deploys to ensure deterministic ordering using ByteVector
      rejectedDeploysSorted = rejectedUserDeploys.toList.sorted(
        Ordering.by((bs: ByteString) => ByteVector(bs.toByteArray))
      )

      // Log merge summary at debug level
      _ <- Log[F].debug(
            s"DagMerger.merge: LFB=${ByteVector.view(lfb.toByteArray).toHex.take(16)}..., " +
              s"scope=${scope.fold("ALL")(s => s"${s.size}")}, " +
              s"actual=${actualBlocks.size}, late=${lateBlocks.size}, " +
              s"rejected=${rejectedDeploysSorted.size}"
          )
      // Log rejected deploys at info level only if there are any
      _ <- if (rejectedDeploysSorted.nonEmpty) {
            import coop.rchain.casper.PrettyPrinter
            Log[F].info(
              s"DagMerger rejected ${rejectedDeploysSorted.size} deploys: " +
                rejectedDeploysSorted.map(bs => PrettyPrinter.buildString(bs)).mkString(", ")
            )
          } else {
            ().pure[F]
          }
    } yield (newState, rejectedDeploysSorted)
}
