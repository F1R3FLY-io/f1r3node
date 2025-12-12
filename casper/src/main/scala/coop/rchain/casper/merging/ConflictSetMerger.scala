package coop.rchain.casper.merging

import cats.effect.Concurrent
import cats.syntax.all._
import coop.rchain.models.ListParWithRandom
import coop.rchain.rspace.HotStoreTrieAction
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.history.HistoryReaderBinary
import coop.rchain.rspace.merger.MergingLogic._
import coop.rchain.rspace.merger.StateChange._
import coop.rchain.rspace.merger._
import coop.rchain.rspace.serializers.ScodecSerialize.DatumB
import coop.rchain.shared.{Log, Stopwatch}
import coop.rchain.rholang.interpreter.merging.RholangMergingLogic.convertToReadNumber
import coop.rchain.rspace.internal.Datum

object ConflictSetMerger {

  /** R is a type for minimal rejection unit.
    * IMPORTANT: actualSeq and lateSeq must be passed in sorted order to ensure
    * deterministic processing across all validators. */
  def merge[F[_]: Concurrent: Log, R: Ordering](
      actualSeq: Seq[R],
      lateSeq: Seq[R],
      depends: (R, R) => Boolean,
      conflicts: (Set[R], Set[R]) => Boolean,
      cost: R => Long,
      stateChanges: R => F[StateChange],
      mergeableChannels: R => NumberChannelsDiff,
      computeTrieActions: (StateChange, NumberChannelsDiff) => F[Vector[HotStoreTrieAction]],
      applyTrieActions: Seq[HotStoreTrieAction] => F[Blake2b256Hash],
      getData: Blake2b256Hash => F[Seq[Datum[ListParWithRandom]]]
  ): F[(Blake2b256Hash, Set[R])] = {

    // Convert to Sets for set operations, but use Seq for ordered iteration
    val actualSet = actualSeq.toSet
    val lateSet   = lateSeq.toSet

    type Branch = Set[R]

    /** compute optimal rejection configuration */
    def getOptimalRejection(
        options: Set[Set[Branch]],
        targetF: Branch => Long
    ): Set[Branch] = {
      require(
        options.map(_.map(_.head)).size == options.size,
        "Same rejection unit is found in two rejection options. Please report this to code maintainer."
      )
      options.toList
      // reject set with min sum of target function output,
      // if equal value - min size of a branch,
      // if equal size - sorted by head of rejection set option
        .sortBy(b => (b.map(targetF).sum, b.size, b.head.head))
        .headOption
        .getOrElse(Set.empty)
    }

    def calMergedResult(
        branch: Branch,
        originResult: Map[Blake2b256Hash, Long]
    ): Option[Map[Blake2b256Hash, Long]] = {
      val diff = branch.map(mergeableChannels).toList.combineAll
      diff.foldLeft[Option[Map[Blake2b256Hash, Long]]](Some(originResult)) {
        case (baOpt, br) =>
          baOpt.flatMap {
            case ba =>
              try {
                val result = Math.addExact(ba.getOrElse(br._1, 0L), br._2)
                if (result < 0) {
                  none
                } else {
                  Some(ba.updated(br._1, result))
                }
              } catch {
                case _: ArithmeticException => none
              }
          }

      }
    }

    // Ordering for Branch (Set[R]) to ensure deterministic iteration
    implicit val branchOrdering: Ordering[Branch] = (x: Branch, y: Branch) => {
      // Compare by sorted elements
      val xSorted = x.toVector.sorted
      val ySorted = y.toVector.sorted
      val lenCmp  = xSorted.length.compareTo(ySorted.length)
      if (lenCmp != 0) lenCmp
      else {
        xSorted
          .zip(ySorted)
          .map { case (a, b) => Ordering[R].compare(a, b) }
          .find(_ != 0)
          .getOrElse(0)
      }
    }

    def foldRejection(
        baseBalance: Map[Blake2b256Hash, Long],
        branches: Set[Branch]
    ): Set[Branch] = {
      // Sort branches to ensure deterministic processing order
      val sortedBranches = branches.toVector.sorted
      val (_, rejected) = sortedBranches.foldLeft((baseBalance, Set.empty[Branch])) {
        case ((balances, rejected), deploy) =>
          // TODO come up with a better algorithm to solve below case
          // currently we are accumulating result from some order and reject the deploy once negative result happens
          // which doesn't seem perfect cases below
          //
          // base result 10 and folding the result from order like [-10, -1, 20]
          // which on the second case `-1`, the calculation currently would reject it because the result turns
          // into negative.However, if you look at the all the item view 10 - 10 -1 + 20 is not negative
          calMergedResult(deploy, balances).fold((balances, rejected + deploy))((_, rejected))
      }
      rejected
    }

    def getMergedResultRejection(
        branches: Set[Branch],
        rejectOptions: Set[Set[Branch]],
        base: Map[Blake2b256Hash, Long]
    ): Set[Set[Branch]] =
      if (rejectOptions.isEmpty) {
        Set(foldRejection(base, branches))
      } else {
        rejectOptions.map {
          case normalRejectOptions =>
            val rejected = foldRejection(base, branches diff normalRejectOptions)
            rejected ++ normalRejectOptions
        }
      }

    val (rejectedAsDependents, mergeSet) =
      actualSet.partition(t => lateSet.exists(depends(t, _)))

    // Debug: log late set and dependents
    val lateSetInfo    = lateSet.size
    val dependentsInfo = rejectedAsDependents.size

    /** split merging set into branches without cross dependencies
      * TODO make dependencies directional, maintain dependency graph. Now if l depends on r or otherwise - it does not matter. */
    val (branches, branchesTime) =
      Stopwatch.profile(computeRelatedSets[R](mergeSet, (l, r) => depends(l, r)))

    /** map of conflicting branches */
    val (conflictMap, conflictsMapTime) =
      Stopwatch.profile(computeRelationMap[Set[R]](branches, conflicts))

    // TODO reject only units that are conflicting + dependent, its not necessary to reject the whole branch
    /** rejection options that leave only non conflicting branches */
    val (rejectionOptions, rejectionOptionsTime) =
      Stopwatch.profile(computeRejectionOptions(conflictMap))

    /** target function for rejection is minimising cost of deploys rejected */
    val rejectionTargetF = (dc: Branch) => dc.map(cost).sum

    for {
      // Sort keys for deterministic ordering across JVM instances
      baseMergeableChRes <- branches.flatten
                             .flatMap(mergeableChannels(_).keys)
                             .toVector
                             .sorted
                             .traverse(
                               channelHash =>
                                 convertToReadNumber(getData)
                                   .apply(channelHash)
                                   .map(res => (channelHash, res.getOrElse(0L)))
                             )
                             .map(_.toMap)
      rejectionOptionsWithOverflow = getMergedResultRejection(
        branches,
        rejectionOptions,
        baseMergeableChRes
      )
      optimalRejection = getOptimalRejection(rejectionOptionsWithOverflow, rejectionTargetF)
      rejected         = lateSet ++ rejectedAsDependents ++ optimalRejection.flatten
      toMerge          = branches diff optimalRejection

      // Detailed INFO logging for rejection breakdown (always visible)
      _ <- Log[F].info(
            s"ConflictSetMerger rejection breakdown: " +
              s"lateSet=${lateSet.size}, " +
              s"rejectedAsDependents=${rejectedAsDependents.size}, " +
              s"optimalRejection=${optimalRejection.flatten.size}, " +
              s"total rejected=${rejected.size}, " +
              s"branches=${branches.size}, " +
              s"toMerge=${toMerge.size}, " +
              s"conflictMap entries with conflicts=${conflictMap.count(_._2.nonEmpty)}, " +
              s"rejectionOptions=${rejectionOptions.size}, " +
              s"rejectionOptionsWithOverflow=${rejectionOptionsWithOverflow.size}"
          )
      // Sort toMerge for deterministic processing order
      toMergeSorted = toMerge.toVector.sorted.flatMap(_.toVector.sorted)
      r             <- Stopwatch.duration(toMergeSorted.traverse(stateChanges).map(_.combineAll))

      (allChanges, combineAllChanges) = r

      // All number channels merged
      // TODO: Negative or overflow should be rejected before!
      allMergeableChannels = toMergeSorted.map(mergeableChannels).combineAll

      r                                 <- Stopwatch.duration(computeTrieActions(allChanges, allMergeableChannels))
      (trieActions, computeActionsTime) = r
      r                                 <- Stopwatch.duration(applyTrieActions(trieActions))
      (newState, applyActionsTime)      = r
      overallChanges                    = s"${allChanges.datumsChanges.size} D, ${allChanges.kontChanges.size} K, ${allChanges.consumeChannelsToJoinSerializedMap.size} J"
      logStr = s"Merging done: " +
        s"late set size ${lateSet.size}; " +
        s"actual set size ${actualSet.size}; " +
        s"computed branches (${branches.size}) in ${branchesTime}; " +
        s"conflicts map in ${conflictsMapTime}; " +
        s"rejection options (${rejectionOptions.size}) in ${rejectionOptionsTime}; " +
        s"optimal rejection set size ${optimalRejection.size}; " +
        s"rejected as late dependency ${rejectedAsDependents.size}; " +
        s"changes combined (${overallChanges}) in ${combineAllChanges}; " +
        s"trie actions (${trieActions.size}) in ${computeActionsTime}; " +
        s"actions applied in ${applyActionsTime}"
      _ <- Log[F].debug(logStr)
    } yield (newState, rejected)
  }
}
