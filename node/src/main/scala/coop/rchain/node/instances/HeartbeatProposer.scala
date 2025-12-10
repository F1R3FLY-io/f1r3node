package coop.rchain.node.instances

import cats.effect.{Concurrent, Timer}
import cats.effect.concurrent.{Deferred, Ref}
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.casper.{
  HeartbeatConf,
  HeartbeatSignal,
  MultiParentCasper,
  ProposeFunction,
  ValidatorIdentity
}
import coop.rchain.casper.blocks.proposer.{
  ProposerEmpty,
  ProposerFailure,
  ProposerStarted,
  ProposerSuccess
}
import coop.rchain.casper.engine.EngineCell
import coop.rchain.casper.engine.EngineCell._
import coop.rchain.shared.{Log, Time}
import fs2.Stream
import scala.concurrent.duration._

object HeartbeatProposer {

  /**
    * Create a heartbeat proposer stream that periodically checks if a block
    * needs to be proposed to maintain liveness, or wakes immediately when signaled.
    *
    * This integrates with the existing propose queue mechanism for thread safety.
    * The heartbeat simply calls the same triggerPropose function that user deploys
    * and explicit propose calls use, ensuring serialization through ProposerInstance.
    *
    * The heartbeat can be triggered in two ways:
    * 1. Timer-based: wakes every checkInterval to check for stale LFB
    * 2. Signal-based: wakes immediately when deploy is submitted (via returned HeartbeatSignal)
    *
    * To prevent lock-step behavior between validators, the stream waits a random
    * amount of time (0 to checkInterval) before starting the periodic checks.
    *
    * The heartbeat only runs on bonded validators. It checks the active validators
    * set before proposing to avoid unnecessary attempts by unbonded nodes.
    *
    * @param triggerPropose The propose function from Setup.scala
    * @param validatorIdentity The validator identity to check if bonded
    * @param config Heartbeat configuration
    * @return A tuple of (heartbeat stream, signal handle for external wake triggers)
    */
  def create[F[_]: Concurrent: Timer: Time: Log: EngineCell](
      triggerPropose: ProposeFunction[F],
      validatorIdentity: ValidatorIdentity,
      config: HeartbeatConf
  ): F[(Stream[F, Unit], HeartbeatSignal[F])] =
    if (!config.enabled) {
      val noopSignal = new HeartbeatSignal[F] {
        def triggerWake(): F[Unit] = Concurrent[F].unit
      }
      (Stream.empty.covary[F].asInstanceOf[Stream[F, Unit]], noopSignal).pure[F]
    } else {
      for {
        initialDelay <- randomInitialDelay(config.checkInterval)

        // Create signal mechanism using Ref to track current Deferred
        initialDeferred <- Deferred[F, Unit]
        signalRef       <- Ref[F].of(initialDeferred)

        signal = new HeartbeatSignal[F] {
          def triggerWake(): F[Unit] =
            signalRef.get.flatMap(_.complete(()).attempt.void) // complete current deferred, ignore if already completed
        }

        stream = Stream.eval(
          Log[F].info(
            s"Heartbeat: Starting with random initial delay of ${initialDelay.toSeconds}s " +
              s"(check interval: ${config.checkInterval.toSeconds}s, max LFB age: ${config.maxLfbAge.toSeconds}s, signal-based wake enabled)"
          )
        ) >> Stream.sleep[F](initialDelay).flatMap { _ =>
          // Recursive stream that handles both timer and signal wakes
          def heartbeatLoop: Stream[F, Unit] =
            Stream.eval(signalRef.get).flatMap { currentDeferred =>
              val timerWake  = Stream.sleep[F](config.checkInterval).as("timer")
              val signalWake = Stream.eval(currentDeferred.get).as("signal")

              // Race between timer and signal - whichever completes first triggers wake
              timerWake.merge(signalWake).head.evalMap { source =>
                for {
                  _ <- Log[F].debug(s"Heartbeat: Woke from $source")
                  _ <- checkAndMaybePropose(triggerPropose, validatorIdentity, config)
                  // Create new deferred for next iteration
                  newDeferred <- Deferred[F, Unit]
                  _           <- signalRef.set(newDeferred)
                } yield ()
              } >> heartbeatLoop // Continue loop
            }

          heartbeatLoop
        }
      } yield (stream, signal)
    }

  /**
    * Generate a random initial delay between 0 and checkInterval to prevent
    * validators from synchronizing their heartbeat checks.
    */
  private def randomInitialDelay[F[_]: Concurrent](
      checkInterval: FiniteDuration
  ): F[FiniteDuration] =
    Concurrent[F].delay {
      val maxMillis    = checkInterval.toMillis
      val randomMillis = (math.random() * maxMillis).toLong
      randomMillis.millis
    }

  private def checkAndMaybePropose[F[_]: Concurrent: Time: Log: EngineCell](
      triggerPropose: ProposeFunction[F],
      validatorIdentity: ValidatorIdentity,
      config: HeartbeatConf
  ): F[Unit] =
    // Read from EngineCell to get Casper instance
    // This is safe even if Casper is temporarily unavailable
    Log[F].debug("Heartbeat: Checking if propose is needed") >>
      EngineCell[F].read >>= {
      _.withCasper(
        casper => doHeartbeatCheck(triggerPropose, validatorIdentity, casper, config),
        // If Casper not available yet, just skip this heartbeat check
        Log[F].debug("Heartbeat: Casper not available yet, skipping check")
      )
    }

  private def doHeartbeatCheck[F[_]: Concurrent: Time: Log](
      triggerPropose: ProposeFunction[F],
      validatorIdentity: ValidatorIdentity,
      casper: MultiParentCasper[F],
      config: HeartbeatConf
  ): F[Unit] =
    for {
      // Get current snapshot to check if validator is bonded
      snapshot <- casper.getSnapshot

      // Check if this validator is in the active validators set (bonded)
      validatorId = ByteString.copyFrom(validatorIdentity.publicKey.bytes)
      isBonded    = snapshot.onChainState.activeValidators.contains(validatorId)

      _ <- if (!isBonded) {
            Log[F].info(
              "Heartbeat: Validator is not bonded, skipping heartbeat propose"
            )
          } else {
            // Validator is bonded, proceed with heartbeat check
            Log[F].debug("Heartbeat: Validator is bonded, checking LFB age") >>
              checkLfbAndPropose(triggerPropose, validatorId, casper, config)
          }
    } yield ()

  private def checkLfbAndPropose[F[_]: Concurrent: Time: Log](
      triggerPropose: ProposeFunction[F],
      validatorId: ByteString,
      casper: MultiParentCasper[F],
      config: HeartbeatConf
  ): F[Unit] = {
    import coop.rchain.models.BlockHash.BlockHash
    import coop.rchain.dag.DagOps

    for {
      // Get current snapshot
      snapshot <- casper.getSnapshot

      // Check if we have pending user deploys
      hasPendingDeploys = snapshot.deploysInScope.nonEmpty

      // Get last finalized block
      lfb <- casper.lastFinalizedBlock

      // Check if LFB is stale
      now          <- Time[F].currentMillis
      lfbTimestamp = lfb.header.timestamp
      timeSinceLFB = now - lfbTimestamp
      lfbIsStale   = timeSinceLFB > config.maxLfbAge.toMillis

      // Check if we have new parents (new blocks since our last block)
      hasNewParents <- snapshot.dag.latestMessageHash(validatorId).flatMap {
                        case None =>
                          // Validator has no blocks yet - can propose
                          true.pure[F]
                        case Some(lastBlockHash) =>
                          // Check if this is genesis block (allows breaking deadlock after genesis)
                          snapshot.dag.lookup(lastBlockHash).flatMap {
                            case Some(blockMeta) if blockMeta.parents.isEmpty =>
                              // This is genesis - allow proposal to break post-genesis deadlock
                              Log[F].debug(
                                "Heartbeat: Validator's last block is genesis, allowing proposal"
                              ) >> true.pure[F]
                            case _ =>
                              // Get all blocks validator knew about when creating last block
                              for {
                                ancestorHashes <- DagOps
                                                   .bfTraverseF[F, BlockHash](
                                                     List(lastBlockHash)
                                                   )(
                                                     hash =>
                                                       snapshot.dag
                                                         .lookup(hash)
                                                         .map(
                                                           _.map(_.parents)
                                                             .getOrElse(List.empty)
                                                         )
                                                   )
                                                   .toList
                                ancestorSet = ancestorHashes.toSet

                                // Get current latest blocks (frontier)
                                allLatestBlocks <- snapshot.dag.latestMessageHashes

                                // Check if any of the latest blocks are new (not in ancestor set)
                                newBlocksExist = allLatestBlocks.values
                                  .exists(hash => !ancestorSet.contains(hash))
                              } yield newBlocksExist
                          }
                      }

      // Proposal logic: propose if (pending deploys) OR (LFB stale AND new parents)
      shouldPropose = hasPendingDeploys || (lfbIsStale && hasNewParents)

      _ <- if (shouldPropose) {
            val reason = if (hasPendingDeploys) {
              s"${snapshot.deploysInScope.size} pending user deploys"
            } else {
              s"LFB is stale (${timeSinceLFB}ms old, threshold: ${config.maxLfbAge.toMillis}ms) and new parents exist"
            }

            for {
              _ <- Log[F].info(s"Heartbeat: Proposing block - reason: $reason")
              // Trigger propose - this goes through the same queue as user proposes
              // The ProposerInstance semaphore ensures thread safety
              result <- triggerPropose(casper, false)
              _ <- result match {
                    case ProposerEmpty =>
                      Log[F].debug(
                        "Heartbeat: Propose already in progress, will retry next check"
                      )
                    case ProposerFailure(status, seqNum) =>
                      Log[F].warn(s"Heartbeat: Propose failed with $status (seqNum $seqNum)")
                    case ProposerSuccess(_, _) =>
                      Log[F].info(s"Heartbeat: Successfully created block")
                    case ProposerStarted(seqNum) =>
                      Log[F].info(s"Heartbeat: Async propose started (seqNum $seqNum)")
                  }
            } yield ()
          } else {
            val reason = if (!hasNewParents) {
              "no new parents (would violate validation)"
            } else if (!lfbIsStale) {
              s"LFB age is ${timeSinceLFB}ms (threshold: ${config.maxLfbAge.toMillis}ms)"
            } else {
              "unknown"
            }
            Log[F].debug(s"Heartbeat: No action needed - reason: $reason")
          }
    } yield ()
  }
}
