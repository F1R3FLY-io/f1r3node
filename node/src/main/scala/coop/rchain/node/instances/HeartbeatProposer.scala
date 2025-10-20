package coop.rchain.node.instances

import cats.effect.{Concurrent, Timer}
import cats.syntax.all._
import coop.rchain.casper.{HeartbeatConf, MultiParentCasper, ProposeFunction}
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

object HeartbeatProposer {

  /**
    * Create a heartbeat proposer stream that periodically checks if a block
    * needs to be proposed to maintain liveness.
    *
    * This integrates with the existing propose queue mechanism for thread safety.
    * The heartbeat simply calls the same triggerPropose function that user deploys
    * and explicit propose calls use, ensuring serialization through ProposerInstance.
    *
    * @param triggerPropose The propose function from Setup.scala
    * @param config Heartbeat configuration
    * @return A stream that runs the heartbeat loop
    */
  def create[F[_]: Concurrent: Timer: Time: Log: EngineCell](
      triggerPropose: ProposeFunction[F],
      config: HeartbeatConf
  ): Stream[F, Unit] =
    if (!config.enabled) {
      Stream.empty
    } else {
      Stream
        .awakeEvery[F](config.checkInterval)
        .evalMap(_ => checkAndMaybePropose(triggerPropose, config))
    }

  private def checkAndMaybePropose[F[_]: Concurrent: Time: Log: EngineCell](
      triggerPropose: ProposeFunction[F],
      config: HeartbeatConf
  ): F[Unit] =
    // Read from EngineCell to get Casper instance
    // This is safe even if Casper is temporarily unavailable
    EngineCell[F].read >>= {
      _.withCasper(
        casper => doHeartbeatCheck(triggerPropose, casper, config),
        // If Casper not available yet, just skip this heartbeat check
        Log[F].debug("Heartbeat: Casper not available yet, skipping check")
      )
    }

  private def doHeartbeatCheck[F[_]: Concurrent: Time: Log](
      triggerPropose: ProposeFunction[F],
      casper: MultiParentCasper[F],
      config: HeartbeatConf
  ): F[Unit] =
    for {
      // Get last finalized block
      lfb <- casper.lastFinalizedBlock

      // Check if LFB is stale
      now           <- Time[F].currentMillis
      lfbTimestamp  = lfb.header.timestamp
      timeSinceLFB  = now - lfbTimestamp
      shouldPropose = timeSinceLFB > config.maxLfbAge.toMillis

      _ <- if (shouldPropose) {
            for {
              _ <- Log[F].info(
                    s"Heartbeat: LFB is ${timeSinceLFB}ms old (threshold: ${config.maxLfbAge.toMillis}ms), triggering propose"
                  )
              // Trigger propose - this goes through the same queue as user proposes
              // The ProposerInstance semaphore ensures thread safety
              result <- triggerPropose(casper, false)
              _ <- result match {
                    case ProposerEmpty =>
                      Log[F].debug("Heartbeat: Propose already in progress, will retry next check")
                    case ProposerFailure(status, seqNum) =>
                      Log[F].warn(s"Heartbeat: Propose failed with $status (seqNum $seqNum)")
                    case ProposerSuccess(_, _) =>
                      Log[F].info(
                        s"Heartbeat: Successfully created block"
                      )
                    case ProposerStarted(seqNum) =>
                      Log[F].info(s"Heartbeat: Async propose started (seqNum $seqNum)")
                  }
            } yield ()
          } else {
            Log[F].debug(s"Heartbeat: LFB age is ${timeSinceLFB}ms, no action needed")
          }
    } yield ()
}
