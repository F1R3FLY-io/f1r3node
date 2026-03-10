package coop.rchain.casper

/**
  * Signal handle for triggering heartbeat wakes from external sources (e.g., deploy submission).
  * Call triggerWake() to wake the heartbeat immediately for fast block proposal.
  */
trait HeartbeatSignal[F[_]] {
  def triggerWake(): F[Unit]
}
