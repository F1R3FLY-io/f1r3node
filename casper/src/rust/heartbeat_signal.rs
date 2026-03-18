// See casper/src/main/scala/coop/rchain/casper/HeartbeatSignal.scala

use std::sync::Arc;
use std::sync::OnceLock;

/// Signal handle for triggering heartbeat wakes from external sources (e.g., deploy submission).
/// Call trigger_wake() to wake the heartbeat immediately for fast block proposal.
pub trait HeartbeatSignal: Send + Sync {
    /// Trigger the heartbeat to wake up immediately for block proposal.
    fn trigger_wake(&self);
}

/// A shared reference to an optional HeartbeatSignal.
/// This allows the signal to be set after Casper is created but before heartbeat starts.
pub type HeartbeatSignalRef = Arc<OnceLock<Arc<dyn HeartbeatSignal>>>;

/// Create a new empty heartbeat signal reference.
/// The signal will be set later when the heartbeat proposer is created.
pub fn new_heartbeat_signal_ref() -> HeartbeatSignalRef {
    Arc::new(OnceLock::new())
}

pub fn install_heartbeat_signal(
    signal_ref: &HeartbeatSignalRef,
    signal: Arc<dyn HeartbeatSignal>,
) -> bool {
    signal_ref.set(signal).is_ok()
}
