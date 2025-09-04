// See casper/src/main/scala/coop/rchain/casper/util/rholang/ReplayFailure.scala

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayFailure {
    InternalError {
        msg: String,
    },

    ReplayStatusMismatch {
        initial_failed: bool,
        replay_failed: bool,
    },

    UnusedCOMMEvent {
        msg: String,
    },

    ReplayCostMismatch {
        initial_cost: u64,
        replay_cost: u64,
    },

    SystemDeployErrorMismatch {
        play_error: String,
        replay_error: String,
    },
}

impl ReplayFailure {
    pub fn internal_error(msg: String) -> Self {
        ReplayFailure::InternalError { msg }
    }

    pub fn replay_status_mismatch(initial_failed: bool, replay_failed: bool) -> Self {
        ReplayFailure::ReplayStatusMismatch {
            initial_failed,
            replay_failed,
        }
    }

    pub fn unused_comm_event(msg: String) -> Self {
        ReplayFailure::UnusedCOMMEvent { msg }
    }

    pub fn replay_cost_mismatch(initial_cost: u64, replay_cost: u64) -> Self {
        ReplayFailure::ReplayCostMismatch {
            initial_cost,
            replay_cost,
        }
    }

    pub fn system_deploy_error_mismatch(play_error: String, replay_error: String) -> Self {
        ReplayFailure::SystemDeployErrorMismatch {
            play_error,
            replay_error,
        }
    }
}

impl std::fmt::Display for ReplayFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayFailure::InternalError { msg } => {
                write!(f, "Internal error: {}", msg)
            }
            ReplayFailure::ReplayStatusMismatch {
                initial_failed,
                replay_failed,
            } => {
                write!(
                    f,
                    "Replay status mismatch: initial_failed={}, replay_failed={}",
                    initial_failed, replay_failed
                )
            }
            ReplayFailure::UnusedCOMMEvent { msg } => {
                write!(f, "Unused COMM event: {}", msg)
            }
            ReplayFailure::ReplayCostMismatch {
                initial_cost,
                replay_cost,
            } => {
                write!(
                    f,
                    "Replay cost mismatch: initial_cost={}, replay_cost={}",
                    initial_cost, replay_cost
                )
            }
            ReplayFailure::SystemDeployErrorMismatch {
                play_error,
                replay_error,
            } => {
                write!(
                    f,
                    "System deploy error mismatch:\n  Play error: {}\n  Replay error: {}",
                    play_error, replay_error
                )
            }
        }
    }
}
