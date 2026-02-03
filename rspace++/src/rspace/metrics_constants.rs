// RSpace metrics sources
pub const RSPACE_METRICS_SOURCE: &str = "f1r3fly.rspace";
pub const REPLAY_RSPACE_METRICS_SOURCE: &str = "f1r3fly.rspace.replay";
pub const REPORTING_RSPACE_METRICS_SOURCE: &str = "f1r3fly.rspace.reporting";
pub const HISTORY_RSPACE_METRICS_SOURCE: &str = "f1r3fly.rspace.history";
pub const TWO_STEP_LOCK_PHASE_A_METRICS_SOURCE: &str = "f1r3fly.rspace.two-step-lock.phase-a";
pub const TWO_STEP_LOCK_PHASE_B_METRICS_SOURCE: &str = "f1r3fly.rspace.two-step-lock.phase-b";

// RSpace communication labels
pub const CONSUME_COMM_LABEL: &str = "comm.consume";
pub const PRODUCE_COMM_LABEL: &str = "comm.produce";

// RSpace timer/histogram metrics
pub const COMM_CONSUME_TIME_METRIC: &str = "comm.consume-time";
pub const COMM_PRODUCE_TIME_METRIC: &str = "comm.produce-time";
pub const INSTALL_TIME_METRIC: &str = "install-time";
pub const LOCK_ACQUIRE_TIME_METRIC: &str = "lock.acquire";

// RSpace gauge metrics
pub const LOCK_QUEUE_METRIC: &str = "lock.queue";

// RSpace tracing span names
pub const LOCKED_CONSUME_SPAN: &str = "locked-consume";
pub const LOCKED_PRODUCE_SPAN: &str = "locked-produce";
pub const RESET_SPAN: &str = "reset";
pub const REVERT_SOFT_CHECKPOINT_SPAN: &str = "revert-soft-checkpoint";
pub const CREATE_CHECKPOINT_SPAN: &str = "create-checkpoint";
pub const CHANGES_SPAN: &str = "changes";
pub const HISTORY_CHECKPOINT_SPAN: &str = "history-checkpoint";
