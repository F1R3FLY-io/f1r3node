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
pub const HOT_STORE_HISTORY_CONT_CACHE_SIZE_METRIC: &str = "hot-store.history.continuations-cache.size";
pub const HOT_STORE_HISTORY_DATA_CACHE_SIZE_METRIC: &str = "hot-store.history.data-cache.size";
pub const HOT_STORE_HISTORY_JOINS_CACHE_SIZE_METRIC: &str = "hot-store.history.joins-cache.size";
pub const HOT_STORE_HISTORY_CONT_CACHE_ITEMS_METRIC: &str =
    "hot-store.history.continuations-cache.items";
pub const HOT_STORE_HISTORY_DATA_CACHE_ITEMS_METRIC: &str = "hot-store.history.data-cache.items";
pub const HOT_STORE_HISTORY_JOINS_CACHE_ITEMS_METRIC: &str =
    "hot-store.history.joins-cache.items";
pub const HOT_STORE_STATE_CONT_SIZE_METRIC: &str = "hot-store.state.continuations.size";
pub const HOT_STORE_STATE_DATA_SIZE_METRIC: &str = "hot-store.state.data.size";
pub const HOT_STORE_STATE_JOINS_SIZE_METRIC: &str = "hot-store.state.joins.size";
pub const HOT_STORE_STATE_INSTALLED_CONT_SIZE_METRIC: &str =
    "hot-store.state.installed-continuations.size";
pub const HOT_STORE_STATE_INSTALLED_JOINS_SIZE_METRIC: &str =
    "hot-store.state.installed-joins.size";
pub const HOT_STORE_STATE_CONT_ITEMS_METRIC: &str = "hot-store.state.continuations.items";
pub const HOT_STORE_STATE_DATA_ITEMS_METRIC: &str = "hot-store.state.data.items";
pub const HOT_STORE_STATE_JOINS_ITEMS_METRIC: &str = "hot-store.state.joins.items";
pub const HOT_STORE_STATE_INSTALLED_CONT_ITEMS_METRIC: &str =
    "hot-store.state.installed-continuations.items";
pub const HOT_STORE_STATE_INSTALLED_JOINS_ITEMS_METRIC: &str =
    "hot-store.state.installed-joins.items";

// Replay waiting-continuation observability metrics
pub const REPLAY_WAITING_CONTINUATIONS_STORED_TOTAL_METRIC: &str =
    "replay.waiting-continuations.stored-total";
pub const REPLAY_WAITING_CONTINUATIONS_MATCHED_TOTAL_METRIC: &str =
    "replay.waiting-continuations.matched-total";
pub const REPLAY_WAITING_CONTINUATIONS_ESTIMATE_METRIC: &str =
    "replay.waiting-continuations.estimate";
pub const REPLAY_WAITING_CONTINUATIONS_CHANNEL_DEPTH_METRIC: &str =
    "replay.waiting-continuations.channel-depth";

// RSpace tracing span names
pub const LOCKED_CONSUME_SPAN: &str = "locked-consume";
pub const LOCKED_PRODUCE_SPAN: &str = "locked-produce";
pub const RESET_SPAN: &str = "reset";
pub const REVERT_SOFT_CHECKPOINT_SPAN: &str = "revert-soft-checkpoint";
pub const CREATE_CHECKPOINT_SPAN: &str = "create-checkpoint";
pub const CHANGES_SPAN: &str = "changes";
pub const HISTORY_CHECKPOINT_SPAN: &str = "history-checkpoint";
