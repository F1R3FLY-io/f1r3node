// Rholang metrics sources
pub const RHOLANG_METRICS_SOURCE: &str = "f1r3fly.rholang";
pub const INTERPRETER_METRICS_SOURCE: &str = "f1r3fly.rholang.interpreter";
pub const RUNTIME_METRICS_SOURCE: &str = "f1r3fly.rholang.runtime";
pub const CREATE_PLAY_METRICS_SOURCE: &str = "f1r3fly.rholang.runtime.create-play";
pub const CREATE_REPLAY_METRICS_SOURCE: &str = "f1r3fly.rholang.runtime.create-replay";

// Rholang interpreter tracing span names
pub const SET_INITIAL_COST_SPAN: &str = "set-initial-cost";
pub const CHARGE_PARSING_COST_SPAN: &str = "charge-parsing-cost";
pub const BUILD_NORMALIZED_TERM_SPAN: &str = "build-normalized-term";
pub const REDUCE_TERM_SPAN: &str = "reduce-term";

// Rholang runtime tracing span names
pub const CREATE_CHECKPOINT_SPAN: &str = "create-checkpoint";
pub const CREATE_SOFT_CHECKPOINT_SPAN: &str = "create-soft-checkpoint";
pub const CREATE_PLAY_SPAN: &str = "create-play";
pub const CREATE_REPLAY_SPAN: &str = "create-replay";

// Rholang histogram metrics
pub const EVALUATE_TIME_METRIC: &str = "evaluate";
pub const CREATE_CHECKPOINT_TIME_METRIC: &str = "create-checkpoint";
pub const CREATE_SOFT_CHECKPOINT_TIME_METRIC: &str = "create-soft-checkpoint";
pub const REDUCE_TIME_METRIC: &str = "reduce";
