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

// inj_attempt phase histograms — split the `evaluate` cost into the four
// internal phases so the relative weight of source parsing vs. AST reduction
// is observable in production metrics.
pub const INJ_ATTEMPT_SET_INITIAL_COST_TIME_METRIC: &str = "inj-attempt.set-initial-cost.time";
pub const INJ_ATTEMPT_CHARGE_PARSING_COST_TIME_METRIC: &str =
    "inj-attempt.charge-parsing-cost.time";
pub const INJ_ATTEMPT_BUILD_NORMALIZED_TERM_TIME_METRIC: &str =
    "inj-attempt.build-normalized-term.time";
pub const INJ_ATTEMPT_REDUCE_TERM_TIME_METRIC: &str = "inj-attempt.reduce-term.time";

// fold_match recursion diagnostics — bottleneck #2 in fold_match.rs.
// Counts every recursive frame and accumulates time spent in tail-vec clones.
pub const RHOLANG_MATCHER_FOLD_MATCH_CALLS_METRIC: &str = "rholang.matcher.fold_match.calls";
pub const RHOLANG_MATCHER_FOLD_MATCH_RECURSION_DEPTH_TOTAL_METRIC: &str =
    "rholang.matcher.fold_match.recursion_depth_total";
pub const RHOLANG_MATCHER_FOLD_MATCH_TAIL_CLONE_NS_METRIC: &str =
    "rholang.matcher.fold_match.tail_clone_ns";

// Reducer per-op-type counters — split reduce_term cost by the Rholang
// AST node kind dispatched in DebruijnInterpreter::generated_message_eval.
// Calls counter increments once per dispatch; time_ns accumulates wall-clock
// nanoseconds spent inside that branch. Same pattern as rspace.produce.*.
pub const REDUCER_EVAL_SEND_CALLS_METRIC: &str = "reducer.eval_send.calls";
pub const REDUCER_EVAL_SEND_TIME_NS_METRIC: &str = "reducer.eval_send.time_ns";
pub const REDUCER_EVAL_RECEIVE_CALLS_METRIC: &str = "reducer.eval_receive.calls";
pub const REDUCER_EVAL_RECEIVE_TIME_NS_METRIC: &str = "reducer.eval_receive.time_ns";
pub const REDUCER_EVAL_NEW_CALLS_METRIC: &str = "reducer.eval_new.calls";
pub const REDUCER_EVAL_NEW_TIME_NS_METRIC: &str = "reducer.eval_new.time_ns";
pub const REDUCER_EVAL_MATCH_CALLS_METRIC: &str = "reducer.eval_match.calls";
pub const REDUCER_EVAL_MATCH_TIME_NS_METRIC: &str = "reducer.eval_match.time_ns";

// Runtime counters/gauges for checkpoint and event-log churn diagnostics
pub const RUNTIME_SOFT_CHECKPOINT_TOTAL_METRIC: &str = "runtime_soft_checkpoint_total";
pub const RUNTIME_CHECKPOINT_TOTAL_METRIC: &str = "runtime_checkpoint_total";
pub const RUNTIME_REVERT_SOFT_CHECKPOINT_TOTAL_METRIC: &str =
    "runtime_revert_soft_checkpoint_total";
pub const RUNTIME_TAKE_EVENT_LOG_TOTAL_METRIC: &str = "runtime_take_event_log_total";
pub const RUNTIME_TAKE_EVENT_LOG_EVENTS_TOTAL_METRIC: &str = "runtime_take_event_log_events_total";
pub const RUNTIME_TAKE_EVENT_LOG_LAST_EVENTS_METRIC: &str = "runtime_take_event_log_last_events";
