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
pub const HOT_STORE_HISTORY_CONT_CACHE_SIZE_METRIC: &str =
    "hot-store.history.continuations-cache.size";
pub const HOT_STORE_HISTORY_DATA_CACHE_SIZE_METRIC: &str = "hot-store.history.data-cache.size";
pub const HOT_STORE_HISTORY_JOINS_CACHE_SIZE_METRIC: &str = "hot-store.history.joins-cache.size";
pub const HOT_STORE_HISTORY_CONT_CACHE_ITEMS_METRIC: &str =
    "hot-store.history.continuations-cache.items";
pub const HOT_STORE_HISTORY_DATA_CACHE_ITEMS_METRIC: &str = "hot-store.history.data-cache.items";
pub const HOT_STORE_HISTORY_JOINS_CACHE_ITEMS_METRIC: &str = "hot-store.history.joins-cache.items";
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

// Hot-store put/get diagnostics — bottleneck #3 instrumentation.
// All counters share `source = RSPACE_METRICS_SOURCE`. ns counters accumulate
// wall-clock; *.calls + *_count counters provide call-rate context.
pub const HOT_STORE_PUT_CONT_CALLS_METRIC: &str = "hot-store.put_continuation.calls";
pub const HOT_STORE_PUT_CONT_TIME_NS_METRIC: &str = "hot-store.put_continuation.time_ns";
pub const HOT_STORE_PUT_CONT_IDENTITY_BUILD_NS_METRIC: &str =
    "hot-store.put_continuation.identity_build_ns";
pub const HOT_STORE_PUT_CONT_IDENTITY_COMPARE_NS_METRIC: &str =
    "hot-store.put_continuation.identity_compare_ns";
pub const HOT_STORE_PUT_CONT_EXISTING_COUNT_METRIC: &str =
    "hot-store.put_continuation.existing_count_sum";
pub const HOT_STORE_PUT_CONT_DUPLICATES_METRIC: &str = "hot-store.put_continuation.duplicates";
pub const HOT_STORE_PUT_CONT_HISTORY_FILL_METRIC: &str = "hot-store.put_continuation.history_fill";

pub const HOT_STORE_PUT_JOIN_CALLS_METRIC: &str = "hot-store.put_join.calls";
pub const HOT_STORE_PUT_JOIN_TIME_NS_METRIC: &str = "hot-store.put_join.time_ns";
pub const HOT_STORE_PUT_JOIN_HISTORY_FILL_METRIC: &str = "hot-store.put_join.history_fill";

pub const HOT_STORE_PUT_DATUM_CALLS_METRIC: &str = "hot-store.put_datum.calls";
pub const HOT_STORE_PUT_DATUM_TIME_NS_METRIC: &str = "hot-store.put_datum.time_ns";
pub const HOT_STORE_PUT_DATUM_HISTORY_FILL_METRIC: &str = "hot-store.put_datum.history_fill";

pub const HOT_STORE_GET_CONT_CALLS_METRIC: &str = "hot-store.get_continuations.calls";
pub const HOT_STORE_GET_CONT_HISTORY_FILL_METRIC: &str = "hot-store.get_continuations.history_fill";
pub const HOT_STORE_GET_DATA_CALLS_METRIC: &str = "hot-store.get_data.calls";
pub const HOT_STORE_GET_DATA_HISTORY_FILL_METRIC: &str = "hot-store.get_data.history_fill";
pub const HOT_STORE_GET_JOINS_CALLS_METRIC: &str = "hot-store.get_joins.calls";
pub const HOT_STORE_GET_JOINS_HISTORY_FILL_METRIC: &str = "hot-store.get_joins.history_fill";

pub const HOT_STORE_HISTORY_CACHE_BULK_CLEAR_CONT_METRIC: &str =
    "hot-store.history_cache.bulk_clear.continuations";
pub const HOT_STORE_HISTORY_CACHE_BULK_CLEAR_DATUMS_METRIC: &str =
    "hot-store.history_cache.bulk_clear.datums";
pub const HOT_STORE_HISTORY_CACHE_BULK_CLEAR_JOINS_METRIC: &str =
    "hot-store.history_cache.bulk_clear.joins";

// Space matcher — bottleneck #2 instrumentation.
pub const RSPACE_MATCHER_EXTRACT_FIRST_MATCH_CALLS_METRIC: &str =
    "rspace.matcher.extract_first_match.calls";
pub const RSPACE_MATCHER_EXTRACT_FIRST_MATCH_SUCCESS_METRIC: &str =
    "rspace.matcher.extract_first_match.success";
pub const RSPACE_MATCHER_EXTRACT_FIRST_MATCH_CANDIDATES_ITERATED_METRIC: &str =
    "rspace.matcher.extract_first_match.candidates_iterated";
pub const RSPACE_MATCHER_EXTRACT_FIRST_MATCH_PAIR_CONSTRUCTION_NS_METRIC: &str =
    "rspace.matcher.extract_first_match.pair_construction_ns";

// Cold-path history reader — bottleneck #2 backing-store instrumentation.
pub const HISTORY_FETCH_DATA_CALLS_METRIC: &str = "history.fetch_data.calls";
pub const HISTORY_FETCH_DATA_TIME_NS_METRIC: &str = "history.fetch_data.time_ns";
pub const HISTORY_FETCH_DATA_LEGACY_FALLBACK_METRIC: &str =
    "history.fetch_data.legacy_fallback_fired";
pub const HISTORY_FETCH_DATA_TRIE_READ_NS_METRIC: &str =
    "history.fetch_data.target_history_read_ns";
pub const HISTORY_FETCH_DATA_LEAF_GET_NS_METRIC: &str = "history.fetch_data.leaf_store_get_ns";
pub const HISTORY_FETCH_DATA_DESERIALIZE_NS_METRIC: &str =
    "history.fetch_data.bincode_deserialize_ns";

// RSpace tracing span names
pub const LOCKED_CONSUME_SPAN: &str = "locked-consume";
pub const LOCKED_PRODUCE_SPAN: &str = "locked-produce";
pub const RESET_SPAN: &str = "reset";
pub const REVERT_SOFT_CHECKPOINT_SPAN: &str = "revert-soft-checkpoint";
pub const CREATE_CHECKPOINT_SPAN: &str = "create-checkpoint";
pub const CHANGES_SPAN: &str = "changes";
pub const HISTORY_CHECKPOINT_SPAN: &str = "history-checkpoint";
