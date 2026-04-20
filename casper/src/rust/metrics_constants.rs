// Casper metrics sources
pub const CASPER_METRICS_SOURCE: &str = "f1r3fly.casper";
pub const MERGING_METRICS_SOURCE: &str = "f1r3fly.casper.merging";
pub const RUNNING_METRICS_SOURCE: &str = "f1r3fly.casper.running";
pub const BLOCK_RETRIEVER_METRICS_SOURCE: &str = "f1r3fly.casper.block-retriever";
pub const APPROVE_BLOCK_METRICS_SOURCE: &str = "f1r3fly.casper.approve-block";
pub const REPORT_REPLAY_METRICS_SOURCE: &str = "f1r3fly.casper.report-replay";
pub const ESTIMATOR_METRICS_SOURCE: &str = "f1r3fly.casper.estimator";
pub const TIPS0_METRICS_SOURCE: &str = "f1r3fly.casper.estimator.tips0";
pub const TIPS1_METRICS_SOURCE: &str = "f1r3fly.casper.estimator.tips1";
pub const VALIDATOR_METRICS_SOURCE: &str = "f1r3fly.casper.validator";
pub const RHO_RUNTIME_METRICS_SOURCE: &str = "f1r3fly.casper.rho-runtime";
pub const REPLAY_RHO_RUNTIME_METRICS_SOURCE: &str = "f1r3fly.casper.replay-rho-runtime";
pub const BLOCK_PROCESSOR_METRICS_SOURCE: &str = "f1r3fly.casper.block-processor";
pub const CREATE_BLOCK_METRICS_SOURCE: &str = "f1r3fly.create-block";
pub const BLOCK_API_METRICS_SOURCE: &str = "f1r3fly.block-api";
pub const DEPLOY_API_METRICS_SOURCE: &str = "f1r3fly.block-api.deploy";
pub const GET_BLOCK_API_METRICS_SOURCE: &str = "f1r3fly.block-api.get-block";
pub const REPORTING_RUNTIME_METRICS_SOURCE: &str = "f1r3fly.rholang.reportingRuntime";

// Casper counter metrics
pub const BLOCK_HASH_RECEIVED_METRIC: &str = "block.hash.received";
pub const BLOCK_REQUEST_RECEIVED_METRIC: &str = "block.request.received";
pub const BLOCK_REQUESTS_TOTAL_METRIC: &str = "block.requests.total";
pub const BLOCK_REQUESTS_RETRIES_METRIC: &str = "block.requests.retries";
pub const BLOCK_REQUESTS_RETRY_ACTION_METRIC: &str = "block.requests.retry.action";
pub const BLOCK_REQUESTS_STALE_EVICTIONS_METRIC: &str = "block.requests.stale-evictions";
pub const BLOCK_RETRIEVER_DEP_RECOVERY_TRACKING_SIZE_METRIC: &str =
    "block.retriever.dep-recovery-tracking.size";
pub const BLOCK_RETRIEVER_BROADCAST_TRACKING_SIZE_METRIC: &str =
    "block.retriever.broadcast-tracking.size";
pub const BLOCK_RETRIEVER_REQUESTED_BLOCKS_SIZE_METRIC: &str =
    "block.retriever.requested-blocks.size";
pub const BLOCK_RETRIEVER_WAITING_LIST_TOTAL_SIZE_METRIC: &str =
    "block.retriever.waiting-list.total.size";
pub const BLOCK_RETRIEVER_PEERS_TOTAL_SIZE_METRIC: &str = "block.retriever.peers.total.size";
pub const ACTIVE_VALIDATORS_CACHE_SIZE_METRIC: &str = "active-validators-cache.size";
pub const DEPLOYS_IN_SCOPE_SIZE_METRIC: &str = "deploys-in-scope.size";
pub const DEPLOYS_IN_SCOPE_SIG_BYTES_ESTIMATE_METRIC: &str = "deploys-in-scope.sig-bytes-estimate";
pub const BLOCK_INDEX_CACHE_SIZE_METRIC: &str = "block-index-cache.size";
pub const PARENTS_POST_STATE_CACHE_SIZE_METRIC: &str = "parents-post-state-cache.size";
pub const PROPOSER_QUEUE_PENDING_METRIC: &str = "proposer.queue.pending";
pub const PROPOSER_QUEUE_REJECTED_TOTAL_METRIC: &str = "proposer.queue.rejected.total";
pub const INIT_BLOCK_MESSAGE_QUEUE_PENDING_METRIC: &str = "init.block-message.queue.pending";
pub const INIT_TUPLE_SPACE_QUEUE_PENDING_METRIC: &str = "init.tuple-space.queue.pending";
pub const DAG_BLOCKS_SIZE_METRIC: &str = "dag.blocks.size";
pub const DAG_CHILDREN_INDEX_SIZE_METRIC: &str = "dag.children-index.size";
pub const DAG_HEIGHTS_SIZE_METRIC: &str = "dag.heights.size";
pub const DAG_FINALIZED_BLOCKS_SIZE_METRIC: &str = "dag.finalized-blocks.size";
pub const GENESIS_METRIC: &str = "genesis";
pub const BLOCK_VALIDATION_SUCCESS_METRIC: &str = "block.validation.success";
pub const BLOCK_VALIDATION_FAILED_METRIC: &str = "block.validation.failed";
pub const CASPER_INIT_ATTEMPTS_METRIC: &str = "casper.init.attempts";
pub const CASPER_INIT_RETRY_NO_APPROVED_BLOCK_METRIC: &str = "casper.init.retry.no-approved-block";
pub const CASPER_INIT_APPROVED_BLOCK_RECEIVED_METRIC: &str = "casper.init.approved-block.received";
pub const CASPER_INIT_TRANSITION_TO_RUNNING_METRIC: &str = "casper.init.transition-to-running";
pub const ALLOCATOR_TRIM_TOTAL_METRIC: &str = "allocator.trim.total";
// TODO: Port MergeableChannelsGC metric when PR #367 is merged
// See: https://github.com/F1R3FLY-io/f1r3node/pull/367
// pub const MERGEABLE_CHANNELS_GC_DELETED_METRIC: &str = "mergeable.channels.gc.deleted";

// Casper timer metrics (recorded as histograms with _seconds suffix)
pub const BLOCK_PROCESSING_VALIDATION_SETUP_TIME_METRIC: &str =
    "block.processing.stage.validation-setup.time";
pub const BLOCK_VALIDATION_TIME_METRIC: &str = "block.validation.time";
pub const BLOCK_PROCESSING_STORAGE_TIME_METRIC: &str = "block.processing.stage.storage.time";
pub const BLOCK_PROCESSING_REPLAY_TIME_METRIC: &str = "block.processing.stage.replay.time";
pub const BLOCK_PROCESSING_PARENTS_POST_STATE_TIME_METRIC: &str =
    "block.processing.stage.parents-post-state.time";
pub const DAG_MERGE_TOTAL_TIME_METRIC: &str = "dag.merge.total.time";
pub const DAG_MERGE_INDEX_TIME_METRIC: &str = "dag.merge.index.time";
pub const DAG_MERGE_CONFLICT_TIME_METRIC: &str = "dag.merge.conflict.time";
pub const DAG_MERGE_COMPUTE_TRIE_ACTIONS_TIME_METRIC: &str = "dag.merge.compute-trie-actions.time";
pub const DAG_MERGE_APPLY_TRIE_ACTIONS_TIME_METRIC: &str = "dag.merge.apply-trie-actions.time";
pub const DAG_MERGE_SCOPE_METRIC: &str = "dag.merge.scope";
pub const DAG_MERGE_BRANCHES_TIME_METRIC: &str = "dag.merge.branches.time";
pub const DAG_MERGE_CONFLICTS_MAP_TIME_METRIC: &str = "dag.merge.conflicts-map.time";
pub const DAG_MERGE_REJECTION_OPTIONS_TIME_METRIC: &str = "dag.merge.rejection-options.time";
pub const BLOCK_REPLAY_SYSDEPLOY_EVAL_TIME_METRIC: &str = "block.replay.sysdeploy.eval.time";
pub const BLOCK_REPLAY_SYSDEPLOY_CHECK_TIME_METRIC: &str = "block.replay.sysdeploy.check.time";
pub const CASPER_INIT_TIME_TO_APPROVED_BLOCK_METRIC: &str = "casper.init.time-to-approved-block";
pub const CASPER_INIT_TIME_TO_RUNNING_METRIC: &str = "casper.init.time-to-running";

// Casper record/histogram metrics
pub const BLOCK_SIZE_METRIC: &str = "block.size";
pub const BLOCK_DOWNLOAD_END_TO_END_TIME_METRIC: &str = "block.download.end-to-end-time";
pub const BLOCK_REPLAY_PHASE_RESET_TIME_METRIC: &str = "block.replay.phase.reset.time";
pub const BLOCK_REPLAY_PHASE_USER_DEPLOYS_TIME_METRIC: &str =
    "block.replay.phase.user-deploys.time";
pub const BLOCK_REPLAY_PHASE_SYSTEM_DEPLOYS_TIME_METRIC: &str =
    "block.replay.phase.system-deploys.time";
pub const BLOCK_REPLAY_PHASE_CREATE_CHECKPOINT_TIME_METRIC: &str =
    "block.replay.phase.create-checkpoint.time";
pub const BLOCK_REPLAY_SYSDEPLOY_CHECKPOINT_MERGEABLE_TIME_METRIC: &str =
    "block.replay.sysdeploy.checkpoint-mergeable.time";
pub const BLOCK_REPLAY_SYSDEPLOY_RIG_TIME_METRIC: &str = "block.replay.sysdeploy.rig.time";
pub const BLOCK_REPLAY_SYSDEPLOY_EVAL_EVALUATE_SOURCE_TIME_METRIC: &str =
    "block.replay.sysdeploy.eval.evaluate-source.time";
pub const BLOCK_REPLAY_SYSDEPLOY_EVAL_CONSUME_RESULT_TIME_METRIC: &str =
    "block.replay.sysdeploy.eval.consume-result.time";

// Per-deploy replay breakdown metrics
pub const BLOCK_REPLAY_DEPLOY_RIG_TIME_METRIC: &str = "block.replay.deploy.rig.time";
pub const BLOCK_REPLAY_DEPLOY_PRECHARGE_TIME_METRIC: &str = "block.replay.deploy.precharge.time";
pub const BLOCK_REPLAY_DEPLOY_EVALUATE_TIME_METRIC: &str = "block.replay.deploy.evaluate.time";
pub const BLOCK_REPLAY_DEPLOY_REFUND_TIME_METRIC: &str = "block.replay.deploy.refund.time";
pub const BLOCK_REPLAY_DEPLOY_DISCARD_EVENT_LOG_TIME_METRIC: &str =
    "block.replay.deploy.discard-event-log.time";
pub const BLOCK_REPLAY_DEPLOY_CHECK_REPLAY_DATA_TIME_METRIC: &str =
    "block.replay.deploy.check-replay-data.time";

// Runtime spawn timing metrics
pub const RUNTIME_SPAWN_TIME_METRIC: &str = "runtime.spawn.time";
pub const RUNTIME_SPAWN_REPLAY_TIME_METRIC: &str = "runtime.spawn-replay.time";

// Block validation step time metrics (7 variants)
pub const BLOCK_VALIDATION_STEP_BLOCK_SUMMARY_TIME_METRIC: &str =
    "block.validation.step.block-summary.time";
pub const BLOCK_VALIDATION_STEP_CHECKPOINT_TIME_METRIC: &str =
    "block.validation.step.checkpoint.time";
pub const BLOCK_VALIDATION_STEP_BONDS_CACHE_TIME_METRIC: &str =
    "block.validation.step.bonds-cache.time";
pub const BLOCK_VALIDATION_STEP_NEGLECTED_INVALID_BLOCK_TIME_METRIC: &str =
    "block.validation.step.neglected-invalid-block.time";
pub const BLOCK_VALIDATION_STEP_NEGLECTED_EQUIVOCATION_TIME_METRIC: &str =
    "block.validation.step.neglected-equivocation.time";
pub const BLOCK_VALIDATION_STEP_PHLO_PRICE_TIME_METRIC: &str =
    "block.validation.step.phlo-price.time";
pub const BLOCK_VALIDATION_STEP_SIMPLE_EQUIVOCATION_TIME_METRIC: &str =
    "block.validation.step.simple-equivocation.time";

// Casper tracing span names
pub const TIPS0_SPAN: &str = "tips0";
pub const TIPS1_SPAN: &str = "tips1";
pub const DEPLOY_SPAN: &str = "deploy";
pub const GET_BLOCK_SPAN: &str = "get-block";
pub const CREATE_BLOCK_SPAN: &str = "create-block";
pub const DO_PROPOSE_SPAN: &str = "do-propose";
pub const COMPUTE_STATE_SPAN: &str = "compute-state";
pub const PLAY_DEPLOYS_SPAN: &str = "play-deploys";
pub const COMPUTE_GENESIS_SPAN: &str = "compute-genesis";
pub const PRECHARGE_SPAN: &str = "precharge";
pub const REFUND_SPAN: &str = "refund";
pub const USER_DEPLOY_SPAN: &str = "user-deploy";
pub const PLAY_DEPLOY_SPAN: &str = "play-deploy";
pub const EVALUATE_SYSTEM_SOURCE_SPAN: &str = "evaluate-system-source";
pub const CONSUME_SYSTEM_RESULT_SPAN: &str = "consume-system-result";
pub const REPLAY_COMPUTE_STATE_SPAN: &str = "replay-compute-state";
pub const REPLAY_DEPLOY_SPAN: &str = "replay-deploy";
pub const REPLAY_SYS_DEPLOY_SPAN: &str = "replay-sys-deploy";
pub const CREATE_CHECKPOINT_SPAN: &str = "create-checkpoint";
pub const REPLAY_SYSTEM_DEPLOY_SPAN: &str = "replay-system-deploy";
pub const COMPUTE_MAX_CLIQUE_WEIGHT_SPAN: &str = "compute-max-clique-weight";
pub const NORMALIZED_FAULT_TOLERANCE_SPAN: &str = "normalized-fault-tolerance";
pub const FINALIZER_RUN_SPAN: &str = "finalizer-run";
