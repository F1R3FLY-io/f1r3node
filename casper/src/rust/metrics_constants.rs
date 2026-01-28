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
pub const GENESIS_METRIC: &str = "genesis";
pub const BLOCK_VALIDATION_SUCCESS_METRIC: &str = "block.validation.success";
pub const BLOCK_VALIDATION_FAILED_METRIC: &str = "block.validation.failed";
// TODO: Port MergeableChannelsGC metric when PR #367 is merged
// See: https://github.com/F1R3FLY-io/f1r3node/pull/367
// pub const MERGEABLE_CHANNELS_GC_DELETED_METRIC: &str = "mergeable.channels.gc.deleted";

// Casper timer metrics (recorded as histograms with _seconds suffix)
pub const BLOCK_PROCESSING_VALIDATION_SETUP_TIME_METRIC: &str = "block.processing.stage.validation-setup.time";
pub const BLOCK_VALIDATION_TIME_METRIC: &str = "block.validation.time";
pub const BLOCK_PROCESSING_STORAGE_TIME_METRIC: &str = "block.processing.stage.storage.time";
pub const BLOCK_PROCESSING_REPLAY_TIME_METRIC: &str = "block.processing.stage.replay.time";
pub const BLOCK_REPLAY_SYSDEPLOY_EVAL_TIME_METRIC: &str = "block.replay.sysdeploy.eval.time";
pub const BLOCK_REPLAY_SYSDEPLOY_CHECK_TIME_METRIC: &str = "block.replay.sysdeploy.check.time";

// Casper record/histogram metrics
pub const BLOCK_SIZE_METRIC: &str = "block.size";
pub const BLOCK_DOWNLOAD_END_TO_END_TIME_METRIC: &str = "block.download.end-to-end-time";
pub const BLOCK_REPLAY_PHASE_RESET_TIME_METRIC: &str = "block.replay.phase.reset.time";
pub const BLOCK_REPLAY_PHASE_USER_DEPLOYS_TIME_METRIC: &str = "block.replay.phase.user-deploys.time";
pub const BLOCK_REPLAY_PHASE_SYSTEM_DEPLOYS_TIME_METRIC: &str = "block.replay.phase.system-deploys.time";
pub const BLOCK_REPLAY_PHASE_CREATE_CHECKPOINT_TIME_METRIC: &str = "block.replay.phase.create-checkpoint.time";
pub const BLOCK_REPLAY_SYSDEPLOY_CHECKPOINT_MERGEABLE_TIME_METRIC: &str = "block.replay.sysdeploy.checkpoint-mergeable.time";
pub const BLOCK_REPLAY_SYSDEPLOY_RIG_TIME_METRIC: &str = "block.replay.sysdeploy.rig.time";
pub const BLOCK_REPLAY_SYSDEPLOY_EVAL_EVALUATE_SOURCE_TIME_METRIC: &str = "block.replay.sysdeploy.eval.evaluate-source.time";
pub const BLOCK_REPLAY_SYSDEPLOY_EVAL_CONSUME_RESULT_TIME_METRIC: &str = "block.replay.sysdeploy.eval.consume-result.time";

// Block validation step time metrics (7 variants)
pub const BLOCK_VALIDATION_STEP_BLOCK_SUMMARY_TIME_METRIC: &str = "block.validation.step.block-summary.time";
pub const BLOCK_VALIDATION_STEP_CHECKPOINT_TIME_METRIC: &str = "block.validation.step.checkpoint.time";
pub const BLOCK_VALIDATION_STEP_BONDS_CACHE_TIME_METRIC: &str = "block.validation.step.bonds-cache.time";
pub const BLOCK_VALIDATION_STEP_NEGLECTED_INVALID_BLOCK_TIME_METRIC: &str = "block.validation.step.neglected-invalid-block.time";
pub const BLOCK_VALIDATION_STEP_NEGLECTED_EQUIVOCATION_TIME_METRIC: &str = "block.validation.step.neglected-equivocation.time";
pub const BLOCK_VALIDATION_STEP_PHLO_PRICE_TIME_METRIC: &str = "block.validation.step.phlo-price.time";
pub const BLOCK_VALIDATION_STEP_SIMPLE_EQUIVOCATION_TIME_METRIC: &str = "block.validation.step.simple-equivocation.time";

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
