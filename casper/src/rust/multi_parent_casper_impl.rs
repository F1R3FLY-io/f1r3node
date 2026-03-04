// See casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala

use async_trait::async_trait;
use rspace_plus_plus::rspace::state::rspace_exporter::RSpaceExporter;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::{
        BlockDagKeyValueStorage, DeployId, KeyValueDagRepresentation,
    },
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::transport::transport_layer::TransportLayer;
use crypto::rust::signatures::signed::Signed;
use models::rust::{
    block_hash::{BlockHash, BlockHashSerde},
    casper::{
        pretty_printer::PrettyPrinter,
        protocol::casper_message::{BlockMessage, DeployData, Justification},
    },
    equivocation_record::EquivocationRecord,
    normalizer_env::normalizer_env_from_deploy,
    validator::Validator,
};
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::{hashing::blake2b256_hash::Blake2b256Hash, history::Either};
use shared::rust::{
    dag::dag_ops,
    shared::{
        f1r3fly_event::{DeployEvent, F1r3flyEvent},
        f1r3fly_events::F1r3flyEvents,
    },
    store::key_value_store::KvStoreError,
};

use crate::rust::{
    block_status::{BlockError, InvalidBlock, ValidBlock},
    casper::{
        Casper, CasperShardConf, CasperSnapshot, DeployError, MultiParentCasper, OnChainCasperState,
    },
    engine::block_retriever::{AdmitHashReason, BlockRetriever},
    equivocation_detector::EquivocationDetector,
    errors::CasperError,
    estimator::Estimator,
    finality::finalizer::Finalizer,
    metrics_constants::{
        ACTIVE_VALIDATORS_CACHE_SIZE_METRIC, BLOCK_VALIDATION_STEP_BLOCK_SUMMARY_TIME_METRIC,
        BLOCK_VALIDATION_STEP_BONDS_CACHE_TIME_METRIC,
        BLOCK_VALIDATION_STEP_CHECKPOINT_TIME_METRIC,
        BLOCK_VALIDATION_STEP_NEGLECTED_EQUIVOCATION_TIME_METRIC,
        BLOCK_VALIDATION_STEP_NEGLECTED_INVALID_BLOCK_TIME_METRIC,
        BLOCK_VALIDATION_STEP_PHLO_PRICE_TIME_METRIC,
        BLOCK_VALIDATION_STEP_SIMPLE_EQUIVOCATION_TIME_METRIC, CASPER_METRICS_SOURCE,
        DAG_BLOCKS_SIZE_METRIC, DAG_CHILDREN_INDEX_SIZE_METRIC, DAG_FINALIZED_BLOCKS_SIZE_METRIC,
        DAG_HEIGHTS_SIZE_METRIC, DEPLOYS_IN_SCOPE_SIG_BYTES_ESTIMATE_METRIC,
        DEPLOYS_IN_SCOPE_SIZE_METRIC,
    },
    util::{
        proto_util,
        rholang::{
            interpreter_util::{self, validate_block_checkpoint},
            runtime_manager::RuntimeManager,
        },
    },
    validate::Validate,
    validator_identity::ValidatorIdentity,
};

const FINALIZER_BLOCKING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
const MAX_ACTIVE_VALIDATORS_CACHE_ENTRIES: usize = 4096;

/// RAII guard that ensures the finalization flag is reset on drop.
/// This prevents the flag from being stuck in `true` state if the async block
/// panics or returns early via `?` operator.
struct FinalizationGuard<'a>(&'a AtomicBool);

impl Drop for FinalizationGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

pub struct MultiParentCasperImpl<T: TransportLayer + Send + Sync> {
    pub block_retriever: BlockRetriever<T>,
    pub event_publisher: F1r3flyEvents,
    pub runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    pub estimator: Estimator,
    pub block_store: KeyValueBlockStore,
    pub block_dag_storage: BlockDagKeyValueStorage,
    pub deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    pub casper_buffer_storage: CasperBufferKeyValueStorage,
    pub validator_id: Option<ValidatorIdentity>,
    // TODO: this should be read from chain, for now read from startup options - OLD
    pub casper_shard_conf: CasperShardConf,
    pub approved_block: BlockMessage,
    /// Flag to track finalization status - block proposals fail fast if finalization is running.
    /// This prevents validators from creating blocks with stale snapshots during finalization.
    pub finalization_in_progress: Arc<AtomicBool>,
    /// Single-flight guard for background finalizer scheduling from propose path.
    pub finalizer_task_in_progress: Arc<AtomicBool>,
    /// Indicates a finalizer run was requested while another run was still in progress.
    /// The next queued run will execute immediately after the current one finishes.
    pub finalizer_task_queued: Arc<AtomicBool>,
    /// Shared reference to heartbeat signal for triggering immediate wake on deploy
    pub heartbeat_signal_ref: crate::rust::heartbeat_signal::HeartbeatSignalRef,
    /// Cache for deploys_in_scope BFS result keyed by DAG generation.
    /// Avoids re-traversing the full block window on every snapshot when the DAG hasn't changed.
    pub deploys_in_scope_cache: Arc<Mutex<Option<(u64, Arc<dashmap::DashSet<Bytes>>)>>>,
    /// Cache for get_active_validators results keyed by post_state_hash bytes.
    /// Avoids re-reading from RSpace when the main parent block hasn't changed.
    pub active_validators_cache: Arc<tokio::sync::Mutex<HashMap<Vec<u8>, Vec<Validator>>>>,
}

#[async_trait]
impl<T: TransportLayer + Send + Sync> Casper for MultiParentCasperImpl<T> {
    async fn get_snapshot(&self) -> Result<CasperSnapshot, CasperError> {
        if self.finalization_in_progress.load(Ordering::SeqCst) {
            tracing::debug!(
                "Finalization in progress while creating snapshot; using best-effort snapshot"
            );
        }

        let mut dag = self.block_dag_storage.get_representation();

        // Parent selection: Use latest block from EACH bonded validator.
        // Every block should have one parent per validator to ensure all deploy effects
        // are included in the merged state. Apply maxNumberOfParents and maxParentDepth limits.
        let latest_msgs_hashes: HashMap<Validator, BlockHash> = dag
            .latest_message_hashes()
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        // Filter out invalid latest messages (e.g., from slashed validators)
        let invalid_latest_msgs =
            dag.invalid_latest_messages_from_hashes(latest_msgs_hashes.clone())?;
        let valid_latest_msgs: HashMap<Validator, BlockHash> = latest_msgs_hashes
            .iter()
            .filter(|entry| !invalid_latest_msgs.contains_key(entry.0))
            .map(|(validator, hash)| (validator.clone(), hash.clone()))
            .collect();
        let valid_latest_metas: HashMap<Validator, models::rust::block_metadata::BlockMetadata> =
            valid_latest_msgs
                .iter()
                .filter_map(|(validator, hash)| {
                    dag.lookup_unsafe(hash).ok().map(|meta| (validator.clone(), meta))
                })
                .collect();
        // Deduplicate: multiple validators may have the same latest block (e.g., genesis)
        let unique_parent_hashes: HashSet<BlockHash> =
            valid_latest_msgs.values().cloned().collect();
        let parent_blocks_list: Vec<BlockMessage> = unique_parent_hashes
            .iter()
            .filter_map(|hash| self.block_store.get(hash).ok().flatten())
            .collect();

        // Sort parents deterministically with a near-tip tolerance:
        // - if both parents are near the maximum parent height, order by hash only to keep
        //   main-parent choice stable across validators even under slight view skew;
        // - otherwise prefer higher block number, then hash.
        //
        // Without this, validators tend to pick their own freshest block as main parent,
        // which can keep latest messages on disjoint main-parent chains and starve finalization.
        let mut sorted_parents_list = parent_blocks_list;
        let max_parent_block_number = sorted_parents_list
            .iter()
            .map(|b| b.body.state.block_number as i64)
            .max()
            .unwrap_or(0);
        let near_tip_tolerance_blocks = std::env::var("F1R3_MAIN_PARENT_NEAR_TIP_TOLERANCE")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|v| *v >= 0)
            .unwrap_or(2);
        sorted_parents_list.sort_by(|a, b| {
            let a_num = a.body.state.block_number as i64;
            let b_num = b.body.state.block_number as i64;
            let a_is_near_tip = max_parent_block_number.saturating_sub(a_num) <= near_tip_tolerance_blocks;
            let b_is_near_tip = max_parent_block_number.saturating_sub(b_num) <= near_tip_tolerance_blocks;

            if a_is_near_tip && b_is_near_tip {
                a.block_hash.cmp(&b.block_hash)
            } else {
                let block_num_cmp = b_num.cmp(&a_num);
                if block_num_cmp != std::cmp::Ordering::Equal {
                    block_num_cmp
                } else {
                    a.block_hash.cmp(&b.block_hash)
                }
            }
        });

        // Filter to blocks with matching bond maps (required for merge compatibility)
        // If no parent blocks exist (genesis case), use approved block as the parent
        let mut unfiltered_parents = if sorted_parents_list.is_empty() {
            vec![self.approved_block.clone()]
        } else {
            // Use the newest block as the bond-reference baseline.
            // Relying on the first (hash-sorted near-tip) block can select an older
            // parent and regress snapshot max_block_num when a joiner/fresh bond-map
            // divergence is present.
            let reference_bonds = sorted_parents_list
                .iter()
                .max_by(|a, b| {
                    a.body
                        .state
                        .block_number
                        .cmp(&b.body.state.block_number)
                        .then_with(|| a.block_hash.cmp(&b.block_hash))
                })
                .expect("sorted_parents_list is non-empty after is_empty() check")
                .body
                .state
                .bonds
                .clone();

            sorted_parents_list
                .into_iter()
                .filter(|block| block.body.state.bonds == reference_bonds)
                .collect()
        };

        let unfiltered_parents_count = unfiltered_parents.len();

        // Apply maxNumberOfParents limit
        const UNLIMITED_PARENTS: i32 = -1;
        let mut parents_after_count_limit = unfiltered_parents;
        if self.casper_shard_conf.max_number_of_parents != UNLIMITED_PARENTS {
            parents_after_count_limit
                .truncate(self.casper_shard_conf.max_number_of_parents as usize);
        }

        // Apply maxParentDepth filtering (similar to Estimator.filterDeepParents)
        // Find the parent with highest block number to use as reference for depth filtering
        let parents = if self.casper_shard_conf.max_parent_depth != i32::MAX
            && parents_after_count_limit.len() > 1
        {
            let parents_with_meta: Vec<(
                BlockMessage,
                models::rust::block_metadata::BlockMetadata,
            )> = parents_after_count_limit
                .into_iter()
                .filter_map(|b| dag.lookup_unsafe(&b.block_hash).ok().map(|meta| (b, meta)))
                .collect();

            // Find the parent with max block number as the reference point
            let max_block_num = parents_with_meta
                .iter()
                .map(|(_, meta)| meta.block_number)
                .max()
                .unwrap_or(0);

            // Filter to keep only parents within maxParentDepth of the highest block
            parents_with_meta
                .into_iter()
                .filter(|(_, meta)| {
                    max_block_num - meta.block_number
                        <= self.casper_shard_conf.max_parent_depth as i64
                })
                .map(|(b, _)| b)
                .collect()
        } else {
            parents_after_count_limit
        };

        // Calculate LCA via fold over parent pairs (for DagMerger)
        let parent_metas_for_lca: Vec<models::rust::block_metadata::BlockMetadata> = parents
            .iter()
            .filter_map(|b| dag.lookup_unsafe(&b.block_hash).ok())
            .collect();

        let lca = if parent_metas_for_lca.len() > 1 {
            // Fold to find LCA of all parents
            let mut current_lca = parent_metas_for_lca[0].clone();
            for meta in parent_metas_for_lca.iter().skip(1) {
                current_lca = crate::rust::util::dag_operations::DagOperations::lowest_universal_common_ancestor(
                    &current_lca,
                    meta,
                    &dag,
                )
                .await?;
            }
            current_lca.block_hash
        } else {
            // Single parent or genesis case - use that block as LCA
            parent_metas_for_lca
                .first()
                .map(|m| m.block_hash.clone())
                .unwrap_or_else(|| self.approved_block.block_hash.clone())
        };

        let tips: Vec<BlockHash> = parents.iter().map(|b| b.block_hash.clone()).collect();

        // Log parent selection for debugging
        tracing::debug!(
            "Parent selection: {} validators, {} invalid, {} valid, {} after bond filter, {} parents",
            latest_msgs_hashes.len(),
            invalid_latest_msgs.len(),
            valid_latest_msgs.len(),
            unfiltered_parents_count,
            parents.len()
        );

        let on_chain_state = self
            .get_on_chain_state(
                parents
                    .first()
                    .expect("parents should never be empty after approved block"),
            )
            .await?;

        // We ensure that only the justifications given in the block are those
        // which are bonded validators in the chosen parent. This is safe because
        // any latest message not from a bonded validator will not change the
        // final fork-choice.
        let justifications = {
            let bonded_validators = &on_chain_state.bonds_map;

            valid_latest_metas
                .iter()
                .filter(|(validator, _)| bonded_validators.contains_key(*validator))
                .map(|(validator, block_metadata)| Justification {
                    validator: validator.clone(),
                    latest_block_hash: block_metadata.block_hash.clone(),
                })
                .collect::<dashmap::DashSet<_>>()
        };

        let parent_hashes: Vec<BlockHash> = parents.iter().map(|b| b.block_hash.clone()).collect();
        let parent_metas = dag.lookups_unsafe(parent_hashes)?;
        let max_block_num = proto_util::max_block_number_metadata(&parent_metas);

        let max_seq_nums = valid_latest_metas
            .iter()
            .map(|(validator, block_metadata)| {
                (validator.clone(), block_metadata.sequence_number as u64)
            })
            .collect::<dashmap::DashMap<_, _>>();

        let deploys_in_scope = {
            let current_dag_generation = self.block_dag_storage.current_generation();

            // Phase 1: check cache under a short-lived lock.
            let cached: Option<Arc<dashmap::DashSet<Bytes>>> = {
                let cache_guard = self.deploys_in_scope_cache.lock().map_err(|_| {
                    CasperError::RuntimeError("deploys_in_scope_cache lock failed".to_string())
                })?;
                cache_guard.as_ref().and_then(|(gen, set)| {
                    if *gen == current_dag_generation {
                        Some(set.clone())
                    } else {
                        None
                    }
                })
            };

            // Phase 2: return cached or compute.
            if let Some(deploys) = cached {
                deploys
            } else {
                let current_block_number = max_block_num + 1;
                let earliest_block_number =
                    current_block_number - on_chain_state.shard_conf.deploy_lifespan;

                let neighbor_fn = |block_metadata: &models::rust::block_metadata::BlockMetadata| -> Vec<models::rust::block_metadata::BlockMetadata> {
                    match proto_util::get_parent_metadatas_above_block_number(block_metadata, earliest_block_number, &mut dag) {
                        Ok(parents) => parents,
                        Err(_) => vec![],
                    }
                };

                let traversal_result = dag_ops::bf_traverse(parent_metas, neighbor_fn);

                let all_deploys = Arc::new(dashmap::DashSet::new());
                for block_metadata in traversal_result {
                    let block_deploy_sigs = self
                        .block_store
                        .deploy_sigs(&block_metadata.block_hash)?
                        .ok_or_else(|| {
                            CasperError::RuntimeError(format!(
                                "Missing block {} during deploys_in_scope traversal",
                                PrettyPrinter::build_string_bytes(&block_metadata.block_hash)
                            ))
                        })?;
                    for deploy_sig in block_deploy_sigs {
                        all_deploys.insert(deploy_sig.into());
                    }
                }

                let mut cache_guard = self.deploys_in_scope_cache.lock().map_err(|_| {
                    CasperError::RuntimeError("deploys_in_scope_cache lock failed".to_string())
                })?;
                *cache_guard = Some((current_dag_generation, all_deploys.clone()));
                all_deploys
            }
        };
        let deploys_in_scope_len = deploys_in_scope.len();
        // Approximate memory in bytes for signatures only cache (cheap O(1) estimate).
        let deploys_in_scope_sig_bytes_estimate = (deploys_in_scope_len as f64) * 65.0;
        metrics::gauge!(DEPLOYS_IN_SCOPE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(deploys_in_scope_len as f64);
        metrics::gauge!(
            DEPLOYS_IN_SCOPE_SIG_BYTES_ESTIMATE_METRIC,
            "source" => CASPER_METRICS_SOURCE
        )
        .set(deploys_in_scope_sig_bytes_estimate);

        let invalid_blocks = dag.invalid_blocks_map()?;
        let last_finalized_block = dag.last_finalized_block();
        self.record_dag_cardinality_metrics(&dag);

        Ok(CasperSnapshot {
            dag,
            last_finalized_block,
            lca,
            tips,
            parents,
            justifications,
            invalid_blocks,
            deploys_in_scope,
            max_block_num,
            max_seq_nums,
            on_chain_state,
        })
    }

    fn contains(&self, hash: &BlockHash) -> bool {
        self.buffer_contains(hash) || self.dag_contains(hash)
    }

    fn dag_contains(&self, hash: &BlockHash) -> bool {
        self.block_dag_storage.get_representation().contains(hash)
    }

    fn buffer_contains(&self, hash: &BlockHash) -> bool {
        let block_hash_serde = BlockHashSerde(hash.clone());
        self.casper_buffer_storage.contains(&block_hash_serde)
    }

    fn get_approved_block(&self) -> Result<&BlockMessage, CasperError> {
        Ok(&self.approved_block)
    }
    fn deploy(
        &self,
        deploy: Signed<DeployData>,
    ) -> Result<Either<DeployError, DeployId>, CasperError> {
        // Create normalizer environment from deploy
        let normalizer_env = normalizer_env_from_deploy(&deploy);
        let parse_started_at = std::time::Instant::now();

        // Try to parse the deploy term
        match interpreter_util::mk_term(&deploy.data.term, normalizer_env) {
            // Parse failed - return parsing error
            Err(interpreter_error) => {
                tracing::debug!(
                    target: "f1r3fly.deploy.latency",
                    parse_ms = parse_started_at.elapsed().as_millis(),
                    "Deploy parse failed"
                );
                Ok(Either::Left(DeployError::parsing_error(format!(
                    "Error in parsing term: \n{}",
                    interpreter_error
                ))))
            }
            // Parse succeeded - call add_deploy
            Ok(_parsed_term) => {
                let parse_elapsed_ms = parse_started_at.elapsed().as_millis();
                let add_started_at = std::time::Instant::now();
                let deploy_id = self.add_deploy(deploy)?;
                tracing::debug!(
                    target: "f1r3fly.deploy.latency",
                    parse_ms = parse_elapsed_ms,
                    add_deploy_ms = add_started_at.elapsed().as_millis(),
                    "Deploy parse/add completed"
                );
                Ok(Either::Right(deploy_id))
            }
        }
    }

    async fn estimator(
        &self,
        dag: &mut KeyValueDagRepresentation,
    ) -> Result<Vec<BlockHash>, CasperError> {
        // Use latest message from each validator (matching get_snapshot behavior)
        // No fork choice ranking - all validators' latest blocks included
        // Filter out invalid messages (from slashed validators)
        // When latestMessages is empty, return genesis block hash
        let latest_message_hashes: HashMap<Validator, BlockHash> = dag
            .latest_message_hashes()
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        let invalid_latest_messages =
            dag.invalid_latest_messages_from_hashes(latest_message_hashes.clone())?;

        // Filter out invalid validators
        let valid_latest: HashMap<Validator, BlockHash> = latest_message_hashes
            .iter()
            .filter(|entry| !invalid_latest_messages.contains_key(entry.0))
            .map(|(validator, hash)| (validator.clone(), hash.clone()))
            .collect();

        if valid_latest.is_empty() {
            Ok(vec![self.approved_block.block_hash.clone()])
        } else {
            // Deduplicate: multiple validators may have the same latest block (e.g., genesis)
            let unique_hashes: HashSet<BlockHash> = valid_latest.values().cloned().collect();
            Ok(unique_hashes.into_iter().collect())
        }
    }

    fn get_version(&self) -> i64 {
        self.casper_shard_conf.casper_version
    }

    async fn validate(
        &self,
        block: &BlockMessage,
        snapshot: &mut CasperSnapshot,
    ) -> Result<Either<BlockError, ValidBlock>, CasperError> {
        fn timed_step<A, Fut>(
            step_name: &'static str,
            metric_name: &'static str,
            future: Fut,
        ) -> impl std::future::Future<Output = Result<(Either<BlockError, A>, String), CasperError>>
        where
            Fut: std::future::Future<Output = Result<Either<BlockError, A>, CasperError>>,
        {
            async move {
                tracing::debug!(target: "f1r3fly.casper", "before-{}", step_name);
                let start = std::time::Instant::now();
                let result = future.await?;
                let elapsed = start.elapsed();
                let elapsed_str = format!("{:?}", elapsed);
                let step_time_seconds = elapsed.as_secs_f64();
                metrics::histogram!(metric_name, "source" => CASPER_METRICS_SOURCE)
                    .record(step_time_seconds);
                tracing::debug!(target: "f1r3fly.casper", "after-{}", step_name);
                Ok((result, elapsed_str))
            }
        }

        tracing::info!(
            "Validating block {}",
            PrettyPrinter::build_string_block_message(block, true)
        );

        let start = std::time::Instant::now();
        let val_result = {
            let (block_summary_result, t1) = timed_step(
                "block-summary",
                BLOCK_VALIDATION_STEP_BLOCK_SUMMARY_TIME_METRIC,
                async {
                    Ok(Validate::block_summary(
                        block,
                        &self.approved_block,
                        snapshot,
                        &self.casper_shard_conf.shard_name,
                        self.casper_shard_conf.deploy_lifespan as i32,
                        self.casper_shard_conf.max_number_of_parents,
                        &self.block_store,
                        self.casper_shard_conf.disable_validator_progress_check,
                    )
                    .await)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "post-validation-block-summary");
            if let Either::Left(block_error) = block_summary_result {
                return Ok(Either::Left(block_error));
            }

            let (validate_block_checkpoint_result, t2) = timed_step(
                "checkpoint",
                BLOCK_VALIDATION_STEP_CHECKPOINT_TIME_METRIC,
                validate_block_checkpoint(
                    block,
                    &self.block_store,
                    snapshot,
                    &mut *self.runtime_manager.lock().await,
                ),
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "transactions-validated");
            if let Either::Left(block_error) = validate_block_checkpoint_result {
                return Ok(Either::Left(block_error));
            }
            if let Either::Right(None) = validate_block_checkpoint_result {
                return Ok(Either::Left(BlockError::Invalid(
                    InvalidBlock::InvalidTransaction,
                )));
            }

            let (bonds_cache_result, t3) = timed_step(
                "bonds-cache",
                BLOCK_VALIDATION_STEP_BONDS_CACHE_TIME_METRIC,
                async {
                    Ok(Validate::bonds_cache(block, &*self.runtime_manager.lock().await).await)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "bonds-cache-validated");
            if let Either::Left(block_error) = bonds_cache_result {
                return Ok(Either::Left(block_error));
            }

            let (neglected_invalid_block_result, t4) = timed_step(
                "neglected-invalid-block",
                BLOCK_VALIDATION_STEP_NEGLECTED_INVALID_BLOCK_TIME_METRIC,
                async { Ok(Validate::neglected_invalid_block(block, snapshot)) },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "neglected-invalid-block-validated");
            if let Either::Left(block_error) = neglected_invalid_block_result {
                return Ok(Either::Left(block_error));
            }

            let (equivocation_detector_result, t5) = timed_step(
                "neglected-equivocation",
                BLOCK_VALIDATION_STEP_NEGLECTED_EQUIVOCATION_TIME_METRIC,
                async {
                    EquivocationDetector::check_neglected_equivocations_with_update(
                        block,
                        &snapshot.dag,
                        &self.block_store,
                        &self.approved_block,
                        &self.block_dag_storage,
                    )
                    .await
                    .map_err(CasperError::from)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "neglected-equivocation-validated");
            if let Either::Left(block_error) = equivocation_detector_result {
                return Ok(Either::Left(block_error));
            }

            let (phlo_price_result, t6) = timed_step(
                "phlo-price",
                BLOCK_VALIDATION_STEP_PHLO_PRICE_TIME_METRIC,
                async {
                    Ok(Validate::phlo_price(
                        block,
                        self.casper_shard_conf.min_phlo_price,
                    ))
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "phlogiston-price-validated");
            if let Either::Left(_) = phlo_price_result {
                tracing::warn!(
                    "One or more deploys has phloPrice lower than {}",
                    self.casper_shard_conf.min_phlo_price
                );
            }

            let requested_as_dependency = self.casper_buffer_storage.requested_as_dependency(
                &models::rust::block_hash::BlockHashSerde(block.block_hash.clone()),
            );

            let (equivocation_result, t7) = timed_step(
                "simple-equivocation",
                BLOCK_VALIDATION_STEP_SIMPLE_EQUIVOCATION_TIME_METRIC,
                async {
                    EquivocationDetector::check_equivocations(
                        requested_as_dependency,
                        block,
                        &snapshot.dag,
                    )
                    .await
                    .map_err(CasperError::from)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "equivocation-validated");

            tracing::debug!(
                target: "f1r3fly.casper",
                "Validation timing breakdown: summary={}, checkpoint={}, bonds={}, neglected-invalid={}, neglected-equiv={}, phlo={}, simple-equiv={}",
                t1, t2, t3, t4, t5, t6, t7
            );

            equivocation_result
        };

        let elapsed = start.elapsed();

        if let Either::Right(ref status) = val_result {
            let block_info = PrettyPrinter::build_string_block_message(block, true);
            let deploy_count = block.body.deploys.len();
            tracing::info!(
                "Block replayed: {} ({}d) ({:?}) [{:?}]",
                block_info,
                deploy_count,
                status,
                elapsed
            );

            if self.casper_shard_conf.max_number_of_parents > 1 {
                let maybe_mergeable = self.runtime_manager.lock().await.load_mergeable_channels(
                    &block.body.state.post_state_hash,
                    block.sender.clone(),
                    block.seq_num,
                );

                match maybe_mergeable {
                    Ok(mergeable_chs) => {
                        if let Err(err) = self
                            .runtime_manager
                            .lock()
                            .await
                            .get_or_compute_block_index(
                                &block.block_hash,
                                &block.body.deploys,
                                &block.body.system_deploys,
                                &Blake2b256Hash::from_bytes_prost(&block.body.state.pre_state_hash),
                                &Blake2b256Hash::from_bytes_prost(
                                    &block.body.state.post_state_hash,
                                ),
                                &mergeable_chs,
                            )
                        {
                            tracing::warn!(
                                "Skipping block index cache update for block {}: {}",
                                PrettyPrinter::build_string_bytes(&block.block_hash),
                                err
                            );
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            "Skipping mergeable/index cache update for block {}: {}",
                            PrettyPrinter::build_string_bytes(&block.block_hash),
                            err
                        );
                    }
                }
            }
        }

        Ok(val_result)
    }

    async fn validate_self_created(
        &self,
        block: &BlockMessage,
        snapshot: &mut CasperSnapshot,
        pre_state_hash: Bytes,
        post_state_hash: Bytes,
    ) -> Result<Either<BlockError, ValidBlock>, CasperError> {
        fn timed_step<A, Fut>(
            step_name: &'static str,
            metric_name: &'static str,
            future: Fut,
        ) -> impl std::future::Future<Output = Result<(Either<BlockError, A>, String), CasperError>>
        where
            Fut: std::future::Future<Output = Result<Either<BlockError, A>, CasperError>>,
        {
            async move {
                tracing::debug!(target: "f1r3fly.casper", "before-{}", step_name);
                let start = std::time::Instant::now();
                let result = future.await?;
                let elapsed = start.elapsed();
                let elapsed_str = format!("{:?}", elapsed);
                let step_time_seconds = elapsed.as_secs_f64();
                metrics::histogram!(metric_name, "source" => CASPER_METRICS_SOURCE)
                    .record(step_time_seconds);
                tracing::debug!(target: "f1r3fly.casper", "after-{}", step_name);
                Ok((result, elapsed_str))
            }
        }

        tracing::info!(
            "Validating self-created block {}",
            PrettyPrinter::build_string_block_message(block, true)
        );

        // Safety: verify the block carries the hashes we computed (should always match).
        assert_eq!(
            block.body.state.pre_state_hash, pre_state_hash,
            "Self-created block pre_state_hash mismatch"
        );
        assert_eq!(
            block.body.state.post_state_hash, post_state_hash,
            "Self-created block post_state_hash mismatch"
        );

        let start = std::time::Instant::now();
        let val_result = {
            let (block_summary_result, t1) = timed_step(
                "block-summary",
                BLOCK_VALIDATION_STEP_BLOCK_SUMMARY_TIME_METRIC,
                async {
                    Ok(Validate::block_summary(
                        block,
                        &self.approved_block,
                        snapshot,
                        &self.casper_shard_conf.shard_name,
                        self.casper_shard_conf.deploy_lifespan as i32,
                        self.casper_shard_conf.max_number_of_parents,
                        &self.block_store,
                        self.casper_shard_conf.disable_validator_progress_check,
                    )
                    .await)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "post-validation-block-summary");
            if let Either::Left(block_error) = block_summary_result {
                return Ok(Either::Left(block_error));
            }

            // SKIP validate_block_checkpoint: hashes were computed in block_creator::create.
            // SKIP Validate::bonds_cache: bonds were computed from the same post_state_hash.

            let (neglected_invalid_block_result, t4) = timed_step(
                "neglected-invalid-block",
                BLOCK_VALIDATION_STEP_NEGLECTED_INVALID_BLOCK_TIME_METRIC,
                async { Ok(Validate::neglected_invalid_block(block, snapshot)) },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "neglected-invalid-block-validated");
            if let Either::Left(block_error) = neglected_invalid_block_result {
                return Ok(Either::Left(block_error));
            }

            let (equivocation_detector_result, t5) = timed_step(
                "neglected-equivocation",
                BLOCK_VALIDATION_STEP_NEGLECTED_EQUIVOCATION_TIME_METRIC,
                async {
                    EquivocationDetector::check_neglected_equivocations_with_update(
                        block,
                        &snapshot.dag,
                        &self.block_store,
                        &self.approved_block,
                        &self.block_dag_storage,
                    )
                    .await
                    .map_err(CasperError::from)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "neglected-equivocation-validated");
            if let Either::Left(block_error) = equivocation_detector_result {
                return Ok(Either::Left(block_error));
            }

            let (phlo_price_result, t6) = timed_step(
                "phlo-price",
                BLOCK_VALIDATION_STEP_PHLO_PRICE_TIME_METRIC,
                async {
                    Ok(Validate::phlo_price(
                        block,
                        self.casper_shard_conf.min_phlo_price,
                    ))
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "phlogiston-price-validated");
            if let Either::Left(_) = phlo_price_result {
                tracing::warn!(
                    "One or more deploys has phloPrice lower than {}",
                    self.casper_shard_conf.min_phlo_price
                );
            }

            let requested_as_dependency = self.casper_buffer_storage.requested_as_dependency(
                &models::rust::block_hash::BlockHashSerde(block.block_hash.clone()),
            );

            let (equivocation_result, t7) = timed_step(
                "simple-equivocation",
                BLOCK_VALIDATION_STEP_SIMPLE_EQUIVOCATION_TIME_METRIC,
                async {
                    EquivocationDetector::check_equivocations(
                        requested_as_dependency,
                        block,
                        &snapshot.dag,
                    )
                    .await
                    .map_err(CasperError::from)
                },
            )
            .await?;
            tracing::debug!(target: "f1r3fly.casper", "equivocation-validated");

            tracing::debug!(
                target: "f1r3fly.casper",
                "Self-validation timing breakdown: summary={}, neglected-invalid={}, neglected-equiv={}, phlo={}, simple-equiv={} (checkpoint and bonds-cache skipped)",
                t1, t4, t5, t6, t7
            );

            equivocation_result
        };

        let elapsed = start.elapsed();

        if let Either::Right(ref status) = val_result {
            let block_info = PrettyPrinter::build_string_block_message(block, true);
            let deploy_count = block.body.deploys.len();
            tracing::info!(
                "Self-created block validated: {} ({}d) ({:?}) [{:?}]",
                block_info,
                deploy_count,
                status,
                elapsed
            );

            if self.casper_shard_conf.max_number_of_parents > 1 {
                let maybe_mergeable = self.runtime_manager.lock().await.load_mergeable_channels(
                    &block.body.state.post_state_hash,
                    block.sender.clone(),
                    block.seq_num,
                );

                match maybe_mergeable {
                    Ok(mergeable_chs) => {
                        if let Err(err) = self
                            .runtime_manager
                            .lock()
                            .await
                            .get_or_compute_block_index(
                                &block.block_hash,
                                &block.body.deploys,
                                &block.body.system_deploys,
                                &Blake2b256Hash::from_bytes_prost(&block.body.state.pre_state_hash),
                                &Blake2b256Hash::from_bytes_prost(
                                    &block.body.state.post_state_hash,
                                ),
                                &mergeable_chs,
                            )
                        {
                            tracing::warn!(
                                "Skipping block index cache update for self-created block {}: {}",
                                PrettyPrinter::build_string_bytes(&block.block_hash),
                                err
                            );
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            "Skipping mergeable/index cache update for self-created block {}: {}",
                            PrettyPrinter::build_string_bytes(&block.block_hash),
                            err
                        );
                    }
                }
            }
        }

        Ok(val_result)
    }

    async fn handle_valid_block(
        &self,
        block: &BlockMessage,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        // Insert block as valid into DAG storage
        let updated_dag = self.block_dag_storage.insert(block, false, false)?;
        self.record_dag_cardinality_metrics(&updated_dag);

        // Remove user deploys from pending deploy storage as soon as the block is
        // accepted into the DAG. This keeps pending deploy pool bounded and avoids
        // repeating selection scans on already-finalized deploys.
        let deploys: Vec<_> = block
            .body
            .deploys
            .iter()
            .map(|pd| pd.deploy.clone())
            .collect();
        if !deploys.is_empty() {
            let deploys_count = deploys.len();
            let block_hash = PrettyPrinter::build_string_bytes(&block.block_hash);
            let block_number = block.body.state.block_number;
            self.deploy_storage
                .lock()
                .map_err(|_| {
                    CasperError::RuntimeError("Failed to acquire deploy_storage lock".to_string())
                })?
                .remove(deploys)?;

            tracing::debug!(
                "Removed {} deploys from pending pool for accepted block {} at {}.",
                deploys_count,
                block_hash,
                block_number
            );
        }

        // Remove block from casper buffer
        let block_hash_serde = BlockHashSerde(block.block_hash.clone());
        self.casper_buffer_storage.remove(block_hash_serde)?;

        // Publish BlockAdded event
        self.event_publisher
            .publish(added_event(block))
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        // Update last finalized block if needed
        self.update_last_finalized_block(block).await?;

        // Wake heartbeat immediately when a new peer block is accepted.
        // This lets bonded validators react to fresh parents without waiting
        // for the next heartbeat timer tick.
        if let Some(validator_id) = &self.validator_id {
            if block.sender != validator_id.public_key.bytes {
                if let Ok(signal_guard) = self.heartbeat_signal_ref.try_read() {
                    if let Some(ref signal) = *signal_guard {
                        tracing::debug!(
                            "Triggering heartbeat wake for accepted peer block {}",
                            PrettyPrinter::build_string_bytes(&block.block_hash)
                        );
                        signal.trigger_wake();
                    }
                }
            }
        }

        Ok(updated_dag)
    }

    fn handle_invalid_block(
        &self,
        block: &BlockMessage,
        status: &InvalidBlock,
        dag: &KeyValueDagRepresentation,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        // Helper function to handle invalid block effect (logging + storage operations)
        let handle_invalid_block_effect =
            |block_dag_storage: &BlockDagKeyValueStorage,
             casper_buffer_storage: &CasperBufferKeyValueStorage,
             status: &InvalidBlock,
             block: &BlockMessage|
             -> Result<KeyValueDagRepresentation, CasperError> {
                tracing::warn!(
                    "Recording invalid block {} for {:?}.",
                    PrettyPrinter::build_string_bytes(&block.block_hash),
                    status
                );

                // TODO: should be nice to have this transition of a block from casper buffer to dag storage atomic - OLD
                let updated_dag = block_dag_storage.insert(block, true, false)?;
                self.record_dag_cardinality_metrics(&updated_dag);
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                casper_buffer_storage.remove(block_hash_serde)?;
                Ok(updated_dag)
            };

        match status {
            InvalidBlock::AdmissibleEquivocation => {
                let base_equivocation_block_seq_num = block.seq_num - 1;

                // Check if equivocation record already exists for this validator and sequence number
                let equivocation_records = self.block_dag_storage.equivocation_records()?;
                let record_exists = equivocation_records.iter().any(|record| {
                    record.equivocator == block.sender
                        && record.equivocation_base_block_seq_num == base_equivocation_block_seq_num
                });

                if !record_exists {
                    // Create and insert new equivocation record
                    let new_equivocation_record = EquivocationRecord::new(
                        block.sender.clone(),
                        base_equivocation_block_seq_num,
                        BTreeSet::new(),
                    );
                    self.block_dag_storage
                        .insert_equivocation_record(new_equivocation_record)?;
                }

                // We can only treat admissible equivocations as invalid blocks if
                // casper is single threaded.
                handle_invalid_block_effect(
                    &self.block_dag_storage,
                    &self.casper_buffer_storage,
                    status,
                    block,
                )
            }

            InvalidBlock::IgnorableEquivocation => {
                /*
                 * We don't have to include these blocks to the equivocation tracker because if any validator
                 * will build off this side of the equivocation, we will get another attempt to add this block
                 * through the admissible equivocations.
                 */
                tracing::info!(
                    "Did not add block {} as that would add an equivocation to the BlockDAG",
                    PrettyPrinter::build_string_bytes(&block.block_hash)
                );
                Ok(dag.clone())
            }

            status if status.is_slashable() => {
                // TODO: Slash block for status except InvalidUnslashableBlock - OLD
                // This should implement actual slashing mechanism (reducing stake, etc.)
                handle_invalid_block_effect(
                    &self.block_dag_storage,
                    &self.casper_buffer_storage,
                    status,
                    block,
                )
            }

            _ => {
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                self.casper_buffer_storage.remove(block_hash_serde)?;
                tracing::warn!(
                    "Recording invalid block {} for {:?}.",
                    PrettyPrinter::build_string_bytes(&block.block_hash),
                    status
                );
                Ok(dag.clone())
            }
        }
    }

    fn get_dependency_free_from_buffer(&self) -> Result<Vec<BlockMessage>, CasperError> {
        let equivocation_hashes: HashSet<BlockHash> = self
            .block_dag_storage
            .access_equivocations_tracker(|tracker| {
                let equivocation_records = tracker.data()?;
                let hashes: HashSet<BlockHash> = equivocation_records
                    .iter()
                    .flat_map(|record| record.equivocation_detected_block_hashes.iter())
                    .cloned()
                    .collect();
                Ok(hashes)
            })
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        let invalid_block_hashes: HashSet<BlockHash> = self
            .block_dag_storage
            .get_representation()
            .invalid_blocks_map()
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?
            .into_keys()
            .collect();

        // Build candidate set from both pendants and buffered children.
        // Some race/interleaving paths can leave stale parent links in buffer
        // while block-level dependencies are already satisfied.
        let mut candidate_hashes: HashSet<BlockHash> = HashSet::new();

        let pendants = self.casper_buffer_storage.get_pendants();
        for pendant_serde in pendants.iter() {
            candidate_hashes.insert(BlockHash::from(pendant_serde.0.clone()));
        }

        let buffer_dag = self.casper_buffer_storage.to_doubly_linked_dag();
        for child_entry in buffer_dag.child_to_parent_adjacency_list.iter() {
            candidate_hashes.insert(BlockHash::from(child_entry.key().0.clone()));
        }

        // Keep only candidates that exist in block store.
        let mut candidates_stored = Vec::new();
        for candidate_hash in candidate_hashes {
            if self.block_store.get(&candidate_hash)?.is_some() {
                candidates_stored.push(candidate_hash);
            }
        }

        // Filter to dependency-free candidates by real block dependencies
        // (parents + justifications), independent of buffer edge shape.
        let mut dep_free_pendants = Vec::new();
        for candidate_hash in candidates_stored {
            let block = self.block_store.get(&candidate_hash)?.unwrap();
            let all_deps = proto_util::dependencies_hashes_of(&block);
            let all_deps_available = all_deps.into_iter().all(|dep| {
                self.dag_contains(&dep)
                    || equivocation_hashes.contains(&dep)
                    || invalid_block_hashes.contains(&dep)
            });

            if all_deps_available {
                dep_free_pendants.push(candidate_hash);
            }
        }

        // Get the actual BlockMessages
        let result = dep_free_pendants
            .into_iter()
            .map(|hash| self.block_store.get(&hash).unwrap())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                CasperError::RuntimeError("Failed to get blocks from store".to_string())
            })?;

        Ok(result)
    }

    fn get_all_from_buffer(&self) -> Result<Vec<BlockMessage>, CasperError> {
        let dag = self.casper_buffer_storage.to_doubly_linked_dag();
        let all_hashes = dag
            .child_to_parent_adjacency_list
            .iter()
            .map(|entry| BlockHash::from(entry.key().clone()));

        let mut blocks = Vec::new();
        for hash in all_hashes {
            if let Some(block) = self.block_store.get(&hash)? {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }
}

async fn run_queued_finalizer(
    block_dag_storage: BlockDagKeyValueStorage,
    block_store: KeyValueBlockStore,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    event_publisher: F1r3flyEvents,
    finalization_in_progress: Arc<AtomicBool>,
    finalizer_task_in_progress: Arc<AtomicBool>,
    finalizer_task_queued: Arc<AtomicBool>,
    enable_mergeable_channel_gc: bool,
    fault_tolerance_threshold: f32,
) {
    let _task_guard = FinalizationGuard(finalizer_task_in_progress.as_ref());
    tracing::info!(target: "f1r3fly.casper", "finalizer-run-started");

    loop {
        match tokio::time::timeout(
            FINALIZER_BLOCKING_TIMEOUT,
            compute_last_finalized_block(
                block_dag_storage.clone(),
                block_store.clone(),
                deploy_storage.clone(),
                runtime_manager.clone(),
                event_publisher.clone(),
                finalization_in_progress.clone(),
                enable_mergeable_channel_gc,
                fault_tolerance_threshold,
            ),
        )
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => {
                tracing::warn!("finalizer-run failed: {:?}", err);
            }
            Err(_) => {
                tracing::warn!(
                    "finalizer-run timed out after {:?}; skipping this cycle to avoid blocking propose",
                    FINALIZER_BLOCKING_TIMEOUT
                );
            }
        }

        if finalizer_task_queued.swap(false, Ordering::SeqCst) {
            tracing::debug!("finalizer-run-queued; continuing finalizer loop");
            continue;
        }

        tracing::info!(target: "f1r3fly.casper", "finalizer-run-finished");
        return;
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync> MultiParentCasper for MultiParentCasperImpl<T> {
    async fn fetch_dependencies(&self) -> Result<(), CasperError> {
        // Get pendants from CasperBuffer
        let pendants = self.casper_buffer_storage.get_pendants();

        // Filter to get unseen pendants (not in block store)
        let mut pendants_unseen = Vec::new();
        for pendant_serde in pendants.iter() {
            let pendant_hash = BlockHash::from(pendant_serde.0.clone());
            if self.block_store.get(&pendant_hash)?.is_none() {
                pendants_unseen.push(pendant_hash);
            }
        }

        // Log debug info about pendant count
        tracing::debug!(
            "Requesting CasperBuffer pendant hashes, {} items.",
            pendants_unseen.len()
        );

        // Send each unseen pendant to BlockRetriever
        for dependency in pendants_unseen {
            tracing::debug!(
                "Sending dependency {} to BlockRetriever",
                PrettyPrinter::build_string_bytes(&dependency)
            );

            self.block_retriever
                .admit_hash(
                    dependency,
                    None,
                    AdmitHashReason::MissingDependencyRequested,
                )
                .await?;
        }

        Ok(())
    }

    fn normalized_initial_fault(
        &self,
        weights: HashMap<Validator, u64>,
    ) -> Result<f32, CasperError> {
        // Access equivocations tracker to get equivocation records
        let equivocating_weight =
            self.block_dag_storage
                .access_equivocations_tracker(|tracker| {
                    let equivocation_records = tracker.data()?;

                    // Extract equivocators and sum their weights
                    let equivocating_weight: u64 = equivocation_records
                        .iter()
                        .map(|record| &record.equivocator)
                        .filter_map(|equivocator| weights.get(equivocator))
                        .sum();

                    Ok(equivocating_weight)
                })?;

        // Calculate total weight from the weights map
        let total_weight: u64 = weights.values().sum();

        // Return normalized fault (equivocating weight / total weight)
        if total_weight == 0 {
            Ok(0.0)
        } else {
            Ok(equivocating_weight as f32 / total_weight as f32)
        }
    }

    async fn last_finalized_block(&self) -> Result<BlockMessage, CasperError> {
        compute_last_finalized_block(
            self.block_dag_storage.clone(),
            self.block_store.clone(),
            self.deploy_storage.clone(),
            self.runtime_manager.clone(),
            self.event_publisher.clone(),
            self.finalization_in_progress.clone(),
            self.casper_shard_conf.enable_mergeable_channel_gc,
            self.casper_shard_conf.fault_tolerance_threshold,
        )
        .await
    }

    // Equivalent to Scala's def blockDag: F[BlockDagRepresentation[F]] = BlockDagStorage[F].getRepresentation
    async fn block_dag(&self) -> Result<KeyValueDagRepresentation, CasperError> {
        Ok(self.block_dag_storage.get_representation())
    }

    fn block_store(&self) -> &KeyValueBlockStore {
        &self.block_store
    }

    fn get_validator(&self) -> Option<ValidatorIdentity> {
        self.validator_id.clone()
    }

    async fn get_history_exporter(&self) -> Arc<dyn RSpaceExporter> {
        self.runtime_manager
            .lock()
            .await
            .get_history_repo()
            .exporter()
    }

    fn runtime_manager(&self) -> Arc<tokio::sync::Mutex<RuntimeManager>> {
        self.runtime_manager.clone()
    }

    async fn has_pending_deploys_in_storage(&self) -> Result<bool, CasperError> {
        let snapshot = self.get_snapshot().await?;
        self.has_pending_deploys_in_storage_for_snapshot(&snapshot)
            .await
    }

    async fn has_pending_deploys_in_storage_for_snapshot(
        &self,
        snapshot: &CasperSnapshot,
    ) -> Result<bool, CasperError> {
        let latest_block_number = snapshot.dag.latest_block_number();
        let earliest_block_number =
            latest_block_number - snapshot.on_chain_state.shard_conf.deploy_lifespan;
        let current_time_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let mut storage = self.deploy_storage.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire deploy_storage lock".to_string())
        })?;
        let unfinalized = storage.read_all().map_err(|e| {
            CasperError::RuntimeError(format!("Failed to read deploy storage: {:?}", e))
        })?;

        let mut expired_to_remove = Vec::new();
        let mut has_eligible_pending = false;

        for deploy in unfinalized.iter() {
            let block_expired = deploy.data.valid_after_block_number <= earliest_block_number;
            let time_expired = deploy.data.is_expired_at(current_time_millis);

            if block_expired || time_expired {
                expired_to_remove.push(deploy.clone());
                continue;
            }

            // Align with BlockCreator::not_future_deploy(next_block_num):
            // a deploy is eligible for the *next* block when valid_after < next_block_num,
            // i.e. valid_after <= latest_block_number.
            let is_future = pending_deploy_is_future_for_next_block(
                latest_block_number,
                deploy.data.valid_after_block_number,
            );
            let already_in_scope = snapshot.deploys_in_scope.contains(&deploy.sig);

            if !is_future && !already_in_scope {
                has_eligible_pending = true;
                break;
            }
        }

        if !expired_to_remove.is_empty() {
            storage.remove(expired_to_remove).map_err(|e| {
                CasperError::RuntimeError(format!(
                    "Failed to prune expired deploys from storage: {:?}",
                    e
                ))
            })?;
        }

        Ok(has_eligible_pending)
    }
}

#[inline]
fn pending_deploy_is_future_for_next_block(
    latest_block_number: i64,
    valid_after_block_number: i64,
) -> bool {
    valid_after_block_number > latest_block_number
}

async fn compute_last_finalized_block(
    block_dag_storage: BlockDagKeyValueStorage,
    block_store: KeyValueBlockStore,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    event_publisher: F1r3flyEvents,
    finalization_in_progress: Arc<AtomicBool>,
    enable_mergeable_channel_gc: bool,
    fault_tolerance_threshold: f32,
) -> Result<BlockMessage, CasperError> {
    let lfb_lookup_started = std::time::Instant::now();
    // Get current LFB hash and height
    let dag = block_dag_storage.get_representation();
    let last_finalized_block_hash = dag.last_finalized_block();
    let last_finalized_block_height = dag.lookup_unsafe(&last_finalized_block_hash)?.block_number;

    // Keep effect closure FnMut-compatible by cloning captured state on each invocation.
    let block_dag_storage_for_effect = block_dag_storage.clone();
    let block_store_for_effect = block_store.clone();
    let deploy_storage_for_effect = deploy_storage.clone();
    let runtime_manager_for_effect = runtime_manager.clone();
    let event_publisher_for_effect = event_publisher.clone();
    let finalization_in_progress_for_effect = finalization_in_progress.clone();

    // Create simple finalization effect closure
    let new_lfb_found_effect = move |new_lfb: BlockHash| {
        let block_dag_storage = block_dag_storage_for_effect.clone();
        let block_store = block_store_for_effect.clone();
        let deploy_storage = deploy_storage_for_effect.clone();
        let runtime_manager = runtime_manager_for_effect.clone();
        let event_publisher = event_publisher_for_effect.clone();
        let finalization_in_progress = finalization_in_progress_for_effect.clone();
        async move {
            let effect_started = std::time::Instant::now();
            block_dag_storage
                .record_directly_finalized(new_lfb.clone(), |finalized_set: &HashSet<BlockHash>| {
                    let finalized_set = finalized_set.clone();
                    let block_store = block_store.clone();
                    let deploy_storage = deploy_storage.clone();
                    let runtime_manager = runtime_manager.clone();
                    let event_publisher = event_publisher.clone();
                    let finalization_in_progress = finalization_in_progress.clone();
                    Box::pin(async move {
                        let process_finalized_started = std::time::Instant::now();
                        // Use RAII guard to ensure flag is reset even if we return early or panic
                        finalization_in_progress.store(true, Ordering::SeqCst);
                        let _guard = FinalizationGuard(finalization_in_progress.as_ref());
                        tracing::debug!("Finalization started for {} blocks", finalized_set.len());

                        // process_finalized
                        for block_hash in &finalized_set {
                            let block = block_store.get(block_hash)?.unwrap();
                            let deploys: Vec<_> = block
                                .body
                                .deploys
                                .iter()
                                .map(|pd| pd.deploy.clone())
                                .collect();

                            // Remove block deploys from persistent store
                            let deploys_count = deploys.len();
                            deploy_storage
                                .lock()
                                .map_err(|_| {
                                    KvStoreError::LockError(
                                        "Failed to acquire deploy_storage lock".to_string(),
                                    )
                                })?
                                .remove(deploys)?;
                            let finalized_set_str = PrettyPrinter::build_string_hashes(
                                &finalized_set.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
                            );
                            let removed_deploy_msg = format!(
                                "Removed {} deploys from deploy history as we finalized block {}.",
                                deploys_count, finalized_set_str
                            );
                            tracing::info!("{}", removed_deploy_msg);

                            // Remove block index from cache
                            runtime_manager
                                .lock()
                                .await
                                .remove_block_index_cache(block_hash);

                            // Keep mergeable data on finalization to preserve deterministic
                            // parent-state reconstruction. Safe deletion is handled only by
                            // reachability-based background GC when enabled.
                            if !enable_mergeable_channel_gc {
                                tracing::debug!(
                                    "Mergeable channel GC disabled; retaining mergeable data for finalized block {} (sender={}, seq={})",
                                    PrettyPrinter::build_string_bytes(&block.block_hash),
                                    PrettyPrinter::build_string_bytes(&block.sender),
                                    block.seq_num
                                );
                            }

                            // Publish BlockFinalised event for each newly finalized block
                            event_publisher
                                .publish(finalised_event(&block))
                                .map_err(|e| KvStoreError::IoError(e.to_string()))?;
                        }

                        // Guard will reset finalization_in_progress flag on drop
                        tracing::debug!("Finalization completed");
                        tracing::info!(
                            target: "f1r3fly.finalizer.effect.timing",
                            "Finalization effect timing: finalized_blocks={}, process_finalized_ms={}",
                            finalized_set.len(),
                            process_finalized_started.elapsed().as_millis()
                        );

                        Ok(())
                    })
                })
                .await?;
            tracing::info!(
                target: "f1r3fly.finalizer.effect.timing",
                "record_directly_finalized_total_ms={}",
                effect_started.elapsed().as_millis()
            );
            Ok(())
        }
    };

    // Run finalizer
    let finalizer_started = std::time::Instant::now();
    let new_finalized_hash_opt = Finalizer::run(
        &dag,
        fault_tolerance_threshold,
        last_finalized_block_height,
        new_lfb_found_effect,
    )
    .await
    .map_err(|e| CasperError::KvStoreError(e))?;
    let finalizer_ms = finalizer_started.elapsed().as_millis();
    let new_lfb_found = new_finalized_hash_opt.is_some();

    // Get the final LFB hash (either new or existing)
    let final_lfb_hash = new_finalized_hash_opt.unwrap_or(last_finalized_block_hash);

    // Return the finalized block
    let read_started = std::time::Instant::now();
    let block_message = block_store.get(&final_lfb_hash)?.unwrap();
    tracing::info!(
        target: "f1r3fly.last_finalized_block.timing",
        "last_finalized_block timing: finalizer_ms={}, read_block_ms={}, total_ms={}, new_lfb_found={}",
        finalizer_ms,
        read_started.elapsed().as_millis(),
        lfb_lookup_started.elapsed().as_millis(),
        new_lfb_found
    );
    Ok(block_message)
}

impl<T: TransportLayer + Send + Sync> MultiParentCasperImpl<T> {
    fn record_dag_cardinality_metrics(&self, dag: &KeyValueDagRepresentation) {
        metrics::gauge!(DAG_BLOCKS_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(dag.dag_set.len() as f64);
        metrics::gauge!(DAG_CHILDREN_INDEX_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(dag.child_map.len() as f64);
        metrics::gauge!(DAG_HEIGHTS_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE).set(
            dag.height_map
                .read()
                .map(|hm| hm.len() as f64)
                .unwrap_or_default(),
        );
        metrics::gauge!(DAG_FINALIZED_BLOCKS_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
            .set(dag.finalized_blocks_set.len() as f64);
    }

    async fn update_last_finalized_block(
        &self,
        new_block: &BlockMessage,
    ) -> Result<(), CasperError> {
        if self.casper_shard_conf.finalization_rate <= 0 {
            return Ok(());
        }

        if new_block.body.state.block_number % self.casper_shard_conf.finalization_rate as i64 == 0 {
            if self
                .finalizer_task_in_progress
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                if !self
                    .finalizer_task_queued
                    .swap(true, Ordering::SeqCst)
                {
                    tracing::debug!(
                        "Finalizer already running; queued follow-up finalization run"
                    );
                }
                return Ok(());
            }

            let block_dag_storage = self.block_dag_storage.clone();
            let block_store = self.block_store.clone();
            let deploy_storage = self.deploy_storage.clone();
            let runtime_manager = self.runtime_manager.clone();
            let event_publisher = self.event_publisher.clone();
            let finalization_in_progress = self.finalization_in_progress.clone();
            let finalizer_task_in_progress = self.finalizer_task_in_progress.clone();
            let finalizer_task_queued = self.finalizer_task_queued.clone();
            let enable_mergeable_channel_gc = self.casper_shard_conf.enable_mergeable_channel_gc;
            let fault_tolerance_threshold = self.casper_shard_conf.fault_tolerance_threshold;

            tokio::spawn(async move {
                run_queued_finalizer(
                    block_dag_storage,
                    block_store,
                    deploy_storage,
                    runtime_manager,
                    event_publisher,
                    finalization_in_progress,
                    finalizer_task_in_progress,
                    finalizer_task_queued,
                    enable_mergeable_channel_gc,
                    fault_tolerance_threshold,
                )
                .await;
            });
        }
        Ok(())
    }

    async fn get_on_chain_state(
        &self,
        block: &BlockMessage,
    ) -> Result<OnChainCasperState, CasperError> {
        let cache_key = block.body.state.post_state_hash.to_vec();
        let av = {
            let mut cache = self.active_validators_cache.lock().await;
            let result = if let Some(cached) = cache.get(&cache_key) {
                cached.clone()
            } else {
                // Bound cache growth across long-running validators.
                if cache.len() >= MAX_ACTIVE_VALIDATORS_CACHE_ENTRIES {
                    cache.clear();
                }
                let fetched = self
                    .runtime_manager
                    .lock()
                    .await
                    .get_active_validators(&block.body.state.post_state_hash)
                    .await?;
                cache.insert(cache_key, fetched.clone());
                fetched
            };
            metrics::gauge!(ACTIVE_VALIDATORS_CACHE_SIZE_METRIC, "source" => CASPER_METRICS_SOURCE)
                .set(cache.len() as f64);
            result
        };

        // bonds are available in block message, but please remember this is just a cache, source of truth is RSpace.
        let bm = &block.body.state.bonds;

        Ok(OnChainCasperState {
            shard_conf: self.casper_shard_conf.clone(),
            bonds_map: bm
                .iter()
                .map(|v| (v.validator.clone(), v.stake))
                .collect::<HashMap<_, _>>(),
            active_validators: av,
        })
    }

    fn add_deploy(&self, deploy: Signed<DeployData>) -> Result<DeployId, CasperError> {
        // Add deploy to storage
        self.deploy_storage
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError("Failed to acquire deploy_storage lock".to_string())
            })?
            .add(vec![deploy.clone()])?;

        // Log the received deploy
        let deploy_info = PrettyPrinter::build_string_signed_deploy_data(&deploy);
        tracing::info!("Received {}", deploy_info);

        // Trigger heartbeat signal to propose block immediately with this deploy
        if let Ok(signal_guard) = self.heartbeat_signal_ref.try_read() {
            if let Some(ref signal) = *signal_guard {
                tracing::debug!("Triggering heartbeat wake for immediate block proposal");
                signal.trigger_wake();
            } else {
                tracing::debug!("No heartbeat signal available (heartbeat may be disabled)");
            }
        }

        // Return deploy signature as DeployId
        Ok(deploy.sig.to_vec())
    }
}

/// Extract common block event data.
fn block_event(
    block: &BlockMessage,
) -> (
    String,
    Vec<String>,
    Vec<(String, String)>,
    Vec<DeployEvent>,
    String,
    i32,
) {
    let block_hash = hex::encode(block.block_hash.clone());

    let parent_hashes = block
        .header
        .parents_hash_list
        .iter()
        .map(|h| hex::encode(h))
        .collect::<Vec<_>>();

    let justification_hashes = block
        .justifications
        .iter()
        .map(|j| {
            (
                hex::encode(j.validator.clone()),
                hex::encode(j.latest_block_hash.clone()),
            )
        })
        .collect::<Vec<_>>();

    // Build DeployEvent with full information
    let deploys = block
        .body
        .deploys
        .iter()
        .map(|pd| {
            DeployEvent::new(
                hex::encode(pd.deploy.sig.clone()),
                pd.cost.cost as i64,
                hex::encode(pd.deploy.pk.bytes.clone()),
                pd.is_failed,
            )
        })
        .collect::<Vec<_>>();

    let creator = hex::encode(block.sender.clone());
    let seq_num = block.seq_num;

    (
        block_hash,
        parent_hashes,
        justification_hashes,
        deploys,
        creator,
        seq_num,
    )
}

/// Create BlockCreated event for a block.
pub fn created_event(block: &BlockMessage) -> F1r3flyEvent {
    let (block_hash, parent_hashes, justification_hashes, deploys, creator, seq_num) =
        block_event(block);
    F1r3flyEvent::block_created(
        block_hash,
        parent_hashes,
        justification_hashes,
        deploys,
        creator,
        seq_num,
    )
}

/// Create BlockAdded event for a block.
pub fn added_event(block: &BlockMessage) -> F1r3flyEvent {
    let (block_hash, parent_hashes, justification_hashes, deploys, creator, seq_num) =
        block_event(block);
    F1r3flyEvent::block_added(
        block_hash,
        parent_hashes,
        justification_hashes,
        deploys,
        creator,
        seq_num,
    )
}

/// Create BlockFinalised event for a block.
pub fn finalised_event(block: &BlockMessage) -> F1r3flyEvent {
    let (block_hash, parent_hashes, justification_hashes, deploys, creator, seq_num) =
        block_event(block);
    F1r3flyEvent::block_finalised(
        block_hash,
        parent_hashes,
        justification_hashes,
        deploys,
        creator,
        seq_num,
    )
}

#[cfg(test)]
mod pending_deploy_tests {
    use super::pending_deploy_is_future_for_next_block;

    #[test]
    fn pending_deploy_equal_to_latest_block_is_not_future_for_next_block() {
        assert!(!pending_deploy_is_future_for_next_block(100, 100));
    }

    #[test]
    fn pending_deploy_above_latest_block_is_future_for_next_block() {
        assert!(pending_deploy_is_future_for_next_block(100, 101));
    }

    #[test]
    fn pending_deploy_below_latest_block_is_not_future_for_next_block() {
        assert!(!pending_deploy_is_future_for_next_block(100, 99));
    }
}
