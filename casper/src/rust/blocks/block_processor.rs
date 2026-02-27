// See casper/src/main/scala/coop/rchain/casper/blocks/BlockProcessor.scala

/*
 * ARCHITECTURAL CHOICE: Trait-based Dependency Injection
 *
 * This implementation uses trait-based dependency injection instead of functional closures
 * because Rust's ownership model and async system work better with traits than with complex
 * closure captures. Traits provide zero-cost abstractions, better testability, and seamless
 * async support while maintaining the same flexibility as the original Scala version.
 */

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::{
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::{
    block_hash::{BlockHash, BlockHashSerde},
    casper::pretty_printer::PrettyPrinter,
    casper::protocol::casper_message::{BlockMessage, CasperMessage},
};
use prost::Message;
use rspace_plus_plus::rspace::history::Either;

use crate::rust::block_status::BlockError;
use crate::rust::engine::block_retriever::{AdmitHashReason, BlockRetriever};
use crate::rust::metrics_constants::{
    ALLOCATOR_TRIM_TOTAL_METRIC, BLOCK_PROCESSING_STORAGE_TIME_METRIC,
    BLOCK_PROCESSING_VALIDATION_SETUP_TIME_METRIC, BLOCK_PROCESSOR_METRICS_SOURCE,
    BLOCK_SIZE_METRIC, BLOCK_VALIDATION_FAILED_METRIC, BLOCK_VALIDATION_SUCCESS_METRIC,
    BLOCK_VALIDATION_TIME_METRIC,
};
use crate::rust::{
    block_status::InvalidBlock,
    casper::{Casper, CasperSnapshot},
    errors::CasperError,
    util::proto_util,
    validate::Validate,
    ValidBlockProcessing,
};

/// Logic for processing incoming blocks
/// Blocks created by node itself are not held here, but in Proposer.
#[derive(Clone)]
pub struct BlockProcessor<T: TransportLayer + Send + Sync> {
    dependencies: BlockProcessorDependencies<T>,
}

const CASPER_BUFFER_PRUNE_INTERVAL_MS: u64 = 5_000;
const CASPER_BUFFER_STALE_TTL_MS: u64 = 180_000;
const CASPER_BUFFER_MAX_APPROX_NODES: usize = 16_384;
const CASPER_BUFFER_MAX_PRUNE_BATCH: usize = 512;
const CASPER_BUFFER_MAX_APPROX_NODES_ENV: &str = "F1R3_CASPER_BUFFER_MAX_APPROX_NODES";
const CASPER_BUFFER_STALE_TTL_MS_ENV: &str = "F1R3_CASPER_BUFFER_STALE_TTL_MS";
const CASPER_BUFFER_MAX_PRUNE_BATCH_ENV: &str = "F1R3_CASPER_BUFFER_MAX_PRUNE_BATCH";
const CASPER_BUFFER_PRUNE_INTERVAL_MS_ENV: &str = "F1R3_CASPER_BUFFER_PRUNE_INTERVAL_MS";
const CASPER_BUFFER_STALE_PRUNED_METRIC: &str = "casper.buffer.stale-pruned";
const CASPER_BUFFER_OVERFLOW_PRUNED_METRIC: &str = "casper.buffer.overflow-pruned";
const CASPER_BUFFER_APPROX_NODES_METRIC: &str = "casper.buffer.approx-nodes";
const CASPER_BUFFER_DEPENDENCY_LOOP_PRUNED_METRIC: &str = "casper.buffer.dependency-loop-pruned";
const MISSING_DEPENDENCY_ATTEMPTS_MAX_DEFAULT: u32 = 32;
const MISSING_DEPENDENCY_ATTEMPTS_MAX_ENV: &str = "F1R3_MISSING_DEPENDENCY_ATTEMPTS_MAX";
const MISSING_DEPENDENCY_QUARANTINE_MS_DEFAULT: u64 = 120_000;
const MISSING_DEPENDENCY_QUARANTINE_MS_ENV: &str = "F1R3_MISSING_DEPENDENCY_QUARANTINE_MS";
const MALLOC_TRIM_INTERVAL_BLOCKS_DEFAULT: u64 = 64;
const MALLOC_TRIM_INTERVAL_BLOCKS_ENV: &str = "F1R3_MALLOC_TRIM_EVERY_BLOCKS";
static MALLOC_TRIM_BLOCK_COUNTER: AtomicU64 = AtomicU64::new(0);
static MALLOC_TRIM_INTERVAL_BLOCKS: OnceLock<u64> = OnceLock::new();
static CASPER_BUFFER_MAX_APPROX_NODES_CFG: OnceLock<usize> = OnceLock::new();
static CASPER_BUFFER_STALE_TTL_MS_CFG: OnceLock<u64> = OnceLock::new();
static CASPER_BUFFER_MAX_PRUNE_BATCH_CFG: OnceLock<usize> = OnceLock::new();
static CASPER_BUFFER_PRUNE_INTERVAL_MS_CFG: OnceLock<u64> = OnceLock::new();
static MISSING_DEPENDENCY_ATTEMPTS_MAX_CFG: OnceLock<u32> = OnceLock::new();
static MISSING_DEPENDENCY_QUARANTINE_MS_CFG: OnceLock<u64> = OnceLock::new();

#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

fn malloc_trim_interval_blocks() -> u64 {
    *MALLOC_TRIM_INTERVAL_BLOCKS.get_or_init(|| {
        std::env::var(MALLOC_TRIM_INTERVAL_BLOCKS_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(MALLOC_TRIM_INTERVAL_BLOCKS_DEFAULT)
    })
}

fn casper_buffer_max_approx_nodes() -> usize {
    *CASPER_BUFFER_MAX_APPROX_NODES_CFG.get_or_init(|| {
        std::env::var(CASPER_BUFFER_MAX_APPROX_NODES_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(CASPER_BUFFER_MAX_APPROX_NODES)
    })
}

fn casper_buffer_stale_ttl_ms() -> u64 {
    *CASPER_BUFFER_STALE_TTL_MS_CFG.get_or_init(|| {
        std::env::var(CASPER_BUFFER_STALE_TTL_MS_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(CASPER_BUFFER_STALE_TTL_MS)
    })
}

fn casper_buffer_max_prune_batch() -> usize {
    *CASPER_BUFFER_MAX_PRUNE_BATCH_CFG.get_or_init(|| {
        std::env::var(CASPER_BUFFER_MAX_PRUNE_BATCH_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(CASPER_BUFFER_MAX_PRUNE_BATCH)
    })
}

fn casper_buffer_prune_interval_ms() -> u64 {
    *CASPER_BUFFER_PRUNE_INTERVAL_MS_CFG.get_or_init(|| {
        std::env::var(CASPER_BUFFER_PRUNE_INTERVAL_MS_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(CASPER_BUFFER_PRUNE_INTERVAL_MS)
    })
}

fn maybe_trim_allocator_after_block() {
    let interval = malloc_trim_interval_blocks();
    if interval == 0 {
        return;
    }
    let n = MALLOC_TRIM_BLOCK_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    if n % interval != 0 {
        return;
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // Best-effort return of free heap pages to OS to limit RSS ratcheting.
        unsafe {
            let _ = malloc_trim(0);
        }
        metrics::counter!(ALLOCATOR_TRIM_TOTAL_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
            .increment(1);
    }
}

fn missing_dependency_attempts_max() -> u32 {
    *MISSING_DEPENDENCY_ATTEMPTS_MAX_CFG.get_or_init(|| {
        std::env::var(MISSING_DEPENDENCY_ATTEMPTS_MAX_ENV)
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MISSING_DEPENDENCY_ATTEMPTS_MAX_DEFAULT)
    })
}

fn missing_dependency_quarantine_ms() -> u64 {
    *MISSING_DEPENDENCY_QUARANTINE_MS_CFG.get_or_init(|| {
        std::env::var(MISSING_DEPENDENCY_QUARANTINE_MS_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MISSING_DEPENDENCY_QUARANTINE_MS_DEFAULT)
    })
}

impl<T: TransportLayer + Send + Sync> BlockProcessor<T> {
    pub fn new(dependencies: BlockProcessorDependencies<T>) -> Self {
        Self { dependencies }
    }

    /// check if block should be processed
    pub fn check_if_of_interest(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
    ) -> Result<bool, CasperError> {
        // TODO casper.dag_contains does not take into account equivocation tracker
        let already_processed =
            casper.dag_contains(&block.block_hash) || casper.buffer_contains(&block.block_hash);

        let shard_of_interest = casper.get_approved_block().map(|approved_block| {
            approved_block
                .shard_id
                .eq_ignore_ascii_case(&block.shard_id)
        })?;

        let version_of_interest = casper
            .get_approved_block()
            .map(|approved_block| Validate::version(block, approved_block.header.version))?;

        let old_block = casper.get_approved_block().map(|approved_block| {
            proto_util::block_number(block) < proto_util::block_number(approved_block)
        })?;

        Ok(!already_processed && shard_of_interest && version_of_interest && !old_block)
    }

    /// check block format and store if check passed
    pub async fn check_if_well_formed_and_store(
        &self,
        block: &BlockMessage,
    ) -> Result<bool, CasperError> {
        let valid_format = Validate::format_of_fields(block);
        let valid_sig = Validate::block_signature(block);
        let is_valid = valid_format && valid_sig;

        if is_valid {
            // Time storage operation
            let storage_start = Instant::now();
            self.dependencies.store_block(block).await?;
            metrics::histogram!(BLOCK_PROCESSING_STORAGE_TIME_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
                .record(storage_start.elapsed().as_secs_f64());
        }

        Ok(is_valid)
    }

    /// check if block has all dependencies available and can be validated
    pub async fn check_dependencies_with_effects(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
    ) -> Result<bool, CasperError> {
        self.dependencies.prune_casper_buffer_if_needed()?;
        self.dependencies
            .sweep_expired_missing_dependency_quarantine()?;
        self.dependencies
            .sweep_orphaned_missing_dependency_attempts()?;
        self.dependencies
            .sweep_orphaned_missing_dependency_quarantine()?;

        if self
            .dependencies
            .is_missing_dependency_quarantined(&block.block_hash)?
        {
            tracing::debug!(
                "Skipping block {} due to missing-dependency quarantine ({}ms).",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                missing_dependency_quarantine_ms()
            );
            metrics::counter!(CASPER_BUFFER_DEPENDENCY_LOOP_PRUNED_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE, "reason" => "quarantine")
                .increment(1);
            self.dependencies
                .drop_dependency_loop_block(block)
                .await?;
            return Ok(false);
        }

        let (is_ready, deps_to_fetch, deps_in_buffer) = self
            .dependencies
            .get_non_validated_dependencies(casper, block)
            .await?;

        if is_ready {
            self.dependencies
                .clear_missing_dependency_attempts(&block.block_hash)?;
            // store pendant block in buffer, it will be removed once block is validated and added to DAG
            self.dependencies.commit_to_buffer(block, None).await?;
        } else {
            if self
                .dependencies
                .register_missing_dependency_attempt(&block.block_hash)?
            {
                tracing::warn!(
                    "Dropping block {} after {} missing-dependency checks.",
                    PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                    missing_dependency_attempts_max()
                );
                metrics::counter!(CASPER_BUFFER_DEPENDENCY_LOOP_PRUNED_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE, "reason" => "attempts")
                    .increment(1);
                self.dependencies
                    .drop_dependency_loop_block(block)
                    .await?;
                return Ok(false);
            }

            // associate parents with new block in casper buffer
            let mut all_deps = deps_to_fetch.clone();
            all_deps.extend(deps_in_buffer.clone());
            self.dependencies
                .commit_to_buffer(block, Some(all_deps))
                .await?;
            self.dependencies
                .request_missing_dependencies(&deps_to_fetch)
                .await?;
            // Recovery path: if dependency graph is stuck in buffer (no fresh deps to fetch),
            // force a network re-request for buffered dependencies.
            if deps_to_fetch.is_empty() && !deps_in_buffer.is_empty() {
                self.dependencies
                    .recover_stale_buffer_dependencies(&deps_in_buffer)
                    .await?;
            }
            self.dependencies.ack_processed(block).await?;
        }

        Ok(is_ready)
    }

    /// validate block and invoke all effects required
    pub async fn validate_with_effects(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
        // this option is required for tests, as sometimes block without parents available are added, so
        // CasperSnapshot cannot be constructed
        snapshot_opt: Option<CasperSnapshot>,
    ) -> Result<ValidBlockProcessing, CasperError> {
        // Record block size
        let block_size = block.to_proto().encode_to_vec().len();
        metrics::histogram!(BLOCK_SIZE_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
            .record(block_size as f64);

        // Time validation setup
        let setup_start = Instant::now();
        let mut snapshot = match snapshot_opt {
            Some(snapshot) => snapshot,
            None => {
                self.dependencies
                    .get_casper_state_snapshot(casper.clone())
                    .await?
            }
        };
        metrics::histogram!(BLOCK_PROCESSING_VALIDATION_SETUP_TIME_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
            .record(setup_start.elapsed().as_secs_f64());

        // Time block validation
        let validation_start = Instant::now();
        let status = self
            .dependencies
            .validate_block(casper.clone(), &mut snapshot, block)
            .await?;
        metrics::histogram!(BLOCK_VALIDATION_TIME_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
            .record(validation_start.elapsed().as_secs_f64());

        // Record validation outcome
        let _ = match &status {
            Either::Right(_valid_block) => {
                metrics::counter!(BLOCK_VALIDATION_SUCCESS_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
                    .increment(1);
                self.dependencies
                    .effects_for_valid_block(casper, block)
                    .await
            }
            Either::Left(invalid_block) => {
                metrics::counter!(BLOCK_VALIDATION_FAILED_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
                    .increment(1);
                // this is to maintain backward compatibility with casper validate method.
                // as it returns not only InvalidBlock or ValidBlock
                match invalid_block {
                    BlockError::Invalid(i) => {
                        self.dependencies
                            .effects_for_invalid_block(casper, block, i, &snapshot)
                            .await
                    }
                    _ => {
                        // this should never happen
                        Ok(snapshot.dag.clone())
                    }
                }
            }
        }?;

        // once block is validated and effects are invoked, it should be removed from buffer
        self.dependencies.remove_from_buffer(block).await?;
        self.dependencies.ack_processed(block).await?;
        maybe_trim_allocator_after_block();

        Ok(status)
    }

    /// Equivalent to Scala's: ackProcessed = (b: BlockMessage) => BlockRetriever[F].ackInCasper(b.blockHash)
    pub async fn ack_processed(&self, block: &BlockMessage) -> Result<(), CasperError> {
        self.dependencies.ack_processed(block).await
    }
}

/// Unified dependencies structure - equivalent to Scala companion object approach
/// Contains all dependencies needed for block processing in one place
#[derive(Clone)]
pub struct BlockProcessorDependencies<T: TransportLayer + Send + Sync> {
    block_store: KeyValueBlockStore,
    casper_buffer: CasperBufferKeyValueStorage,
    block_dag_storage: BlockDagKeyValueStorage,
    block_retriever: BlockRetriever<T>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
    casper_buffer_last_prune_ms: Arc<AtomicU64>,
    missing_dependency_attempts: Arc<Mutex<HashMap<BlockHash, u32>>>,
    missing_dependency_quarantine_until: Arc<Mutex<HashMap<BlockHash, u64>>>,
}

impl<T: TransportLayer + Send + Sync> BlockProcessorDependencies<T> {
    pub fn new(
        block_store: KeyValueBlockStore,
        casper_buffer: CasperBufferKeyValueStorage,
        block_dag_storage: BlockDagKeyValueStorage,
        block_retriever: BlockRetriever<T>,
        transport: Arc<T>,
        connections_cell: ConnectionsCell,
        conf: RPConf,
    ) -> Self {
        Self {
            block_store,
            casper_buffer,
            block_dag_storage,
            block_retriever,
            transport,
            connections_cell,
            conf,
            casper_buffer_last_prune_ms: Arc::new(AtomicU64::new(0)),
            missing_dependency_attempts: Arc::new(Mutex::new(HashMap::new())),
            missing_dependency_quarantine_until: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    // Public getters for tests
    pub fn transport(&self) -> &Arc<T> {
        &self.transport
    }

    pub fn casper_buffer(&self) -> &CasperBufferKeyValueStorage {
        &self.casper_buffer
    }

    fn prune_casper_buffer_if_needed(&self) -> Result<(), CasperError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last_prune = self.casper_buffer_last_prune_ms.load(Ordering::Relaxed);
        let prune_interval_ms = casper_buffer_prune_interval_ms();
        if now_ms.saturating_sub(last_prune) < prune_interval_ms {
            return Ok(());
        }
        self.casper_buffer_last_prune_ms
            .store(now_ms, Ordering::Relaxed);

        let (stale_pruned, overflow_pruned) = self
            .casper_buffer
            .enforce_limits(
                casper_buffer_max_approx_nodes(),
                casper_buffer_stale_ttl_ms(),
                casper_buffer_max_prune_batch(),
                prune_interval_ms,
            )
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        let approx_nodes = self.casper_buffer.approx_node_count();

        metrics::gauge!(CASPER_BUFFER_APPROX_NODES_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
            .set(approx_nodes as f64);
        if stale_pruned > 0 {
            metrics::counter!(CASPER_BUFFER_STALE_PRUNED_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
                .increment(stale_pruned as u64);
        }
        if overflow_pruned > 0 {
            metrics::counter!(CASPER_BUFFER_OVERFLOW_PRUNED_METRIC, "source" => BLOCK_PROCESSOR_METRICS_SOURCE)
                .increment(overflow_pruned as u64);
        }
        if stale_pruned > 0 || overflow_pruned > 0 {
            tracing::warn!(
                "Pruned CasperBuffer entries: stale={}, overflow={}, approx_nodes={}",
                stale_pruned,
                overflow_pruned,
                approx_nodes
            );
        }

        Ok(())
    }

    /// Equivalent to Scala's: storeBlock = (b: BlockMessage) => BlockStore[F].put(b)
    pub async fn store_block(&self, block: &BlockMessage) -> Result<(), CasperError> {
        self.block_store
            .put_block_message(block)
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        Ok(())
    }

    /// Equivalent to Scala's: getCasperStateSnapshot = (c: Casper[F]) => c.getSnapshot
    pub async fn get_casper_state_snapshot(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
    ) -> Result<CasperSnapshot, CasperError> {
        casper.get_snapshot().await
    }

    /// Equivalent to Scala's: getNonValidatedDependencies = (c: Casper[F], b: BlockMessage) => { ... }
    pub async fn get_non_validated_dependencies(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
    ) -> Result<(bool, HashSet<BlockHash>, HashSet<BlockHash>), CasperError> {
        let all_deps = proto_util::dependencies_hashes_of(block);

        // in addition, equivocation tracker has to be checked, as admissible equivocations are not stored in DAG
        let equivocation_hashes: HashSet<BlockHash> = {
            self.block_dag_storage
                .access_equivocations_tracker(|tracker| {
                    let equivocation_records = tracker.data()?;
                    // Use HashSet to ensure uniqueness and O(1) lookup, just like Scala's Set
                    let hashes: HashSet<BlockHash> = equivocation_records
                        .iter()
                        .flat_map(|record| record.equivocation_detected_block_hashes.iter())
                        .cloned()
                        .collect();
                    Ok(hashes)
                })
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?
        };
        // Invalid blocks are already known/built into Casper state and should not be re-fetched
        // as unresolved dependencies.
        let invalid_block_hashes: HashSet<BlockHash> = {
            self.block_dag_storage
                .get_representation()
                .invalid_blocks_map()
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?
                .into_keys()
                .collect()
        };

        let deps_in_buffer: Vec<BlockHash> = {
            all_deps
                .iter()
                .filter_map(|dep| {
                    let block_hash_serde = BlockHashSerde(dep.clone());
                    if self.casper_buffer.contains(&block_hash_serde)
                        || self.casper_buffer.is_pendant(&block_hash_serde)
                    {
                        Some(dep.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };

        let deps_in_dag: Vec<BlockHash> = all_deps
            .iter()
            .filter_map(|dep| {
                if casper.dag_contains(dep) {
                    Some(dep.clone())
                } else {
                    None
                }
            })
            .collect();

        let deps_in_eq_tracker: Vec<BlockHash> = all_deps
            .iter()
            .filter(|&dep| equivocation_hashes.contains(dep))
            .cloned()
            .collect();
        let deps_in_invalid_set: Vec<BlockHash> = all_deps
            .iter()
            .filter(|&dep| invalid_block_hashes.contains(dep))
            .cloned()
            .collect();

        let mut deps_validated: Vec<BlockHash> = deps_in_dag.clone();
        deps_validated.extend(deps_in_eq_tracker.iter().cloned());
        deps_validated.extend(deps_in_invalid_set.iter().cloned());

        let deps_to_fetch: Vec<BlockHash> = all_deps
            .iter()
            .filter(|&dep| !deps_in_buffer.contains(dep))
            .filter(|&dep| !deps_validated.contains(dep))
            .cloned()
            .collect();

        let ready = deps_to_fetch.is_empty() && deps_in_buffer.is_empty();

        if !ready {
            tracing::debug!(
                "Block {} waiting on missing dependencies. To fetch: {}. In buffer: {}. Validated: {}.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                PrettyPrinter::build_string_hashes(
                    &deps_to_fetch
                        .iter()
                        .map(|h| h.as_ref().to_vec())
                        .collect::<Vec<_>>()
                ),
                PrettyPrinter::build_string_hashes(
                    &deps_in_buffer
                        .iter()
                        .map(|h| h.as_ref().to_vec())
                        .collect::<Vec<_>>()
                ),
                PrettyPrinter::build_string_hashes(
                    &deps_validated
                        .iter()
                        .map(|h| h.as_ref().to_vec())
                        .collect::<Vec<_>>()
                )
            );
        }

        Ok((
            ready,
            deps_to_fetch.into_iter().collect::<HashSet<BlockHash>>(),
            deps_in_buffer.into_iter().collect::<HashSet<BlockHash>>(),
        ))
    }

    /// Equivalent to Scala's: commitToBuffer = (b: BlockMessage, deps: Option[Set[BlockHash]]) => { ... }
    pub async fn commit_to_buffer(
        &self,
        block: &BlockMessage,
        deps: Option<HashSet<BlockHash>>,
    ) -> Result<(), CasperError> {
        match deps {
            None => {
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                self.casper_buffer
                    .put_pendant(block_hash_serde)
                    .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
            }
            Some(dependencies) => {
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                dependencies.iter().try_for_each(|dep| {
                    let dep_serde = BlockHashSerde(dep.clone());
                    self.casper_buffer
                        .add_relation(dep_serde, block_hash_serde.clone())
                        .map_err(|e| CasperError::RuntimeError(e.to_string()))
                })?;
            }
        }

        Ok(())
    }

    /// Equivalent to Scala's: removeFromBuffer = (b: BlockMessage) => casperBuffer.remove(b.blockHash)
    pub async fn remove_from_buffer(&self, block: &BlockMessage) -> Result<(), CasperError> {
        let block_hash_serde = BlockHashSerde(block.block_hash.clone());
        self.casper_buffer
            .remove(block_hash_serde)
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        self.clear_missing_dependency_attempts(&block.block_hash)?;
        self.clear_missing_dependency_quarantine(&block.block_hash)?;

        Ok(())
    }

    fn sweep_orphaned_missing_dependency_attempts(&self) -> Result<(), CasperError> {
        let to_clear: Vec<BlockHash> = {
            let attempts = self
                .missing_dependency_attempts
                .lock()
                .map_err(|_| {
                    CasperError::RuntimeError(
                        "Failed to acquire missing_dependency_attempts lock".to_string(),
                    )
                })?;

            attempts
                .keys()
                .filter_map(|block_hash| {
                    let block_hash_serde = BlockHashSerde(block_hash.clone());
                    let is_active = self.casper_buffer.contains(&block_hash_serde)
                        || self.casper_buffer.is_pendant(&block_hash_serde);

                    if is_active {
                        None
                    } else {
                        Some(block_hash.clone())
                    }
                })
                .collect()
        };

        if to_clear.is_empty() {
            return Ok(());
        }

        let mut attempts = self
            .missing_dependency_attempts
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire missing_dependency_attempts lock".to_string(),
                )
            })?;

        for block_hash in to_clear {
            attempts.remove(&block_hash);
        }

        Ok(())
    }

    fn sweep_orphaned_missing_dependency_quarantine(&self) -> Result<(), CasperError> {
        let to_clear: Vec<BlockHash> = {
            let quarantine: Vec<BlockHash> = self
                .missing_dependency_quarantine_until
                .lock()
                .map_err(|_| {
                    CasperError::RuntimeError(
                        "Failed to acquire missing_dependency_quarantine_until lock".to_string(),
                    )
                })?
                .keys()
                .cloned()
                .collect();

            quarantine
                .into_iter()
                .filter_map(|block_hash| {
                    let block_hash_serde = BlockHashSerde(block_hash.clone());
                    let is_active = self.casper_buffer.contains(&block_hash_serde)
                        || self.casper_buffer.is_pendant(&block_hash_serde);

                    if is_active {
                        None
                    } else {
                        Some(block_hash)
                    }
                })
                .collect()
        };

        if to_clear.is_empty() {
            return Ok(());
        }

        let mut quarantine = self
            .missing_dependency_quarantine_until
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire missing_dependency_quarantine_until lock".to_string(),
                )
            })?;

        for block_hash in to_clear {
            quarantine.remove(&block_hash);
        }

        Ok(())
    }

    fn register_missing_dependency_attempt(
        &self,
        block_hash: &BlockHash,
    ) -> Result<bool, CasperError> {
        let mut attempts = self.missing_dependency_attempts.lock().map_err(|_| {
            CasperError::RuntimeError(
                "Failed to acquire missing_dependency_attempts lock".to_string(),
            )
        })?;
        let next = attempts.entry(block_hash.clone()).or_insert(0);
        *next = next.saturating_add(1);
        Ok(*next >= missing_dependency_attempts_max())
    }

    fn clear_missing_dependency_attempts(&self, block_hash: &BlockHash) -> Result<(), CasperError> {
        let mut attempts = self.missing_dependency_attempts.lock().map_err(|_| {
            CasperError::RuntimeError(
                "Failed to acquire missing_dependency_attempts lock".to_string(),
            )
        })?;
        attempts.remove(block_hash);
        Ok(())
    }

    fn clear_missing_dependency_quarantine(&self, block_hash: &BlockHash) -> Result<(), CasperError> {
        let mut quarantine = self
            .missing_dependency_quarantine_until
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire missing_dependency_quarantine_until lock".to_string(),
                )
            })?;
        quarantine.remove(block_hash);
        Ok(())
    }

    fn mark_missing_dependency_quarantine(
        &self,
        block_hash: &BlockHash,
    ) -> Result<(), CasperError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let until = now_ms.saturating_add(missing_dependency_quarantine_ms());
        let mut quarantine = self
            .missing_dependency_quarantine_until
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire missing_dependency_quarantine_until lock".to_string(),
                )
            })?;
        quarantine.insert(block_hash.clone(), until);
        Ok(())
    }

    fn is_missing_dependency_quarantined(
        &self,
        block_hash: &BlockHash,
    ) -> Result<bool, CasperError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let quarantine = self
            .missing_dependency_quarantine_until
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire missing_dependency_quarantine_until lock".to_string(),
                )
            })?;
        Ok(quarantine
            .get(block_hash)
            .copied()
            .is_some_and(|until| now_ms < until))
    }

    fn sweep_expired_missing_dependency_quarantine(&self) -> Result<(), CasperError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mut quarantine = self
            .missing_dependency_quarantine_until
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire missing_dependency_quarantine_until lock".to_string(),
                )
            })?;
        quarantine.retain(|_, until| *until > now_ms);
        Ok(())
    }

    async fn drop_dependency_loop_block(&self, block: &BlockMessage) -> Result<(), CasperError> {
        self.remove_from_buffer(block).await?;
        self.mark_missing_dependency_quarantine(&block.block_hash)?;
        self.ack_processed(block).await?;
        Ok(())
    }

    /// Equivalent to Scala's: requestMissingDependencies = (deps: Set[BlockHash]) => { ... }
    pub async fn request_missing_dependencies(
        &self,
        deps: &HashSet<BlockHash>,
    ) -> Result<(), CasperError> {
        for dep in deps {
            self.block_retriever
                .admit_hash(
                    dep.clone(),
                    None,
                    AdmitHashReason::MissingDependencyRequested,
                )
                .await
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        }

        Ok(())
    }

    /// Recovery helper for deadlock scenarios where dependencies remain in CasperBuffer
    /// but there are no newly discovered hashes to fetch.
    pub async fn recover_stale_buffer_dependencies(
        &self,
        deps: &HashSet<BlockHash>,
    ) -> Result<(), CasperError> {
        for dep in deps {
            self.block_retriever
                .recover_dependency(dep.clone())
                .await
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        }

        Ok(())
    }

    /// Equivalent to Scala's: validateBlock = (c: Casper[F], s: CasperSnapshot[F], b: BlockMessage) => c.validate(b, s)
    pub async fn validate_block(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        snapshot: &mut CasperSnapshot,
        block: &BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        casper.validate(block, snapshot).await
    }

    /// Equivalent to Scala's: ackProcessed = (b: BlockMessage) => BlockRetriever[F].ackInCasper(b.blockHash)
    pub async fn ack_processed(&self, block: &BlockMessage) -> Result<(), CasperError> {
        self.block_retriever
            .ack_in_casper(block.block_hash.clone())
            .await
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        Ok(())
    }

    /// Equivalent to Scala's: effectsForInvalidBlock = (c: Casper[F], b: BlockMessage, r: InvalidBlock, s: CasperSnapshot[F]) => { ... }
    pub async fn effects_for_invalid_block(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
        invalid_block: &InvalidBlock,
        snapshot: &CasperSnapshot,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        let dag = casper.handle_invalid_block(block, invalid_block, &snapshot.dag)?;

        // Equivalent to Scala's: CommUtil[F].sendBlockHash(b.blockHash, b.sender)
        if let Err(err) = self
            .transport
            .send_block_hash(
                &self.connections_cell,
                &self.conf,
                &block.block_hash,
                &block.sender,
            )
            .await
        {
            tracing::warn!(
                "Failed to send block hash {} to sender during invalid-block effects: {}",
                PrettyPrinter::build_string_bytes(&block.block_hash),
                err
            );
        }

        Ok(dag)
    }

    /// Equivalent to Scala's: effectsForValidBlock = (c: Casper[F], b: BlockMessage) => { ... }
    pub async fn effects_for_valid_block(
        &self,
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block: &BlockMessage,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        let dag = { casper.handle_valid_block(block).await? };

        // Equivalent to Scala's: CommUtil[F].sendBlockHash(b.blockHash, b.sender)
        if let Err(err) = self
            .transport
            .send_block_hash(
                &self.connections_cell,
                &self.conf,
                &block.block_hash,
                &block.sender,
            )
            .await
        {
            tracing::warn!(
                "Failed to send block hash {} to sender during valid-block effects: {}",
                PrettyPrinter::build_string_bytes(&block.block_hash),
                err
            );
        }

        Ok(dag)
    }
}

/// Constructor function equivalent to Scala's companion object apply method
/// Creates unified dependencies and BlockProcessor
pub fn new_block_processor<T: TransportLayer + Send + Sync>(
    block_store: KeyValueBlockStore,
    casper_buffer: CasperBufferKeyValueStorage,
    block_dag_storage: BlockDagKeyValueStorage,
    block_retriever: BlockRetriever<T>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
) -> BlockProcessor<T> {
    let dependencies = BlockProcessorDependencies::new(
        block_store,
        casper_buffer,
        block_dag_storage,
        block_retriever,
        transport,
        connections_cell,
        conf,
    );

    BlockProcessor::new(dependencies)
}
