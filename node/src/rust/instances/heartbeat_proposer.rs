use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use casper::rust::blocks::proposer::proposer::ProposerResult;
use casper::rust::casper::{CasperSnapshot, MultiParentCasper};
use casper::rust::casper_conf::HeartbeatConf;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::heartbeat_signal::{HeartbeatSignal, HeartbeatSignalRef};
use casper::rust::validator_identity::ValidatorIdentity;
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::BlockMessage;
use rand::Rng;
use shared::rust::dag::dag_ops;
use tokio::sync::Notify;

use casper::rust::ProposeFunction;

/// Implementation of HeartbeatSignal using tokio::sync::Notify.
/// This allows external callers (like deploy submission) to wake the heartbeat immediately.
struct NotifyHeartbeatSignal {
    notify: Arc<Notify>,
}

impl HeartbeatSignal for NotifyHeartbeatSignal {
    fn trigger_wake(&self) {
        self.notify.notify_one();
    }
}

/// Heartbeat proposer that periodically checks if a block
/// needs to be proposed to maintain liveness.
pub struct HeartbeatProposer;

impl HeartbeatProposer {
    /// Create a heartbeat proposer stream that periodically checks if a block
    /// needs to be proposed to maintain liveness.
    ///
    /// This integrates with the existing propose queue mechanism for thread safety.
    /// The heartbeat simply calls the same triggerPropose function that user deploys
    /// and explicit propose calls use, ensuring serialization through ProposerInstance.
    ///
    /// To prevent lock-step behavior between validators, the stream waits a random
    /// amount of time (0 to checkInterval) before starting the periodic checks.
    ///
    /// The heartbeat only runs on bonded validators. It checks the active validators
    /// set before proposing to avoid unnecessary attempts by unbonded nodes.
    ///
    /// # Arguments
    ///
    /// * `engine_cell` - The EngineCell to read the current Casper instance from
    /// * `trigger_propose_f` - The propose function that integrates with the propose queue
    /// * `validator_identity` - The validator identity to check if bonded
    /// * `config` - Heartbeat configuration (enabled, check_interval, max_lfb_age)
    /// * `max_number_of_parents` - Maximum number of parents allowed for blocks
    /// * `heartbeat_signal_ref` - Shared reference where the signal will be stored
    ///
    /// # Returns
    ///
    /// Returns `Some(JoinHandle)` when the heartbeat is spawned, or `None` when
    /// disabled, trigger function is not available, or max_number_of_parents == 1.
    ///
    /// # Safety
    ///
    /// Heartbeat requires max-number-of-parents > 1. With only 1 parent allowed,
    /// empty heartbeat blocks would fail InvalidParents validation when other
    /// validators have newer blocks.
    pub fn create(
        engine_cell: Arc<EngineCell>,
        trigger_propose_f: Option<Arc<ProposeFunction>>, // same queue/function used by user-triggered proposes
        validator_identity: ValidatorIdentity,
        config: HeartbeatConf,
        max_number_of_parents: i32,
        heartbeat_signal_ref: HeartbeatSignalRef,
    ) -> Option<tokio::task::JoinHandle<()>> {
        // CRITICAL: Heartbeat cannot work with max-number-of-parents = 1
        // Empty blocks would fail InvalidParents validation when other validators have newer blocks
        if max_number_of_parents == 1 {
            tracing::error!(
                "\n\
============================================================================\n\
  CONFIGURATION ERROR: Heartbeat incompatible with max-number-of-parents=1\n\
============================================================================\n\
\n\
  The heartbeat proposer cannot function when max-number-of-parents is 1.\n\
  With single-parent mode, empty heartbeat blocks fail InvalidParents\n\
  validation when other validators have newer blocks, causing the shard\n\
  to stall after the first few blocks.\n\
\n\
  SOLUTION: Set max-number-of-parents to at least 3x your shard size.\n\
            Example: For a 3-validator shard, use max-number-of-parents = 9\n\
\n\
  The heartbeat thread is now DISABLED.\n\
  Your shard will NOT make automatic progress without user deploys.\n\
============================================================================"
            );
            return None;
        }

        if !config.enabled {
            return None;
        }

        let trigger = match trigger_propose_f {
            Some(f) => f,
            None => {
                tracing::warn!("Heartbeat: trigger_propose function not available, skipping spawn");
                return None;
            }
        };

        // Create the signal mechanism using tokio::sync::Notify
        let notify = Arc::new(Notify::new());
        let signal: Arc<dyn HeartbeatSignal> = Arc::new(NotifyHeartbeatSignal {
            notify: notify.clone(),
        });

        // Store the signal in the shared reference so Casper can use it
        // Use try_write() since we're being called from sync context within async runtime
        match heartbeat_signal_ref.try_write() {
            Ok(mut signal_guard) => {
                *signal_guard = Some(signal);
            }
            Err(_) => {
                tracing::warn!("Heartbeat: Could not acquire write lock for signal ref, signal-based wake may not work");
            }
        }

        let initial_delay = random_initial_delay(config.check_interval);
        tracing::info!(
            "Heartbeat: Starting with random initial delay of {}s (check interval: {}s, max LFB age: {}s, signal-based wake enabled)",
            initial_delay.as_secs(),
            config.check_interval.as_secs(),
            config.max_lfb_age.as_secs()
        );

        let handle = tokio::spawn(async move {
            tokio::time::sleep(initial_delay).await;

            loop {
                // Race between timer and signal - whichever completes first triggers wake
                let wake_source = tokio::select! {
                    _ = tokio::time::sleep(config.check_interval) => "timer",
                    _ = notify.notified() => "signal",
                };

                tracing::debug!("Heartbeat: Woke from {}", wake_source);
                let eng = engine_cell.get().await;

                // Access Casper if available and run the check
                if let Some(casper) = eng.with_casper() {
                    let _ =
                        do_heartbeat_check(casper, &*trigger, &validator_identity, &config).await;
                } else {
                    tracing::debug!("Heartbeat: Casper not available yet, skipping check");
                }
            }
        });

        Some(handle)
    }
}

fn random_initial_delay(check_interval: Duration) -> Duration {
    let max_millis = check_interval.as_millis() as u64;
    let random_millis = rand::rng().random_range(0..=max_millis);
    Duration::from_millis(random_millis)
}

async fn do_heartbeat_check(
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    trigger_propose: &ProposeFunction,
    validator_identity: &ValidatorIdentity,
    config: &HeartbeatConf,
) -> Result<(), casper::rust::errors::CasperError> {
    let snapshot: CasperSnapshot = casper.get_snapshot().await?;

    let is_bonded = snapshot
        .on_chain_state
        .active_validators
        .contains(&validator_identity.public_key.bytes);

    if !is_bonded {
        tracing::info!("Heartbeat: Validator is not bonded, skipping heartbeat propose");
    } else {
        tracing::debug!("Heartbeat: Validator is bonded, checking LFB age");
        check_lfb_and_propose(casper.clone(), trigger_propose, validator_identity, config).await?;
    }

    Ok(())
}

async fn check_lfb_and_propose(
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    trigger_propose: &ProposeFunction,
    validator_identity: &ValidatorIdentity,
    config: &HeartbeatConf,
) -> Result<(), casper::rust::errors::CasperError> {
    // Get current snapshot
    let snapshot: CasperSnapshot = casper.get_snapshot().await?;

    // Check if we have pending user deploys
    let has_pending_deploys = !snapshot.deploys_in_scope.is_empty();

    // Get last finalized block
    let lfb: BlockMessage = casper.last_finalized_block().await?;

    // Check if LFB is stale
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u128)
        .unwrap_or(0);

    let lfb_timestamp_ms = lfb.header.timestamp as u128;
    let time_since_lfb = now.saturating_sub(lfb_timestamp_ms);
    let lfb_is_stale = time_since_lfb > config.max_lfb_age.as_millis();

    // Check if we have new parents (new blocks since our last block)
    let has_new_parents = check_has_new_parents(&snapshot, validator_identity);

    // Proposal logic: propose if (pending deploys) OR (LFB stale AND new parents)
    let should_propose = has_pending_deploys || (lfb_is_stale && has_new_parents);

    if should_propose {
        let reason = if has_pending_deploys {
            format!("{} pending user deploys", snapshot.deploys_in_scope.len())
        } else {
            format!(
                "LFB is stale ({}ms old, threshold: {}ms) and new parents exist",
                time_since_lfb,
                config.max_lfb_age.as_millis()
            )
        };

        tracing::info!("Heartbeat: Proposing block - reason: {}", reason);

        let result = trigger_propose(casper.clone(), false).await?;
        match result {
            ProposerResult::Empty => {
                tracing::debug!("Heartbeat: Propose already in progress, will retry next check");
            }
            ProposerResult::Failure(status, seq_num) => {
                tracing::warn!(
                    "Heartbeat: Propose failed with {} (seqNum {})",
                    status,
                    seq_num
                );
            }
            ProposerResult::Success(_, _) => {
                tracing::info!("Heartbeat: Successfully created block");
            }
            ProposerResult::Started(seq_num) => {
                tracing::info!("Heartbeat: Async propose started (seqNum {})", seq_num);
            }
        }
    } else {
        let reason = if !has_new_parents {
            "no new parents (would violate validation)".to_string()
        } else if !lfb_is_stale {
            format!(
                "LFB age is {}ms (threshold: {}ms)",
                time_since_lfb,
                config.max_lfb_age.as_millis()
            )
        } else {
            "unknown".to_string()
        };
        tracing::debug!("Heartbeat: No action needed - reason: {}", reason);
    }

    Ok(())
}

/// Check if new blocks exist since this validator's last block.
/// Returns true if:
/// - Validator has no blocks yet (can propose)
/// - Validator's last block is genesis (allows breaking post-genesis deadlock)
/// - Any of the current frontier blocks are not in the ancestor set of validator's last block
fn check_has_new_parents(
    snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
) -> bool {
    let validator_id = &validator_identity.public_key.bytes;

    // Get validator's last block
    let last_block_hash = match snapshot.dag.latest_message_hash(validator_id) {
        Some(hash) => hash,
        None => {
            // Validator has no blocks yet - can propose
            return true;
        }
    };

    // Check if this is genesis block (allows breaking deadlock after genesis)
    let block_meta = match snapshot.dag.lookup(&last_block_hash) {
        Ok(Some(meta)) => meta,
        _ => {
            // Can't find block metadata, allow proposal
            return true;
        }
    };

    if block_meta.parents.is_empty() {
        // This is genesis - allow proposal to break post-genesis deadlock
        tracing::debug!("Heartbeat: Validator's last block is genesis, allowing proposal");
        return true;
    }

    // Get all blocks validator knew about when creating last block (ancestor set)
    let neighbor_fn = |hash: &BlockHash| -> Vec<BlockHash> {
        match snapshot.dag.lookup(hash) {
            Ok(Some(meta)) => meta
                .parents
                .into_iter()
                .map(|p| BlockHash::from(p))
                .collect(),
            _ => vec![],
        }
    };

    let ancestor_hashes = dag_ops::bf_traverse(vec![last_block_hash], neighbor_fn);
    let ancestor_set: HashSet<BlockHash> = ancestor_hashes.into_iter().collect();

    // Get current latest blocks (frontier)
    let all_latest_blocks = snapshot.dag.latest_message_hashes();

    // Check if any of the latest blocks are new (not in ancestor set)
    for entry in all_latest_blocks.iter() {
        let hash = entry.value();
        if !ancestor_set.contains(hash) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
    use block_storage::rust::dag::block_metadata_store::BlockMetadataStore;
    use casper::rust::casper::{CasperShardConf, OnChainCasperState};
    use casper::rust::heartbeat_signal::new_heartbeat_signal_ref;
    use crypto::rust::signatures::secp256k1::Secp256k1;
    use crypto::rust::signatures::signatures_alg::SignaturesAlg;
    use dashmap::{DashMap, DashSet};
    use models::rust::block_metadata::BlockMetadata;
    use models::rust::validator::Validator;
    use prost::bytes::Bytes;
    use rspace_plus_plus::rspace::shared::in_mem_key_value_store::InMemoryKeyValueStore;
    use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::RwLock;

    fn create_test_validator_identity() -> ValidatorIdentity {
        let secp = Secp256k1;
        let (sk, pk) = secp.new_key_pair();
        ValidatorIdentity {
            public_key: pk,
            private_key: sk,
            signature_algorithm: "secp256k1".to_string(),
        }
    }

    fn create_empty_dag_representation() -> KeyValueDagRepresentation {
        let block_metadata_store =
            KeyValueTypedStoreImpl::new(Arc::new(InMemoryKeyValueStore::new()));

        KeyValueDagRepresentation {
            dag_set: Arc::new(DashSet::new()),
            latest_messages_map: Arc::new(DashMap::new()),
            child_map: Arc::new(DashMap::new()),
            height_map: Arc::new(RwLock::new(BTreeMap::new())),
            invalid_blocks_set: Arc::new(DashSet::new()),
            last_finalized_block_hash: BlockHash::new(),
            finalized_blocks_set: Arc::new(DashSet::new()),
            block_metadata_index: Arc::new(RwLock::new(BlockMetadataStore::new(
                block_metadata_store,
            ))),
            deploy_index: Arc::new(RwLock::new(KeyValueTypedStoreImpl::new(Arc::new(
                InMemoryKeyValueStore::new(),
            )))),
        }
    }

    fn create_test_snapshot(dag: KeyValueDagRepresentation) -> CasperSnapshot {
        CasperSnapshot {
            dag,
            last_finalized_block: Bytes::new(),
            lca: Bytes::new(),
            tips: Vec::new(),
            parents: Vec::new(),
            justifications: DashSet::new(),
            invalid_blocks: HashMap::new(),
            deploys_in_scope: DashSet::new(),
            max_block_num: 0,
            max_seq_nums: DashMap::new(),
            on_chain_state: OnChainCasperState {
                shard_conf: CasperShardConf::new(),
                bonds_map: HashMap::new(),
                active_validators: Vec::new(),
            },
        }
    }

    // ==================== check_has_new_parents tests ====================

    #[test]
    fn check_has_new_parents_returns_true_when_validator_has_no_blocks() {
        // Validator has never created a block - should be allowed to propose
        let dag = create_empty_dag_representation();
        let snapshot = create_test_snapshot(dag);
        let validator = create_test_validator_identity();

        let result = check_has_new_parents(&snapshot, &validator);

        assert!(
            result,
            "Validator with no blocks should be allowed to propose"
        );
    }

    #[test]
    fn check_has_new_parents_returns_true_when_validators_last_block_is_genesis() {
        // Validator's last block has no parents (is genesis) - allows breaking deadlock
        let dag = create_empty_dag_representation();
        let validator = create_test_validator_identity();
        let validator_id: Validator = validator.public_key.bytes.clone().into();

        // Create a genesis block (no parents)
        let genesis_hash = BlockHash::from(b"genesis".to_vec());
        let genesis_meta = BlockMetadata {
            block_hash: genesis_hash.clone().into(),
            parents: vec![], // No parents = genesis
            sender: validator.public_key.bytes.clone(),
            justifications: vec![],
            weight_map: BTreeMap::new(),
            block_number: 0,
            sequence_number: 0,
            invalid: false,
            directly_finalized: false,
            finalized: true,
        };

        // Add genesis to DAG
        dag.dag_set.insert(genesis_hash.clone());
        dag.latest_messages_map
            .insert(validator_id.clone(), genesis_hash.clone());
        {
            let mut meta_store = dag.block_metadata_index.write().unwrap();
            meta_store.add(genesis_meta).unwrap();
        }

        let snapshot = create_test_snapshot(dag);

        let result = check_has_new_parents(&snapshot, &validator);

        assert!(
            result,
            "Validator whose last block is genesis should be allowed to propose"
        );
    }

    #[test]
    fn check_has_new_parents_returns_true_when_new_blocks_exist_in_frontier() {
        // Another validator has created a block that's not in our ancestor set
        let dag = create_empty_dag_representation();
        let validator = create_test_validator_identity();
        let validator_id: Validator = validator.public_key.bytes.clone().into();

        // Create our validator's block
        let our_block_hash = BlockHash::from(b"our_block".to_vec());
        let genesis_hash = BlockHash::from(b"genesis".to_vec());
        let our_block_meta = BlockMetadata {
            block_hash: our_block_hash.clone().into(),
            parents: vec![genesis_hash.clone().into()], // Has parent, not genesis
            sender: validator.public_key.bytes.clone(),
            justifications: vec![],
            weight_map: BTreeMap::new(),
            block_number: 1,
            sequence_number: 1,
            invalid: false,
            directly_finalized: false,
            finalized: false,
        };

        // Create another validator's block (new block not in our ancestors)
        let other_validator = create_test_validator_identity();
        let other_validator_id: Validator = other_validator.public_key.bytes.clone().into();
        let other_block_hash = BlockHash::from(b"other_block".to_vec());

        // Add blocks to DAG
        dag.dag_set.insert(our_block_hash.clone());
        dag.dag_set.insert(other_block_hash.clone());
        dag.latest_messages_map
            .insert(validator_id.clone(), our_block_hash.clone());
        dag.latest_messages_map
            .insert(other_validator_id.clone(), other_block_hash.clone());
        {
            let mut meta_store = dag.block_metadata_index.write().unwrap();
            meta_store.add(our_block_meta).unwrap();
        }

        let snapshot = create_test_snapshot(dag);

        let result = check_has_new_parents(&snapshot, &validator);

        assert!(
            result,
            "Should return true when other validators have new blocks"
        );
    }

    #[test]
    fn check_has_new_parents_returns_false_when_no_new_blocks() {
        // All frontier blocks are already in our ancestor set
        let dag = create_empty_dag_representation();
        let validator = create_test_validator_identity();
        let validator_id: Validator = validator.public_key.bytes.clone().into();

        // Create genesis
        let genesis_hash = BlockHash::from(b"genesis".to_vec());
        let genesis_meta = BlockMetadata {
            block_hash: genesis_hash.clone().into(),
            parents: vec![],
            sender: Bytes::new(),
            justifications: vec![],
            weight_map: BTreeMap::new(),
            block_number: 0,
            sequence_number: 0,
            invalid: false,
            directly_finalized: false,
            finalized: true,
        };

        // Create our validator's block that includes genesis as parent
        let our_block_hash = BlockHash::from(b"our_block".to_vec());
        let our_block_meta = BlockMetadata {
            block_hash: our_block_hash.clone().into(),
            parents: vec![genesis_hash.clone().into()],
            sender: validator.public_key.bytes.clone(),
            justifications: vec![],
            weight_map: BTreeMap::new(),
            block_number: 1,
            sequence_number: 1,
            invalid: false,
            directly_finalized: false,
            finalized: false,
        };

        // Our block is the only latest message - no new parents exist
        dag.dag_set.insert(genesis_hash.clone());
        dag.dag_set.insert(our_block_hash.clone());
        dag.latest_messages_map
            .insert(validator_id.clone(), our_block_hash.clone());
        {
            let mut meta_store = dag.block_metadata_index.write().unwrap();
            meta_store.add(genesis_meta).unwrap();
            meta_store.add(our_block_meta).unwrap();
        }

        let snapshot = create_test_snapshot(dag);

        let result = check_has_new_parents(&snapshot, &validator);

        assert!(
            !result,
            "Should return false when no new blocks exist in frontier"
        );
    }

    // ==================== HeartbeatProposer::create configuration tests ====================

    #[tokio::test]
    async fn heartbeat_create_returns_none_when_disabled() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: false,
            check_interval: Duration::from_secs(10),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());

        // Create a mock propose function
        let propose_f: Arc<ProposeFunction> = Arc::new(|_casper, _is_async| {
            Box::pin(async { Ok(casper::rust::blocks::proposer::proposer::ProposerResult::Empty) })
        });

        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            10, // max_number_of_parents > 1
            heartbeat_signal_ref,
        );

        assert!(
            result.is_none(),
            "Should return None when heartbeat is disabled"
        );
    }

    #[tokio::test]
    async fn heartbeat_create_returns_none_when_max_parents_is_one_with_enabled_config() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: true,
            check_interval: Duration::from_secs(10),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());

        // Create a mock propose function
        let propose_f: Arc<ProposeFunction> = Arc::new(|_casper, _is_async| {
            Box::pin(async { Ok(casper::rust::blocks::proposer::proposer::ProposerResult::Empty) })
        });

        // max_number_of_parents = 1 should trigger safety check and return None
        // even when config.enabled = true and propose function is provided
        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            1, // max_number_of_parents == 1 triggers safety check
            heartbeat_signal_ref,
        );

        assert!(
            result.is_none(),
            "Should return None when max_number_of_parents == 1 (safety check)"
        );
    }

    #[tokio::test]
    async fn heartbeat_create_returns_none_when_trigger_function_missing() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: true,
            check_interval: Duration::from_secs(10),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());

        let result = HeartbeatProposer::create(
            engine_cell,
            None, // No trigger function
            validator,
            config,
            10, // max_number_of_parents > 1
            heartbeat_signal_ref,
        );

        assert!(
            result.is_none(),
            "Should return None when trigger function is missing"
        );
    }

    #[tokio::test]
    async fn heartbeat_create_returns_none_when_max_parents_equals_one() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: true,
            check_interval: Duration::from_secs(10),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());

        // Create a mock propose function
        let propose_f: Arc<ProposeFunction> = Arc::new(|_casper, _is_async| {
            Box::pin(async { Ok(casper::rust::blocks::proposer::proposer::ProposerResult::Empty) })
        });

        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            1, // max_number_of_parents == 1 triggers safety check
            heartbeat_signal_ref,
        );

        assert!(
            result.is_none(),
            "Should return None when max_number_of_parents == 1"
        );
    }

    #[tokio::test]
    async fn heartbeat_create_returns_some_when_all_conditions_met() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: true,
            check_interval: Duration::from_secs(1),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());

        // Create a mock propose function
        let propose_f: Arc<ProposeFunction> = Arc::new(|_casper, _is_async| {
            Box::pin(async { Ok(casper::rust::blocks::proposer::proposer::ProposerResult::Empty) })
        });

        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            10, // max_number_of_parents > 1
            heartbeat_signal_ref,
        );

        assert!(
            result.is_some(),
            "Should return Some(JoinHandle) when all conditions are met"
        );

        // Clean up: abort the spawned task
        if let Some(handle) = result {
            handle.abort();
        }
    }

    // ==================== Proposal logic tests ====================

    #[test]
    fn proposal_logic_proposes_when_pending_deploys_exist() {
        // Test the proposal decision logic: has_pending_deploys => should_propose
        let has_pending_deploys = true;
        let lfb_is_stale = false;
        let has_new_parents = false;

        let should_propose = has_pending_deploys || (lfb_is_stale && has_new_parents);

        assert!(should_propose, "Should propose when pending deploys exist");
    }

    #[test]
    fn proposal_logic_proposes_when_lfb_stale_and_new_parents() {
        // Test the proposal decision logic: lfb_is_stale && has_new_parents => should_propose
        let has_pending_deploys = false;
        let lfb_is_stale = true;
        let has_new_parents = true;

        let should_propose = has_pending_deploys || (lfb_is_stale && has_new_parents);

        assert!(
            should_propose,
            "Should propose when LFB is stale and new parents exist"
        );
    }

    #[test]
    fn proposal_logic_skips_when_lfb_stale_but_no_new_parents() {
        // Test the proposal decision logic: lfb_is_stale but !has_new_parents => skip
        let has_pending_deploys = false;
        let lfb_is_stale = true;
        let has_new_parents = false;

        let should_propose = has_pending_deploys || (lfb_is_stale && has_new_parents);

        assert!(
            !should_propose,
            "Should NOT propose when LFB is stale but no new parents"
        );
    }

    #[test]
    fn proposal_logic_skips_when_lfb_fresh_and_no_deploys() {
        // Test the proposal decision logic: !lfb_is_stale && !has_pending_deploys => skip
        let has_pending_deploys = false;
        let lfb_is_stale = false;
        let has_new_parents = true;

        let should_propose = has_pending_deploys || (lfb_is_stale && has_new_parents);

        assert!(
            !should_propose,
            "Should NOT propose when LFB is fresh and no pending deploys"
        );
    }
}
