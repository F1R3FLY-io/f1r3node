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

/// Check if a heartbeat propose is needed and trigger one if so.
///
/// This is the core decision logic for heartbeat proposals. It:
/// 1. Gets the current Casper snapshot
/// 2. Checks if the validator is bonded
/// 3. Checks for pending deploys or stale LFB with new parents
/// 4. Triggers a propose if conditions are met
///
/// Exposed for testing - allows direct testing of decision logic without spawning tasks.
pub async fn do_heartbeat_check(
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

/// Unit tests for HeartbeatProposer configuration validation.
///
/// These tests verify the create() function properly handles configuration:
/// - Disabled config returns None
/// - Invalid max-number-of-parents returns None (with error log)
/// - Valid config returns Some(JoinHandle)
///
/// Note: Actual proposal behavior is tested via integration tests (Python/Docker)
/// which can properly set up a full Casper environment.
#[cfg(test)]
mod tests {
    use super::*;
    use casper::rust::heartbeat_signal::new_heartbeat_signal_ref;
    use crypto::rust::signatures::secp256k1::Secp256k1;
    use crypto::rust::signatures::signatures_alg::SignaturesAlg;

    fn create_test_validator_identity() -> ValidatorIdentity {
        let secp = Secp256k1;
        let (sk, pk) = secp.new_key_pair();
        ValidatorIdentity {
            public_key: pk,
            private_key: sk,
            signature_algorithm: "secp256k1".to_string(),
        }
    }

    fn create_mock_propose_function() -> Arc<ProposeFunction> {
        Arc::new(|_casper, _is_async| {
            Box::pin(async { Ok(casper::rust::blocks::proposer::proposer::ProposerResult::Empty) })
        })
    }

    // ==================== Configuration validation tests ====================

    #[tokio::test]
    async fn heartbeat_create_returns_none_when_config_disabled() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: false,
            check_interval: Duration::from_secs(10),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());
        let propose_f = create_mock_propose_function();

        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            10,
            heartbeat_signal_ref,
        );

        assert!(
            result.is_none(),
            "Should return None when heartbeat is disabled"
        );
    }

    #[tokio::test]
    async fn heartbeat_create_returns_none_when_max_parents_is_one() {
        use casper::rust::engine::engine_cell::EngineCell;

        let config = HeartbeatConf {
            enabled: true,
            check_interval: Duration::from_secs(10),
            max_lfb_age: Duration::from_secs(60),
        };
        let validator = create_test_validator_identity();
        let heartbeat_signal_ref = new_heartbeat_signal_ref();
        let engine_cell = Arc::new(EngineCell::init());
        let propose_f = create_mock_propose_function();

        // max_number_of_parents = 1 triggers safety check
        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            1,
            heartbeat_signal_ref,
        );

        assert!(
            result.is_none(),
            "Should return None when max_number_of_parents == 1 (safety check)"
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
        let propose_f = create_mock_propose_function();

        let result = HeartbeatProposer::create(
            engine_cell,
            Some(propose_f),
            validator,
            config,
            10,
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

    // ==================== Decision Logic Tests (Direct Method Calls) ====================
    // Tests that call do_heartbeat_check directly for deterministic behavior

    mod decision_logic_tests {
        use super::*;
        use casper::rust::casper::MultiParentCasper;
        use crypto::rust::signatures::secp256k1::Secp256k1;
        use crypto::rust::signatures::signed::Signed;
        use models::rust::casper::protocol::casper_message::DeployData;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Helper to create LFB with controllable timestamp (age in ms)
        fn create_lfb_with_age(
            age_ms: u64,
        ) -> models::rust::casper::protocol::casper_message::BlockMessage {
            let mut block = models::rust::block_implicits::get_random_block_default();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64;
            block.header.timestamp = now - (age_ms as i64);
            block
        }

        // Helper to create a propose function that tracks call count
        fn create_counting_propose_function() -> (Arc<AtomicUsize>, Arc<ProposeFunction>) {
            use casper::rust::blocks::proposer::propose_result::{ProposeStatus, ProposeSuccess};
            use casper::rust::blocks::proposer::proposer::ProposerResult;

            let count = Arc::new(AtomicUsize::new(0));
            let count_clone = count.clone();
            let func: Arc<ProposeFunction> = Arc::new(move |_casper, _is_async| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                Box::pin(async {
                    Ok(ProposerResult::Success(
                        ProposeStatus::Success(ProposeSuccess {
                            result: casper::rust::block_status::ValidBlock::Valid,
                        }),
                        models::rust::block_implicits::get_random_block_default(),
                    ))
                })
            });
            (count, func)
        }

        // Helper to create a mock signed deploy
        fn create_mock_deploy(validator: &ValidatorIdentity) -> Signed<DeployData> {
            let deploy_data = DeployData {
                term: "new x in { x!(0) }".to_string(),
                time_stamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64,
                phlo_price: 1,
                phlo_limit: 1000,
                valid_after_block_number: 0,
                shard_id: "test-shard".to_string(),
            };
            Signed::create(
                deploy_data,
                Box::new(Secp256k1),
                validator.private_key.clone(),
            )
            .expect("Failed to sign deploy")
        }

        #[tokio::test]
        async fn do_heartbeat_check_triggers_propose_with_pending_deploys() {
            // Create validator identity
            let validator = create_test_validator_identity();
            let validator_id = validator.public_key.bytes.clone();

            // Create a mock deploy
            let signed_deploy = create_mock_deploy(&validator);

            // Create snapshot with pending deploys (validator is bonded)
            let mut snapshot =
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::create_empty_snapshot();
            snapshot.deploys_in_scope.insert(signed_deploy);
            snapshot
                .on_chain_state
                .active_validators
                .push(validator_id.into());

            // Fresh LFB (100ms old)
            let lfb = create_lfb_with_age(100);

            // Create casper with snapshot
            let casper: Arc<dyn MultiParentCasper + Send + Sync> = Arc::new(
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::new(snapshot, lfb),
            );

            // Create counting propose function
            let (propose_count, propose_func) = create_counting_propose_function();

            // Config with long max_lfb_age so LFB is NOT stale
            let config = HeartbeatConf {
                enabled: true,
                check_interval: Duration::from_secs(1),
                max_lfb_age: Duration::from_secs(10),
            };

            // Call do_heartbeat_check directly
            let result = do_heartbeat_check(casper, &*propose_func, &validator, &config).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                1,
                "Should trigger propose when pending deploys exist"
            );
        }

        #[tokio::test]
        async fn do_heartbeat_check_triggers_propose_when_lfb_stale() {
            // Create validator identity
            let validator = create_test_validator_identity();
            let validator_id = validator.public_key.bytes.clone();

            // Create snapshot with no deploys but validator is bonded
            let mut snapshot =
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::create_empty_snapshot();
            snapshot
                .on_chain_state
                .active_validators
                .push(validator_id.into());

            // Stale LFB (60 seconds old)
            let lfb = create_lfb_with_age(60000);

            // Create casper with snapshot
            let casper: Arc<dyn MultiParentCasper + Send + Sync> = Arc::new(
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::new(snapshot, lfb),
            );

            // Create counting propose function
            let (propose_count, propose_func) = create_counting_propose_function();

            // Config with short max_lfb_age so LFB IS stale
            let config = HeartbeatConf {
                enabled: true,
                check_interval: Duration::from_secs(1),
                max_lfb_age: Duration::from_secs(1),
            };

            // Call do_heartbeat_check directly
            let result = do_heartbeat_check(casper, &*propose_func, &validator, &config).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                1,
                "Should trigger propose when LFB is stale and new parents exist"
            );
        }

        #[tokio::test]
        async fn do_heartbeat_check_skips_when_not_bonded() {
            // Create validator identity
            let validator = create_test_validator_identity();

            // Create snapshot with NO active validators (validator not bonded)
            let snapshot =
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::create_empty_snapshot();

            // Stale LFB
            let lfb = create_lfb_with_age(60000);

            // Create casper with snapshot
            let casper: Arc<dyn MultiParentCasper + Send + Sync> = Arc::new(
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::new(snapshot, lfb),
            );

            // Create counting propose function
            let (propose_count, propose_func) = create_counting_propose_function();

            let config = HeartbeatConf {
                enabled: true,
                check_interval: Duration::from_secs(1),
                max_lfb_age: Duration::from_secs(1),
            };

            // Call do_heartbeat_check directly
            let result = do_heartbeat_check(casper, &*propose_func, &validator, &config).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                0,
                "Should NOT trigger propose when validator is not bonded"
            );
        }

        #[tokio::test]
        async fn do_heartbeat_check_skips_when_lfb_fresh() {
            // Create validator identity
            let validator = create_test_validator_identity();
            let validator_id = validator.public_key.bytes.clone();

            // Create snapshot with no deploys but validator is bonded
            let mut snapshot =
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::create_empty_snapshot();
            snapshot
                .on_chain_state
                .active_validators
                .push(validator_id.into());

            // Fresh LFB (100ms old)
            let lfb = create_lfb_with_age(100);

            // Create casper with snapshot
            let casper: Arc<dyn MultiParentCasper + Send + Sync> = Arc::new(
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::new(snapshot, lfb),
            );

            // Create counting propose function
            let (propose_count, propose_func) = create_counting_propose_function();

            // Config with long max_lfb_age so LFB is NOT stale
            let config = HeartbeatConf {
                enabled: true,
                check_interval: Duration::from_secs(1),
                max_lfb_age: Duration::from_secs(10),
            };

            // Call do_heartbeat_check directly
            let result = do_heartbeat_check(casper, &*propose_func, &validator, &config).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                0,
                "Should NOT trigger propose when LFB is fresh and no pending deploys"
            );
        }
    }
}
