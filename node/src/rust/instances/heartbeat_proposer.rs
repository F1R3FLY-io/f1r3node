use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use casper::rust::blocks::proposer::{
    propose_result::{ProposeFailure, ProposeStatus},
    proposer::ProposerResult,
};
use casper::rust::casper::{CasperSnapshot, MultiParentCasper};
use casper::rust::casper_conf::HeartbeatConf;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::heartbeat_signal::{
    install_heartbeat_signal, HeartbeatSignal, HeartbeatSignalRef,
};
use casper::rust::system_deploy::is_system_deploy_id;
use casper::rust::validator_identity::ValidatorIdentity;
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use rand::Rng;
use shared::rust::env;
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

const HEARTBEAT_FRONTIER_CHASE_MAX_LAG_ENV: &str = "F1R3_HEARTBEAT_FRONTIER_CHASE_MAX_LAG";
const DEFAULT_HEARTBEAT_FRONTIER_CHASE_MAX_LAG: i64 = 0;
const HEARTBEAT_PENDING_DEPLOY_MAX_LAG_ENV: &str = "F1R3_HEARTBEAT_PENDING_DEPLOY_MAX_LAG";
const DEFAULT_HEARTBEAT_PENDING_DEPLOY_MAX_LAG: i64 = 20;
const HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG_ENV: &str = "F1R3_HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG";
const DEFAULT_HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG: i64 = 64;
const HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS_ENV: &str = "F1R3_HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS";
const DEFAULT_HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS: u128 = 0;
const HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS_ENV: &str =
    "F1R3_HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS";
const DEFAULT_HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS: u128 = 12_000;
const HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS_ENV: &str =
    "F1R3_HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS";
const DEFAULT_HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS: u128 = 25_000;

fn heartbeat_frontier_chase_max_lag() -> i64 {
    env::var_or_filtered(
        HEARTBEAT_FRONTIER_CHASE_MAX_LAG_ENV,
        DEFAULT_HEARTBEAT_FRONTIER_CHASE_MAX_LAG,
        |value: &i64| *value >= 0,
    )
}

fn heartbeat_pending_deploy_max_lag() -> i64 {
    env::var_or_filtered(
        HEARTBEAT_PENDING_DEPLOY_MAX_LAG_ENV,
        DEFAULT_HEARTBEAT_PENDING_DEPLOY_MAX_LAG,
        |value: &i64| *value >= 0,
    )
}

fn heartbeat_deploy_recovery_max_lag() -> i64 {
    env::var_or_filtered(
        HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG_ENV,
        DEFAULT_HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG,
        |value: &i64| *value >= 0,
    )
}

fn heartbeat_self_propose_cooldown_ms() -> u128 {
    env::var_or(
        HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS_ENV,
        DEFAULT_HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS,
    )
}

fn heartbeat_stale_recovery_min_interval_ms() -> u128 {
    env::var_or(
        HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS_ENV,
        DEFAULT_HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS,
    )
}

fn heartbeat_deploy_finalization_grace_ms() -> u128 {
    env::var_or(
        HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS_ENV,
        DEFAULT_HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS,
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct HeartbeatCheckResult {
    bug_failure: bool,
    refresh_deploy_grace_window: bool,
}

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
        standalone: bool,
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
            tracing::warn!("Heartbeat: config is not enabled!");
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

        // Store the signal in the shared reference so Casper can use it.
        if !install_heartbeat_signal(&heartbeat_signal_ref, signal) {
            tracing::warn!(
                "Heartbeat: signal ref already initialized; keeping existing signal handle"
            );
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
            let mut consecutive_failures: u32 = 0;
            let mut backoff_until: Option<std::time::Instant> = None;
            let mut deploy_grace_until: Option<std::time::Instant> = None;

            loop {
                // Race between timer and signal - whichever completes first triggers wake
                let wake_source = tokio::select! {
                    _ = tokio::time::sleep(config.check_interval) => "timer",
                    _ = notify.notified() => "signal",
                };

                tracing::debug!("Heartbeat: Woke from {}", wake_source);
                let eng = engine_cell.get().await;

                // Access Casper if available and run the check
                // Errors are logged but don't stop the heartbeat loop - transient errors
                // (DB contention, lock timeouts) should not kill the heartbeat
                if let Some(casper) = eng.with_casper() {
                    let now = std::time::Instant::now();
                    if backoff_until.is_some_and(|deadline| now < deadline) {
                        continue;
                    }
                    if deploy_grace_until.is_some_and(|deadline| now >= deadline) {
                        deploy_grace_until = None;
                    }
                    let deploy_grace_active = deploy_grace_until.is_some();

                    match do_heartbeat_check(
                        casper,
                        &*trigger,
                        &validator_identity,
                        &config,
                        standalone,
                        deploy_grace_active,
                    )
                    .await
                    {
                        Ok(outcome) => {
                            if outcome.refresh_deploy_grace_window {
                                let grace_ms = heartbeat_deploy_finalization_grace_ms();
                                let grace_duration = Duration::from_millis(std::cmp::min(
                                    grace_ms,
                                    u128::from(u64::MAX),
                                )
                                    as u64);
                                let deadline = std::time::Instant::now() + grace_duration;
                                deploy_grace_until = Some(deadline);
                                tracing::debug!(
                                    "Heartbeat: refreshed deploy finalization grace window for {:?}",
                                    grace_duration
                                );
                            }

                            if !outcome.bug_failure {
                                consecutive_failures = 0;
                                backoff_until = None;
                                continue;
                            }

                            consecutive_failures = consecutive_failures.saturating_add(1);
                            // Exponential backoff capped at 60s to avoid invalid-propose churn.
                            let shift = consecutive_failures.min(4);
                            let scale = 1u32 << shift;
                            let mut delay = config.check_interval.saturating_mul(scale);
                            let max_delay = Duration::from_secs(60);
                            if delay > max_delay {
                                delay = max_delay;
                            }
                            backoff_until = Some(std::time::Instant::now() + delay);
                            tracing::warn!(
                                "Heartbeat: Entering backoff for {:?} after {} consecutive failures",
                                delay,
                                consecutive_failures
                            );
                        }
                        Err(err) => {
                            tracing::warn!(
                                "Heartbeat: Check failed with error: {:?}, will retry next cycle",
                                err
                            );
                        }
                    }
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
///
/// # Arguments
/// * `standalone` - If true, skips hasNewParents check (single validator can always propose)
async fn do_heartbeat_check(
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    trigger_propose: &ProposeFunction,
    validator_identity: &ValidatorIdentity,
    config: &HeartbeatConf,
    standalone: bool,
    deploy_grace_active: bool,
) -> Result<HeartbeatCheckResult, casper::rust::errors::CasperError> {
    let snapshot: CasperSnapshot = casper.get_snapshot().await?;

    let is_bonded = snapshot
        .on_chain_state
        .active_validators
        .contains(&validator_identity.public_key.bytes);

    if !is_bonded {
        tracing::info!("Heartbeat: Validator is not bonded, skipping heartbeat propose");
        return Ok(HeartbeatCheckResult::default());
    } else {
        tracing::debug!("Heartbeat: Validator is bonded, checking LFB age");
        return Ok(check_lfb_and_propose(
            casper.clone(),
            snapshot,
            trigger_propose,
            validator_identity,
            config,
            standalone,
            deploy_grace_active,
        )
        .await?);
    }
}

async fn check_lfb_and_propose(
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    snapshot: CasperSnapshot,
    trigger_propose: &ProposeFunction,
    validator_identity: &ValidatorIdentity,
    config: &HeartbeatConf,
    standalone: bool,
    deploy_grace_active: bool,
) -> Result<HeartbeatCheckResult, casper::rust::errors::CasperError> {
    // Check if we have pending user deploys in storage (not yet included in blocks)
    let has_pending_deploys = casper
        .has_pending_deploys_in_storage_for_snapshot(&snapshot)
        .await?;

    // Check if LFB is stale
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u128)
        .unwrap_or(0);

    // Avoid running heavyweight finalizer path from heartbeat loop.
    // Use the snapshot's latest finalized block hash and read it from block store directly.
    let lfb_timestamp_ms = match casper.block_store().get(&snapshot.last_finalized_block) {
        Ok(Some(lfb)) => lfb.header.timestamp as u128,
        Err(err) => {
            tracing::warn!(
                "Heartbeat: Failed to read latest finalized block {} for timestamp: {:?}, treating as stale",
                PrettyPrinter::build_string_bytes(&snapshot.last_finalized_block),
                err
            );
            0
        }
        Ok(None) => {
            tracing::warn!(
                "Heartbeat: Finalized block {} missing in block store, treating as stale",
                PrettyPrinter::build_string_bytes(&snapshot.last_finalized_block)
            );
            0
        }
    };
    let time_since_lfb = if now >= lfb_timestamp_ms {
        now - lfb_timestamp_ms
    } else {
        tracing::warn!(
            "LFB timestamp {} is in the future (now: {}), possible clock skew",
            lfb_timestamp_ms,
            now
        );
        0
    };
    let lfb_is_stale = time_since_lfb > config.max_lfb_age.as_millis();
    // Use latest observed frontier timestamp as a second staleness signal.
    // LFB may remain behind head under healthy operation; proposing recovery blocks
    // while the frontier is already active creates avoidable empty-block churn.
    let frontier_latest_timestamp_ms = snapshot
        .dag
        .latest_message_hashes()
        .iter()
        .filter_map(|(_, block_hash)| {
            casper
                .block_store()
                .get(block_hash)
                .ok()
                .flatten()
                .map(|block| block.header.timestamp as u128)
        })
        .max()
        .unwrap_or(lfb_timestamp_ms);
    let frontier_age_ms = now.saturating_sub(frontier_latest_timestamp_ms);
    let frontier_is_stale = frontier_age_ms > config.max_lfb_age.as_millis();
    let last_finalized_block_number = snapshot
        .dag
        .lookup(&snapshot.last_finalized_block)?
        .map(|meta| meta.block_number)
        .unwrap_or(0);
    let latest_block_number = snapshot.dag.latest_block_number();
    let lfb_lag_blocks = latest_block_number.saturating_sub(last_finalized_block_number);

    // If this validator is already ahead of finalized height, avoid repeatedly proposing
    // stale-LFB recovery blocks every heartbeat tick. Keep 1s heartbeat checks, but gate
    // recovery proposals unless finalized catches up or pending deploys exist.
    let self_recently_proposed = match (
        snapshot
            .dag
            .latest_message(&validator_identity.public_key.bytes)?,
        snapshot.dag.lookup(&snapshot.last_finalized_block)?,
    ) {
        (Some(latest_self), Some(last_finalized)) => {
            latest_self.block_number > last_finalized.block_number
        }
        _ => false,
    };
    let self_latest_block_timestamp_ms = snapshot
        .dag
        .latest_message_hash(&validator_identity.public_key.bytes)
        .and_then(|hash| match casper.block_store().get(&hash) {
            Ok(Some(block)) => Some(block.header.timestamp as u128),
            _ => None,
        });
    let self_proposed_too_recently = self_latest_block_timestamp_ms.is_some_and(|timestamp_ms| {
        now.saturating_sub(timestamp_ms) < heartbeat_self_propose_cooldown_ms()
    });

    // Check if we have new parents (new blocks since our last block) and whether
    // they include user deploys (to keep deploy-driven finality fast).
    let parent_update = inspect_parent_updates(&snapshot, validator_identity, &casper);
    let has_new_parents = parent_update.has_new_parents;
    let has_new_parent_with_user_deploys = parent_update.has_new_parent_with_user_deploys;
    // Treat "deploy recovery" as actively deploy-driven conditions only.
    // Keeping grace-only mode out of this hint avoids prolonged frontier-chase churn
    // once deploy pressure is gone.
    let deploy_recovery_hint = has_pending_deploys || has_new_parent_with_user_deploys;
    let deploy_recovery_max_lag = std::cmp::max(
        heartbeat_pending_deploy_max_lag(),
        heartbeat_deploy_recovery_max_lag(),
    );

    // Under active deploy-finalization recovery, allow a wider bounded chase window so
    // validators can keep up with fast parent growth without stalling on tight lag caps.
    // Outside deploy recovery, keep the tighter cap to avoid idle empty-block churn.
    let deploy_recovery_frontier_chase_cap = if deploy_recovery_hint {
        std::cmp::max(2, deploy_recovery_max_lag)
    } else {
        2
    };
    let effective_frontier_chase_cap = if deploy_recovery_hint {
        std::cmp::max(
            heartbeat_frontier_chase_max_lag(),
            deploy_recovery_frontier_chase_cap,
        )
    } else {
        heartbeat_frontier_chase_max_lag()
    };
    let stale_recovery_interval_elapsed =
        frontier_age_ms >= heartbeat_stale_recovery_min_interval_ms();
    let stale_recovery_window_open = stale_recovery_interval_elapsed || deploy_recovery_hint;

    // Proposal logic:
    // - Prioritize pending deploys, but avoid lag-amplification loops:
    //   - when this validator is already ahead of finalized and lag is above cap,
    //     temporarily stop heartbeat-driven pending-deploy proposes.
    // - Keep frontier moving on peer progress:
    //   - when new parents are observed, allow follow-up propose even before LFB turns stale;
    //   - when already ahead, guard this with the frontier chase lag cap.
    // - For stale-LFB recovery:
    //   - if we are not ahead of finalized, propose;
    //   - if we are ahead and lag is still small, allow frontier-chasing on new parents;
    //   - if frontier-chasing is throttled, allow a deterministic leader-only fallback when
    //     LFB is stale and lag is non-zero so low-lag dead zones do not stall progress;
    //   - if lag is already high, keep explicit leader recovery.
    let can_propose_pending_deploys_while_ahead = if deploy_grace_active {
        lfb_lag_blocks <= deploy_recovery_max_lag
    } else {
        lfb_lag_blocks <= heartbeat_pending_deploy_max_lag()
    };
    let pending_deploys_due =
        has_pending_deploys && (!self_recently_proposed || can_propose_pending_deploys_while_ahead);
    // Backstop: even when high lag throttles pending-deploy proposals, force a bounded
    // retry based on local self-proposal cadence so deploys cannot starve indefinitely.
    let pending_deploy_backstop_due = has_pending_deploys
        && self_recently_proposed
        && !can_propose_pending_deploys_while_ahead
        && self_latest_block_timestamp_ms
            .map(|timestamp_ms| {
                now.saturating_sub(timestamp_ms) >= heartbeat_stale_recovery_min_interval_ms()
            })
            .unwrap_or(true)
        && (!self_proposed_too_recently || deploy_grace_active);
    let can_follow_frontier_without_pending_deploys =
        deploy_recovery_hint || stale_recovery_interval_elapsed;
    // Cooldown protects idle clusters from empty-block churn, but during deploy-driven
    // recovery/finalization we should not wait out the full cooldown before advancing finality.
    let allow_cooldown_override_for_deploy_recovery =
        has_pending_deploys || has_new_parent_with_user_deploys;
    // When a peer parent with user deploys is observed, allow one frontier-follow step
    // while ahead (bounded by pending-deploy lag threshold) to unblock synchrony progress.
    let allow_frontier_follow_while_ahead_for_deploy_parent =
        has_new_parent_with_user_deploys && lfb_lag_blocks <= deploy_recovery_max_lag;
    let can_chase_frontier_while_ahead = lfb_lag_blocks <= effective_frontier_chase_cap
        && has_new_parents
        && (!self_proposed_too_recently || allow_cooldown_override_for_deploy_recovery);
    let frontier_follow_due = !has_pending_deploys
        && has_new_parents
        && can_follow_frontier_without_pending_deploys
        && (!self_recently_proposed
            || can_chase_frontier_while_ahead
            || allow_frontier_follow_while_ahead_for_deploy_parent);
    let stale_lfb_recovery_due = lfb_is_stale
        && stale_recovery_window_open
        && (!self_recently_proposed || can_chase_frontier_while_ahead || deploy_grace_active);
    let lag_recovery_leader = is_lag_recovery_leader(&snapshot, validator_identity);
    let lag_recovery_threshold = heartbeat_pending_deploy_max_lag();
    let moderate_lag_recovery_threshold = std::cmp::max(1, lag_recovery_threshold / 2);
    let stale_lfb_leader_recovery_due = lfb_is_stale
        && (frontier_is_stale || lfb_lag_blocks > moderate_lag_recovery_threshold)
        && !has_pending_deploys
        && lfb_lag_blocks > 0
        && lag_recovery_leader
        && stale_recovery_window_open
        && (!self_proposed_too_recently || deploy_grace_active)
        && !stale_lfb_recovery_due;
    let high_lag_recovery_due = !has_pending_deploys
        && lfb_lag_blocks > lag_recovery_threshold
        && lag_recovery_leader
        && stale_recovery_window_open
        && (!self_proposed_too_recently || deploy_grace_active);
    let should_propose = pending_deploys_due
        || pending_deploy_backstop_due
        || frontier_follow_due
        || stale_lfb_recovery_due
        || stale_lfb_leader_recovery_due
        || high_lag_recovery_due;

    if should_propose {
        let reason = if pending_deploy_backstop_due {
            format!(
                "pending deploy recovery backstop: lag={} exceeds cap={} while ahead; forcing propose after {}ms",
                lfb_lag_blocks,
                if deploy_grace_active {
                    deploy_recovery_max_lag
                } else {
                    heartbeat_pending_deploy_max_lag()
                },
                heartbeat_stale_recovery_min_interval_ms()
            )
        } else if has_pending_deploys && !pending_deploys_due {
            format!(
                "pending deploys exist but lag={} exceeds pending-deploy cap={} while already ahead of finalized (throttling)",
                lfb_lag_blocks,
                heartbeat_pending_deploy_max_lag()
            )
        } else if has_pending_deploys {
            "pending user deploys in storage".to_string()
        } else if frontier_follow_due {
            format!(
                "new parents observed (lag={}, self_recently_proposed={}, cooldown_active={}, cooldown_ms={}, cooldown_override_for_deploy_recovery={}, frontier_chase_cap={}, user_deploy_parent={}, deploy_grace_active={}, stale_recovery_interval_ms={}); proposing to keep frontier moving",
                lfb_lag_blocks,
                self_recently_proposed,
                self_proposed_too_recently,
                heartbeat_self_propose_cooldown_ms(),
                allow_cooldown_override_for_deploy_recovery,
                effective_frontier_chase_cap,
                has_new_parent_with_user_deploys,
                deploy_grace_active,
                heartbeat_stale_recovery_min_interval_ms()
            )
        } else if stale_lfb_leader_recovery_due {
            format!(
                "LFB is stale ({}ms) with lag={}; regular stale recovery is throttled, selected recovery leader proposing (frontier_stale={}, moderate_lag_threshold={}, deploy_grace_active={}, stale_recovery_interval_ms={})",
                time_since_lfb,
                lfb_lag_blocks,
                frontier_is_stale,
                moderate_lag_recovery_threshold,
                deploy_grace_active,
                heartbeat_stale_recovery_min_interval_ms()
            )
        } else if high_lag_recovery_due {
            format!(
                "Finality lag recovery: lag={} exceeds threshold={} and this validator is selected recovery leader (deploy_grace_active={}, stale_recovery_interval_ms={})",
                lfb_lag_blocks,
                lag_recovery_threshold,
                deploy_grace_active,
                heartbeat_stale_recovery_min_interval_ms()
            )
        } else if self_recently_proposed && has_new_parents && !can_chase_frontier_while_ahead {
            format!(
                "LFB is stale but frontier-follow is throttled (lag={}, cooldown_active={}, frontier_chase_cap={})",
                lfb_lag_blocks,
                self_proposed_too_recently,
                effective_frontier_chase_cap
            )
        } else if self_recently_proposed && !has_new_parents {
            format!(
                "LFB is stale but validator is already ahead of finalized height (cooling down stale-LFB recovery)"
            )
        } else if !standalone && !has_new_parents {
            format!(
                "LFB is stale ({}ms old, threshold: {}ms) and no new parents (recovery heartbeat)",
                time_since_lfb,
                config.max_lfb_age.as_millis()
            )
        } else {
            format!(
                "LFB is stale ({}ms old, threshold: {}ms) and new parents exist",
                time_since_lfb,
                config.max_lfb_age.as_millis()
            )
        };

        tracing::info!("Heartbeat: Proposing block - reason: {}", reason);

        // Heartbeat proposals are liveness-driven and may need empty-block capability.
        // We route them through async propose mode to enable empty blocks only for heartbeat.
        let result = trigger_propose(casper.clone(), true).await?;
        match result {
            ProposerResult::Empty => {
                tracing::debug!("Heartbeat: Propose already in progress, will retry next check");
                return Ok(HeartbeatCheckResult {
                    bug_failure: false,
                    refresh_deploy_grace_window: has_pending_deploys
                        || has_new_parent_with_user_deploys,
                });
            }
            ProposerResult::Failure(status, seq_num) => {
                tracing::warn!(
                    "Heartbeat: Propose failed with {} (seqNum {})",
                    status,
                    seq_num
                );
                // Only escalate backoff for explicit bug failures.
                // Recoverable propose races should retry on the normal heartbeat cadence.
                return Ok(HeartbeatCheckResult {
                    bug_failure: matches!(status, ProposeStatus::Failure(ProposeFailure::BugError)),
                    refresh_deploy_grace_window: has_pending_deploys
                        || has_new_parent_with_user_deploys,
                });
            }
            ProposerResult::Success(_, _) => {
                tracing::info!("Heartbeat: Successfully created block");
                return Ok(HeartbeatCheckResult {
                    bug_failure: false,
                    refresh_deploy_grace_window: has_pending_deploys
                        || has_new_parent_with_user_deploys,
                });
            }
            ProposerResult::Started(seq_num) => {
                tracing::info!("Heartbeat: Async propose started (seqNum {})", seq_num);
                return Ok(HeartbeatCheckResult {
                    bug_failure: false,
                    refresh_deploy_grace_window: has_pending_deploys
                        || has_new_parent_with_user_deploys,
                });
            }
        }
    } else {
        let reason = if !lfb_is_stale {
            if has_pending_deploys
                && self_recently_proposed
                && !can_propose_pending_deploys_while_ahead
            {
                let pending_backstop_remaining_ms = self_latest_block_timestamp_ms
                    .map(|timestamp_ms| {
                        heartbeat_stale_recovery_min_interval_ms()
                            .saturating_sub(now.saturating_sub(timestamp_ms))
                    })
                    .unwrap_or(0);
                format!(
                    "pending deploy lag throttle active: lag {} exceeds cap {} while already ahead (next backstop in {}ms)",
                    lfb_lag_blocks,
                    if deploy_grace_active {
                        deploy_recovery_max_lag
                    } else {
                        heartbeat_pending_deploy_max_lag()
                    },
                    pending_backstop_remaining_ms
                )
            } else if has_new_parents && self_recently_proposed && !can_chase_frontier_while_ahead {
                format!(
                    "frontier-follow throttled: lag {}, cooldown_active={}, cooldown_override_for_deploy_recovery={}, deploy_parent_override={}, cap {} while already ahead",
                    lfb_lag_blocks,
                    self_proposed_too_recently,
                    allow_cooldown_override_for_deploy_recovery,
                    allow_frontier_follow_while_ahead_for_deploy_parent,
                    effective_frontier_chase_cap
                )
            } else if has_new_parents && !can_follow_frontier_without_pending_deploys {
                format!(
                    "frontier-follow throttled by stale-recovery cadence: frontier_age_ms={}, min_interval_ms={}, user_deploy_parent={}, deploy_grace_active={}",
                    frontier_age_ms,
                    heartbeat_stale_recovery_min_interval_ms(),
                    has_new_parent_with_user_deploys,
                    deploy_grace_active
                )
            } else {
                format!(
                    "LFB age is {}ms (threshold: {}ms)",
                    time_since_lfb,
                    config.max_lfb_age.as_millis()
                )
            }
        } else if lfb_is_stale
            && !has_pending_deploys
            && lfb_lag_blocks > 0
            && !lag_recovery_leader
            && !stale_lfb_recovery_due
        {
            format!(
                "LFB is stale with lag {}, regular stale recovery is throttled, waiting for selected recovery leader",
                lfb_lag_blocks
            )
        } else if lfb_is_stale
            && !has_pending_deploys
            && !frontier_is_stale
            && lfb_lag_blocks <= moderate_lag_recovery_threshold
            && !stale_lfb_recovery_due
        {
            format!(
                "LFB is stale ({}ms) but frontier is active ({}ms old) and lag={} <= moderate threshold {}; skipping leader recovery",
                time_since_lfb, frontier_age_ms, lfb_lag_blocks, moderate_lag_recovery_threshold
            )
        } else if lfb_is_stale && !stale_recovery_window_open {
            format!(
                "LFB is stale but stale-recovery cadence gate is active: frontier_age_ms={}, min_interval_ms={}, user_deploy_parent={}, deploy_grace_active={}",
                frontier_age_ms,
                heartbeat_stale_recovery_min_interval_ms(),
                has_new_parent_with_user_deploys,
                deploy_grace_active
            )
        } else if !has_pending_deploys
            && lfb_lag_blocks > lag_recovery_threshold
            && !lag_recovery_leader
        {
            format!(
                "finality lag {} exceeds threshold {}, waiting for selected recovery leader",
                lfb_lag_blocks, lag_recovery_threshold
            )
        } else if self_recently_proposed && has_new_parents && !can_chase_frontier_while_ahead {
            format!(
                "frontier-follow throttled while ahead (lag {}, cooldown_active={}, cap {})",
                lfb_lag_blocks, self_proposed_too_recently, effective_frontier_chase_cap
            )
        } else if !standalone && !has_new_parents {
            "no new parents".to_string()
        } else {
            "unknown".to_string()
        };
        tracing::debug!("Heartbeat: No action needed - reason: {}", reason);
        return Ok(HeartbeatCheckResult {
            bug_failure: false,
            refresh_deploy_grace_window: has_pending_deploys || has_new_parent_with_user_deploys,
        });
    }
}

/// Check if new blocks exist since this validator's last block.
/// Returns parent update details where:
/// - Validator has no blocks yet (can propose)
/// - Validator's last block is genesis (allows breaking post-genesis deadlock)
/// - Any latest message hash diverges from what this validator observed in its last justifications
#[derive(Default)]
struct ParentUpdate {
    has_new_parents: bool,
    has_new_parent_with_user_deploys: bool,
}

fn inspect_parent_updates(
    snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
    casper: &Arc<dyn MultiParentCasper + Send + Sync>,
) -> ParentUpdate {
    let validator_id = &validator_identity.public_key.bytes;

    // Get validator's last block
    let last_block_hash = match snapshot.dag.latest_message_hash(validator_id) {
        Some(hash) => hash,
        None => {
            // Validator has no blocks yet - can propose
            return ParentUpdate {
                has_new_parents: true,
                has_new_parent_with_user_deploys: false,
            };
        }
    };

    // Check if this is genesis block (allows breaking deadlock after genesis)
    let block_meta = match snapshot.dag.lookup(&last_block_hash) {
        Ok(Some(meta)) => meta,
        _ => {
            // Can't find block metadata, allow proposal
            return ParentUpdate {
                has_new_parents: true,
                has_new_parent_with_user_deploys: false,
            };
        }
    };

    if block_meta.parents.is_empty() {
        // This is genesis - allow proposal to break post-genesis deadlock
        tracing::debug!("Heartbeat: Validator's last block is genesis, allowing proposal");
        return ParentUpdate {
            has_new_parents: true,
            has_new_parent_with_user_deploys: false,
        };
    }

    // Fast path: compare current validator latest messages against the latest messages
    // referenced in this validator's own latest block justifications.
    // If any validator advanced since then (or newly appeared), we have new parents.
    let justified_latest: std::collections::HashMap<Vec<u8>, BlockHash> = block_meta
        .justifications
        .iter()
        .map(|j| (j.validator.to_vec(), j.latest_block_hash.clone()))
        .collect();

    let mut update = ParentUpdate::default();

    for (validator, current_hash) in snapshot.dag.latest_message_hashes().iter() {
        let known_hash_opt = if *validator == *validator_id {
            Some(&last_block_hash)
        } else {
            justified_latest.get(validator.as_ref())
        };

        if known_hash_opt != Some(current_hash) {
            update.has_new_parents = true;
            if !update.has_new_parent_with_user_deploys {
                if let Ok(Some(block)) = casper.block_store().get(current_hash) {
                    let has_user_deploys = block
                        .body
                        .deploys
                        .iter()
                        .any(|processed| !is_system_deploy_id(&processed.deploy.sig));
                    if has_user_deploys {
                        update.has_new_parent_with_user_deploys = true;
                    }
                }
            }
            if update.has_new_parent_with_user_deploys {
                break;
            }
        }
    }

    update
}

fn is_lag_recovery_leader(
    snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
) -> bool {
    let mut active_validators = snapshot.on_chain_state.active_validators.clone();
    if active_validators.is_empty() {
        return true;
    }

    // Deterministic ordering across validators.
    active_validators.sort();

    // Rotate leader by next block number so recovery proposes are spread across validators.
    let next_block_number = snapshot.max_block_num.saturating_add(1);
    let leader_index = (next_block_number as usize) % active_validators.len();
    let leader = &active_validators[leader_index];
    *leader == validator_identity.public_key.bytes
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
            false,
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
            false,
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
            false,
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

        #[tokio::test]
        async fn do_heartbeat_check_triggers_propose_with_pending_deploys() {
            let validator = create_test_validator_identity();
            let validator_id = validator.public_key.bytes.clone();

            // Snapshot with bonded validator
            let mut snapshot =
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::create_empty_snapshot();
            snapshot
                .on_chain_state
                .active_validators
                .push(validator_id.into());

            // Fresh LFB (100ms old)
            let lfb = create_lfb_with_age(100);

            // Casper with 1 pending deploy in storage
            let casper: Arc<dyn MultiParentCasper + Send + Sync> = Arc::new(
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::new_with_pending_deploys(
                    snapshot, lfb, 1,
                ),
            );

            // Create counting propose function
            let (propose_count, propose_func) = create_counting_propose_function();

            // Config with long max_lfb_age so LFB is NOT stale
            let config = HeartbeatConf {
                enabled: true,
                check_interval: Duration::from_secs(1),
                max_lfb_age: Duration::from_secs(10),
            };

            // Call do_heartbeat_check directly (standalone=false for multi-node test)
            let result =
                do_heartbeat_check(casper, &*propose_func, &validator, &config, false, false).await;

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

            // Call do_heartbeat_check directly (standalone=false for multi-node test)
            let result =
                do_heartbeat_check(casper, &*propose_func, &validator, &config, false, false).await;

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

            // Call do_heartbeat_check directly (standalone=false for multi-node test)
            let result =
                do_heartbeat_check(casper, &*propose_func, &validator, &config, false, false).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                0,
                "Should NOT trigger propose when validator is not bonded"
            );
        }

        #[tokio::test]
        async fn do_heartbeat_check_proposes_when_lfb_fresh_and_validator_has_no_latest_block() {
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

            // Call do_heartbeat_check directly (standalone=false for multi-node test)
            let result =
                do_heartbeat_check(casper, &*propose_func, &validator, &config, false, false).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                1,
                "Should trigger propose when validator has no latest block (frontier-follow path), even if LFB is fresh"
            );
        }

        #[tokio::test]
        async fn do_heartbeat_check_proposes_when_storage_has_deploys_but_deploys_in_scope_empty() {
            // Reproduces bug: deploys in storage but deploysInScope empty (aged out).
            // Current: checks deploysInScope -> empty -> no propose (BUG)
            // Fixed: checks storage -> has deploy -> propose

            let validator = create_test_validator_identity();
            let validator_id = validator.public_key.bytes.clone();

            // Snapshot with EMPTY deploys_in_scope but validator is bonded
            let mut snapshot =
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::create_empty_snapshot();
            snapshot
                .on_chain_state
                .active_validators
                .push(validator_id.into());

            // Fresh LFB so LFB is NOT stale
            let lfb = create_lfb_with_age(100);

            // Casper with pending deploy in storage (but deploys_in_scope is empty)
            let casper: Arc<dyn MultiParentCasper + Send + Sync> = Arc::new(
                casper::rust::casper::test_helpers::TestCasperWithSnapshot::new_with_pending_deploys(
                    snapshot, lfb, 1,
                ),
            );

            let (propose_count, propose_func) = create_counting_propose_function();

            let config = HeartbeatConf {
                enabled: true,
                check_interval: Duration::from_secs(1),
                max_lfb_age: Duration::from_secs(10),
            };

            // Call do_heartbeat_check directly (standalone=false for multi-node test)
            let result =
                do_heartbeat_check(casper, &*propose_func, &validator, &config, false, false).await;

            assert!(result.is_ok(), "do_heartbeat_check should succeed");
            // FAILS before fix: heartbeat checks deploys_in_scope (empty) instead of storage
            assert_eq!(
                propose_count.load(Ordering::SeqCst),
                1,
                "Should propose when storage has pending deploys, even if deploys_in_scope is empty"
            );
        }
    }
}
