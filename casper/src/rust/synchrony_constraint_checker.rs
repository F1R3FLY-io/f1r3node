// See casper/src/main/scala/coop/rchain/casper/SynchronyConstraintChecker.scala

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use lazy_static::lazy_static;

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use models::rust::{block_metadata::BlockMetadata, validator::Validator};

use crate::rust::util::proto_util;

use super::{
    blocks::proposer::propose_result::CheckProposeConstraintsResult, casper::CasperSnapshot,
    errors::CasperError, util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
};

const SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS_ENV: &str = "F1R3_SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS";
const SYNCHRONY_RECOVERY_COOLDOWN_SECONDS_ENV: &str = "F1R3_SYNCHRONY_RECOVERY_COOLDOWN_SECONDS";
const SYNCHRONY_RECOVERY_MAX_BYPASSES_ENV: &str = "F1R3_SYNCHRONY_RECOVERY_MAX_BYPASSES";
const SYNCHRONY_CONSTRAINT_THRESHOLD_ENV: &str = "F1R3_SYNCHRONY_CONSTRAINT_THRESHOLD";
const DEFAULT_SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS: u64 = 8;
const DEFAULT_SYNCHRONY_RECOVERY_COOLDOWN_SECONDS: u64 = 20;
const DEFAULT_SYNCHRONY_RECOVERY_MAX_BYPASSES: u32 = 2;

static SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS: OnceLock<u64> = OnceLock::new();
static SYNCHRONY_RECOVERY_COOLDOWN_SECONDS: OnceLock<u64> = OnceLock::new();
static SYNCHRONY_RECOVERY_MAX_BYPASSES: OnceLock<u32> = OnceLock::new();
static SYNCHRONY_CONSTRAINT_THRESHOLD_OVERRIDE: OnceLock<Option<f64>> = OnceLock::new();

fn read_i64_from_env(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(default)
}

fn read_non_negative_u64_from_env(name: &str, default: u64) -> u64 {
    let value = read_i64_from_env(name, default as i64);
    if value <= 0 {
        0
    } else {
        value as u64
    }
}

fn synchrony_recovery_stall_window_seconds() -> u64 {
    *SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS.get_or_init(|| {
        read_non_negative_u64_from_env(
            SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS_ENV,
            DEFAULT_SYNCHRONY_RECOVERY_STALL_WINDOW_SECONDS,
        )
    })
}

fn synchrony_recovery_cooldown_seconds() -> u64 {
    *SYNCHRONY_RECOVERY_COOLDOWN_SECONDS.get_or_init(|| {
        read_non_negative_u64_from_env(
            SYNCHRONY_RECOVERY_COOLDOWN_SECONDS_ENV,
            DEFAULT_SYNCHRONY_RECOVERY_COOLDOWN_SECONDS,
        )
    })
}

fn synchrony_recovery_max_bypasses() -> u32 {
    *SYNCHRONY_RECOVERY_MAX_BYPASSES.get_or_init(|| {
        let value = std::env::var(SYNCHRONY_RECOVERY_MAX_BYPASSES_ENV)
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(DEFAULT_SYNCHRONY_RECOVERY_MAX_BYPASSES as i32);

        if value <= 0 {
            0
        } else {
            value as u32
        }
    })
}

fn synchrony_constraint_threshold_override() -> Option<f64> {
    *SYNCHRONY_CONSTRAINT_THRESHOLD_OVERRIDE.get_or_init(|| {
        std::env::var(SYNCHRONY_CONSTRAINT_THRESHOLD_ENV)
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .and_then(|value| {
                if value.is_finite() {
                    Some(value.clamp(0.0, 1.0))
                } else {
                    None
                }
            })
    })
}

#[derive(Debug)]
struct SynchronyRecoveryState {
    last_known_hash: Vec<u8>,
    first_failure_at: Option<Instant>,
    consecutive_failures: u32,
    bypass_count: u32,
    last_bypass_at: Option<Instant>,
}

impl SynchronyRecoveryState {
    fn reset_for_hash(&mut self, last_hash: &[u8], now: Instant) {
        self.last_known_hash = last_hash.to_vec();
        self.first_failure_at = Some(now);
        self.consecutive_failures = 1;
        self.bypass_count = 0;
        self.last_bypass_at = None;
    }

    fn mark_success(&mut self) {
        self.consecutive_failures = 0;
        self.first_failure_at = None;
        self.bypass_count = 0;
        self.last_bypass_at = None;
    }

    fn should_bypass(&mut self, now: Instant) -> bool {
        let first_failure_at = self.first_failure_at.unwrap_or(now);
        self.first_failure_at = Some(first_failure_at);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);

        let stalled_long_enough = now.duration_since(first_failure_at)
            >= Duration::from_secs(synchrony_recovery_stall_window_seconds());

        let in_cooldown = self
            .last_bypass_at
            .is_some_and(|last| now.duration_since(last) < Duration::from_secs(synchrony_recovery_cooldown_seconds()));
        if !stalled_long_enough || in_cooldown {
            return false;
        }

        if self.bypass_count >= synchrony_recovery_max_bypasses() {
            return false;
        }

        self.consecutive_failures = 0;
        self.first_failure_at = None;
        self.bypass_count = self.bypass_count.saturating_add(1);
        self.last_bypass_at = Some(now);
        true
    }
}

lazy_static! {
    static ref SYNCHRONY_RECOVERY_STATES: Mutex<HashMap<Validator, SynchronyRecoveryState>> =
        Mutex::new(HashMap::new());
}

fn update_recovery_state_on_success(validator: &Validator) {
    if let Ok(mut states) = SYNCHRONY_RECOVERY_STATES.lock() {
        if let Some(state) = states.get_mut(validator) {
            state.mark_success();
        }
    }
}

fn should_bypass_synchrony_constraint(
    validator: &Validator,
    last_proposed_block_hash: &[u8],
) -> bool {
    let now = Instant::now();

    let mut states = match SYNCHRONY_RECOVERY_STATES.lock() {
        Ok(states) => states,
        Err(_) => return false,
    };

    match states.get_mut(validator) {
        Some(state) if state.last_known_hash == last_proposed_block_hash => {
            state.should_bypass(now)
        }
        Some(state) => {
            state.reset_for_hash(last_proposed_block_hash, now);
            false
        }
        None => {
            states.insert(
                validator.clone(),
                SynchronyRecoveryState {
                    last_known_hash: last_proposed_block_hash.to_vec(),
                    first_failure_at: Some(now),
                    consecutive_failures: 1,
                    bypass_count: 0,
                    last_bypass_at: None,
                },
            );
            false
        }
    }
}

pub async fn check(
    snapshot: &CasperSnapshot,
    runtime_manager: &RuntimeManager,
    block_store: &KeyValueBlockStore,
    validator_identity: &ValidatorIdentity,
) -> Result<CheckProposeConstraintsResult, CasperError> {
    let validator = validator_identity.public_key.bytes.clone();
    let main_parent_opt = snapshot.parents.first();
    let mut synchrony_constraint_threshold = snapshot
        .on_chain_state
        .shard_conf
        .synchrony_constraint_threshold as f64;
    if let Some(override_threshold) = synchrony_constraint_threshold_override() {
        synchrony_constraint_threshold = override_threshold;
    }

    match snapshot.dag.latest_message_hash(&validator) {
        Some(last_proposed_block_hash) => {
            let last_proposed_block_meta = snapshot.dag.lookup_unsafe(&last_proposed_block_hash)?;

            // If validator's latest block is genesis, it's not proposed any block yet and hence allowed to propose once.
            let latest_block_is_genesis = last_proposed_block_meta.block_number == 0;
            if latest_block_is_genesis {
                update_recovery_state_on_success(&validator);
                Ok(CheckProposeConstraintsResult::success())
            } else {
                if synchrony_constraint_threshold <= 0.0 {
                    update_recovery_state_on_success(&validator);
                    return Ok(CheckProposeConstraintsResult::success());
                }

                let main_parent = main_parent_opt.ok_or(CasperError::Other(
                    "Synchrony constraint checker: Parent blocks not found".to_string(),
                ))?;

                let main_parent_meta = snapshot.dag.lookup_unsafe(&main_parent.block_hash)?;

                // Loading the whole block is only needed to get post-state hash
                let main_parent_block = block_store.get_unsafe(&main_parent_meta.block_hash);
                let main_parent_state_hash = proto_util::post_state_hash(&main_parent_block);

                // Get bonds map from PoS
                // NOTE: It would be useful to have active validators cached in the block in the same way as bonds.
                let active_validators = runtime_manager
                    .get_active_validators(&main_parent_state_hash)
                    .await?;

                // Validators weight map filtered by active validators only.
                let validator_weight_map: HashMap<Validator, i64> = main_parent_meta
                    .weight_map
                    .into_iter()
                    .filter(|(validator, _)| active_validators.contains(validator))
                    .collect();

                // Guaranteed to be present since last proposed block was present
                let seen_senders =
                    calculate_seen_senders_since(last_proposed_block_meta, snapshot.dag.clone());

                let seen_senders_weight: i64 = seen_senders
                    .iter()
                    .map(|validator| validator_weight_map.get(validator).unwrap_or(&0))
                    .sum();

                // This method can be called on readonly node or not active validator.
                // So map validator -> stake might not have key associated with the node,
                // that's why we need `getOrElse`
                let validator_own_stake = validator_weight_map.get(&validator).unwrap_or(&0);
                let other_validators_weight =
                    validator_weight_map.values().sum::<i64>() - validator_own_stake;

                // If there is no other active validators, do not put any constraint (value = 1)
                // Use f64 for precision matching Scala's Double type
                let synchrony_constraint_value = if other_validators_weight == 0 {
                    1.0
                } else {
                    seen_senders_weight as f64 / other_validators_weight as f64
                };

                let threshold_f64 = synchrony_constraint_threshold as f64;

                tracing::warn!(
                    "Seen {} senders with weight {} out of total {} ({:.2} out of {:.2} needed)",
                    seen_senders.len(),
                    seen_senders_weight,
                    other_validators_weight,
                    synchrony_constraint_value,
                    threshold_f64
                );

                if synchrony_constraint_value >= synchrony_constraint_threshold {
                    update_recovery_state_on_success(&validator);
                    Ok(CheckProposeConstraintsResult::success())
                } else {
                    let bypass = should_bypass_synchrony_constraint(&validator, last_proposed_block_hash.as_ref());

                    if bypass {
                        tracing::warn!(
                            "Synchrony constraint bypassed after sustained stall (validator {}, seen {} senders with ratio {:.2} < {:.2})",
                            hex::encode(&validator[..8]),
                            seen_senders.len(),
                            synchrony_constraint_value,
                            threshold_f64
                        );
                        update_recovery_state_on_success(&validator);
                        Ok(CheckProposeConstraintsResult::success())
                    } else {
                        Ok(CheckProposeConstraintsResult::not_enough_new_block())
                    }
                }
            }
        }
        None => Err(CasperError::Other(
            "Synchrony constraint checker: Validator does not have a latest message".to_string(),
        )),
    }
}

fn calculate_seen_senders_since(
    last_proposed: BlockMetadata,
    dag: KeyValueDagRepresentation,
) -> HashSet<Validator> {
    let latest_messages = dag.latest_message_hashes();

    // Match Scala implementation exactly: only check validators that are in justifications
    last_proposed
        .justifications
        .iter()
        .filter_map(|justification| {
            let validator = &justification.validator;
            let justification_hash = &justification.latest_block_hash;

            // Skip the sender itself
            if validator == &last_proposed.sender {
                return None;
            }

            // Get the latest block hash for this validator from the DAG
            // In Scala: latestMessages(validator) - this throws if validator is not in map
            // In Rust: we use get() which returns Option, but since justifications come from
            // latest_messages, the validator should always be present
            let latest_block_hash = match latest_messages.get(validator) {
                Some(hash) => hash,
                None => {
                    tracing::warn!(
                        "Validator {} not found in latest_messages, skipping",
                        hex::encode(&validator[..8])
                    );
                    return None;
                }
            };

            let hash_changed = *latest_block_hash != *justification_hash;

            // If the latest block hash is different from what was in justifications,
            // we've seen a new block from this validator since the last proposed block
            // Scala: latestMessages(validator) != latestBlockHash
            if hash_changed {
                Some(validator.clone())
            } else {
                None
            }
        })
        .collect()
}
