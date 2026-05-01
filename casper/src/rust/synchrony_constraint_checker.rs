// See casper/src/main/scala/coop/rchain/casper/SynchronyConstraintChecker.scala

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

use lazy_static::lazy_static;

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use models::rust::{block_metadata::BlockMetadata, validator::Validator};

use crate::rust::util::proto_util;

use super::{
    blocks::proposer::propose_result::CheckProposeConstraintsResult,
    casper::{CasperShardConf, CasperSnapshot},
    errors::CasperError,
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
};

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

    fn should_bypass(&mut self, now: Instant, conf: &CasperShardConf) -> bool {
        let first_failure_at = self.first_failure_at.unwrap_or(now);
        self.first_failure_at = Some(first_failure_at);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);

        let stalled_long_enough =
            now.duration_since(first_failure_at) >= conf.synchrony_recovery_stall_window;

        let in_cooldown = self
            .last_bypass_at
            .is_some_and(|last| now.duration_since(last) < conf.synchrony_recovery_cooldown);
        if !stalled_long_enough || in_cooldown {
            return false;
        }

        let max_bypasses = conf.synchrony_recovery_max_bypasses;
        if max_bypasses == 0 {
            return false;
        }

        if self.bypass_count >= max_bypasses {
            let can_recycle_budget = self.last_bypass_at.is_some_and(|last| {
                now.duration_since(last) >= conf.synchrony_recovery_stall_window
            });
            if can_recycle_budget {
                self.bypass_count = 0;
            } else {
                return false;
            }
        }

        self.consecutive_failures = 0;
        self.first_failure_at = None;
        self.bypass_count = self.bypass_count.saturating_add(1);
        self.last_bypass_at = Some(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn test_conf() -> CasperShardConf {
        let mut conf = CasperShardConf::new();
        conf.synchrony_recovery_stall_window = Duration::from_secs(60);
        conf.synchrony_recovery_cooldown = Duration::from_secs(20);
        conf.synchrony_recovery_max_bypasses = 2;
        conf.synchrony_finalized_baseline_enabled = true;
        conf.synchrony_finalized_baseline_max_distance = 2048;
        conf
    }

    #[test]
    fn should_not_bypass_before_stall_window() {
        let conf = test_conf();
        let now = Instant::now();
        let mut state = SynchronyRecoveryState {
            last_known_hash: vec![1],
            first_failure_at: Some(now),
            consecutive_failures: 0,
            bypass_count: 0,
            last_bypass_at: None,
        };

        assert!(
            !state.should_bypass(now, &conf),
            "should not bypass before stall window elapsed"
        );
    }

    #[test]
    fn should_recycle_bypass_budget_after_stall_window() {
        let conf = test_conf();
        let max_bypasses = conf.synchrony_recovery_max_bypasses;
        if max_bypasses == 0 {
            return;
        }

        let now = Instant::now();
        let stall_window = conf.synchrony_recovery_stall_window.as_secs();
        let cooldown = conf.synchrony_recovery_cooldown.as_secs();
        let elapsed =
            Duration::from_secs(stall_window.max(cooldown)).saturating_add(Duration::from_secs(1));

        let mut state = SynchronyRecoveryState {
            last_known_hash: vec![1],
            first_failure_at: Some(now - elapsed),
            consecutive_failures: 0,
            bypass_count: max_bypasses,
            last_bypass_at: Some(now - elapsed),
        };

        assert!(
            state.should_bypass(now, &conf),
            "should bypass again after budget recycle window elapsed"
        );
        assert_eq!(
            state.bypass_count, 1,
            "bypass budget should be recycled then incremented"
        );
    }

    #[test]
    fn fallback_sender_detection_counts_equal_height_different_hash() {
        let latest_hash = [2u8; 32];
        let proposed_hash = [1u8; 32];
        assert!(
            super::should_count_fallback_sender(&latest_hash, 10, &proposed_hash, 10),
            "equal-height different hash should count as sender progress"
        );
    }

    #[test]
    fn fallback_sender_detection_ignores_equal_height_same_hash() {
        let hash = [1u8; 32];
        assert!(
            !super::should_count_fallback_sender(&hash, 10, &hash, 10),
            "equal-height same hash should not count as sender progress"
        );
    }

    #[test]
    fn compute_synchrony_value_respects_weight_ratio() {
        let mut validator_weight_map = HashMap::new();
        let validator1 = Validator::from(vec![1u8]);
        let validator2 = Validator::from(vec![2u8]);
        validator_weight_map.insert(validator1.clone(), 1000);
        validator_weight_map.insert(validator2, 1000);

        let mut seen_senders = HashSet::new();
        let _ = seen_senders.insert(validator1);

        let (seen_weight, ratio) =
            super::compute_synchrony_value(&seen_senders, &validator_weight_map, 2000);
        assert_eq!(seen_weight, 1000);
        assert!(
            (ratio - 0.5).abs() < f64::EPSILON,
            "single sender should be 50% of total other validators weight"
        );
    }

    #[test]
    fn compute_synchrony_value_with_all_senders_reaches_full_ratio() {
        let mut validator_weight_map = HashMap::new();
        let validator1 = Validator::from(vec![1u8]);
        let validator2 = Validator::from(vec![2u8]);
        validator_weight_map.insert(validator1.clone(), 1000);
        validator_weight_map.insert(validator2.clone(), 1000);

        let mut seen_senders = HashSet::new();
        let _ = seen_senders.insert(validator1);
        let _ = seen_senders.insert(validator2);

        let (seen_weight, ratio) =
            super::compute_synchrony_value(&seen_senders, &validator_weight_map, 2000);
        assert_eq!(seen_weight, 2000);
        assert!(
            (ratio - 1.0).abs() < f64::EPSILON,
            "all senders should be 100% of total other validators weight"
        );
    }

    #[test]
    fn finalized_baseline_is_used_when_proposer_is_near_finalized_height() {
        assert!(
            super::can_use_finalized_baseline(12, 10, 2, 2048),
            "proposer at threshold distance from finalized should use finalized fallback"
        );
    }

    #[test]
    fn finalized_baseline_is_not_used_when_proposer_is_far_ahead() {
        assert!(
            !super::can_use_finalized_baseline(30, 10, 2, 2048),
            "proposer far ahead of finalized should not use finalized fallback"
        );
    }

    #[test]
    fn finalized_baseline_distance_is_capped_even_if_height_threshold_is_large() {
        assert!(
            !super::can_use_finalized_baseline(20, 10, 1000, 8),
            "finalized fallback must stay tightly bounded and not inherit large height thresholds"
        );
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

fn compute_synchrony_value(
    seen_senders: &HashSet<Validator>,
    validator_weight_map: &HashMap<Validator, i64>,
    other_validators_weight: i64,
) -> (i64, f64) {
    let seen_senders_weight: i64 = seen_senders
        .iter()
        .map(|validator| validator_weight_map.get(validator).unwrap_or(&0))
        .sum();

    let synchrony_constraint_value = if other_validators_weight == 0 {
        1.0
    } else {
        seen_senders_weight as f64 / other_validators_weight as f64
    };

    (seen_senders_weight, synchrony_constraint_value)
}

fn can_use_finalized_baseline(
    last_proposed_block_number: i64,
    last_finalized_block_number: i64,
    height_constraint_threshold: i64,
    max_distance: u64,
) -> bool {
    let allowed_ahead = height_constraint_threshold.max(0).min(max_distance as i64);
    let proposer_ahead_of_finalized = last_proposed_block_number - last_finalized_block_number;
    proposer_ahead_of_finalized <= allowed_ahead
}

fn should_bypass_synchrony_constraint(
    validator: &Validator,
    last_proposed_block_hash: &[u8],
    conf: &CasperShardConf,
) -> bool {
    let now = Instant::now();

    let mut states = match SYNCHRONY_RECOVERY_STATES.lock() {
        Ok(states) => states,
        Err(_) => return false,
    };

    match states.get_mut(validator) {
        Some(state) if state.last_known_hash == last_proposed_block_hash => {
            state.should_bypass(now, conf)
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
    let shard_conf = &snapshot.on_chain_state.shard_conf;
    let synchrony_constraint_threshold = shard_conf.synchrony_constraint_threshold as f64;

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
                let seen_senders = calculate_seen_senders_since(
                    last_proposed_block_meta.clone(),
                    snapshot.dag.clone(),
                    &validator,
                );

                // This method can be called on readonly node or not active validator.
                // So map validator -> stake might not have key associated with the node,
                // that's why we need `getOrElse`
                let validator_own_stake = validator_weight_map.get(&validator).unwrap_or(&0);
                let other_validators_weight =
                    validator_weight_map.values().sum::<i64>() - validator_own_stake;

                let (seen_senders_weight, synchrony_constraint_value) = compute_synchrony_value(
                    &seen_senders,
                    &validator_weight_map,
                    other_validators_weight,
                );

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
                    // Keep the same synchrony threshold, but evaluate it against finalized baseline
                    // before any explicit bypass logic when proposer is still near finalized height.
                    // This avoids circular waits where all validators block on each other's latest
                    // block while preventing over-permissive proposing far ahead of finalization.
                    let last_finalized_block_hash = snapshot.dag.last_finalized_block();
                    let last_finalized_block_meta =
                        snapshot.dag.lookup_unsafe(&last_finalized_block_hash)?;
                    let can_use_finalized = can_use_finalized_baseline(
                        last_proposed_block_meta.block_number,
                        last_finalized_block_meta.block_number,
                        shard_conf.height_constraint_threshold as i64,
                        shard_conf.synchrony_finalized_baseline_max_distance,
                    );

                    if can_use_finalized && shard_conf.synchrony_finalized_baseline_enabled {
                        let finalized_seen_senders = calculate_seen_senders_since(
                            last_finalized_block_meta,
                            snapshot.dag.clone(),
                            &validator,
                        );
                        let (finalized_seen_senders_weight, finalized_synchrony_constraint_value) =
                            compute_synchrony_value(
                                &finalized_seen_senders,
                                &validator_weight_map,
                                other_validators_weight,
                            );

                        tracing::warn!(
                            "Finalized-baseline synchrony: seen {} senders with weight {} out of total {} ({:.2} out of {:.2} needed)",
                            finalized_seen_senders.len(),
                            finalized_seen_senders_weight,
                            other_validators_weight,
                            finalized_synchrony_constraint_value,
                            threshold_f64
                        );

                        if finalized_synchrony_constraint_value >= synchrony_constraint_threshold {
                            tracing::warn!(
                                "Synchrony constraint satisfied via finalized-block baseline (primary {:.2} < {:.2}, finalized {:.2} >= {:.2})",
                                synchrony_constraint_value,
                                threshold_f64,
                                finalized_synchrony_constraint_value,
                                threshold_f64
                            );
                            update_recovery_state_on_success(&validator);
                            return Ok(CheckProposeConstraintsResult::success());
                        }
                    } else if can_use_finalized {
                        tracing::debug!("Finalized-baseline synchrony fallback disabled in config");
                    } else {
                        tracing::warn!(
                            "Skipping finalized-baseline synchrony fallback: validator is too far ahead of finalized (proposed #{}, finalized #{})",
                            last_proposed_block_meta.block_number,
                            last_finalized_block_meta.block_number
                        );
                    }

                    let bypass = should_bypass_synchrony_constraint(
                        &validator,
                        last_proposed_block_hash.as_ref(),
                        shard_conf,
                    );

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
    excluded_validator: &Validator,
) -> HashSet<Validator> {
    let latest_messages = dag.latest_message_hashes();
    let mut seen_senders: HashSet<Validator> = HashSet::new();

    // Primary path: compare latest messages against validators referenced in the
    // last proposed block's justifications.
    for justification in &last_proposed.justifications {
        let validator = &justification.validator;
        let justification_hash = &justification.latest_block_hash;

        // Skip the sender itself
        if validator == excluded_validator {
            continue;
        }

        match latest_messages.get(validator) {
            Some(latest_block_hash) if *latest_block_hash != *justification_hash => {
                let _ = seen_senders.insert(validator.clone());
            }
            Some(_) => {}
            None => {
                tracing::warn!(
                    "Validator {} not found in latest_messages, skipping",
                    hex::encode(&validator[..8])
                );
            }
        }
    }

    // Fallback path: when a validator has no entry in justifications (common for
    // sparse/legacy justifications), still count it as seen if its latest message
    // advanced beyond the last proposed block height.
    for (validator, latest_block_hash) in latest_messages.iter() {
        if validator == excluded_validator {
            continue;
        }

        let present_in_justifications = last_proposed
            .justifications
            .iter()
            .any(|j| &j.validator == validator);
        if present_in_justifications {
            continue;
        }

        if let Ok(latest_block_number) = dag.block_number_unsafe(latest_block_hash) {
            if should_count_fallback_sender(
                latest_block_hash.as_ref(),
                latest_block_number,
                last_proposed.block_hash.as_ref(),
                last_proposed.block_number,
            ) {
                let _ = seen_senders.insert(validator.clone());
            }
        }
    }

    seen_senders
}

fn should_count_fallback_sender(
    latest_block_hash: &[u8],
    latest_block_number: i64,
    last_proposed_hash: &[u8],
    last_proposed_block_number: i64,
) -> bool {
    let advanced_height = latest_block_number > last_proposed_block_number;
    let same_or_higher_height_new_hash = latest_block_number >= last_proposed_block_number
        && latest_block_hash != last_proposed_hash;
    advanced_height || same_or_higher_height_new_hash
}
