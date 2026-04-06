// See casper/src/main/scala/coop/rchain/casper/blocks/proposer/BlockCreator.scala

use prost::bytes::Bytes;
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::{collections::HashSet, time::SystemTime};
use tracing;

use block_storage::rust::{
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use crypto::rust::{private_key::PrivateKey, signatures::signed::Signed};
use models::rust::casper::pretty_printer;
use models::rust::casper::protocol::casper_message::{
    BlockMessage, Body, Bond, DeployData, F1r3flyState, Header, Justification, ProcessedDeploy,
    ProcessedSystemDeploy, RejectedDeploy,
};

use rholang::rust::interpreter::system_processes::BlockData;

use crate::rust::util::construct_deploy;
use crate::rust::util::rholang::{
    costacc::{close_block_deploy::CloseBlockDeploy, slash_deploy::SlashDeploy},
    interpreter_util,
    system_deploy_enum::SystemDeployEnum,
    system_deploy_user_error::SystemDeployPlatformFailure,
    system_deploy_util,
};
use crate::rust::{
    blocks::proposer::propose_result::BlockCreatorResult,
    casper::CasperSnapshot,
    errors::CasperError,
    util::{proto_util, rholang::runtime_manager::RuntimeManager},
    validator_identity::ValidatorIdentity,
};

/*
 * Overview of createBlock
 *
 *  1. Rank each of the block cs's latest messages (blocks) via the LMD GHOST estimator.
 *  2. Let each latest message have a score of 2^(-i) where i is the index of that latest message in the ranking.
 *     Take a subset S of the latest messages such that the sum of scores is the greatest and
 *     none of the blocks in S conflicts with each other. S will become the parents of the
 *     about-to-be-created block.
 *  3. Extract all valid deploys that aren't already in all ancestors of S (the parents).
 *  4. Create a new block that contains the deploys from the previous step.
 */
struct PreparedUserDeploys {
    deploys: HashSet<Signed<DeployData>>,
    effective_cap: usize,
    cap_hit: bool,
}

fn deploy_selection_reserve_tail_enabled() -> bool {
    true
}

async fn prepare_user_deploys(
    casper_snapshot: &CasperSnapshot,
    block_number: i64,
    current_time_millis: i64,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
) -> Result<PreparedUserDeploys, CasperError> {
    let mut deploy_storage_guard = deploy_storage
        .lock()
        .map_err(|e| CasperError::LockError(e.to_string()))?;

    // Read all unfinalized deploys from storage
    let unfinalized: HashSet<Signed<DeployData>> = deploy_storage_guard.read_all()?;

    let earliest_block_number =
        block_number - casper_snapshot.on_chain_state.shard_conf.deploy_lifespan;

    // Categorize deploys for logging
    let future_deploys: Vec<_> = unfinalized
        .iter()
        .filter(|d| !not_future_deploy(block_number, &d.data))
        .collect();
    let block_expired_deploys: Vec<_> = unfinalized
        .iter()
        .filter(|d| !not_expired_deploy(earliest_block_number, &d.data))
        .collect();
    let time_expired_deploys: Vec<_> = unfinalized
        .iter()
        .filter(|d| d.data.is_expired_at(current_time_millis))
        .collect();

    // Filter valid deploys (not expired by block, not expired by time, and not future)
    let valid: HashSet<Signed<DeployData>> = unfinalized
        .iter()
        .filter(|deploy| {
            not_future_deploy(block_number, &deploy.data)
                && not_expired_deploy(earliest_block_number, &deploy.data)
                && !deploy.data.is_expired_at(current_time_millis)
        })
        .cloned()
        .collect();

    let valid_count = valid.len();

    // Remove deploys that are already in scope to prevent resending
    let already_in_scope: Vec<Signed<DeployData>> = valid
        .iter()
        .filter(|deploy| casper_snapshot.deploys_in_scope.contains(&deploy.sig))
        .map(|deploy| (*deploy).clone())
        .collect();
    let valid_unique: HashSet<Signed<DeployData>> = valid
        .into_iter()
        .filter(|deploy| !casper_snapshot.deploys_in_scope.contains(&deploy.sig))
        .collect();

    let already_in_scope_count = already_in_scope.len();

    // Log deploy selection details when there are any deploys in the pool
    if !unfinalized.is_empty() || !casper_snapshot.deploys_in_scope.is_empty() {
        tracing::info!(
            "Deploy selection for block #{}: pool={}, future={} (validAfterBlockNumber >= {}), \
             blockExpired={} (validAfterBlockNumber <= {}), timeExpired={} (expirationTimestamp <= {}), \
             valid={}, alreadyInScope={}, selected={}",
            block_number,
            unfinalized.len(),
            future_deploys.len(),
            block_number,
            block_expired_deploys.len(),
            earliest_block_number,
            time_expired_deploys.len(),
            current_time_millis,
            valid_count,
            already_in_scope_count,
            valid_unique.len()
        );
    }

    // Log details for filtered-out deploys (to help debug why deploys aren't included)
    for d in &future_deploys {
        tracing::warn!(
            "Deploy {}... FILTERED (future): validAfterBlockNumber={} >= currentBlock={}",
            hex::encode(&d.sig[..std::cmp::min(8, d.sig.len())]),
            d.data.valid_after_block_number,
            block_number
        );
    }
    for d in &block_expired_deploys {
        tracing::warn!(
            "Deploy {}... FILTERED (block-expired): validAfterBlockNumber={} <= earliestBlock={}",
            hex::encode(&d.sig[..std::cmp::min(8, d.sig.len())]),
            d.data.valid_after_block_number,
            earliest_block_number
        );
    }
    for d in &time_expired_deploys {
        tracing::warn!(
            "Deploy {}... FILTERED (time-expired): expirationTimestamp={:?} <= currentTime={}",
            hex::encode(&d.sig[..std::cmp::min(8, d.sig.len())]),
            d.data.expiration_timestamp,
            current_time_millis
        );
    }
    for d in &already_in_scope {
        tracing::warn!(
            "Deploy {}... FILTERED (already in scope): deploy already exists in DAG within lifespan window",
            hex::encode(&d.sig[..std::cmp::min(8, d.sig.len())])
        );
    }

    // Remove all expired deploys from storage to prevent them from triggering future proposals
    // Combine block-expired and time-expired, avoiding duplicates
    let all_expired: HashSet<&Signed<DeployData>> = block_expired_deploys
        .iter()
        .chain(time_expired_deploys.iter())
        .cloned()
        .collect();
    if !all_expired.is_empty() {
        tracing::info!(
            "Removing {} expired deploy(s) from storage",
            all_expired.len()
        );
        let expired_list: Vec<Signed<DeployData>> = all_expired.into_iter().cloned().collect();
        deploy_storage_guard.remove(expired_list)?;
    }

    let pending_unique_count = valid_unique.len();
    let max_deploys = casper_snapshot.on_chain_state.shard_conf.max_user_deploys_per_block as usize;
    if should_bypass_adaptive_cap_for_small_batch(
        pending_unique_count,
        max_deploys,
        adaptive_small_batch_bypass_threshold(),
    ) {
        return Ok(PreparedUserDeploys {
            deploys: valid_unique,
            effective_cap: pending_unique_count,
            cap_hit: false,
        });
    }

    let max_user_deploys = effective_user_deploys_per_block_cap(pending_unique_count, max_deploys);
    if valid_unique.len() <= max_user_deploys {
        return Ok(PreparedUserDeploys {
            deploys: valid_unique,
            effective_cap: max_user_deploys,
            cap_hit: false,
        });
    }

    // Deterministically order deploys by age so selection remains stable across validators.
    let mut ordered: Vec<Signed<DeployData>> = valid_unique.into_iter().collect();
    ordered.sort_by(|a, b| {
        a.data
            .valid_after_block_number
            .cmp(&b.data.valid_after_block_number)
            .then_with(|| a.data.time_stamp.cmp(&b.data.time_stamp))
            .then_with(|| {
                // Stable deterministic tie-breaker for identical timestamps/windows.
                a.sig.cmp(&b.sig)
            })
    });

    // To avoid head-of-line blocking after stress bursts, reserve one slot for
    // the freshest deploy when capping is active. The remaining slots still drain
    // oldest deploys first to preserve fairness.
    let (selected, selection_strategy): (HashSet<Signed<DeployData>>, &'static str) =
        if deploy_selection_reserve_tail_enabled() {
            if max_user_deploys == 1 {
                (
                    ordered.iter().last().cloned().into_iter().collect(),
                    "newest-only",
                )
            } else {
                let oldest_take = max_user_deploys.saturating_sub(1);
                let mut picked: HashSet<Signed<DeployData>> =
                    ordered.iter().take(oldest_take).cloned().collect();
                if let Some(newest) = ordered.iter().last().cloned() {
                    picked.insert(newest);
                }
                if max_user_deploys <= ordered.len() {
                    debug_assert_eq!(picked.len(), max_user_deploys);
                }
                (picked, "oldest-plus-newest")
            }
        } else {
            (
                ordered.iter().take(max_user_deploys).cloned().collect(),
                "oldest-only",
            )
        };
    let deferred = valid_count
        .saturating_sub(already_in_scope_count)
        .saturating_sub(selected.len());

    tracing::info!(
        "Deploy selection capped for block #{}: selected={}, deferred={}, cap={}, strategy={}",
        block_number,
        selected.len(),
        deferred,
        max_user_deploys,
        selection_strategy
    );

    Ok(PreparedUserDeploys {
        deploys: selected,
        effective_cap: max_user_deploys,
        cap_hit: true,
    })
}


#[derive(Debug)]
struct AdaptiveDeployCapState {
    current_cap: usize,
    ema_create_block_ms: Option<f64>,
}

fn adaptive_user_deploy_cap_enabled() -> bool {
    true
}

fn adaptive_user_deploy_target_ms() -> u64 {
    1_000
}

fn adaptive_user_deploy_min_cap(max_cap: usize) -> usize {
    1_usize.clamp(1, max_cap)
}

fn adaptive_small_batch_bypass_threshold() -> usize {
    3
}

fn should_bypass_adaptive_cap_for_small_batch(
    pending_count: usize,
    max_cap: usize,
    bypass_threshold: usize,
) -> bool {
    if pending_count == 0 {
        return false;
    }

    let effective_threshold = bypass_threshold.min(max_cap);
    pending_count <= effective_threshold
}

fn adaptive_user_deploy_cap_state(
    max_cap: usize,
    min_cap: usize,
) -> &'static Mutex<AdaptiveDeployCapState> {
    static VALUE: OnceLock<Mutex<AdaptiveDeployCapState>> = OnceLock::new();
    VALUE.get_or_init(|| {
        Mutex::new(AdaptiveDeployCapState {
            current_cap: max_cap.clamp(min_cap, max_cap),
            ema_create_block_ms: None,
        })
    })
}

fn adaptive_backlog_floor_enabled() -> bool {
    true
}

fn adaptive_backlog_floor_trigger() -> usize {
    2
}

fn adaptive_backlog_floor_divisor() -> usize {
    2
}

fn adaptive_backlog_floor_min(max_cap: usize) -> usize {
    2_usize.clamp(1, max_cap)
}

fn adaptive_backlog_floor_max(max_cap: usize) -> usize {
    8_usize.clamp(1, max_cap)
}

fn backlog_floor_for_pending(
    pending_count: usize,
    max_cap: usize,
    trigger: usize,
    divisor: usize,
    min_floor: usize,
    max_floor: usize,
) -> usize {
    if pending_count < trigger {
        return 1;
    }

    let divisor = divisor.max(1);
    let ceil_div = pending_count.saturating_add(divisor - 1) / divisor;
    ceil_div
        .clamp(min_floor.max(1), max_floor.max(min_floor))
        .clamp(1, max_cap)
}

fn adaptive_backlog_floor(max_cap: usize, pending_count: usize) -> usize {
    if !adaptive_backlog_floor_enabled() {
        return 1;
    }

    let trigger = adaptive_backlog_floor_trigger();
    let divisor = adaptive_backlog_floor_divisor();
    let min_floor = adaptive_backlog_floor_min(max_cap);
    let max_floor = adaptive_backlog_floor_max(max_cap);

    backlog_floor_for_pending(
        pending_count,
        max_cap,
        trigger,
        divisor,
        min_floor,
        max_floor,
    )
}

fn effective_user_deploys_per_block_cap(pending_count: usize, max_cap: usize) -> usize {
    if !adaptive_user_deploy_cap_enabled() {
        return max_cap;
    }

    let min_cap = adaptive_user_deploy_min_cap(max_cap);
    let backlog_floor = adaptive_backlog_floor(max_cap, pending_count);
    let state = adaptive_user_deploy_cap_state(max_cap, min_cap);
    match state.lock() {
        Ok(mut guard) => {
            guard.current_cap = guard.current_cap.clamp(min_cap, max_cap);
            std::cmp::max(guard.current_cap, backlog_floor)
        }
        Err(err) => {
            tracing::warn!(
                "Adaptive deploy cap lock poisoned while reading current cap: {}",
                err
            );
            max_cap
        }
    }
}

fn next_adaptive_cap(
    current_cap: usize,
    min_cap: usize,
    max_cap: usize,
    target_ms: f64,
    ema_ms: f64,
    cap_hit: bool,
) -> usize {
    if min_cap >= max_cap {
        return max_cap;
    }

    let current_cap = current_cap.clamp(min_cap, max_cap);
    if !(target_ms.is_finite() && target_ms > 0.0 && ema_ms.is_finite() && ema_ms > 0.0) {
        return current_cap;
    }

    if ema_ms > target_ms {
        // Reduce cap proportionally when observed block creation time exceeds the target.
        let ratio = (target_ms / ema_ms).clamp(0.1, 0.99);
        let scaled = ((current_cap as f64) * ratio).floor() as usize;
        let max_step_down = current_cap.saturating_sub(1).max(min_cap);
        return scaled.clamp(min_cap, max_step_down);
    }

    const INCREASE_THRESHOLD_RATIO: f64 = 0.75;
    if cap_hit && ema_ms < target_ms * INCREASE_THRESHOLD_RATIO {
        // Increase only when we saturated the cap and have enough latency headroom.
        let ratio = (target_ms / ema_ms).clamp(1.0, 1.5);
        let scaled = ((current_cap as f64) * ratio).ceil() as usize;
        let min_step_up = current_cap.saturating_add(1).min(max_cap);
        return scaled.max(min_step_up).min(max_cap);
    }

    current_cap
}

fn update_adaptive_user_deploy_cap(
    observed_create_block_ms: u128,
    selected_user_deploys: usize,
    cap_hit: bool,
    max_deploys: usize,
) {
    if !adaptive_user_deploy_cap_enabled() || selected_user_deploys == 0 {
        return;
    }

    let max_cap = max_deploys;
    let min_cap = adaptive_user_deploy_min_cap(max_cap);
    if min_cap >= max_cap {
        return;
    }

    let target_ms = adaptive_user_deploy_target_ms() as f64;
    let sample_ms = observed_create_block_ms as f64;
    let state = adaptive_user_deploy_cap_state(max_cap, min_cap);

    let mut guard = match state.lock() {
        Ok(guard) => guard,
        Err(err) => {
            tracing::warn!(
                "Adaptive deploy cap lock poisoned while updating cap: {}",
                err
            );
            return;
        }
    };

    guard.current_cap = guard.current_cap.clamp(min_cap, max_cap);

    const EMA_ALPHA: f64 = 0.35;
    let prev_ema = guard.ema_create_block_ms.unwrap_or(target_ms);
    let ema_ms = prev_ema + EMA_ALPHA * (sample_ms - prev_ema);
    guard.ema_create_block_ms = Some(ema_ms);

    let prev_cap = guard.current_cap;
    let next_cap = next_adaptive_cap(prev_cap, min_cap, max_cap, target_ms, ema_ms, cap_hit);

    if next_cap != prev_cap {
        guard.current_cap = next_cap;
        tracing::info!(
            "Adaptive deploy cap update: prev_cap={}, next_cap={}, sample_create_block_ms={}, ema_create_block_ms={:.2}, target_ms={}, selected_user_deploys={}, cap_hit={}",
            prev_cap,
            next_cap,
            observed_create_block_ms,
            ema_ms,
            target_ms as u64,
            selected_user_deploys,
            cap_hit
        );
    }
}

fn collect_self_chain_deploy_sigs(
    casper_snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
    block_store: &KeyValueBlockStore,
) -> Result<HashSet<Bytes>, CasperError> {
    let self_validator = validator_identity.public_key.bytes.clone();
    let current_hash_from_justifications = casper_snapshot
        .justifications
        .iter()
        .find(|j| j.validator == self_validator)
        .map(|j| j.latest_block_hash.clone());
    let current_hash_from_dag = casper_snapshot.dag.latest_message_hash(&self_validator);

    let Some(mut current_hash) = current_hash_from_justifications.or(current_hash_from_dag) else {
        return Ok(HashSet::new());
    };

    let mut deploy_sigs: HashSet<Bytes> = HashSet::new();
    let max_depth = std::cmp::max(casper_snapshot.on_chain_state.shard_conf.deploy_lifespan, 1);

    for _ in 0..(max_depth as usize) {
        let Some(block) = block_store.get(&current_hash)? else {
            break;
        };

        for processed in &block.body.deploys {
            deploy_sigs.insert(processed.deploy.sig.clone());
        }

        let Some(main_parent) = block.header.parents_hash_list.first().cloned() else {
            break;
        };
        current_hash = main_parent;
    }

    Ok(deploy_sigs)
}

async fn prepare_slashing_deploys(
    casper_snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
    seq_num: i32,
) -> Result<Vec<SlashDeploy>, CasperError> {
    let self_id = Bytes::copy_from_slice(&validator_identity.public_key.bytes);

    // Get invalid latest messages from DAG
    let invalid_latest_messages = casper_snapshot.dag.invalid_latest_messages()?;

    // Filter to only include bonded validators
    let bonded_invalid_messages: Vec<_> = invalid_latest_messages
        .into_iter()
        .filter(|(validator, _)| {
            casper_snapshot
                .on_chain_state
                .bonds_map
                .get(validator)
                .map(|stake| *stake > 0)
                .unwrap_or(false)
        })
        .collect();

    // TODO: Add `slashingDeploys` to DeployStorage - OLD
    // Create SlashDeploy objects
    let mut slashing_deploys = Vec::new();
    for (_, invalid_block_hash) in bonded_invalid_messages {
        let slash_deploy = SlashDeploy {
            invalid_block_hash: invalid_block_hash.clone(),
            pk: validator_identity.public_key.clone(),
            initial_rand: system_deploy_util::generate_slash_deploy_random_seed(
                self_id.clone(),
                seq_num,
            ),
        };

        tracing::info!(
            "Issuing slashing deploy justified by block {}",
            pretty_printer::PrettyPrinter::build_string_bytes(&invalid_block_hash)
        );

        slashing_deploys.push(slash_deploy);
    }

    Ok(slashing_deploys)
}

fn prepare_dummy_deploy(
    block_number: i64,
    shard_id: String,
    dummy_deploy_opt: Option<(PrivateKey, String)>,
) -> Result<Vec<Signed<DeployData>>, CasperError> {
    match dummy_deploy_opt {
        Some((private_key, term)) => {
            let deploy = construct_deploy::source_deploy_now(
                term,
                Some(private_key),
                Some(block_number - 1),
                Some(shard_id),
            )
            .map_err(|e| {
                CasperError::RuntimeError(format!("Failed to create dummy deploy: {}", e))
            })?;
            Ok(vec![deploy])
        }
        None => Ok(Vec::new()),
    }
}

fn extract_deploy_sig_from_refund_failure(msg: &str) -> Option<Vec<u8>> {
    let marker = "deploy_sig=";
    let start = msg.find(marker)? + marker.len();
    let tail = &msg[start..];
    let end = tail.find(',').unwrap_or(tail.len());
    let sig_hex = tail[..end].trim();
    hex::decode(sig_hex).ok()
}

fn quarantine_refund_failure_deploy(
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    failure_msg: &str,
) -> Result<bool, CasperError> {
    let Some(sig) = extract_deploy_sig_from_refund_failure(failure_msg) else {
        return Ok(false);
    };

    let mut guard = deploy_storage
        .lock()
        .map_err(|e| CasperError::LockError(e.to_string()))?;
    guard.remove_by_sig(&sig).map_err(CasperError::KvStoreError)
}

pub async fn create(
    casper_snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
    dummy_deploy_opt: Option<(PrivateKey, String)>,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    runtime_manager: &mut RuntimeManager,
    block_store: &mut KeyValueBlockStore,
    allow_empty_blocks: bool,
) -> Result<BlockCreatorResult, CasperError> {
    let create_started = std::time::Instant::now();
    // Capture current time once to ensure consistency between deploy filtering and block timestamp.
    // This prevents race condition where a deploy could pass filtering but expire before block creation.
    let now_u128 = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| CasperError::RuntimeError(format!("Failed to get current time: {}", e)))?
        .as_millis();
    let mut now_millis = i64::try_from(now_u128).map_err(|_| {
        CasperError::RuntimeError(format!(
            "Current timestamp millis {} exceeds i64::MAX",
            now_u128
        ))
    })?;

    let next_seq_num = casper_snapshot
        .max_seq_nums
        .get(&validator_identity.public_key.bytes)
        .map(|seq| *seq + 1)
        .unwrap_or(1) as i32;
    let next_block_num = casper_snapshot.max_block_num + 1;
    let parents = &casper_snapshot.parents;
    let justifications = &casper_snapshot.justifications;
    if let Some(max_parent_ts) = parents.iter().map(|p| p.header.timestamp).max() {
        if now_millis < max_parent_ts {
            tracing::debug!(
                "Adjusting block timestamp from {} to parent timestamp {} to avoid clock-skew regressions",
                now_millis,
                max_parent_ts
            );
            now_millis = max_parent_ts;
        }
    }

    tracing::info!(
        "Creating block #{} (seqNum {})",
        next_block_num,
        next_seq_num
    );

    let shard_id = casper_snapshot.on_chain_state.shard_conf.shard_name.clone();

    // Prepare deploys
    let (user_deploys, selected_user_deploy_cap, selected_user_deploy_cap_hit) = {
        let t = std::time::Instant::now();
        let prepared = prepare_user_deploys(
            casper_snapshot,
            next_block_num,
            now_millis,
            deploy_storage.clone(),
        )
        .await?;
        let mut v = prepared.deploys;
        let self_chain_deploy_sigs =
            collect_self_chain_deploy_sigs(casper_snapshot, validator_identity, block_store)?;
        if !self_chain_deploy_sigs.is_empty() {
            let before = v.len();
            v.retain(|deploy| !self_chain_deploy_sigs.contains(&deploy.sig));
            let skipped = before.saturating_sub(v.len());
            if skipped > 0 {
                tracing::info!(
                    "Filtered {} deploy(s) already present in self latest-message chain",
                    skipped
                );
            }
        }
        tracing::debug!(
            target: "f1r3fly.block_creator.timing",
            "prepare_user_deploys_ms={}, user_deploys_count={}, user_deploy_cap={}, user_deploy_cap_hit={}",
            t.elapsed().as_millis(),
            v.len(),
            prepared.effective_cap,
            prepared.cap_hit
        );
        (v, prepared.effective_cap, prepared.cap_hit)
    };
    let dummy_deploys = {
        let t = std::time::Instant::now();
        let v = prepare_dummy_deploy(next_block_num, shard_id.clone(), dummy_deploy_opt)?;
        tracing::debug!(
            target: "f1r3fly.block_creator.timing",
            "prepare_dummy_deploys_ms={}, dummy_deploys_count={}",
            t.elapsed().as_millis(),
            v.len()
        );
        v
    };
    let slashing_deploys = {
        let t = std::time::Instant::now();
        let v = prepare_slashing_deploys(casper_snapshot, validator_identity, next_seq_num).await?;
        tracing::debug!(
            target: "f1r3fly.block_creator.timing",
            "prepare_slashing_deploys_ms={}, slashing_deploys_count={}",
            t.elapsed().as_millis(),
            v.len()
        );
        v
    };

    let selected_user_deploy_count = user_deploys.len();

    // Combine all deploys. prepare_user_deploys already removed deploys in scope.
    let mut all_deploys: HashSet<Signed<DeployData>> = user_deploys;

    // Add dummy deploys
    all_deploys.extend(dummy_deploys);

    // Check if we have any new work to process.
    // If empty blocks are disabled, skip closeBlock-only proposals to avoid no-op checkpoint cost.
    // If empty blocks are enabled (heartbeat/liveness mode), continue and emit closeBlock.
    let has_slashing_deploys = !slashing_deploys.is_empty();
    if all_deploys.is_empty() && !has_slashing_deploys {
        if !allow_empty_blocks {
            tracing::info!(
                "Skipping empty block creation: no new user deploys and no slashing deploys"
            );
            return Ok(BlockCreatorResult::NoNewDeploys);
        }
    }

    // Make sure closeBlock is the last system Deploy
    let mut system_deploys_converted: Vec<SystemDeployEnum> = Vec::new();

    // Add slashing deploys
    for slash_deploy in slashing_deploys {
        system_deploys_converted.push(SystemDeployEnum::Slash(slash_deploy));
    }

    // Add the actual close block deploy
    system_deploys_converted.push(SystemDeployEnum::Close(CloseBlockDeploy {
        initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
            validator_identity.public_key.clone(),
            next_seq_num,
        ),
    }));

    // Use the adjusted `now_millis` captured at the start of create for block timestamp.
    // The value is clamped to the max parent timestamp to avoid InvalidTimestamp from clock skew.
    // This ensures the same time is used for deploy filtering and block creation.
    let invalid_blocks = casper_snapshot.invalid_blocks.clone();
    let block_data = BlockData {
        time_stamp: now_millis,
        block_number: next_block_num,
        sender: validator_identity.public_key.clone(),
        seq_num: next_seq_num,
    };

    // Compute checkpoint data
    let checkpoint_started = std::time::Instant::now();
    let checkpoint_data = match interpreter_util::compute_deploys_checkpoint(
        block_store,
        parents.clone(),
        all_deploys.into_iter().collect(),
        system_deploys_converted,
        casper_snapshot,
        runtime_manager,
        block_data.clone(),
        invalid_blocks,
    )
    .await
    {
        Ok(data) => data,
        Err(CasperError::SystemRuntimeError(SystemDeployPlatformFailure::GasRefundFailure(
            msg,
        ))) => {
            let removed = quarantine_refund_failure_deploy(deploy_storage.clone(), &msg)?;
            tracing::warn!(
                "Gas refund failure during checkpoint; quarantined_toxic_deploy={} error={}",
                removed,
                msg
            );
            return Ok(BlockCreatorResult::NoNewDeploys);
        }
        Err(err) => return Err(err),
    };
    tracing::debug!(
        target: "f1r3fly.block_creator.timing",
        "compute_deploys_checkpoint_ms={}",
        checkpoint_started.elapsed().as_millis()
    );

    let (
        pre_state_hash,
        post_state_hash,
        processed_deploys,
        rejected_deploys,
        processed_system_deploys,
        new_bonds,
    ) = checkpoint_data;

    let casper_version = casper_snapshot.on_chain_state.shard_conf.casper_version;

    // Span[F].trace(ProcessDeploysAndCreateBlockMetricsSource) from Scala
    let _span =
        tracing::info_span!(target: "f1r3fly.create-block", "process-deploys-and-create-block")
            .entered();

    tracing::event!(tracing::Level::DEBUG, mark = "before-packing-block");
    // Create unsigned block
    let package_started = std::time::Instant::now();
    let pre_state_hash_for_result = pre_state_hash.clone();
    let post_state_hash_for_result = post_state_hash.clone();
    let unsigned_block = package_block(
        &block_data,
        parents.iter().map(|p| p.block_hash.clone()).collect(),
        justifications.iter().map(|j| j.clone()).collect(),
        pre_state_hash,
        post_state_hash,
        processed_deploys,
        rejected_deploys,
        processed_system_deploys,
        new_bonds,
        shard_id,
        casper_version,
    );
    let package_ms = package_started.elapsed().as_millis();

    tracing::event!(tracing::Level::DEBUG, mark = "block-created");
    // Sign the block
    let sign_started = std::time::Instant::now();
    let signed_block = validator_identity.sign_block(&unsigned_block);
    let sign_ms = sign_started.elapsed().as_millis();

    tracing::event!(tracing::Level::DEBUG, mark = "block-signed");

    let block_info = pretty_printer::PrettyPrinter::build_string_block_message(&signed_block, true);
    let deploy_count = signed_block.body.deploys.len();
    tracing::debug!("Block created: {} ({}d)", block_info, deploy_count);
    let total_create_block_ms = create_started.elapsed().as_millis();

    tracing::debug!(
        target: "f1r3fly.block_creator.timing",
        "Block creator timing: package_ms={}, sign_ms={}, total_create_block_ms={}",
        package_ms,
        sign_ms,
        total_create_block_ms
    );

    update_adaptive_user_deploy_cap(
        total_create_block_ms,
        selected_user_deploy_count,
        selected_user_deploy_cap_hit && selected_user_deploy_count >= selected_user_deploy_cap,
        casper_snapshot.on_chain_state.shard_conf.max_user_deploys_per_block as usize,
    );

    RuntimeManager::trim_allocator();

    Ok(BlockCreatorResult::Created(
        signed_block,
        pre_state_hash_for_result,
        post_state_hash_for_result,
    ))
}

fn package_block(
    block_data: &BlockData,
    parents: Vec<Bytes>,
    justifications: Vec<Justification>,
    pre_state_hash: Bytes,
    post_state_hash: Bytes,
    deploys: Vec<ProcessedDeploy>,
    rejected_deploys: Vec<Bytes>,
    system_deploys: Vec<ProcessedSystemDeploy>,
    bonds_map: Vec<Bond>,
    shard_id: String,
    version: i64,
) -> BlockMessage {
    let state = F1r3flyState {
        pre_state_hash,
        post_state_hash,
        bonds: bonds_map,
        block_number: block_data.block_number,
    };

    let rejected_deploys_wrapped: Vec<RejectedDeploy> = rejected_deploys
        .into_iter()
        .map(|r| RejectedDeploy { sig: r })
        .collect();

    let body = Body {
        state,
        deploys,
        rejected_deploys: rejected_deploys_wrapped,
        system_deploys,
        extra_bytes: Bytes::new(),
    };

    let header = Header {
        parents_hash_list: parents,
        timestamp: block_data.time_stamp,
        version,
        extra_bytes: Bytes::new(),
    };

    proto_util::unsigned_block_proto(
        body,
        header,
        justifications,
        shard_id,
        Some(block_data.seq_num as i32),
    )
}

fn not_expired_deploy(earliest_block_number: i64, deploy_data: &DeployData) -> bool {
    deploy_data.valid_after_block_number > earliest_block_number
}

fn not_future_deploy(current_block_number: i64, deploy_data: &DeployData) -> bool {
    deploy_data.valid_after_block_number < current_block_number
}

#[cfg(test)]
mod tests {
    use super::{
        backlog_floor_for_pending, next_adaptive_cap, should_bypass_adaptive_cap_for_small_batch,
    };

    #[test]
    fn adaptive_cap_reduces_when_latency_exceeds_target() {
        let next = next_adaptive_cap(32, 1, 32, 1000.0, 2500.0, true);
        assert_eq!(next, 12);
    }

    #[test]
    fn adaptive_cap_increases_when_capped_and_headroom_exists() {
        let next = next_adaptive_cap(4, 1, 32, 1000.0, 400.0, true);
        assert_eq!(next, 6);
    }

    #[test]
    fn adaptive_cap_does_not_increase_when_not_capped() {
        let next = next_adaptive_cap(8, 1, 32, 1000.0, 300.0, false);
        assert_eq!(next, 8);
    }

    #[test]
    fn adaptive_cap_respects_min_and_max_bounds() {
        let down = next_adaptive_cap(3, 2, 4, 1000.0, 5000.0, true);
        let up = next_adaptive_cap(3, 2, 4, 1000.0, 250.0, true);
        assert_eq!(down, 2);
        assert_eq!(up, 4);
    }

    #[test]
    fn backlog_floor_disabled_below_trigger() {
        let floor = backlog_floor_for_pending(7, 32, 8, 4, 2, 16);
        assert_eq!(floor, 1);
    }

    #[test]
    fn backlog_floor_scales_with_pending_pool() {
        let floor = backlog_floor_for_pending(35, 32, 8, 4, 2, 16);
        assert_eq!(floor, 9);
    }

    #[test]
    fn backlog_floor_respects_bounds() {
        let floor = backlog_floor_for_pending(512, 32, 8, 4, 2, 16);
        assert_eq!(floor, 16);

        let floor_small_cap = backlog_floor_for_pending(64, 6, 8, 4, 2, 16);
        assert_eq!(floor_small_cap, 6);
    }

    #[test]
    fn small_batch_bypass_applies_when_pending_within_threshold() {
        assert!(should_bypass_adaptive_cap_for_small_batch(3, 32, 3));
        assert!(should_bypass_adaptive_cap_for_small_batch(2, 32, 3));
    }

    #[test]
    fn small_batch_bypass_does_not_apply_for_zero_or_large_pending() {
        assert!(!should_bypass_adaptive_cap_for_small_batch(0, 32, 3));
        assert!(!should_bypass_adaptive_cap_for_small_batch(4, 32, 3));
    }

    #[test]
    fn small_batch_bypass_respects_max_cap_bound() {
        assert!(should_bypass_adaptive_cap_for_small_batch(6, 6, 16));
        assert!(!should_bypass_adaptive_cap_for_small_batch(7, 6, 16));
    }
}
