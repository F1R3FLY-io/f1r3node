// See casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::{block_hash::BlockHash, block_metadata::BlockMetadata, validator::Validator};
use shared::rust::store::key_value_store::KvStoreError;

use crate::rust::safety::clique_oracle::CliqueOracle;

/// Block can be recorded as last finalized block (LFB) if Safety oracle outputs fault tolerance (FT)
/// for this block greater then some predefined threshold. This is defined by [`CliqueOracle::compute_output`]
/// function, which requires some target block as input arg.
///
/// Therefore: Finalizer has a scope of search, defined by tips and previous LFB - each of this blocks can be next LFB.
///
/// We know that LFB advancement is not necessary continuous, next LFB might not be direct child of current one.
///
/// Therefore: we cannot start from current LFB children and traverse DAG from the bottom to the top, calculating FT
/// for each block. Also its computationally ineffective.
///
/// But we know that scope of search for potential next LFB is constrained. Block A can be finalized only
/// if it has more then half of total stake in bonds map of A translated from tips throughout main parent chain.
/// IMPORTANT: only main parent relation gives weight to potentially finalized block.
///
/// Therefore: Finalizer should seek for next LFB going through 2 steps:
///   1. Find messages in scope of search that have more then half of the stake translated through main parent chain
///     from tips down to the message.
///   2. Execute [`CliqueOracle::compute_output`] on these targets.
///   3. First message passing FT threshold becomes the next LFB.
pub struct Finalizer;
const FINALIZER_WORK_BUDGET_MS: u64 = 2_000;
const FINALIZER_STEP_TIMEOUT_MS: u64 = 200;
const FINALIZER_CATCHUP_LAG_THRESHOLD_BLOCKS: i64 = 1_024;
const FINALIZER_CATCHUP_WORK_BUDGET_MS: u64 = 10_000;
const FINALIZER_CATCHUP_STEP_TIMEOUT_MS: u64 = 1_000;
const FINALIZER_MAX_CLIQUE_CANDIDATES_DEFAULT: usize = 128;
const FINALIZER_MAX_CLIQUE_CANDIDATES_ENV: &str = "F1R3_FINALIZER_MAX_CLIQUE_CANDIDATES";
const FINALIZER_CANDIDATE_RANKING_ENV: &str = "F1R3_FINALIZER_CANDIDATE_RANKING";

#[derive(Debug, Clone, Copy)]
enum CandidateRankingStrategy {
    RecencySmallSetStake,
    StakeDesc,
    RecencyStake,
}

impl CandidateRankingStrategy {
    fn from_env() -> Self {
        let value = std::env::var(FINALIZER_CANDIDATE_RANKING_ENV)
            .unwrap_or_default()
            .to_lowercase();
        match value.as_str() {
            "stake_desc" => Self::StakeDesc,
            "recency_smallset_stake" => Self::RecencySmallSetStake,
            _ => Self::RecencyStake,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::RecencySmallSetStake => "recency_smallset_stake",
            Self::StakeDesc => "stake_desc",
            Self::RecencyStake => "recency_stake",
        }
    }
}

type WeightMap = HashMap<Validator, i64>;
type SharedWeightMap = Arc<WeightMap>;

impl Finalizer {
    fn finalizer_max_clique_candidates() -> usize {
        std::env::var(FINALIZER_MAX_CLIQUE_CANDIDATES_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(FINALIZER_MAX_CLIQUE_CANDIDATES_DEFAULT)
    }

    /// weight map as per message, look inside [`CliqueOracle::get_corresponding_weight_map`] description for more info
    async fn message_weight_map_f(
        message: &BlockMetadata,
        dag: &KeyValueDagRepresentation,
    ) -> Result<WeightMap, KvStoreError> {
        CliqueOracle::get_corresponding_weight_map(&message.block_hash, dag).await
    }

    /// If more then half of total stake agree on message - it is considered to be safe from orphaning.
    pub fn cannot_be_orphaned(
        message_weight_map: &WeightMap,
        agreeing_weight_map: &WeightMap,
    ) -> bool {
        assert!(
            !agreeing_weight_map.values().any(|&stake| stake <= 0),
            "Agreeing map contains not bonded validators"
        );

        let active_stake_total: i64 = message_weight_map.values().sum();
        let active_stake_agreeing: i64 = agreeing_weight_map.values().sum();

        // in theory if each stake is high enough, e.g. i64::MAX, sum of them might result in negative value
        assert!(
            active_stake_total > 0,
            "Long overflow when computing total stake"
        );
        assert!(
            active_stake_agreeing > 0,
            "Long overflow when computing total stake"
        );

        active_stake_agreeing as f64 > (active_stake_total as f64) / 2.0
    }

    /// Cheap upper bound on FT without clique search.
    /// Since max clique weight <= sum(agreeing stake), this is a safe prune bound.
    fn fault_tolerance_upper_bound(message_weight_map: &WeightMap, agreeing_weight_map: &WeightMap) -> f32 {
        let total_stake = message_weight_map.values().sum::<i64>() as f32;
        let agreeing_stake = agreeing_weight_map.values().sum::<i64>() as f32;
        if total_stake <= 0.0 {
            return f32::MIN;
        }
        (agreeing_stake * 2.0 - total_stake) / total_stake
    }

    /// Create an agreement given validator that agrees on a message and weight map of a message.
    /// If validator is not present in message bonds map or its stake is zero, None is returned
    fn record_agreement(
        message_weight_map: &WeightMap,
        agreeing_validator: &Validator,
    ) -> Option<(Validator, i64)> {
        // if validator is not bonded according to message weight map - there is no agreement translated.
        let stake_agreed = message_weight_map
            .get(agreeing_validator)
            .copied()
            .unwrap_or(0);
        if stake_agreed > 0 {
            Some((agreeing_validator.clone(), stake_agreed))
        } else {
            None
        }
    }

    /// Find the highest finalized message.
    /// Scope of the search is constrained by the lowest height (height of current last finalized message).
    pub async fn run<F, Fut>(
        dag: &KeyValueDagRepresentation,
        fault_tolerance_threshold: f32,
        curr_lfb_height: i64,
        mut new_lfb_found_effect: F,
    ) -> Result<Option<BlockHash>, KvStoreError>
    where
        F: FnMut(BlockHash) -> Fut,
        Fut: std::future::Future<Output = Result<(), KvStoreError>>,
    {
        let total_started = std::time::Instant::now();
        let lfb_lag = dag.latest_block_number().saturating_sub(curr_lfb_height);
        let catchup_mode = lfb_lag > FINALIZER_CATCHUP_LAG_THRESHOLD_BLOCKS;
        let work_budget = Duration::from_millis(if catchup_mode {
            FINALIZER_CATCHUP_WORK_BUDGET_MS
        } else {
            FINALIZER_WORK_BUDGET_MS
        });
        let step_timeout = Duration::from_millis(if catchup_mode {
            FINALIZER_CATCHUP_STEP_TIMEOUT_MS
        } else {
            FINALIZER_STEP_TIMEOUT_MS
        });
        let max_clique_candidates = Self::finalizer_max_clique_candidates();
        let ranking_strategy = CandidateRankingStrategy::from_env();
        /*
         * Stream of agreements passed down from all latest messages to main parents.
         * Starts with agreements of latest message on themselves.
         *
         * The goal here is to create stream of agreements breadth first, so on each step agreements by all
         * validator are recorded, and only after that next level of main parents is visited.
         */
        let lms = dag.latest_messages()?;
        let latest_messages_count = lms.len();

        // sort latest messages by agreeing validator to ensure random ordering does not change output
        let mut sorted_latest_messages: Vec<(Validator, BlockMetadata)> = lms.into_iter().collect();
        sorted_latest_messages.sort_by(|(v1, _), (v2, _)| v1.cmp(v2));

        // Step 1: Traverse agreement layers and aggregate agreements per target block.
        // This avoids materializing a large stream of duplicate (message, weight-map, agreement)
        // tuples that would later be deduplicated by block hash.
        let mut aggregated_agreements: HashMap<BlockHash, (BlockMetadata, SharedWeightMap, WeightMap)> =
            HashMap::new();
        let mut message_weight_map_cache: HashMap<BlockHash, SharedWeightMap> = HashMap::new();
        let mut main_parent_cache: HashMap<BlockHash, Option<BlockMetadata>> = HashMap::new();
        let mut message_weight_map_cache_hit: usize = 0;
        let mut message_weight_map_cache_miss: usize = 0;
        let mut main_parent_cache_hit: usize = 0;
        let mut main_parent_cache_miss: usize = 0;
        let mut current_layer = sorted_latest_messages;
        let mut layers_visited: usize = 0;
        let mut budget_exhausted = false;
        let mut agreements_count: usize = 0;
        let mut weight_map_phase_ns: u128 = 0;
        let mut agreement_record_phase_ns: u128 = 0;
        let mut parent_lookup_phase_ns: u128 = 0;
        let mut next_layer_push_phase_ns: u128 = 0;

        loop {
            if total_started.elapsed() >= work_budget {
                budget_exhausted = true;
                break;
            }
            layers_visited += 1;
            let mut next_layer: Vec<(Validator, BlockMetadata)> = Vec::new();

            for (agreeing_validator, message) in current_layer.into_iter() {
                if total_started.elapsed() >= work_budget {
                    budget_exhausted = true;
                    break;
                }
                let phase_t = std::time::Instant::now();
                let message_weight_map =
                    if let Some(cached) = message_weight_map_cache.get(&message.block_hash).cloned() {
                        message_weight_map_cache_hit += 1;
                        cached
                    } else {
                        message_weight_map_cache_miss += 1;
                        let Ok(fetched) = Self::message_weight_map_f(&message, dag).await else {
                            continue;
                        };
                        let fetched = Arc::new(fetched);
                        message_weight_map_cache.insert(message.block_hash.clone(), fetched.clone());
                        fetched
                    };
                weight_map_phase_ns += phase_t.elapsed().as_nanos();

                let phase_t = std::time::Instant::now();
                let (_, _, agreeing_weight_map) = aggregated_agreements
                    .entry(message.block_hash.clone())
                    .or_insert_with(|| {
                        (message.clone(), message_weight_map.clone(), WeightMap::new())
                    });
                if let Some((agreeing_validator, stake_agreed)) =
                    Self::record_agreement(&message_weight_map, &agreeing_validator)
                {
                    assert!(
                        !agreeing_weight_map.contains_key(&agreeing_validator),
                        "Logical error during finalization: message {:?} got duplicate agreement.",
                        message.block_hash
                    );
                    agreeing_weight_map.insert(agreeing_validator, stake_agreed);
                    agreements_count += 1;
                }
                agreement_record_phase_ns += phase_t.elapsed().as_nanos();

                if let Some(main_parent_hash) = message.parents.first() {
                    let phase_t = std::time::Instant::now();
                    let parent_meta = if let Some(cached) = main_parent_cache.get(main_parent_hash) {
                        main_parent_cache_hit += 1;
                        cached.clone()
                    } else {
                        main_parent_cache_miss += 1;
                        let fetched = dag.lookup_unsafe(main_parent_hash).ok();
                        main_parent_cache.insert(main_parent_hash.clone(), fetched.clone());
                        fetched
                    };
                    parent_lookup_phase_ns += phase_t.elapsed().as_nanos();

                    let phase_t = std::time::Instant::now();
                    if let Some(next_message) =
                        parent_meta.filter(|meta| meta.block_number > curr_lfb_height)
                    {
                        next_layer.push((agreeing_validator, next_message));
                    }
                    next_layer_push_phase_ns += phase_t.elapsed().as_nanos();
                }
            }

            if budget_exhausted {
                break;
            }

            if next_layer.is_empty() {
                break;
            }

            current_layer = next_layer;
        }

        // Step 2: Filter blocks that cannot be orphaned and precompute sort keys.
        let filtered_agreements: Vec<(BlockMetadata, SharedWeightMap, WeightMap, i64, usize)> =
            aggregated_agreements
                .into_values()
                .filter_map(|(message, message_weight_map, agreeing_weight_map)| {
                    Self::cannot_be_orphaned(&message_weight_map, &agreeing_weight_map).then(|| {
                        let stake_sum = agreeing_weight_map.values().sum::<i64>();
                        let agreeing_size = agreeing_weight_map.len();
                        (
                            message,
                            message_weight_map,
                            agreeing_weight_map,
                            stake_sum,
                            agreeing_size,
                        )
                    })
                })
                .collect();
        let filtered_agreements_count = filtered_agreements.len();
        let mut deduped_filtered_agreements: Vec<(
            BlockMetadata,
            SharedWeightMap,
            WeightMap,
            i64,
            usize,
        )> =
            filtered_agreements;
        deduped_filtered_agreements.sort_by(
            |(msg_l, _, _, stake_l, size_l), (msg_r, _, _, stake_r, size_r)| {
            match ranking_strategy {
                CandidateRankingStrategy::RecencySmallSetStake => msg_r
                    .block_number
                    .cmp(&msg_l.block_number)
                    .then_with(|| size_l.cmp(size_r))
                    .then_with(|| stake_r.cmp(stake_l))
                    .then_with(|| msg_l.block_hash.cmp(&msg_r.block_hash)),
                CandidateRankingStrategy::StakeDesc => stake_r
                    .cmp(stake_l)
                    .then_with(|| msg_r.block_number.cmp(&msg_l.block_number))
                    .then_with(|| size_l.cmp(size_r))
                    .then_with(|| msg_l.block_hash.cmp(&msg_r.block_hash)),
                CandidateRankingStrategy::RecencyStake => msg_r
                    .block_number
                    .cmp(&msg_l.block_number)
                    .then_with(|| stake_r.cmp(stake_l))
                    .then_with(|| size_l.cmp(size_r))
                    .then_with(|| msg_l.block_hash.cmp(&msg_r.block_hash)),
            }
        });
        let deduped_filtered_agreements_count = deduped_filtered_agreements.len();
        let candidate_capped = deduped_filtered_agreements_count > max_clique_candidates;
        let capped_agreements: Vec<(BlockMetadata, SharedWeightMap, WeightMap)> =
            deduped_filtered_agreements
            .into_iter()
            .map(
                |(message, message_weight_map, agreeing_weight_map, _, _)| {
                    (message, message_weight_map, agreeing_weight_map)
                },
            )
            .take(max_clique_candidates)
            .collect();

        // Compute fault tolerance lazily and stop at the first candidate that satisfies
        // finalization criteria. Preserves original candidate order while avoiding
        // expensive full-scan FT computation on long chains.
        let clique_started = std::time::Instant::now();
        let mut clique_run_cache = CliqueOracle::new_run_cache();
        let mut clique_eval_count: usize = 0;
        let mut upper_bound_pruned_count: usize = 0;
        let mut upper_bound_passed_count: usize = 0;
        let mut max_ft_upper_bound: f32 = f32::MIN;
        let mut lfb_result: Option<BlockHash> = None;
        for (message, message_weight_map, agreeing_weight_map) in capped_agreements {
            if total_started.elapsed() >= work_budget {
                budget_exhausted = true;
                break;
            }
            let ft_upper_bound =
                Self::fault_tolerance_upper_bound(&message_weight_map, &agreeing_weight_map);
            max_ft_upper_bound = max_ft_upper_bound.max(ft_upper_bound);
            if ft_upper_bound <= fault_tolerance_threshold {
                upper_bound_pruned_count += 1;
                continue;
            }
            upper_bound_passed_count += 1;
            clique_eval_count += 1;
            let ft_result = tokio::time::timeout(
                step_timeout,
                CliqueOracle::compute_output_with_cache(
                    &message.block_hash,
                    &message_weight_map,
                    &agreeing_weight_map,
                    dag,
                    &mut clique_run_cache,
                ),
            )
            .await;
            let Ok(Ok(fault_tolerance)) = ft_result else {
                continue;
            };

            if fault_tolerance > fault_tolerance_threshold {
                let lfb_hash = message.block_hash.clone();
                // Only process blocks that aren't already finalized
                if !dag.is_finalized(&lfb_hash) {
                    new_lfb_found_effect(lfb_hash.clone()).await?;
                }
                lfb_result = Some(lfb_hash);
                break;
            }
        }
        tracing::info!(
            target: "f1r3fly.finalizer.timing",
            "Finalizer timing: latest_messages={}, layers_visited={}, agreements={}, filtered_agreements={}, deduped_filtered_agreements={}, message_weight_map_cache_hit={}, message_weight_map_cache_miss={}, main_parent_cache_hit={}, main_parent_cache_miss={}, candidate_cap={}, ranking_strategy={}, candidate_capped={}, upper_bound_pruned={}, upper_bound_passed={}, max_ft_upper_bound={:.6}, clique_evals={}, clique_ms={}, total_ms={}, budget_ms={}, step_timeout_ms={}, budget_exhausted={}, lfb_lag={}, catchup_mode={}, found_new_lfb={}, weight_map_ns={}, agreement_ns={}, parent_ns={}, next_push_ns={}",
            latest_messages_count,
            layers_visited,
            agreements_count,
            filtered_agreements_count,
            deduped_filtered_agreements_count,
            message_weight_map_cache_hit,
            message_weight_map_cache_miss,
            main_parent_cache_hit,
            main_parent_cache_miss,
            max_clique_candidates,
            ranking_strategy.as_str(),
            candidate_capped,
            upper_bound_pruned_count,
            upper_bound_passed_count,
            max_ft_upper_bound,
            clique_eval_count,
            clique_started.elapsed().as_millis(),
            total_started.elapsed().as_millis(),
            work_budget.as_millis(),
            step_timeout.as_millis(),
            budget_exhausted,
            lfb_lag,
            catchup_mode,
            lfb_result.is_some(),
            weight_map_phase_ns,
            agreement_record_phase_ns,
            parent_lookup_phase_ns,
            next_layer_push_phase_ns
        );

        Ok(lfb_result)
    }
}
