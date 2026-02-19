// See casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala

use std::collections::{BTreeMap, HashMap};
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

type WeightMap = BTreeMap<Validator, i64>;

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
        let work_budget = Duration::from_millis(FINALIZER_WORK_BUDGET_MS);
        let step_timeout = Duration::from_millis(FINALIZER_STEP_TIMEOUT_MS);
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

        // Step 1: Generate stream of agreements
        // Scala's unfoldLoopEval outputs the current layer, then checks if next is non-empty to continue.
        // So ALL layers are output, including the last one where next would be empty.
        let mut mk_agreements_stream = Vec::new();
        let mut message_weight_map_cache: HashMap<BlockHash, WeightMap> = HashMap::new();
        let mut current_layer = sorted_latest_messages;
        let mut layers_visited: usize = 0;
        let mut budget_exhausted = false;

        loop {
            if total_started.elapsed() >= work_budget {
                budget_exhausted = true;
                break;
            }
            layers_visited += 1;
            // output current visits - process agreements for this layer
            let out = current_layer.clone();

            // evalMap: map visits to message agreements: validator v agrees on message m
            // The agreement is on the MESSAGE itself (from the current layer), not on its parent.
            for (validator, message) in out {
                if total_started.elapsed() >= work_budget {
                    budget_exhausted = true;
                    break;
                }
                let message_weight_map =
                    if let Some(cached) = message_weight_map_cache.get(&message.block_hash) {
                        cached.clone()
                    } else {
                        let weight_map_result = tokio::time::timeout(
                            step_timeout,
                            Self::message_weight_map_f(&message, dag),
                        )
                        .await;
                        let Ok(Ok(fetched)) = weight_map_result else {
                            continue;
                        };
                        message_weight_map_cache.insert(message.block_hash.clone(), fetched.clone());
                        fetched
                    };
                if let Some(agreement) = Self::record_agreement(&message_weight_map, &validator) {
                    mk_agreements_stream.push((message, message_weight_map, agreement));
                }
            }

            // proceed to main parents
            let next_layer: Vec<(Validator, BlockMetadata)> = current_layer
                .iter()
                .filter_map(|(validator, message)| {
                    message
                        .parents
                        .first()
                        .and_then(|main_parent_hash| dag.lookup_unsafe(main_parent_hash).ok())
                        // filter out empty results when no main parent and those out of scope
                        .filter(|meta| meta.block_number > curr_lfb_height)
                        .map(|meta| (validator.clone(), meta))
                })
                .collect();

            // Check if we should continue
            // If next is empty, we've processed all layers and should stop
            if next_layer.is_empty() {
                break;
            }

            current_layer = next_layer;
        }
        let agreements_count = mk_agreements_stream.len();

        // Step 2: Process agreements stream

        // while recording each agreement in agreements map
        let mut full_agreements_map: HashMap<BlockMetadata, WeightMap> = HashMap::new();
        let mut mapaccumulate_stream: Vec<(BlockMetadata, WeightMap)> =
            Vec::with_capacity(mk_agreements_stream.len());
        for (message, message_weight_map, agreement) in mk_agreements_stream {
            let entry = full_agreements_map.entry(message.clone()).or_default();
            assert!(
                !entry.contains_key(&agreement.0),
                "Logical error during finalization: message {:?} got duplicate agreement.",
                message.block_hash
            );
            entry.insert(agreement.0, agreement.1);
            mapaccumulate_stream.push((message, message_weight_map));
        }

        // output only target message of current agreement
        let agreements_with_accumulator: Vec<(BlockMetadata, WeightMap, WeightMap)> =
            mapaccumulate_stream
                .into_iter()
                .map(|(message, message_weight_map)| {
                    let agreeing_weight_map = full_agreements_map.get(&message).cloned().unwrap();
                    (message, message_weight_map, agreeing_weight_map)
                })
                .collect();

        // filter only messages that cannot be orphaned
        let filtered_agreements: Vec<(BlockMetadata, WeightMap, WeightMap, i64, usize)> =
            agreements_with_accumulator
                .into_iter()
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
        let mut dedup_filtered_agreements: HashMap<
            BlockHash,
            (BlockMetadata, WeightMap, WeightMap, i64, usize),
        > = HashMap::new();
        for (message, message_weight_map, agreeing_weight_map, stake_sum, agreeing_size) in
            filtered_agreements
        {
            dedup_filtered_agreements
                .entry(message.block_hash.clone())
                .or_insert((
                    message,
                    message_weight_map,
                    agreeing_weight_map,
                    stake_sum,
                    agreeing_size,
                ));
        }
        let mut deduped_filtered_agreements: Vec<(BlockMetadata, WeightMap, WeightMap, i64, usize)> =
            dedup_filtered_agreements.into_values().collect();
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
        let capped_agreements: Vec<(BlockMetadata, WeightMap, WeightMap)> = deduped_filtered_agreements
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
            "Finalizer timing: latest_messages={}, layers_visited={}, agreements={}, filtered_agreements={}, deduped_filtered_agreements={}, candidate_cap={}, ranking_strategy={}, candidate_capped={}, upper_bound_pruned={}, upper_bound_passed={}, max_ft_upper_bound={:.6}, clique_evals={}, clique_ms={}, total_ms={}, budget_ms={}, step_timeout_ms={}, budget_exhausted={}, found_new_lfb={}",
            latest_messages_count,
            layers_visited,
            agreements_count,
            filtered_agreements_count,
            deduped_filtered_agreements_count,
            max_clique_candidates,
            ranking_strategy.as_str(),
            candidate_capped,
            upper_bound_pruned_count,
            upper_bound_passed_count,
            max_ft_upper_bound,
            clique_eval_count,
            clique_started.elapsed().as_millis(),
            total_started.elapsed().as_millis(),
            FINALIZER_WORK_BUDGET_MS,
            FINALIZER_STEP_TIMEOUT_MS,
            budget_exhausted,
            lfb_result.is_some()
        );

        Ok(lfb_result)
    }
}
