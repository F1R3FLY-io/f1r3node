// See casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{Duration, Instant};

use crate::rust::safety_oracle::MIN_FAULT_TOLERANCE;
use crate::rust::util::clique::Clique;
use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::{block_hash::BlockHash, validator::Validator};

use shared::rust::env;
use shared::rust::store::key_value_store::KvStoreError;

pub struct CliqueOracle;

type M = BlockHash; // type for message
type V = Validator; // type for message creator/validator
type WeightMap = HashMap<V, i64>; // stakes per message creator
const COOPERATIVE_YIELD_CHECK_INTERVAL: usize = 8;
const COOPERATIVE_YIELD_TIMESLICE_MS: u64 = 1;
const COOPERATIVE_YIELD_CHECK_INTERVAL_ENV: &str = "F1R3_CLIQUE_YIELD_CHECK_INTERVAL";
const COOPERATIVE_YIELD_TIMESLICE_MS_ENV: &str = "F1R3_CLIQUE_YIELD_TIMESLICE_MS";

pub struct CliqueOracleRunCache {
    latest_message_cache: BTreeMap<V, Option<M>>,
    latest_justifications_cache: BTreeMap<V, BTreeMap<V, M>>,
    self_justification_cache: BTreeMap<M, Option<M>>,
    in_main_chain_cache: BTreeMap<(M, M), bool>,
    yield_check_interval: usize,
    yield_timeslice: Duration,
}

impl CliqueOracle {
    pub fn new_run_cache() -> CliqueOracleRunCache {
        CliqueOracleRunCache {
            latest_message_cache: BTreeMap::new(),
            latest_justifications_cache: BTreeMap::new(),
            self_justification_cache: BTreeMap::new(),
            in_main_chain_cache: BTreeMap::new(),
            yield_check_interval: Self::cooperative_yield_check_interval(),
            yield_timeslice: Duration::from_millis(Self::cooperative_yield_timeslice_ms()),
        }
    }

    fn cooperative_yield_check_interval() -> usize {
        env::var_or_filtered(
            COOPERATIVE_YIELD_CHECK_INTERVAL_ENV,
            COOPERATIVE_YIELD_CHECK_INTERVAL,
            |v: &usize| *v > 0,
        )
    }

    fn cooperative_yield_timeslice_ms() -> u64 {
        env::var_or(
            COOPERATIVE_YIELD_TIMESLICE_MS_ENV,
            COOPERATIVE_YIELD_TIMESLICE_MS,
        )
    }

    /// weight map of main parent (fallbacks to message itself if no parents)
    /// TODO - why not use local weight map but seek for parent?
    /// P.S. This is related to the fact that we create latest message for newly bonded validator
    /// equal to message where bonding deploy has been submitted. So stake from validator that did not create anything is
    /// put behind this message. So here is one more place where this logic makes things more complex.
    pub async fn get_corresponding_weight_map(
        target_msg: &M,
        dag: &KeyValueDagRepresentation,
    ) -> Result<WeightMap, KvStoreError> {
        dag.lookup_unsafe(target_msg)
            .and_then(|meta| match meta.parents.first() {
                Some(main_parent) => dag
                    .lookup_unsafe(main_parent)
                    .map(|parent_meta| parent_meta.weight_map.into_iter().collect()),
                None => Ok(meta.weight_map.into_iter().collect()),
            })
    }

    /// If two validators will never have disagreement on target message
    ///
    /// Prerequisite for this is that latest messages from a and b both are in main chain with target message
    ///
    /// ```text
    ///     a    b
    ///     *    *  <- lmB
    ///      \   *
    ///       \  *
    ///        \ *
    ///         \*
    ///          *  <- lmAjB
    /// ```
    ///
    /// 1. get justification of validator b as per latest message of a (lmAjB)
    /// 2. check if any self justifications between latest message of b (lmB) and lmAjB are NOT in main chain
    ///    with target message.
    ///
    ///    If one found - this is a source of disagreement.
    async fn never_eventually_see_disagreement(
        lm_b: &M,
        lm_a_j_b: &M,
        dag: &KeyValueDagRepresentation,
        target_msg: &M,
        yield_check_interval: usize,
        yield_timeslice: Duration,
        self_justification_cache: &mut BTreeMap<M, Option<M>>,
        in_main_chain_cache: &mut BTreeMap<(M, M), bool>,
    ) -> Result<bool, KvStoreError> {
        /// Check if there might be eventual disagreement between validators
        async fn might_eventually_disagree(
            lm_b: &M,
            lm_a_j_b: &M,
            dag: &KeyValueDagRepresentation,
            target_msg: &M,
            self_justification_cache: &mut BTreeMap<M, Option<M>>,
            in_main_chain_cache: &mut BTreeMap<(M, M), bool>,
            yield_check_interval: usize,
            yield_timeslice: Duration,
        ) -> Result<bool, KvStoreError> {
            // self justification of lmAjB or lmAjB itself. Used as a stopper for traversal
            // TODO not completely clear why try to use self justification and not just message itself
            let stopper = if let Some(cached) = self_justification_cache.get(lm_a_j_b) {
                cached.clone().unwrap_or_else(|| lm_a_j_b.clone())
            } else {
                let value = dag.self_justification(lm_a_j_b)?;
                self_justification_cache.insert(lm_a_j_b.clone(), value.clone());
                value.unwrap_or_else(|| lm_a_j_b.clone())
            };

            // Traverse only until stopper instead of materializing full history to genesis.
            let mut current = if let Some(cached) = self_justification_cache.get(lm_b) {
                cached.clone()
            } else {
                let value = dag.self_justification(lm_b)?;
                self_justification_cache.insert(lm_b.clone(), value.clone());
                value
            };
            let mut last_yield = Instant::now();
            let mut idx: usize = 0;
            while let Some(hash) = current {
                if hash == stopper {
                    break;
                }
                if idx % yield_check_interval == 0 && last_yield.elapsed() >= yield_timeslice {
                    tokio::task::yield_now().await;
                    last_yield = Instant::now();
                }
                idx += 1;
                let in_main_chain_key = (target_msg.clone(), hash.clone());
                let is_in_main_chain =
                    if let Some(cached) = in_main_chain_cache.get(&in_main_chain_key) {
                        *cached
                    } else {
                        let value = dag.is_in_main_chain(target_msg, &hash)?;
                        in_main_chain_cache.insert(in_main_chain_key, value);
                        value
                    };
                if !is_in_main_chain {
                    return Ok(true);
                }

                current = if let Some(cached) = self_justification_cache.get(&hash) {
                    cached.clone()
                } else {
                    let value = dag.self_justification(&hash)?;
                    self_justification_cache.insert(hash, value.clone());
                    value
                };
            }
            Ok(false)
        }

        might_eventually_disagree(
            lm_b,
            lm_a_j_b,
            dag,
            target_msg,
            self_justification_cache,
            in_main_chain_cache,
            yield_check_interval,
            yield_timeslice,
        )
        .await
        .map(|r| !r)
    }

    async fn compute_max_clique_weight(
        target_msg: &M,
        agreeing_weight_map: &WeightMap,
        dag: &KeyValueDagRepresentation,
        run_cache: &mut CliqueOracleRunCache,
    ) -> Result<i64, KvStoreError> {
        // Using tracing events for async - Span[F].traceI("compute-max-clique-weight") from Scala
        tracing::debug!(target: "f1r3fly.casper.safety.clique-oracle", "compute-max-clique-weight-started");
        /// across combination of validators compute pairs that do not have disagreement
        async fn compute_agreeing_validator_pairs(
            target_msg: &M,
            agreeing_weight_map: &WeightMap,
            dag: &KeyValueDagRepresentation,
            run_cache: &mut CliqueOracleRunCache,
        ) -> Result<Vec<(V, V)>, KvStoreError> {
            let yield_check_interval = run_cache.yield_check_interval;
            let yield_timeslice = run_cache.yield_timeslice;

            let agreeing_validators: BTreeSet<V> = agreeing_weight_map.keys().cloned().collect();
            let mut latest_justifications_cache: BTreeMap<V, BTreeMap<V, M>> = BTreeMap::new();
            let mut pairwise_latest_messages: BTreeMap<V, M> = BTreeMap::new();
            // Conservative pruning: if validator has no latest message or no justifications
            // to other agreeing validators, it cannot form an agreeing edge with anyone.
            let mut pairwise_validators: Vec<V> = Vec::new();
            for validator in agreeing_validators.iter() {
                let latest = if let Some(cached) = run_cache.latest_message_cache.get(validator) {
                    cached.clone()
                } else {
                    let value = dag.latest_message_hash(validator);
                    run_cache
                        .latest_message_cache
                        .insert(validator.clone(), value.clone());
                    value
                };
                let Some(latest) = latest else {
                    continue;
                };
                pairwise_latest_messages.insert(validator.clone(), latest.clone());

                let all_justifications =
                    if let Some(cached) = run_cache.latest_justifications_cache.get(validator) {
                        cached.clone()
                    } else {
                        let metadata = dag.lookup_unsafe(&latest)?;
                        let all: BTreeMap<V, M> = metadata
                            .justifications
                            .iter()
                            .map(|j| (j.validator.clone(), j.latest_block_hash.clone()))
                            .collect();
                        run_cache
                            .latest_justifications_cache
                            .insert(validator.clone(), all.clone());
                        all
                    };

                let relevant_justifications: BTreeMap<V, M> = all_justifications
                    .iter()
                    .filter_map(|(validator, hash)| {
                        agreeing_validators
                            .contains(validator)
                            .then_some((validator.clone(), hash.clone()))
                    })
                    .collect();
                if relevant_justifications.is_empty() {
                    continue;
                }

                latest_justifications_cache.insert(validator.clone(), relevant_justifications);
                pairwise_validators.push(validator.clone());
            }

            let mut result = Vec::new();
            let mut last_yield = Instant::now();
            let mut pair_idx: usize = 0;
            for i in 0..pairwise_validators.len() {
                for j in (i + 1)..pairwise_validators.len() {
                    // Keep this loop cooperative so higher-level timeouts can preempt
                    // expensive clique evaluation on deep DAGs.
                    if pair_idx % yield_check_interval == 0
                        && last_yield.elapsed() >= yield_timeslice
                    {
                        tokio::task::yield_now().await;
                        last_yield = Instant::now();
                    }
                    pair_idx += 1;

                    let a = &pairwise_validators[i];
                    let b = &pairwise_validators[j];
                    // Both directions of latest-message justification must exist, otherwise
                    // disagreement check returns false immediately and pair cannot be in clique.
                    let Some(lm_a_j_b) = latest_justifications_cache.get(a).and_then(|m| m.get(b))
                    else {
                        continue;
                    };
                    let Some(lm_b_j_a) = latest_justifications_cache.get(b).and_then(|m| m.get(a))
                    else {
                        continue;
                    };
                    let Some(lm_a) = pairwise_latest_messages.get(a) else {
                        continue;
                    };
                    let Some(lm_b) = pairwise_latest_messages.get(b) else {
                        continue;
                    };
                    let no_a_b_disagreement = CliqueOracle::never_eventually_see_disagreement(
                        lm_b,
                        lm_a_j_b,
                        dag,
                        target_msg,
                        yield_check_interval,
                        yield_timeslice,
                        &mut run_cache.self_justification_cache,
                        &mut run_cache.in_main_chain_cache,
                    )
                    .await?;
                    let no_b_a_disagreement = CliqueOracle::never_eventually_see_disagreement(
                        lm_a,
                        lm_b_j_a,
                        dag,
                        target_msg,
                        yield_check_interval,
                        yield_timeslice,
                        &mut run_cache.self_justification_cache,
                        &mut run_cache.in_main_chain_cache,
                    )
                    .await?;

                    if no_a_b_disagreement && no_b_a_disagreement {
                        result.push((a.clone(), b.clone()));
                    }
                }
            }

            Ok(result)
        }

        let edges =
            compute_agreeing_validator_pairs(target_msg, agreeing_weight_map, dag, run_cache)
                .await?;
        let max_weight = Clique::find_maximum_clique_by_weight(&edges, agreeing_weight_map);

        Ok(max_weight)
    }

    pub async fn compute_output_with_cache(
        target_msg: &M,
        message_weight_map: &WeightMap,
        agreeing_weight_map: &WeightMap,
        dag: &KeyValueDagRepresentation,
        run_cache: &mut CliqueOracleRunCache,
    ) -> Result<f32, KvStoreError> {
        let total_stake = message_weight_map.values().sum::<i64>() as f32;
        assert!(
            total_stake > 0.0,
            "Long overflow when computing total stake"
        );

        // If less than 1/2+ of stake agrees on message - it can be orphaned
        if (agreeing_weight_map.values().sum::<i64>() as f32) <= total_stake / 2.0 {
            Ok(MIN_FAULT_TOLERANCE)
        } else {
            let max_clique_weight = CliqueOracle::compute_max_clique_weight(
                target_msg,
                agreeing_weight_map,
                dag,
                run_cache,
            )
            .await? as f32;

            let result = (max_clique_weight * 2.0 - total_stake) / total_stake;

            Ok(result)
        }
    }

    pub async fn compute_output(
        target_msg: &M,
        message_weight_map: &WeightMap,
        agreeing_weight_map: &WeightMap,
        dag: &KeyValueDagRepresentation,
    ) -> Result<f32, KvStoreError> {
        let mut run_cache = Self::new_run_cache();
        Self::compute_output_with_cache(
            target_msg,
            message_weight_map,
            agreeing_weight_map,
            dag,
            &mut run_cache,
        )
        .await
    }

    pub async fn normalized_fault_tolerance(
        target_msg: &M,
        dag: &KeyValueDagRepresentation,
    ) -> Result<f32, KvStoreError> {
        // Using tracing events for async - Span[F].traceI("normalized-fault-tolerance") from Scala
        tracing::debug!(target: "f1r3fly.casper.safety.clique-oracle", "normalized-fault-tolerance-started");
        /// weight map containing only validators that agree on the message
        async fn agreeing_weight_map_f(
            weight_map: &WeightMap,
            target_msg: &M,
            dag: &KeyValueDagRepresentation,
        ) -> Result<WeightMap, KvStoreError> {
            async fn agree(
                validator: &V,
                message: &M,
                dag: &KeyValueDagRepresentation,
            ) -> Result<bool, KvStoreError> {
                dag.latest_message_hash(validator)
                    .map_or(Ok(false), |hash| dag.is_in_main_chain(message, &hash))
            }

            let mut agreeing_map = HashMap::new();
            for (validator, weight) in weight_map.iter() {
                if agree(validator, target_msg, dag).await? {
                    agreeing_map.insert(validator.clone(), *weight);
                }
            }

            Ok(agreeing_map)
        }

        if dag.contains(target_msg) {
            tracing::debug!("Calculating fault tolerance for {:?}.", target_msg);
            let full_weight_map =
                CliqueOracle::get_corresponding_weight_map(target_msg, dag).await?;
            let agreeing_weight_map =
                agreeing_weight_map_f(&full_weight_map, target_msg, dag).await?;
            let result = CliqueOracle::compute_output(
                target_msg,
                &full_weight_map,
                &agreeing_weight_map,
                dag,
            )
            .await?;

            Ok(result)
        } else {
            tracing::warn!(
                "Fault tolerance for non existing message {:?} requested.",
                target_msg
            );
            Ok(MIN_FAULT_TOLERANCE)
        }
    }
}
