// See casper/src/main/scala/coop/rchain/casper/safety/CliqueOracle.scala

use itertools::Itertools;
use std::collections::BTreeMap;

use crate::rust::safety_oracle::MIN_FAULT_TOLERANCE;
use crate::rust::util::clique::Clique;
use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::{block_hash::BlockHash, validator::Validator};

use shared::rust::store::key_value_store::KvStoreError;

pub struct CliqueOracle;

type M = BlockHash; // type for message
type V = Validator; // type for message creator/validator
type WeightMap = BTreeMap<V, i64>; // stakes per message creator

impl CliqueOracle {
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
                    .map(|parent_meta| parent_meta.weight_map),
                None => Ok(meta.weight_map),
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
        validator_a: &V,
        validator_b: &V,
        dag: &KeyValueDagRepresentation,
        target_msg: &M,
    ) -> Result<bool, KvStoreError> {
        /// Check if there might be eventual disagreement between validators
        async fn might_eventually_disagree(
            lm_b: &M,
            lm_a_j_b: &M,
            dag: &KeyValueDagRepresentation,
            target_msg: &M,
        ) -> Result<bool, KvStoreError> {
            // self justification of lmAjB or lmAjB itself. Used as a stopper for traversal
            // TODO not completely clear why try to use self justification and not just message itself
            let stopper = dag
                .self_justification(lm_a_j_b)?
                .unwrap_or_else(|| lm_a_j_b.clone());

            // stream of self justifications till stopper
            let full_chain = dag.self_justification_chain(lm_b.clone())?;
            let chain: Vec<BlockHash> = full_chain
                .into_iter()
                .take_while(|hash| hash != &stopper)
                .collect();

            // if message is not in main chain with target - this is disagreement
            let disagreements: Vec<_> = chain
                .iter()
                .filter_map(|hash| match dag.is_in_main_chain(target_msg, hash) {
                    Ok(false) => Some(hash.clone()),
                    _ => None,
                })
                .collect();

            // its enough only one disagreement found to declare output true
            Ok(!disagreements.is_empty())
        }

        // justification from `validatorB` as per latest message of `validatorA`
        let lm_a_option = dag.latest_message_hash(validator_a);
        let lm_a_j_b_option = match lm_a_option {
            Some(lm_a) => {
                let lm_a_metadata = dag.lookup_unsafe(&lm_a)?;
                lm_a_metadata
                    .justifications
                    .iter()
                    .find(|j| j.validator == *validator_b)
                    .map(|j| j.latest_block_hash.clone())
            }
            None => None,
        };

        let lm_b_option = dag.latest_message_hash(validator_b);

        // Check for disagreement, if messages are not found return false
        match (lm_b_option, lm_a_j_b_option) {
            (Some(lm_b), Some(lm_a_j_b)) => {
                might_eventually_disagree(&lm_b, &lm_a_j_b, dag, target_msg)
                    .await
                    .map(|r| !r)
            }
            _ => Ok(false),
        }
    }

    async fn compute_max_clique_weight(
        target_msg: &M,
        agreeing_weight_map: &WeightMap,
        dag: &KeyValueDagRepresentation,
    ) -> Result<i64, KvStoreError> {
        // Using tracing events for async - Span[F].traceI("compute-max-clique-weight") from Scala
        tracing::debug!(target: "f1r3fly.casper.safety.clique-oracle", "compute-max-clique-weight-started");
        /// across combination of validators compute pairs that do not have disagreement
        async fn compute_agreeing_validator_pairs(
            target_msg: &M,
            agreeing_weight_map: &WeightMap,
            dag: &KeyValueDagRepresentation,
        ) -> Result<Vec<(V, V)>, KvStoreError> {
            let case_a_b_pairs: Vec<(V, V)> = agreeing_weight_map
                .keys()
                .cloned()
                .combinations(2)
                .map(|pair| (pair[0].clone(), pair[1].clone()))
                .collect();

            let mut result = Vec::new();
            for (a, b) in case_a_b_pairs {
                let no_a_b_disagreement =
                    CliqueOracle::never_eventually_see_disagreement(&a, &b, dag, target_msg)
                        .await?;
                let no_b_a_disagreement =
                    CliqueOracle::never_eventually_see_disagreement(&b, &a, dag, target_msg)
                        .await?;

                if no_a_b_disagreement && no_b_a_disagreement {
                    result.push((a, b));
                }
            }

            Ok(result)
        }

        let edges = compute_agreeing_validator_pairs(target_msg, agreeing_weight_map, dag).await?;
        let max_weight = Clique::find_maximum_clique_by_weight(&edges, agreeing_weight_map);

        Ok(max_weight)
    }

    pub async fn compute_output(
        target_msg: &M,
        message_weight_map: &WeightMap,
        agreeing_weight_map: &WeightMap,
        dag: &KeyValueDagRepresentation,
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
            let max_clique_weight =
                CliqueOracle::compute_max_clique_weight(target_msg, agreeing_weight_map, dag)
                    .await? as f32;

            let result = (max_clique_weight * 2.0 - total_stake) / total_stake;

            Ok(result)
        }
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

            let mut agreeing_map = BTreeMap::new();
            for (validator, weight) in weight_map.iter() {
                if agree(validator, target_msg, dag).await? {
                    agreeing_map.insert(validator.clone(), *weight);
                }
            }

            Ok(agreeing_map)
        }

        let target_msg_not_in_dag = {
            let error_message = format!(
                "Fault tolerance for non existing message {:?} requested.",
                target_msg
            );
            tracing::warn!("{}", error_message);
            Ok(MIN_FAULT_TOLERANCE)
        };

        let do_compute = async {
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
        };

        if dag.contains(target_msg) {
            do_compute.await
        } else {
            target_msg_not_in_dag
        }
    }
}
