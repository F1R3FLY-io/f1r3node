// See casper/src/main/scala/coop/rchain/casper/finality/Finalizer.scala

use std::collections::{BTreeMap, HashMap};

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

type WeightMap = BTreeMap<Validator, i64>;

/// Message that is agreed on + weight of this agreement.
#[derive(Debug, Clone)]
pub struct MessageAgreement {
    pub message: BlockMetadata,
    pub message_weight_map: WeightMap,
    pub stake_agreed: WeightMap,
}

impl Finalizer {
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
        /*
         * Stream of agreements passed down from all latest messages to main parents.
         * Starts with agreements of latest message on themselves.
         *
         * The goal here is to create stream of agreements breadth first, so on each step agreements by all
         * validator are recorded, and only after that next level of main parents is visited.
         */
        let lms = dag.latest_messages()?;

        // sort latest messages by agreeing validator to ensure random ordering does not change output
        let mut sorted_latest_messages: Vec<(Validator, BlockMetadata)> = lms.into_iter().collect();
        sorted_latest_messages.sort_by(|(v1, _), (v2, _)| v1.cmp(v2));

        // Step 1: Generate stream of agreements
        // Scala's unfoldLoopEval outputs the current layer, then checks if next is non-empty to continue.
        // So ALL layers are output, including the last one where next would be empty.
        let mut mk_agreements_stream = Vec::new();
        let mut current_layer = sorted_latest_messages;

        loop {
            // output current visits - process agreements for this layer
            let out = current_layer.clone();

            // evalMap: map visits to message agreements: validator v agrees on message m
            // The agreement is on the MESSAGE itself (from the current layer), not on its parent.
            for (validator, message) in out {
                if let Ok(message_weight_map) = Self::message_weight_map_f(&message, dag).await {
                    if let Some(agreement) = Self::record_agreement(&message_weight_map, &validator)
                    {
                        mk_agreements_stream.push(MessageAgreement {
                            message: message,
                            message_weight_map,
                            stake_agreed: [agreement].into_iter().collect(),
                        });
                    }
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

        // Step 2: Process agreements stream

        // while recording each agreement in agreements map
        let mut full_agreements_map: HashMap<BlockMetadata, WeightMap> = HashMap::new();
        let mapaccumulate_stream: Vec<(BlockMetadata, WeightMap)> = mk_agreements_stream
            .into_iter()
            .map(|agreement| {
                let MessageAgreement {
                    message,
                    message_weight_map,
                    stake_agreed,
                } = agreement;

                let cur_val = full_agreements_map
                    .get(&message)
                    .cloned()
                    .unwrap_or_default();

                assert!(
                    stake_agreed.keys().all(|v| !cur_val.contains_key(v)),
                    "Logical error during finalization: message {:?} got duplicate agreement.",
                    message.block_hash
                );

                let new_val = cur_val.into_iter().chain(stake_agreed).collect();

                full_agreements_map.insert(message.clone(), new_val);
                (message, message_weight_map)
            })
            .collect();

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
        let filtered_agreements: Vec<(BlockMetadata, WeightMap, WeightMap)> =
            agreements_with_accumulator
                .into_iter()
                .filter(|(_, message_weight_map, agreeing_weight_map)| {
                    Self::cannot_be_orphaned(message_weight_map, agreeing_weight_map)
                })
                .collect();

        // compute fault tolerance
        let mut fault_tolerance_results: Vec<(BlockMetadata, f32)> = Vec::new();
        for (message, message_weight_map, agreeing_weight_map) in filtered_agreements {
            if let Ok(fault_tolerance) = CliqueOracle::compute_output(
                &message.block_hash,
                &message_weight_map,
                &agreeing_weight_map,
                dag,
            )
            .await
            {
                fault_tolerance_results.push((message, fault_tolerance));
            }
        }

        // first candidate that meets finalization criteria is new LFB
        let lfb_result = if let Some((lfb, _)) = fault_tolerance_results
            .into_iter()
            .filter(|(_, fault_tolerance)| *fault_tolerance > fault_tolerance_threshold)
            .next()
        {
            let lfb_hash = lfb.block_hash;
            // Only process blocks that aren't already finalized
            if !dag.is_finalized(&lfb_hash) {
                new_lfb_found_effect(lfb_hash.clone()).await?;
            }
            Some(lfb_hash)
        } else {
            None
        };

        Ok(lfb_result)
    }
}
