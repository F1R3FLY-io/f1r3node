use std::collections::HashMap;

use block_storage::rust::{
    dag::block_dag_key_value_storage::{BlockDagKeyValueStorage, KeyValueDagRepresentation},
    key_value_block_store::KeyValueBlockStore,
    util::doubly_linked_dag_operations::BlockDependencyDag,
};
use models::rust::{
    block_hash::BlockHash,
    casper::{pretty_printer::PrettyPrinter, protocol::casper_message::BlockMessage},
    equivocation_record::{EquivocationDiscoveryStatus, EquivocationRecord},
    validator::Validator,
};
use rspace_plus_plus::rspace::history::Either;
use shared::rust::{dag::dag_ops, store::key_value_store::KvStoreError};

use crate::rust::{
    block_status::{BlockError, InvalidBlock, ValidBlock},
    util::proto_util,
    ValidBlockProcessing,
};

/// Equivocation detection logic for blockchain consensus
pub struct EquivocationDetector;

impl EquivocationDetector {
    pub async fn check_equivocations(
        block_buffer_dependency_dag: &BlockDependencyDag,
        block: &BlockMessage,
        dag: &KeyValueDagRepresentation,
    ) -> Result<ValidBlockProcessing, KvStoreError> {
        log::info!("Calculate checkEquivocations.");

        let maybe_latest_message_of_creator_hash = dag.latest_message_hash(&block.sender);
        let maybe_creator_justification = Self::creator_justification_hash(block);
        let is_not_equivocation =
            maybe_creator_justification == maybe_latest_message_of_creator_hash;

        if is_not_equivocation {
            Ok(Either::Right(ValidBlock::Valid))
        } else if Self::requested_as_dependency(block, block_buffer_dependency_dag) {
            Ok(Either::Left(BlockError::Invalid(
                InvalidBlock::AdmissibleEquivocation,
            )))
        } else {
            let sender = PrettyPrinter::build_string_no_limit(&block.sender);
            let creator_justification_hash = PrettyPrinter::build_string_no_limit(
                &maybe_creator_justification.unwrap_or_default(),
            );
            let latest_message_of_creator = PrettyPrinter::build_string_no_limit(
                &maybe_latest_message_of_creator_hash.unwrap_or_default(),
            );

            log::warn!(
                "Ignorable equivocation: sender is {}, creator justification is {}, latest message of creator is {}",
                sender,
                creator_justification_hash,
                latest_message_of_creator
            );

            Ok(Either::Left(BlockError::Invalid(
                InvalidBlock::IgnorableEquivocation,
            )))
        }
    }

    pub fn requested_as_dependency(
        block: &BlockMessage,
        block_buffer_dependency_dag: &BlockDependencyDag,
    ) -> bool {
        use models::rust::block_hash::BlockHashSerde;

        let block_hash_serde = BlockHashSerde(block.block_hash.clone());
        block_buffer_dependency_dag
            .parent_to_child_adjacency_list
            .contains_key(&block_hash_serde)
    }

    pub fn creator_justification_hash(block: &BlockMessage) -> Option<BlockHash> {
        proto_util::creator_justification_block_message(block)
            .map(|justification| justification.latest_block_hash)
    }

    pub async fn check_neglected_equivocations_with_update(
        block: &BlockMessage,
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        genesis: &BlockMessage,
        block_dag_storage: &mut BlockDagKeyValueStorage,
    ) -> Result<ValidBlockProcessing, KvStoreError> {
        log::info!("Calculate checkNeglectedEquivocationsWithUpdate");

        let neglected_equivocation_detected = Self::is_neglected_equivocation_detected_with_update(
            block,
            dag,
            block_store,
            genesis,
            block_dag_storage,
        )
        .await?;

        let status = if neglected_equivocation_detected {
            Either::Left(BlockError::Invalid(InvalidBlock::NeglectedEquivocation))
        } else {
            Either::Right(ValidBlock::Valid)
        };

        Ok(status)
    }

    async fn is_neglected_equivocation_detected_with_update(
        block: &BlockMessage,
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        genesis: &BlockMessage,
        block_dag_storage: &mut BlockDagKeyValueStorage,
    ) -> Result<bool, KvStoreError> {
        let equivocations = block_dag_storage.equivocation_records()?;

        for equivocation_record in equivocations {
            let neglected = Self::update_equivocations_tracker(
                block,
                dag,
                block_store,
                &equivocation_record,
                genesis,
                block_dag_storage,
            )?;

            if neglected {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn update_equivocations_tracker(
        block: &BlockMessage,
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        equivocation_record: &EquivocationRecord,
        genesis: &BlockMessage,
        block_dag_storage: &mut BlockDagKeyValueStorage,
    ) -> Result<bool, KvStoreError> {
        let equivocation_discovery_status = Self::get_equivocation_discovery_status(
            block,
            dag,
            block_store,
            equivocation_record,
            genesis,
        )?;

        let neglected_equivocation_detected = match equivocation_discovery_status {
            EquivocationDiscoveryStatus::EquivocationNeglected => true,
            EquivocationDiscoveryStatus::EquivocationDetected => {
                block_dag_storage.update_equivocation_record(
                    equivocation_record.clone(),
                    block.block_hash.clone(),
                )?;
                log::info!(
                    "Equivocation detected and tracker updated for block {}",
                    PrettyPrinter::build_string_no_limit(&block.block_hash)
                );
                false
            }
            EquivocationDiscoveryStatus::EquivocationOblivious => false,
        };

        Ok(neglected_equivocation_detected)
    }

    fn get_equivocation_discovery_status(
        block: &BlockMessage,
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        equivocation_record: &EquivocationRecord,
        genesis: &BlockMessage,
    ) -> Result<EquivocationDiscoveryStatus, KvStoreError> {
        let equivocating_validator = &equivocation_record.equivocator;
        let latest_messages = Self::to_latest_message_hashes(&block.justifications);
        let bonds = proto_util::bonds(block);

        // Find the bond for the equivocating validator
        let maybe_equivocating_validator_bond = bonds
            .iter()
            .find(|bond| &bond.validator == equivocating_validator);

        match maybe_equivocating_validator_bond {
            Some(bond) => Self::get_equivocation_discovery_status_for_bonded_validator(
                dag,
                block_store,
                equivocation_record,
                &latest_messages,
                bond.stake,
                genesis,
            ),
            None => {
                /*
                 * Since block has dropped equivocatingValidator from the bonds, it has acknowledged the equivocation.
                 * The combination of Validate.transactions and Validate.bondsCache ensure that you can only drop
                 * validators through transactions to the proof of stake contract.
                 */
                Ok(EquivocationDiscoveryStatus::EquivocationDetected)
            }
        }
    }

    fn get_equivocation_discovery_status_for_bonded_validator(
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        equivocation_record: &EquivocationRecord,
        latest_messages: &HashMap<Validator, BlockHash>,
        stake: i64,
        genesis: &BlockMessage,
    ) -> Result<EquivocationDiscoveryStatus, KvStoreError> {
        if stake > 0 {
            let equivocation_detectable = Self::is_equivocation_detectable(
                dag,
                block_store,
                latest_messages,
                equivocation_record,
                &Vec::new(),
                genesis,
            )?;

            if equivocation_detectable {
                Ok(EquivocationDiscoveryStatus::EquivocationNeglected)
            } else {
                Ok(EquivocationDiscoveryStatus::EquivocationOblivious)
            }
        } else {
            // TODO: This case is not necessary if assert(stake > 0) in the PoS contract - OLD
            Ok(EquivocationDiscoveryStatus::EquivocationDetected)
        }
    }

    fn to_latest_message_hashes(
        justifications: &[models::rust::casper::protocol::casper_message::Justification],
    ) -> HashMap<Validator, BlockHash> {
        justifications
            .iter()
            .map(|justification| {
                (
                    justification.validator.clone(),
                    justification.latest_block_hash.clone(),
                )
            })
            .collect()
    }

    fn is_equivocation_detectable(
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        latest_messages: &HashMap<Validator, BlockHash>,
        equivocation_record: &EquivocationRecord,
        equivocation_children: &Vec<BlockMessage>,
        genesis: &BlockMessage,
    ) -> Result<bool, KvStoreError> {
        let latest_messages_vec: Vec<(Validator, BlockHash)> = latest_messages
            .iter()
            .map(|(v, h)| (v.clone(), h.clone()))
            .collect();

        match latest_messages_vec.split_first() {
            None => Ok(false),
            Some(((_, justification_block_hash), remainder)) => {
                Self::is_equivocation_detectable_after_viewing_block(
                    dag,
                    block_store,
                    justification_block_hash,
                    equivocation_record,
                    equivocation_children,
                    remainder,
                    genesis,
                )
            }
        }
    }

    fn is_equivocation_detectable_after_viewing_block(
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        justification_block_hash: &BlockHash,
        equivocation_record: &EquivocationRecord,
        equivocation_children: &Vec<BlockMessage>,
        remainder: &[(Validator, BlockHash)],
        genesis: &BlockMessage,
    ) -> Result<bool, KvStoreError> {
        if equivocation_record
            .equivocation_detected_block_hashes
            .contains(justification_block_hash)
        {
            Ok(true)
        } else {
            let justification_block = block_store.get_unsafe(justification_block_hash);
            Self::is_equivocation_detectable_through_children(
                dag,
                block_store,
                equivocation_record,
                equivocation_children,
                remainder,
                &justification_block,
                genesis,
            )
        }
    }

    fn is_equivocation_detectable_through_children(
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        equivocation_record: &EquivocationRecord,
        equivocation_children: &Vec<BlockMessage>,
        remainder: &[(Validator, BlockHash)],
        justification_block: &BlockMessage,
        genesis: &BlockMessage,
    ) -> Result<bool, KvStoreError> {
        let equivocating_validator = &equivocation_record.equivocator;
        let equivocation_base_block_seq_num = equivocation_record.equivocation_base_block_seq_num;

        let updated_equivocation_children = Self::maybe_add_equivocation_child(
            dag,
            block_store,
            justification_block,
            equivocating_validator,
            equivocation_base_block_seq_num.into(),
            equivocation_children,
            genesis,
        )?;

        if updated_equivocation_children.len() > 1 {
            Ok(true)
        } else {
            let remainder_map: HashMap<Validator, BlockHash> = remainder
                .iter()
                .map(|(v, h)| (v.clone(), h.clone()))
                .collect();

            Self::is_equivocation_detectable(
                dag,
                block_store,
                &remainder_map,
                equivocation_record,
                &updated_equivocation_children,
                genesis,
            )
        }
    }

    fn maybe_add_equivocation_child(
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        justification_block: &BlockMessage,
        equivocating_validator: &Validator,
        equivocation_base_block_seq_num: i64,
        equivocation_children: &Vec<BlockMessage>,
        genesis: &BlockMessage,
    ) -> Result<Vec<BlockMessage>, KvStoreError> {
        // TODO: Is this a safe check? Or should I just check block hash? - OLD
        if justification_block.block_hash == genesis.block_hash {
            return Ok(equivocation_children.clone());
        }

        if justification_block.sender == *equivocating_validator {
            let justification_seq_num = i64::from(justification_block.seq_num);
            if justification_seq_num > equivocation_base_block_seq_num {
                Self::add_equivocation_child(
                    dag,
                    block_store,
                    justification_block,
                    equivocation_base_block_seq_num,
                    equivocation_children,
                )
            } else {
                Ok(equivocation_children.clone())
            }
        } else {
            let latest_messages =
                Self::to_latest_message_hashes(&justification_block.justifications);

            match latest_messages.get(equivocating_validator) {
                Some(latest_equivocating_validator_block_hash) => {
                    let latest_equivocating_validator_block =
                        block_store.get_unsafe(latest_equivocating_validator_block_hash);

                    let latest_seq_num = i64::from(latest_equivocating_validator_block.seq_num);
                    if latest_seq_num > equivocation_base_block_seq_num {
                        Self::add_equivocation_child(
                            dag,
                            block_store,
                            &latest_equivocating_validator_block,
                            equivocation_base_block_seq_num,
                            equivocation_children,
                        )
                    } else {
                        Ok(equivocation_children.clone())
                    }
                }
                None => {
                    Err(KvStoreError::KeyNotFound(
                        "justificationBlock is missing justification pointers to equivocatingValidator even though justificationBlock isn't a part of equivocationDetectedBlockHashes for this equivocation record.".to_string()
                    ))
                }
            }
        }
    }

    fn add_equivocation_child(
        dag: &KeyValueDagRepresentation,
        block_store: &KeyValueBlockStore,
        justification_block: &BlockMessage,
        equivocation_base_block_seq_num: i64,
        equivocation_children: &Vec<BlockMessage>,
    ) -> Result<Vec<BlockMessage>, KvStoreError> {
        let target_seq_num = equivocation_base_block_seq_num + 1;

        match Self::find_creator_justification_ancestor_with_seq_num(
            dag,
            justification_block,
            target_seq_num,
        )? {
            Some(equivocation_child_hash) => {
                let equivocation_child = block_store.get_unsafe(&equivocation_child_hash);
                let mut updated_children = equivocation_children.clone();
                updated_children.push(equivocation_child);
                Ok(updated_children)
            }
            None => {
                Err(KvStoreError::KeyNotFound(
                    "creator justification ancestor with lower sequence number hasn't been added to the blockDAG yet.".to_string()
                ))
            }
        }
    }

    fn find_creator_justification_ancestor_with_seq_num(
        dag: &KeyValueDagRepresentation,
        block: &BlockMessage,
        seq_num: i64,
    ) -> Result<Option<BlockHash>, KvStoreError> {
        if i64::from(block.seq_num) == seq_num {
            return Ok(Some(block.block_hash.clone()));
        }

        let start_nodes = vec![block.block_hash.clone()];

        let neighbors = |block_hash: &BlockHash| -> Vec<BlockHash> {
            proto_util::get_creator_justification_as_list_until_goal_in_memory(
                dag,
                block_hash,
                |_| false,
            )
            .unwrap_or_else(|_| Vec::new())
        };

        let traversal_result = dag_ops::bf_traverse(start_nodes, neighbors);
        let target_validator = &block.sender;

        for candidate_hash in traversal_result {
            match dag.lookup_unsafe(&candidate_hash) {
                Ok(candidate_metadata) => {
                    if i64::from(candidate_metadata.sequence_number) == seq_num
                        && candidate_metadata.sender == *target_validator
                    {
                        return Ok(Some(candidate_hash));
                    }
                }
                Err(_) => {
                    continue;
                }
            }
        }

        Ok(None)
    }
}
