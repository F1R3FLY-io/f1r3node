// See models/src/test/scala/coop/rchain/models/blockImplicits.scala

use proptest::{prelude::*, strategy::ValueTree, test_runner::TestRunner};
use rand::prelude::*;

use crypto::rust::signatures::{
    secp256k1::Secp256k1, signatures_alg::SignaturesAlg, signed::Signed,
};

use crate::rhoapi::PCost;

use super::{
    block::state_hash::{self, StateHash},
    block_hash::{self, BlockHash},
    casper::protocol::casper_message::{
        BlockMessage, Body, Bond, DeployData, F1r3flyState, Header, Justification, ProcessedDeploy,
        ProcessedSystemDeploy,
    },
    validator::{self, Validator},
};

pub fn block_hash_gen() -> impl Strategy<Value = BlockHash> {
    prop::collection::vec(any::<u8>(), block_hash::LENGTH)
        .prop_map(|byte_vec| prost::bytes::Bytes::from(byte_vec))
}

pub fn state_hash_gen() -> impl Strategy<Value = StateHash> {
    prop::collection::vec(any::<u8>(), state_hash::LENGTH)
        .prop_map(|byte_vec| prost::bytes::Bytes::from(byte_vec))
}

pub fn validator_gen() -> impl Strategy<Value = Validator> {
    prop::collection::vec(any::<u8>(), validator::LENGTH)
        .prop_map(|byte_vec| prost::bytes::Bytes::from(byte_vec))
}

pub fn bond_gen() -> impl Strategy<Value = Bond> {
    let validator_gen = prop::collection::vec(any::<u8>(), validator::LENGTH)
        .prop_map(|byte_vec| prost::bytes::Bytes::from(byte_vec));
    let stake_gen = 1i64..=1024i64;
    (validator_gen, stake_gen).prop_map(|(validator, stake)| Bond { validator, stake })
}

pub fn justification_gen() -> impl Strategy<Value = Justification> {
    let validator_gen = prop::collection::vec(any::<u8>(), validator::LENGTH)
        .prop_map(|byte_vec| prost::bytes::Bytes::from(byte_vec));
    let block_hash_gen = block_hash_gen();
    (validator_gen, block_hash_gen).prop_map(|(validator, latest_block_hash)| Justification {
        validator: validator.into(),
        latest_block_hash: latest_block_hash.into(),
    })
}

fn alpha_num_char() -> impl Strategy<Value = char> {
    prop::char::ranges(vec!['a'..='z', 'A'..='Z', '0'..='9'].into_iter().collect())
}

pub fn signed_deploy_data_gen() -> impl Strategy<Value = Signed<DeployData>> {
    let term_length = 32..=1024;
    let term = prop::collection::vec(alpha_num_char(), term_length)
        .prop_map(|chars| chars.into_iter().collect::<String>());

    (any::<i64>(), term, any::<String>()).prop_map(|(timestamp, term, shard_id)| {
        let secp256k1 = Secp256k1;
        let (sec, _) = secp256k1.new_key_pair();

        Signed::create(
            DeployData {
                time_stamp: timestamp,
                phlo_price: 1,
                phlo_limit: 9000000,
                valid_after_block_number: 1,
                language: "rholang".to_string(),
                term,
                shard_id,
            },
            Box::new(secp256k1),
            sec,
        )
        .expect("Failed to create signed deploy data")
    })
}

pub fn processed_deploy_gen() -> impl Strategy<Value = ProcessedDeploy> {
    let deploy_data_gen = signed_deploy_data_gen();
    deploy_data_gen.prop_map(|deploy_data| ProcessedDeploy {
        deploy: deploy_data,
        cost: PCost { cost: 0 },
        deploy_log: Vec::new(),
        is_failed: false,
        system_deploy_error: None,
    })
}

pub fn block_element_gen(
    set_block_number: Option<i64>,
    set_seq_number: Option<i32>,
    set_pre_state_hash: Option<StateHash>,
    set_post_state_hash: Option<StateHash>,
    set_validator: Option<Validator>,
    set_version: Option<i64>,
    set_timestamp: Option<i64>,
    set_parents_hash_list: Option<Vec<BlockHash>>,
    set_justifications: Option<Vec<Justification>>,
    set_deploys: Option<Vec<ProcessedDeploy>>,
    set_sys_deploys: Option<Vec<ProcessedSystemDeploy>>,
    set_bonds: Option<Vec<Bond>>,
    set_shard_id: Option<String>,
    hash_f: Option<Box<dyn Fn(BlockMessage) -> BlockHash>>,
) -> impl Strategy<Value = BlockMessage> {
    // Generate individual components using existing or provided values
    let pre_state_hash_gen = match set_pre_state_hash {
        Some(hash) => Just(hash).boxed(),
        None => state_hash_gen().boxed(),
    };

    let post_state_hash_gen = match set_post_state_hash {
        Some(hash) => Just(hash).boxed(),
        None => state_hash_gen().boxed(),
    };

    let parents_hash_list_gen = match set_parents_hash_list {
        Some(list) => Just(list).boxed(),
        None => prop::collection::vec(block_hash_gen(), 0..5).boxed(),
    };

    let justifications_gen = match set_justifications {
        Some(list) => Just(list).boxed(),
        None => prop::collection::vec(justification_gen(), 0..5).boxed(),
    };

    let deploys_gen = match set_deploys {
        Some(list) => Just(list).boxed(),
        None => prop::collection::vec(processed_deploy_gen(), 0..5).boxed(),
    };

    let bonds_gen = match set_bonds {
        Some(list) => Just(list).boxed(),
        None => prop::collection::vec(bond_gen(), 10).boxed(),
    };

    let validator_gen = match set_validator {
        Some(v) => Just(v).boxed(),
        None => bonds_gen
            .clone()
            .prop_map(|bonds| {
                let mut rng = rand::rng();
                bonds
                    .choose(&mut rng)
                    .map(|bond| bond.validator.clone())
                    .map(|b| b)
                    .unwrap_or_else(|| {
                        validator_gen()
                            .boxed()
                            .new_tree(&mut TestRunner::default())
                            .unwrap()
                            .current()
                    })
            })
            .boxed(),
    };

    let version = set_version.unwrap_or(1);
    let timestamp_gen = match set_timestamp {
        Some(t) => Just(t).boxed(),
        None => any::<i64>().boxed(),
    };
    let shard_id = set_shard_id.unwrap_or_else(|| "root".to_string());
    let block_number = set_block_number.unwrap_or(0);
    let seq_number = set_seq_number.unwrap_or(0);

    (
        pre_state_hash_gen,
        post_state_hash_gen,
        parents_hash_list_gen,
        justifications_gen,
        deploys_gen,
        bonds_gen,
        validator_gen,
        timestamp_gen,
    )
        .prop_map(
            move |(
                pre_state_hash,
                post_state_hash,
                parents_hash_list,
                justifications,
                deploys,
                bonds,
                validator,
                timestamp,
            )| {
                let block = BlockMessage {
                    block_hash: prost::bytes::Bytes::new(),
                    header: Header {
                        parents_hash_list: parents_hash_list.into_iter().map(Into::into).collect(),
                        timestamp,
                        version,
                        extra_bytes: prost::bytes::Bytes::new(),
                    },
                    body: Body {
                        state: F1r3flyState {
                            pre_state_hash,
                            post_state_hash,
                            bonds,
                            block_number,
                        },
                        deploys,
                        system_deploys: set_sys_deploys.clone().unwrap_or_default(),
                        rejected_deploys: Vec::new(),
                        extra_bytes: prost::bytes::Bytes::new(),
                    },
                    justifications,
                    sender: validator.into(),
                    seq_num: seq_number,
                    sig: prost::bytes::Bytes::new(),
                    sig_algorithm: String::new(),
                    shard_id: shard_id.clone(),
                    extra_bytes: prost::bytes::Bytes::new(),
                };

                // Apply custom hash function if provided, otherwise generate random hash
                let block_hash = match hash_f.as_ref() {
                    Some(f) => f(block.clone()),
                    None => block_hash_gen()
                        .new_tree(&mut TestRunner::default())
                        .unwrap()
                        .current(),
                };

                BlockMessage {
                    block_hash: block_hash.into(),
                    ..block
                }
            },
        )
}

pub fn block_elements_with_parents_gen(
    genesis: BlockMessage,
    min_size: usize,
    max_size: usize,
) -> impl Strategy<Value = Vec<BlockMessage>> {
    let bonds = genesis.body.state.bonds.clone();

    prop::collection::vec(any::<u8>(), min_size..max_size).prop_flat_map(move |_| {
        let bonds = bonds.clone();
        let blocks: Vec<BlockMessage> = Vec::new();

        Just(blocks).prop_perturb(move |mut blocks, _| {
            let mut rng = rand::rng();
            let new_block = block_element_gen(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(bonds.clone()),
                None,
                None,
            )
            .new_tree(&mut TestRunner::default())
            .unwrap()
            .current();

            // Select random parents from existing blocks
            let parent_hashes: Vec<BlockHash> = blocks
                .choose_multiple(&mut rng, blocks.len().max(1) / 2)
                .map(|b| b.block_hash.clone().into())
                .collect();

            // Create modified block with parent hashes
            let block = BlockMessage {
                header: Header {
                    parents_hash_list: parent_hashes.into_iter().map(Into::into).collect(),
                    ..new_block.header
                },
                ..new_block
            };

            blocks.push(block);
            blocks
        })
    })
}

pub fn block_with_new_hashes_gen(
    block_elements: Vec<BlockMessage>,
) -> impl Strategy<Value = Vec<BlockMessage>> {
    prop::collection::vec(block_hash_gen(), block_elements.len()).prop_map(move |new_hashes| {
        block_elements
            .iter()
            .zip(new_hashes)
            .map(|(block, hash)| BlockMessage {
                block_hash: hash.into(),
                ..block.clone()
            })
            .collect()
    })
}

pub fn get_random_block(
    set_block_number: Option<i64>,
    set_seq_number: Option<i32>,
    set_pre_state_hash: Option<StateHash>,
    set_post_state_hash: Option<StateHash>,
    set_validator: Option<Validator>,
    set_version: Option<i64>,
    set_timestamp: Option<i64>,
    set_parents_hash_list: Option<Vec<BlockHash>>,
    set_justifications: Option<Vec<Justification>>,
    set_deploys: Option<Vec<ProcessedDeploy>>,
    set_sys_deploys: Option<Vec<ProcessedSystemDeploy>>,
    set_bonds: Option<Vec<Bond>>,
    set_shard_id: Option<String>,
    hash_f: Option<Box<dyn Fn(BlockMessage) -> BlockHash>>,
) -> BlockMessage {
    block_element_gen(
        set_block_number,
        set_seq_number,
        set_pre_state_hash,
        set_post_state_hash,
        set_validator,
        set_version,
        set_timestamp,
        set_parents_hash_list,
        set_justifications,
        set_deploys,
        set_sys_deploys,
        set_bonds,
        set_shard_id,
        hash_f,
    )
    .new_tree(&mut TestRunner::default())
    .unwrap()
    .current()
}

pub fn get_random_block_default() -> BlockMessage {
    get_random_block(
        None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    )
}
