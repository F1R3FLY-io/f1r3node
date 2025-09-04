// See casper/src/test/scala/coop/rchain/casper/batch2/ValidateTest.scala

use crate::helper::{
    block_dag_storage_fixture::{with_genesis, with_storage},
    block_generator::{create_block, create_genesis_block, create_validator_block},
    block_util::generate_validator,
    unlimited_parents_estimator_fixture::UnlimitedParentsEstimatorFixture,
};
use crate::util::genesis_builder::{GenesisBuilder, DEFAULT_VALIDATOR_PKS};
use casper::rust::{
    block_status::{InvalidBlock, ValidBlock},
    casper::CasperSnapshot,
    util::proto_util,
    validate::Validate,
    validator_identity::ValidatorIdentity,
};
use crypto::rust::{
    private_key::PrivateKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg, signed::Signed},
};
use models::rust::casper::protocol::casper_message::{
    BlockMessage, Bond, DeployData, ProcessedDeploy,
};
use prost::bytes::Bytes;
use std::collections::HashMap;
use tempfile::Builder;

use crate::util::rholang::resources::mk_test_rnode_store_manager;
use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::block_status::BlockError;
use casper::rust::genesis::genesis::Genesis;
use casper::rust::util::construct_deploy;
use casper::rust::util::rholang::{interpreter_util, runtime_manager::RuntimeManager};
use casper_message::Justification;
use models::rust::block_implicits::get_random_block;
use models::rust::casper::protocol::casper_message;
use rspace_plus_plus::rspace::history::Either;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;

const SHARD_ID: &str = "root-shard";

fn mk_casper_snapshot(dag: KeyValueDagRepresentation) -> CasperSnapshot {
    CasperSnapshot::new(dag)
}

fn create_chain(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    length: usize,
    bonds: Vec<Bond>,
) -> BlockMessage {
    let genesis = create_genesis_block(
        block_store,
        block_dag_storage,
        None,
        Some(bonds.clone()),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let _final_block = (1..length).fold(genesis.clone(), |block, _| {
        create_block(
            block_store,
            block_dag_storage,
            vec![block.block_hash.clone()],
            &genesis,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    });

    genesis
}

fn create_chain_with_round_robin_validators(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    length: usize,
    validator_length: usize,
) -> BlockMessage {
    let validator_round_robin_cycle = std::iter::repeat(0..validator_length).flatten();

    let validators: Vec<Bytes> = std::iter::repeat_with(|| generate_validator(None))
        .take(validator_length)
        .collect();

    let genesis = create_genesis_block(
        block_store,
        block_dag_storage,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let fold_result = (0..length).zip(validator_round_robin_cycle).fold(
        (genesis.clone(), genesis.clone(), HashMap::new()),
        |acc, (_, validator_num)| {
            let (genesis, block, latest_messages) = acc;
            let creator = validators[validator_num].clone();
            let bnext = create_block(
                block_store,
                block_dag_storage,
                vec![block.block_hash.clone()],
                &genesis,
                Some(creator.clone()),
                None,
                Some(latest_messages.clone()),
                None,
                None,
                None,
                None,
                None,
                None,
            );

            let latest_messages_next = {
                let mut updated = latest_messages.clone();
                updated.insert(bnext.sender.clone(), bnext.block_hash.clone());
                updated
            };

            (genesis, bnext, latest_messages_next)
        },
    );

    fold_result.0 // .map(_._1) in Scala
}

fn signed_block(
    i: usize,
    private_key: &PrivateKey,
    block_dag_storage: &mut IndexedBlockDagStorage,
) -> BlockMessage {
    let secp256k1 = Secp256k1;
    let pk = secp256k1.to_public(private_key);
    let mut block = block_dag_storage.lookup_by_id_unsafe(i as i64);
    let dag = block_dag_storage.get_representation();
    let sender = Bytes::copy_from_slice(&pk.bytes);
    let latest_message_opt = dag.latest_message(&sender).unwrap_or(None);
    let seq_num =
        latest_message_opt.map_or(0, |block_metadata| block_metadata.sequence_number as i32) + 1;

    block.seq_num = seq_num;
    ValidatorIdentity::new(private_key).sign_block(&block)
}

fn with_block_number(block: &BlockMessage, n: i64) -> BlockMessage {
    let mut new_state = block.body.state.clone();
    new_state.block_number = n;
    let mut new_block = block.clone();
    new_block.body.state = new_state;
    new_block
}

//helper functions
fn with_sig_algorithm(block: &BlockMessage, algorithm: &str) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.sig_algorithm = algorithm.to_string();
    new_block
}

fn with_sender(block: &BlockMessage, sender: &Bytes) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.sender = sender.clone();
    new_block
}

fn with_sig(block: &BlockMessage, sig: &Bytes) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.sig = sig.clone();
    new_block
}

fn with_seq_num(block: &BlockMessage, seq_num: i32) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.seq_num = seq_num;
    new_block
}

fn with_shard_id(block: &BlockMessage, shard_id: &str) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.shard_id = shard_id.to_string();
    new_block
}

fn with_post_state_hash(block: &BlockMessage, post_state_hash: &Bytes) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.body.state.post_state_hash = post_state_hash.clone();
    new_block
}

fn with_block_hash(block: &BlockMessage, block_hash: &Bytes) -> BlockMessage {
    let mut new_block = block.clone();
    new_block.block_hash = block_hash.clone();
    new_block
}

fn modified_timestamp_header(block: &BlockMessage, timestamp: i64) -> BlockMessage {
    let mut modified_timestamp_header = block.header.clone();
    modified_timestamp_header.timestamp = timestamp;

    let mut block_with_modified_header = block.clone();
    block_with_modified_header.header = modified_timestamp_header;
    block_with_modified_header
}

fn create_signed_deploy_with_data(
    updated_deploy_data: DeployData,
) -> Result<Signed<DeployData>, String> {
    let secp = Secp256k1;
    Signed::create(
        updated_deploy_data,
        Box::new(secp),
        construct_deploy::DEFAULT_SEC.clone(),
    )
}

fn create_justifications(pairs: Vec<(Bytes, Bytes)>) -> HashMap<Bytes, Bytes> {
    pairs.into_iter().collect()
}

// Many tests use checks that must be added later
// TODO: Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.

#[tokio::test]
async fn block_signature_validation_should_return_false_on_unknown_algorithms() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 2, vec![]);

        let unknown_algorithm = "unknownAlgorithm";
        let rsa = "RSA";

        let block0 =
            with_sig_algorithm(&block_dag_storage.lookup_by_id_unsafe(0), unknown_algorithm);
        let block1 = with_sig_algorithm(&block_dag_storage.lookup_by_id_unsafe(1), rsa);

        let result0 = Validate::block_signature(&block0);
        assert_eq!(result0, false);

        let result1 = Validate::block_signature(&block1);
        assert_eq!(result1, false);

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.last.contains(s"signature algorithm $unknownAlgorithm is unsupported") should be(true)
        // log.warns.last.contains(s"signature algorithm $rsa is unsupported") should be(true)
    })
    .await
}

#[tokio::test]
async fn block_signature_validation_should_return_false_on_invalid_secp256k1_signatures() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let secp256k1 = Secp256k1;
        let (private_key, public_key) = secp256k1.new_key_pair();

        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 6, vec![]);
        let (_wrong_sk, wrong_pk) = secp256k1.new_key_pair();

        assert_ne!(
            public_key.bytes, wrong_pk.bytes,
            "Public keys should be different"
        );
        let empty = Bytes::new();
        let invalid_key = hex::decode("abcdef1234567890").unwrap().into();

        let block0 = with_sender(
            &signed_block(0, &private_key, &mut block_dag_storage),
            &empty,
        );

        let block1 = with_sender(
            &signed_block(1, &private_key, &mut block_dag_storage),
            &invalid_key,
        );

        let block2 = with_sender(
            &signed_block(2, &private_key, &mut block_dag_storage),
            &Bytes::copy_from_slice(&wrong_pk.bytes),
        );

        let block3 = with_sig(
            &signed_block(3, &private_key, &mut block_dag_storage),
            &empty,
        );

        let block4 = with_sig(
            &signed_block(4, &private_key, &mut block_dag_storage),
            &invalid_key,
        );

        let block5 = with_sig(
            &signed_block(5, &private_key, &mut block_dag_storage),
            &block0.sig,
        ); //wrong sig

        let blocks = vec![block0, block1, block2, block3, block4, block5];

        for (i, block) in blocks.iter().enumerate() {
            let result = Validate::block_signature(&block);
            assert_eq!(result, false, "Block {} should have invalid signature", i);
        }

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(blocks.length)
        // log.warns.forall(_.contains("signature is invalid")) should be(true)
    })
    .await
}

#[tokio::test]
async fn block_signature_validation_should_return_true_on_valid_secp256k1_signatures() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let n = 6;
        let secp256k1 = Secp256k1;
        let (private_key, _public_key) = secp256k1.new_key_pair();

        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, n, vec![]);

        let condition = (0..n).all(|i| {
            let block = signed_block(i, &private_key, &mut block_dag_storage);
            Validate::block_signature(&block)
        });

        assert_eq!(condition, true);

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns should be(Nil)
    })
    .await
}

#[tokio::test]
async fn timestamp_validation_should_not_accept_blocks_with_future_time() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 1, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(0);

        // modifiedTimestampHeader = block.header.copy(timestamp = 99999999)
        // Note: In Scala tests LogicalTime starts from 0, but in Rust we use real Unix timestamps
        // So we need a timestamp that's actually in the future relative to current time
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let future_timestamp = current_time + 20000; // 20 seconds in future (> DRIFT of 15 seconds)

        let _dag = block_dag_storage.get_representation();

        let result_invalid = Validate::timestamp(
            &modified_timestamp_header(&block, future_timestamp),
            &mut block_store,
        );
        assert_eq!(
            result_invalid,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidTimestamp))
        );

        let result_valid = Validate::timestamp(&block, &mut block_store);
        assert_eq!(result_valid, Either::Right(ValidBlock::Valid));

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // _ = log.warns.size should be(1)
        // result = log.warns.head.contains("block timestamp") should be(true)
    })
    .await
}

#[tokio::test]
async fn timestamp_validation_should_not_accept_blocks_that_were_published_before_parent_time() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 2, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(1);
        let modified_timestamp_header = modified_timestamp_header(&block, -1);

        let _dag = block_dag_storage.get_representation();

        let result_invalid = Validate::timestamp(&modified_timestamp_header, &mut block_store);
        assert_eq!(
            result_invalid,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidTimestamp))
        );

        let result_valid = Validate::timestamp(&block, &mut block_store);
        assert_eq!(result_valid, Either::Right(ValidBlock::Valid));

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(1)
        // log.warns.head.contains("block timestamp") should be(true)
    })
    .await
}

#[tokio::test]
async fn block_number_validation_should_only_accept_0_as_the_number_for_a_block_with_no_parents() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 1, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(0);
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let result_invalid =
            Validate::block_number(&with_block_number(&block, 1), &mut casper_snapshot);
        assert_eq!(
            result_invalid,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockNumber))
        );

        let result_valid = Validate::block_number(&block, &mut casper_snapshot);
        assert_eq!(result_valid, Either::Right(ValidBlock::Valid));

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(1)
        // log.warns.head.contains("not zero, but block has no parents") should be(true)
    })
    .await
}

#[tokio::test]
async fn block_number_validation_should_return_false_for_non_sequential_numbering() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 2, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(1);
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let result_invalid =
            Validate::block_number(&with_block_number(&block, 17), &mut casper_snapshot);
        assert_eq!(
            result_invalid,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockNumber))
        );

        let result_valid = Validate::block_number(&block, &mut casper_snapshot);
        assert_eq!(result_valid, Either::Right(ValidBlock::Valid));

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(1)
        // log.warns.head.contains("is not one more than maximum parent number") should be(true)
    })
    .await
}

#[tokio::test]
async fn block_number_validation_should_return_true_for_sequential_numbering() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let n = 6;
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, n, vec![]);
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        // Test each block in the chain for sequential numbering
        let condition = (0..n).all(|i| {
            let block = block_dag_storage.lookup_by_id_unsafe(i as i64);
            let result = Validate::block_number(&block, &mut casper_snapshot);
            result == Either::Right(ValidBlock::Valid)
        });

        assert_eq!(condition, true);

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns should be(Nil)
    })
    .await
}

#[tokio::test]
async fn block_number_validation_should_correctly_validate_a_multi_parent_block_where_the_parents_have_different_block_numbers(
) {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let create_block_with_number =
            |block_store: &mut KeyValueBlockStore,
             block_dag_storage: &mut IndexedBlockDagStorage,
             n: i64,
             _genesis: &BlockMessage,
             parent_hashes: Vec<Bytes>| {
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64;

                let block = models::rust::block_implicits::get_random_block(
                    Some(n),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(current_time),
                    Some(parent_hashes),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(Box::new(|block| proto_util::hash_block(&block))),
                );

                block_store
                    .put(block.block_hash.clone(), &block)
                    .expect("Failed to put block");
                block_dag_storage
                    .insert(&block, false, false)
                    .expect("Failed to insert block");

                block
            };

        // Note we need to create a useless chain to satisfy the assert in TopoSort
        let genesis = create_chain(&mut block_store, &mut block_dag_storage, 8, vec![]);

        let b1 = create_block_with_number(
            &mut block_store,
            &mut block_dag_storage,
            3,
            &genesis,
            vec![],
        );

        let b2 = create_block_with_number(
            &mut block_store,
            &mut block_dag_storage,
            7,
            &genesis,
            vec![],
        );

        let b3 = create_block_with_number(
            &mut block_store,
            &mut block_dag_storage,
            8,
            &genesis,
            vec![b1.block_hash.clone(), b2.block_hash.clone()],
        );

        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let s1 = Validate::block_number(&b3, &mut casper_snapshot);
        assert_eq!(s1, Either::Right(ValidBlock::Valid));

        let s2 = Validate::block_number(&with_block_number(&b3, 4), &mut casper_snapshot);
        assert_eq!(
            s2,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockNumber))
        );
    })
    .await
}

#[tokio::test]
async fn future_deploy_validation_should_work() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let deploy = construct_deploy::basic_processed_deploy(0, None).unwrap();

        let updated_processed_deploy = {
            let mut updated_deploy_data = deploy.deploy.data.clone();
            updated_deploy_data.valid_after_block_number = -1;

            let updated_signed_deploy = create_signed_deploy_with_data(updated_deploy_data)
                .expect("Failed to create signed deploy");

            ProcessedDeploy {
                deploy: updated_signed_deploy,
                ..deploy
            }
        };

        let block = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            Some(vec![updated_processed_deploy]),
            None,
            None,
            None,
            None,
        );

        let status = Validate::future_transaction(&block);

        assert_eq!(status, Either::Right(ValidBlock::Valid));
    })
    .await
}

#[tokio::test]
async fn future_deploy_validation_should_not_accept_blocks_with_a_deploy_for_a_future_block_number()
{
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let deploy = construct_deploy::basic_processed_deploy(0, None).unwrap();

        let updated_processed_deploy = {
            let mut updated_deploy_data = deploy.deploy.data.clone();
            updated_deploy_data.valid_after_block_number = i64::MAX;

            let updated_signed_deploy = create_signed_deploy_with_data(updated_deploy_data)
                .expect("Failed to create signed deploy");

            ProcessedDeploy {
                deploy: updated_signed_deploy,
                ..deploy
            }
        };

        let block_with_future_deploy = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            Some(vec![updated_processed_deploy]),
            None,
            None,
            None,
            None,
        );

        let status = Validate::future_transaction(&block_with_future_deploy);

        assert_eq!(
            status,
            Either::Left(BlockError::Invalid(InvalidBlock::ContainsFutureDeploy))
        );
    })
    .await
}

#[tokio::test]
async fn deploy_expiration_validation_should_work() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let deploy = construct_deploy::basic_processed_deploy(0, None).unwrap();
        let block = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            Some(vec![deploy]),
            None,
            None,
            None,
            None,
        );
        let status = Validate::transaction_expiration(&block, 10);
        assert_eq!(status, Either::Right(ValidBlock::Valid));
    })
    .await
}

#[tokio::test]
async fn deploy_expiration_validation_should_not_accept_blocks_with_a_deploy_that_is_expired() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let deploy = construct_deploy::basic_processed_deploy(0, None).unwrap();

        let updated_processed_deploy = {
            let mut updated_deploy_data = deploy.deploy.data.clone();
            updated_deploy_data.valid_after_block_number = i64::MIN;

            let updated_signed_deploy = create_signed_deploy_with_data(updated_deploy_data)
                .expect("Failed to create signed deploy");

            ProcessedDeploy {
                deploy: updated_signed_deploy,
                ..deploy
            }
        };

        let block_with_expired_deploy = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            Some(vec![updated_processed_deploy]),
            None,
            None,
            None,
            None,
        );

        let status = Validate::transaction_expiration(&block_with_expired_deploy, 10);
        assert_eq!(
            status,
            Either::Left(BlockError::Invalid(InvalidBlock::ContainsExpiredDeploy))
        );
    })
    .await
}

#[tokio::test]
async fn sequence_number_validation_should_only_accept_0_as_the_number_for_a_block_with_no_parents()
{
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 1, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(0);
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let block_with_seq_1 = with_seq_num(&block, 1);
        let result_invalid = Validate::sequence_number(&block_with_seq_1, &mut casper_snapshot);
        assert_eq!(
            result_invalid,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidSequenceNumber))
        );

        let result_valid = Validate::sequence_number(&block, &mut casper_snapshot);
        assert_eq!(result_valid, Either::Right(ValidBlock::Valid));

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(1)
    })
    .await
}

#[tokio::test]
async fn sequence_number_validation_should_return_false_for_non_sequential_numbering() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 2, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(1);
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let block_with_seq_1 = with_seq_num(&block, 1);
        let result = Validate::sequence_number(&block_with_seq_1, &mut casper_snapshot);
        assert_eq!(
            result,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidSequenceNumber))
        );

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(1)
    })
    .await
}

#[tokio::test]
async fn sequence_number_validation_should_return_true_for_sequential_numbering() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let n = 20;
        let validator_count = 3;
        let _genesis = create_chain_with_round_robin_validators(
            &mut block_store,
            &mut block_dag_storage,
            n,
            validator_count,
        );
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let condition = (0..n).all(|i| {
            let block = block_dag_storage.lookup_by_id_unsafe(i as i64);
            let result = Validate::sequence_number(&block, &mut casper_snapshot);
            result == Either::Right(ValidBlock::Valid)
        });

        assert_eq!(condition, true);

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns should be(Nil)
    })
    .await
}

#[tokio::test]
async fn repeat_deploy_validation_should_return_valid_for_empty_blocks() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 2, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(0);
        let block2 = block_dag_storage.lookup_by_id_unsafe(1);
        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let result1 = Validate::repeat_deploy(&block, &mut casper_snapshot, &mut block_store, 50);
        assert_eq!(result1, Either::Right(ValidBlock::Valid));

        let result2 = Validate::repeat_deploy(&block2, &mut casper_snapshot, &mut block_store, 50);
        assert_eq!(result2, Either::Right(ValidBlock::Valid));
    })
    .await
}

//Test 18: "Repeat deploy validation" should "not accept blocks with a repeated deploy"
// +
#[tokio::test]
async fn repeat_deploy_validation_should_not_accept_blocks_with_a_repeated_deploy() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let deploy = construct_deploy::basic_processed_deploy(0, None).unwrap();
        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            None,
            None,
            Some(vec![deploy.clone()]),
            None,
            None,
            None,
            None,
        );

        let block1 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            None,
            None,
            None,
            Some(vec![deploy]),
            None,
            None,
            None,
            None,
            None,
        );

        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let result = Validate::repeat_deploy(&block1, &mut casper_snapshot, &mut block_store, 50);
        assert_eq!(
            result,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidRepeatDeploy))
        );
    })
    .await
}

#[tokio::test]
async fn sender_validation_should_return_true_for_genesis_and_blocks_from_bonded_validators_and_false_otherwise(
) {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let validator = generate_validator(Some("Validator"));
        let impostor = generate_validator(Some("Impostor"));

        let _genesis = create_chain(
            &mut block_store,
            &mut block_dag_storage,
            3,
            vec![Bond {
                validator: validator.clone(),
                stake: 1,
            }],
        );

        let genesis = block_dag_storage.lookup_by_id_unsafe(0);
        let valid_block = with_sender(&block_dag_storage.lookup_by_id_unsafe(1), &validator);
        let invalid_block = with_sender(&block_dag_storage.lookup_by_id_unsafe(2), &impostor);
        let _dag = block_dag_storage.get_representation();
        let result_genesis =
            Validate::block_sender_has_weight(&genesis, &genesis, &mut block_store);
        assert_eq!(result_genesis.unwrap(), true);

        let result_valid =
            Validate::block_sender_has_weight(&valid_block, &genesis, &mut block_store);
        assert_eq!(result_valid.unwrap(), true);

        let result_invalid =
            Validate::block_sender_has_weight(&invalid_block, &genesis, &mut block_store);
        assert_eq!(result_invalid.unwrap(), false);
    })
    .await
}

#[tokio::test]
#[ignore]
async fn parent_validation_should_return_true_for_proper_justifications_and_false_otherwise() {
    use crate::helper::block_generator::step;

    let mut genesis_builder = GenesisBuilder::new();
    let genesis_context = genesis_builder
        .build_genesis_with_parameters(None)
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();

    with_genesis(
        genesis_context,
        |mut block_store, mut block_dag_storage, mut runtime_manager| async move {
            let validators = DEFAULT_VALIDATOR_PKS.clone();
            let v0 = validators[0].clone();
            let v1 = validators[1].clone();
            let v2 = validators[2].clone();

            let bonds = vec![
                Bond {
                    validator: Bytes::copy_from_slice(&v0.bytes),
                    stake: 1,
                },
                Bond {
                    validator: Bytes::copy_from_slice(&v1.bytes),
                    stake: 3,
                },
                Bond {
                    validator: Bytes::copy_from_slice(&v2.bytes),
                    stake: 5,
                },
            ];

            let b0 = genesis_block.clone();

            let b1 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b0.clone()],
                &b0,
                vec![],
                Bytes::copy_from_slice(&v0.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            );

            let b2 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b0.clone()],
                &b0,
                vec![],
                Bytes::copy_from_slice(&v1.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            );

            let b3 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b0.clone()],
                &b0,
                vec![],
                Bytes::copy_from_slice(&v2.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            );

            let b4 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b1.clone()],
                &b0,
                vec![b1.clone()],
                Bytes::copy_from_slice(&v0.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            );

            let b5 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b3.clone(), b2.clone(), b1.clone()],
                &b0,
                vec![b1.clone(), b2.clone(), b3.clone()],
                Bytes::copy_from_slice(&v1.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            );

            let b6 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b5.clone(), b4.clone()],
                &b0,
                vec![b1.clone(), b4.clone(), b5.clone()],
                Bytes::copy_from_slice(&v0.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            );

            let b7 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b4.clone()],
                &b0,
                vec![b1.clone(), b4.clone(), b5.clone()],
                Bytes::copy_from_slice(&v1.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            ); // not highest score parent

            let b8 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b1.clone(), b2.clone(), b3.clone()],
                &b0,
                vec![b1.clone(), b2.clone(), b3.clone()],
                Bytes::copy_from_slice(&v2.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            ); // parents wrong order

            let b9 = create_validator_block(
                &mut block_store,
                &mut block_dag_storage,
                vec![b6.clone()],
                &b0,
                vec![],
                Bytes::copy_from_slice(&v0.bytes),
                bonds.clone(),
                None,
                None,
                "root".to_string(),
            ); // empty justification

            // Step through each block to process them with RuntimeManager
            // This is equivalent to Scala's: step[Task](runtimeManager)(b1, b0)
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b1,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b2,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b3,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b4,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b5,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b6,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b7,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b8,
            )
            .await
            .unwrap();
            step(
                &mut block_dag_storage,
                &mut block_store,
                &mut runtime_manager,
                &b9,
            )
            .await
            .unwrap();

            let dag = block_dag_storage.get_representation();
            let mut casper_snapshot = mk_casper_snapshot(dag);

            // Create estimator like in Scala tests: Estimator[Task](Estimator.UnlimitedParents, None)
            let estimator = UnlimitedParentsEstimatorFixture::create_estimator();

            // Test Valid blocks (b0-b6): Validate::parents() should return Valid
            let result_b0 = Validate::parents(&b0, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b0, Either::Right(ValidBlock::Valid));

            let result_b1 = Validate::parents(&b1, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b1, Either::Right(ValidBlock::Valid));

            let result_b2 = Validate::parents(&b2, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b2, Either::Right(ValidBlock::Valid));

            let result_b3 = Validate::parents(&b3, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b3, Either::Right(ValidBlock::Valid));

            let result_b4 = Validate::parents(&b4, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b4, Either::Right(ValidBlock::Valid));

            let result_b5 = Validate::parents(&b5, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b5, Either::Right(ValidBlock::Valid));

            let result_b6 = Validate::parents(&b6, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(result_b6, Either::Right(ValidBlock::Valid));

            // Test Invalid blocks (b7-b9): Validate::parents() should return InvalidParents
            let result_b7 = Validate::parents(&b7, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(
                result_b7,
                Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents))
            );

            let result_b8 = Validate::parents(&b8, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(
                result_b8,
                Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents))
            );

            let result_b9 = Validate::parents(&b9, &b0, &mut casper_snapshot, &estimator).await;
            assert_eq!(
                result_b9,
                Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents))
            );

            // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
            // log.warns.forall(_.matches(".* block parents .* did not match estimate .* based on justification .*")) should be(true)
            // log.warns.size should be(3)
        },
    )
    .await
}

#[tokio::test]
async fn block_summary_validation_should_short_circuit_after_first_invalidity() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let _genesis = create_chain(&mut block_store, &mut block_dag_storage, 2, vec![]);
        let block = block_dag_storage.lookup_by_id_unsafe(1);
        let dag = block_dag_storage.get_representation();

        let secp256k1 = Secp256k1;
        let (sk, pk) = secp256k1.new_key_pair();
        let sender = Bytes::copy_from_slice(&pk.bytes);
        let latest_message_opt = dag.latest_message(&sender).unwrap_or(None);
        let _seq_num = latest_message_opt
            .map_or(0, |block_metadata| block_metadata.sequence_number as i32)
            + 1;

        let signed_block = ValidatorIdentity::new(&sk)
            .sign_block(&with_seq_num(&with_block_number(&block, 17), 1));

        let mut casper_snapshot = mk_casper_snapshot(dag);

        // Create estimator like in Scala tests: Estimator[Task](Estimator.UnlimitedParents, None)
        let estimator = UnlimitedParentsEstimatorFixture::create_estimator();

        let result = Validate::block_summary(
            &signed_block,
            &get_random_block(
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
                None,
                None,
                Some(Box::new(|block| proto_util::hash_block(&block))),
            ),
            &mut casper_snapshot,
            "root",
            i32::MAX,
            &estimator,
            &mut block_store,
        )
        .await;

        assert_eq!(
            result,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockNumber))
        );

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size should be(1)
    })
    .await
}

#[tokio::test]
async fn justification_follow_validation_should_return_valid_for_proper_justifications_and_failed_otherwise(
) {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v1 = generate_validator(Some("Validator One"));
        let v2 = generate_validator(Some("Validator Two"));
        let v1_bond = Bond {
            validator: v1.clone(),
            stake: 2,
        };
        let v2_bond = Bond {
            validator: v2.clone(),
            stake: 3,
        };
        let bonds = vec![v1_bond, v2_bond];

        let genesis = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b2 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            Some(v2.clone()),
            Some(bonds.clone()),
            Some(create_justifications(vec![
                (v1.clone(), genesis.block_hash.clone()),
                (v2.clone(), genesis.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b3 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            Some(v1.clone()),
            Some(bonds.clone()),
            Some(create_justifications(vec![
                (v1.clone(), genesis.block_hash.clone()),
                (v2.clone(), genesis.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b4 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone()],
            &genesis,
            Some(v2.clone()),
            Some(bonds.clone()),
            Some(create_justifications(vec![
                (v1.clone(), genesis.block_hash.clone()),
                (v2.clone(), b2.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b5 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b2.block_hash.clone()],
            &genesis,
            Some(v1.clone()),
            Some(bonds.clone()),
            Some(create_justifications(vec![
                (v1.clone(), b3.block_hash.clone()),
                (v2.clone(), b2.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let _b6 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b4.block_hash.clone()],
            &genesis,
            Some(v2.clone()),
            Some(bonds.clone()),
            Some(create_justifications(vec![
                (v1.clone(), b5.block_hash.clone()),
                (v2.clone(), b4.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b7 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b4.block_hash.clone()],
            &genesis,
            Some(v1.clone()),
            Some(vec![]),
            Some(create_justifications(vec![
                (v1.clone(), b5.block_hash.clone()),
                (v2.clone(), b4.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let _b8 = create_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b7.block_hash.clone()],
            &genesis,
            Some(v1.clone()),
            Some(bonds.clone()),
            Some(create_justifications(vec![
                (v1.clone(), b7.block_hash.clone()),
                (v2.clone(), b4.block_hash.clone()),
            ])),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        // Test blocks 1 to 6 should return Valid
        let condition = (1..=6).all(|i| {
            let block = block_dag_storage.lookup_by_id_unsafe(i as i64);
            let _dag = block_dag_storage.get_representation();
            let result = Validate::justification_follows(&block, &mut block_store);
            result == Either::Right(ValidBlock::Valid)
        });
        assert_eq!(condition, true);

        let block_id_7 = block_dag_storage.lookup_by_id_unsafe(7);
        let _dag = block_dag_storage.get_representation();
        let result = Validate::justification_follows(&block_id_7, &mut block_store);
        assert_eq!(
            result,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidFollows))
        );

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size shouldBe 1
        // log.warns.forall(_.contains("do not match the bonded validators")) shouldBe true
    })
    .await
}

#[tokio::test]
async fn justification_regression_validation_should_return_valid_for_proper_justifications_and_justification_regression_detected_otherwise(
) {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator 1"));
        let v1 = generate_validator(Some("Validator 2"));

        // bonds = List(v0, v1).zipWithIndex.map { case (v, i) => Bond(v, 2L * i.toLong + 1L) }
        let bonds = vec![
            Bond {
                validator: v0.clone(),
                stake: 1, // 2 * 0 (index) + 1 = 1
            },
            Bond {
                validator: v1.clone(),
                stake: 3, // 2 * 1(index) + 1 = 3
            },
        ];

        let b0 = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b1 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b0.clone(), b0.clone()],
            v0.clone(),
            bonds.clone(),
            None,
            None,
            SHARD_ID.to_string(),
        );

        let b2 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b1.clone()],
            &b0,
            vec![b1.clone(), b0.clone()],
            v0.clone(),
            bonds.clone(),
            None,
            None,
            SHARD_ID.to_string(),
        );

        let b3 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b2.clone(), b0.clone()],
            v1.clone(),
            bonds.clone(),
            None,
            None,
            SHARD_ID.to_string(),
        );

        let b4 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b3.clone()],
            &b0,
            vec![b2.clone(), b3.clone()],
            v1.clone(),
            bonds.clone(),
            None,
            None,
            SHARD_ID.to_string(),
        );

        let condition = (0..=4).all(|i| {
            let block = block_dag_storage.lookup_by_id_unsafe(i as i64);
            let dag = block_dag_storage.get_representation();
            let mut casper_snapshot = mk_casper_snapshot(dag);
            let result = Validate::justification_regressions(&block, &mut casper_snapshot);
            result == Either::Right(ValidBlock::Valid)
        });
        assert_eq!(condition, true);

        // The justification block for validator 0 should point to b2 or above.
        let justifications_with_regression = vec![
            Justification {
                validator: v0.clone(),
                latest_block_hash: b1.block_hash.clone(),
            },
            Justification {
                validator: v1.clone(),
                latest_block_hash: b4.block_hash.clone(),
            },
        ];

        let block_with_justification_regression = get_random_block(
            None,
            None,
            None,
            None,
            Some(v1.clone()),
            None,
            None,
            None,
            Some(justifications_with_regression),
            None,
            None,
            None,
            None,
            Some(Box::new(|block| proto_util::hash_block(&block))),
        );

        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let result = Validate::justification_regressions(
            &block_with_justification_regression,
            &mut casper_snapshot,
        );
        assert_eq!(
            result,
            Either::Left(BlockError::Invalid(InvalidBlock::JustificationRegression))
        );

        // Add log validation mechanism when LogStub mechanism from Scala will be implemented on Rust.
        // log.warns.size shouldBe 1
    })
    .await
}

#[tokio::test]
async fn justification_regression_validation_should_return_valid_for_regressive_invalid_blocks() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator 1"));
        let v1 = generate_validator(Some("Validator 2"));

        let bonds = vec![
            Bond {
                validator: v0.clone(),
                stake: 1, // 2 * 0 (index) + 1 = 1
            },
            Bond {
                validator: v1.clone(),
                stake: 3, // 2 * 1 (index) + 1 = 3
            },
        ];

        let b0 = create_genesis_block(
            &mut block_store,
            &mut block_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        let b1 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b0.clone(), b0.clone()],
            v0.clone(),
            bonds.clone(),
            Some(1),
            None,
            SHARD_ID.to_string(),
        );

        let b2 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b1.clone(), b0.clone()],
            v1.clone(),
            bonds.clone(),
            Some(1),
            None,
            SHARD_ID.to_string(),
        );

        let b3 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b1.clone(), b2.clone()],
            v0.clone(),
            bonds.clone(),
            Some(2),
            None,
            SHARD_ID.to_string(),
        );

        let b4 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b3.clone(), b2.clone()],
            v1.clone(),
            bonds.clone(),
            Some(2),
            None,
            SHARD_ID.to_string(),
        );

        let b5 = create_validator_block(
            &mut block_store,
            &mut block_dag_storage,
            vec![b0.clone()],
            &b0,
            vec![b3.clone(), b4.clone()],
            v0.clone(),
            bonds.clone(),
            Some(1),
            Some(true),
            SHARD_ID.to_string(),
        );

        let justifications_with_invalid_block = vec![
            Justification {
                validator: v0.clone(),
                latest_block_hash: b5.block_hash.clone(),
            },
            Justification {
                validator: v1.clone(),
                latest_block_hash: b4.block_hash.clone(),
            },
        ];

        let block_with_invalid_justification = get_random_block(
            None,
            None,
            None,
            None,
            Some(v1.clone()),
            None,
            None,
            None,
            Some(justifications_with_invalid_block),
            None,
            None,
            None,
            None,
            None,
        );

        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        let result = Validate::justification_regressions(
            &block_with_invalid_justification,
            &mut casper_snapshot,
        );
        assert_eq!(result, Either::Right(ValidBlock::Valid));
    })
    .await
}

#[tokio::test]
#[ignore]
async fn bonds_cache_validation_should_succeed_on_a_valid_block_and_fail_on_modified_bonds() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let genesis = GenesisBuilder::new().create_genesis().await.unwrap();

        let storage_directory = Builder::new()
            .prefix("hash-set-casper-test-genesis-")
            .tempdir()
            .expect("Failed to create temporary directory");
        let storage_directory_path = storage_directory.keep();

        block_dag_storage.insert(&genesis, false, true).unwrap();

        let mut kvm = mk_test_rnode_store_manager(storage_directory_path.clone());

        let m_store = RuntimeManager::mergeable_store(&mut kvm).await.unwrap();

        let mut runtime_manager = RuntimeManager::create_with_store(
            kvm.r_space_stores().await.unwrap(),
            m_store,
            Genesis::non_negative_mergeable_tag_name(),
        );

        let dag = block_dag_storage.get_representation();
        let mut casper_snapshot = mk_casper_snapshot(dag);

        interpreter_util::validate_block_checkpoint(
            &genesis,
            &mut block_store,
            &mut casper_snapshot,
            &mut runtime_manager,
        )
        .await
        .unwrap();

        let result_valid = Validate::bonds_cache(&genesis, &runtime_manager).await;
        assert_eq!(result_valid, Either::Right(ValidBlock::Valid));

        let modified_bonds = vec![];

        let mut modified_post_state = genesis.body.state.clone();
        modified_post_state.bonds = modified_bonds;

        let mut modified_body = genesis.body.clone();
        modified_body.state = modified_post_state;

        let mut modified_genesis = genesis.clone();
        modified_genesis.body = modified_body;

        let result_invalid = Validate::bonds_cache(&modified_genesis, &runtime_manager).await;
        assert_eq!(
            result_invalid,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBondsCache))
        );

        // Manual cleanup
        std::fs::remove_dir_all(&storage_directory_path).ok();
    })
    .await
}

#[tokio::test]
async fn field_format_validation_should_succeed_on_a_valid_block_and_fail_on_empty_fields() {
    with_storage(|_block_store, mut block_dag_storage| async move {
        let context = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .unwrap();
        let (sk, pk) = &context.validator_key_pairs[0];

        block_dag_storage
            .insert(&context.genesis_block, false, true)
            .unwrap();
        let dag = block_dag_storage.get_representation();
        let sender = Bytes::copy_from_slice(&pk.bytes);
        let latest_message_opt = dag.latest_message(&sender).unwrap_or(None);
        let seq_num =
            latest_message_opt.map_or(0, |block_metadata| block_metadata.sequence_number) + 1;

        let genesis =
            ValidatorIdentity::new(sk).sign_block(&with_seq_num(&context.genesis_block, seq_num));

        let result = Validate::format_of_fields(&genesis);
        assert_eq!(result, true);

        let invalid_block = with_sig(&genesis, &Bytes::new());
        let result = Validate::format_of_fields(&invalid_block);
        assert_eq!(result, false);

        let invalid_block = with_sig_algorithm(&genesis, "");
        let result = Validate::format_of_fields(&invalid_block);
        assert_eq!(result, false);

        let invalid_block = with_shard_id(&genesis, "");
        let result = Validate::format_of_fields(&invalid_block);
        assert_eq!(result, false);

        let invalid_block = with_post_state_hash(&genesis, &Bytes::new());
        let result = Validate::format_of_fields(&invalid_block);
        assert_eq!(result, false);
    })
    .await
}

#[tokio::test]
async fn block_hash_format_validation_should_fail_on_invalid_hash() {
    with_storage(|_block_store, mut block_dag_storage| async move {
        let context = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .unwrap();
        let (sk, pk) = &context.validator_key_pairs[0];
        let sender = Bytes::copy_from_slice(&pk.bytes);

        block_dag_storage
            .insert(&context.genesis_block, false, true)
            .unwrap();
        let dag = block_dag_storage.get_representation();

        let latest_message_opt = dag.latest_message(&sender).unwrap_or(None);
        let seq_num =
            latest_message_opt.map_or(0, |block_metadata| block_metadata.sequence_number) + 1;

        let genesis =
            ValidatorIdentity::new(sk).sign_block(&with_seq_num(&context.genesis_block, seq_num));

        let result = Validate::block_hash(&genesis);
        assert_eq!(result, Either::Right(ValidBlock::Valid));

        let invalid_block = with_block_hash(&genesis, &Bytes::copy_from_slice("123".as_bytes()));
        let result = Validate::block_hash(&invalid_block);
        assert_eq!(
            result,
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockHash))
        );
    })
    .await
}

#[tokio::test]
async fn block_version_validation_should_work() {
    with_storage(|_block_store, mut block_dag_storage| async move {
        let context = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .unwrap();
        let (sk, pk) = &context.validator_key_pairs[0];
        let sender = Bytes::copy_from_slice(&pk.bytes);

        block_dag_storage
            .insert(&context.genesis_block, false, true)
            .unwrap();
        let dag = block_dag_storage.get_representation();

        let latest_message_opt = dag.latest_message(&sender).unwrap_or(None);
        let seq_num =
            latest_message_opt.map_or(0, |block_metadata| block_metadata.sequence_number) + 1;

        let genesis =
            ValidatorIdentity::new(sk).sign_block(&with_seq_num(&context.genesis_block, seq_num));

        let result = Validate::version(&genesis, -1);
        assert_eq!(result, false);

        let result = Validate::version(&genesis, 1);
        assert_eq!(result, true);
    })
    .await
}
