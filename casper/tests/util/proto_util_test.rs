// See casper/src/test/scala/coop/rchain/casper/util/ProtoUtilTest.scala

use std::collections::HashSet;

use casper::rust::util::{construct_deploy, proto_util};
use models::rust::block_implicits::block_element_gen;
use proptest::prelude::*;
use prost::bytes::Bytes;

use crate::{
    helper::test_node::TestNode,
    util::genesis_builder::{GenesisBuilder, GenesisContext},
};

proptest! {
    #[test]
    fn dependencies_hashes_of_should_return_hashes_of_all_justifications_and_parents_of_a_block(
        block in block_element_gen(None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    ) {
        let result = proto_util::dependencies_hashes_of(&block);

        let justifications_hashes: Vec<Bytes> = block
            .justifications
            .iter()
            .map(|j| j.latest_block_hash.clone())
            .collect();

        let parents_hashes: Vec<Bytes> = block.header.parents_hash_list.clone();

        for hash in &justifications_hashes {
            prop_assert!(result.contains(hash), "Missing justification hash");
        }

        for hash in &parents_hashes {
            prop_assert!(result.contains(hash), "Missing parent hash");
        }

        let result_set: HashSet<Bytes> = result.into_iter().collect();
        let expected: HashSet<Bytes> = justifications_hashes
            .into_iter()
            .chain(parents_hashes.into_iter())
            .collect();

        prop_assert_eq!(result_set, expected);
    }
}

struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        let mut genesis_builder = GenesisBuilder::new();
        let genesis_parameters_tuple =
            GenesisBuilder::build_genesis_parameters_with_defaults(None, None);
        let genesis_context = genesis_builder
            .build_genesis_with_parameters(Some(genesis_parameters_tuple))
            .await
            .expect("Failed to build genesis context");

        Self {
            genesis: genesis_context,
        }
    }
}

#[tokio::test]
async fn unseen_block_hashes_should_return_empty_for_a_single_block_dag() {
    let ctx = TestContext::new().await;
    let mut node = TestNode::standalone(ctx.genesis.clone()).await.unwrap();

    let shard_id = ctx.genesis.genesis_block.shard_id.clone();

    let deploy = construct_deploy::basic_deploy_data(0, None, Some(shard_id)).unwrap();
    let signed_block = node.add_block_from_deploys(&[deploy]).await.unwrap();

    let mut dag = node.block_dag_storage.get_representation();

    let unseen_block_hashes = proto_util::unseen_block_hashes(&mut dag, &signed_block).unwrap();

    assert!(
        unseen_block_hashes.is_empty(),
        "Expected empty set but got {:?}",
        unseen_block_hashes
    );
}

#[tokio::test]
async fn unseen_block_hashes_should_return_all_but_the_first_block_when_passed_the_first_block_in_a_chain(
) {
    let ctx = TestContext::new().await;
    let mut node = TestNode::standalone(ctx.genesis.clone()).await.unwrap();

    let shard_id = ctx.genesis.genesis_block.shard_id.clone();

    let deploy0 = construct_deploy::basic_deploy_data(0, None, Some(shard_id.clone())).unwrap();
    let block0 = node.add_block_from_deploys(&[deploy0]).await.unwrap();

    let deploy1 = construct_deploy::basic_deploy_data(1, None, Some(shard_id)).unwrap();
    let block1 = node.add_block_from_deploys(&[deploy1]).await.unwrap();

    let mut dag = node.block_dag_storage.get_representation();

    let unseen_block_hashes = proto_util::unseen_block_hashes(&mut dag, &block0).unwrap();

    let expected: HashSet<Bytes> = vec![block1.block_hash.clone()].into_iter().collect();
    assert_eq!(
        unseen_block_hashes, expected,
        "Expected {:?} but got {:?}",
        expected, unseen_block_hashes
    );
}
