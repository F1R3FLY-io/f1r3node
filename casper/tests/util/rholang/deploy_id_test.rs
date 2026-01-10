// See casper/src/test/scala/coop/rchain/casper/util/rholang/DeployIdTest.scala

use crate::helper::test_node::TestNode;
use crate::util::{genesis_builder::GenesisBuilder, rholang::resources::with_runtime_manager};
use casper::rust::util::rholang::runtime_manager::RuntimeManager;
use casper::rust::util::{construct_deploy, proto_util};
use crypto::rust::{private_key::PrivateKey, signatures::signed::Signed};
use models::rhoapi::{
    expr::ExprInstance, g_unforgeable::UnfInstance, Expr, GDeployId, GUnforgeable, Par,
};
use models::rust::casper::protocol::casper_message::DeployData;
use std::time::SystemTime;

fn default_sec() -> PrivateKey {
    construct_deploy::DEFAULT_SEC.clone()
}

fn deploy(
    deployer: PrivateKey,
    rho: String,
    _timestamp: Option<i64>,
    shard_id: String,
) -> Signed<DeployData> {
    construct_deploy::source_deploy(
        rho,
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64,
        None,
        None,
        Some(deployer),
        None,
        Some(shard_id),
    )
    .unwrap()
}

#[tokio::test]
async fn deploy_id_should_be_equal_to_deploy_signature() {
    with_runtime_manager(|runtime_manager, genesis_context, _| async move {
        let sk = default_sec();

        let d = deploy(
            sk,
            r#"new return, deployId(`rho:rchain:deployId`) in { return!(*deployId) }"#.to_string(),
            None,
            genesis_context.genesis_block.shard_id.clone(),
        );

        let result = runtime_manager
            .capture_results(&RuntimeManager::empty_state_hash_fixed(), &d)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);

        let expected = Par {
            unforgeables: vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId {
                    sig: d.sig.to_vec(),
                })),
            }],
            ..Default::default()
        };
        assert_eq!(result[0], expected);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn deploy_id_should_be_resolved_during_normalization() {
    let mut genesis_builder = GenesisBuilder::new();
    let genesis_parameters_tuple =
        GenesisBuilder::build_genesis_parameters_with_defaults(None, None);
    let genesis_context = genesis_builder
        .build_genesis_with_parameters(Some(genesis_parameters_tuple))
        .await
        .expect("Failed to build genesis context");

    let mut node = TestNode::standalone(genesis_context)
        .await
        .expect("Failed to create standalone node");

    let sk = default_sec();

    let contract = deploy(
        sk.clone(),
        r#"contract @"check"(input, ret) = { new deployId(`rho:rchain:deployId`) in { ret!(*input == *deployId) }}"#
            .to_string(),
        None,
        node.genesis.shard_id.clone(),
    );

    let contract_call = deploy(
        sk,
        r#"new return, deployId(`rho:rchain:deployId`), ret in { @"check"!(*deployId, *return) }"#
            .to_string(),
        None,
        node.genesis.shard_id.clone(),
    );

    let block = node.add_block_from_deploys(&[contract]).await.unwrap();

    let result = node
        .runtime_manager
        .capture_results(&proto_util::post_state_hash(&block), &contract_call)
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let expected = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GBool(false)),
        }],
        ..Default::default()
    };
    assert_eq!(result[0], expected);
}
