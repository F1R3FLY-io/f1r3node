// See casper/src/test/scala/coop/rchain/casper/util/rholang/DeployerIdTest.scala

use crate::helper::test_node::TestNode;
use crate::util::{genesis_builder::GenesisBuilder, rholang::resources::with_runtime_manager};
use casper::rust::util::rholang::runtime_manager::RuntimeManager;
use casper::rust::util::{construct_deploy, proto_util};
use crypto::rust::{
    private_key::PrivateKey, signatures::secp256k1::Secp256k1,
    signatures::signatures_alg::SignaturesAlg,
};
use models::rhoapi::{
    expr::ExprInstance, g_unforgeable::UnfInstance, Expr, GDeployerId, GUnforgeable, Par,
};
use prost::bytes::Bytes;

fn default_sec() -> PrivateKey {
    construct_deploy::DEFAULT_SEC.clone()
}

fn default_sec2() -> PrivateKey {
    construct_deploy::DEFAULT_SEC2.clone()
}

#[tokio::test]
async fn deployer_id_should_be_equal_to_the_deployers_public_key() {
    with_runtime_manager(|runtime_manager, _, _| async move {
        let sk = PrivateKey::from_bytes(
            &hex::decode("b18e1d0045995ec3d010c387ccfeb984d783af8fbb0f40fa7db126d889f6dadd")
                .unwrap(),
        );
        let pk = Bytes::from(Secp256k1.to_public(&sk).bytes.to_vec());

        let deploy = construct_deploy::source_deploy_now_full(
            r#"new return, auth(`rho:rchain:deployerId`) in { return!(*auth) }"#.to_string(),
            None,
            None,
            Some(sk),
            None,
            None,
        )
        .unwrap();

        let empty_state_hash = RuntimeManager::empty_state_hash_fixed();
        let result = runtime_manager
            .capture_results(&empty_state_hash, &deploy)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);

        let expected = Par {
            unforgeables: vec![GUnforgeable {
                unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                    public_key: pk.to_vec(),
                })),
            }],
            ..Default::default()
        };
        assert_eq!(result[0], expected);
    })
    .await
    .unwrap();
}

async fn check_access_granted(
    deployer: PrivateKey,
    contract_user: PrivateKey,
    is_access_granted: bool,
) {
    let check_deployer_definition = r#"
contract @"checkAuth"(input, ret) = {
  new auth(`rho:rchain:deployerId`) in {
    ret!(*input == *auth)
  }
}"#;

    let check_deployer_call = r#"
new return, auth(`rho:rchain:deployerId`), ret in {
  @"checkAuth"!(*auth, *ret) |
  for(isAuthenticated <- ret) {
    return!(*isAuthenticated)
  }
}"#;

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

    let contract = construct_deploy::source_deploy_now_full(
        check_deployer_definition.to_string(),
        None,
        None,
        Some(deployer),
        None,
        Some(node.genesis.shard_id.clone()),
    )
    .unwrap();

    let block = node.add_block_from_deploys(&[contract]).await.unwrap();

    let check_auth_deploy = construct_deploy::source_deploy_now_full(
        check_deployer_call.to_string(),
        None,
        None,
        Some(contract_user),
        None,
        None,
    )
    .unwrap();

    let result = node
        .runtime_manager
        .capture_results(&proto_util::post_state_hash(&block), &check_auth_deploy)
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let expected = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GBool(is_access_granted)),
        }],
        ..Default::default()
    };
    assert_eq!(result[0], expected);
}

#[tokio::test]
async fn deployer_id_should_make_drain_vault_attacks_impossible() {
    let deployer = default_sec();
    let attacker = default_sec2();

    check_access_granted(deployer.clone(), deployer.clone(), true).await;
    check_access_granted(deployer, attacker, false).await;
}
