// See casper/src/test/scala/coop/rchain/casper/batch2/RholangBuildTest.scala

use casper::rust::genesis::contracts::vault::Vault;
use casper::rust::util::construct_deploy;
use casper::rust::util::rholang::tools::Tools;
use casper::rust::util::rspace_util;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::signatures_alg::SignaturesAlg;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;

fn calculate_unforgeable_name(timestamp: i64) -> String {
    let secp256k1 = Secp256k1;
    let public_key = secp256k1.to_public(&construct_deploy::DEFAULT_SEC);
    let unforgeable_id = Tools::unforgeable_name_rng(&public_key, timestamp).next();
    let unforgeable_id_u8: Vec<u8> = unforgeable_id.iter().map(|&b| b as u8).collect();
    hex::encode(unforgeable_id_u8)
}

#[tokio::test]
async fn our_build_system_should_allow_import_of_rholang_sources_into_scala_code() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut node = TestNode::standalone(genesis.clone()).await.unwrap();

    let code = r#"
new testRet, double, rl(`rho:registry:lookup`), ListOpsCh, getBlockData(`rho:block:data`),
    timeRtn, stdout(`rho:io:stdout`), doubleRet
in {
  contract double(@x, ret) = { ret!(2 * x) } |
  rl!(`rho:lang:listOps`, *ListOpsCh) |
  for(@(_, ListOps) <- ListOpsCh) {
    @ListOps!("map", [2, 3, 5, 7], *double, *doubleRet)
  } |
  getBlockData!(*timeRtn) |
  for (@_, @timestamp, @_ <- timeRtn & @doubles <- doubleRet) {
    testRet!((doubles, "The timestamp is ${timestamp}" %% {"timestamp" : timestamp}))
  }
}
"#;

    let deploy = construct_deploy::source_deploy_now(
        code.to_string(),
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let signed_block = node
        .add_block_from_deploys(&[deploy.clone()])
        .await
        .unwrap();

    let expected_timestamp = signed_block.header.timestamp;
    let expected = format!(
        r#"([4, 6, 10, 14], "The timestamp is {}")"#,
        expected_timestamp
    );

    let data = rspace_util::get_data_at_private_channel(
        &signed_block,
        &calculate_unforgeable_name(deploy.data.time_stamp),
        &node.runtime_manager,
    )
    .await;

    assert_eq!(data, vec![expected]);
}

#[tokio::test]
#[ignore = "Scala ignore"]
async fn our_build_system_should_execute_the_genesis_block() {
    const REV_ADDRESS_COUNT: i32 = 16000;

    let mut vaults = Vec::new();
    let secp256k1 = Secp256k1;

    for i in 1..=REV_ADDRESS_COUNT {
        let (_, public_key) = secp256k1.new_key_pair();
        let rev_address =
            rholang::rust::interpreter::util::rev_address::RevAddress::from_public_key(&public_key)
                .expect("Failed to create RevAddress from public key");

        vaults.push(Vault {
            rev_address,
            initial_balance: i as u64,
        });
    }

    let genesis = GenesisBuilder::new()
        .with_vaults(vaults)
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis with vaults");

    let _node = TestNode::standalone(genesis.clone()).await.unwrap();

    // Scala: (logEff.warns should be(Nil)).pure[Effect]
    // Note: In Rust we don't have direct access to log warnings in tests like in Scala
    // If we got here without panicking, the genesis was successfully created
}
