// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperRholangSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};
use casper::rust::util::{
    construct_deploy, proto_util, rholang::registry_sig_gen::RegistrySigGen, rspace_util,
};
use crypto::rust::signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg};
use tokio::sync::OnceCell;

static GENESIS: OnceCell<GenesisContext> = OnceCell::const_new();

async fn get_genesis() -> &'static GenesisContext {
    GENESIS
        .get_or_init(|| async {
            GenesisBuilder::new()
                .build_genesis_with_parameters(None)
                .await
                .expect("Failed to build genesis")
        })
        .await
}

// Scala comments:
// Uncomment this to use the debugger on M2
// May need to modify if architecture required is different or if path is different. See ./scripts/build_rust_libraries.sh
// System.setProperty("jna.library.path", "../rspace++/target/x86_64-apple-darwin/debug/")

// Scala comments:
//put a new casper instance at the start of each
//test since we cannot reset it
#[tokio::test]
async fn multi_parent_casper_should_create_blocks_based_on_deploys() {
    let genesis = get_genesis().await.clone();

    let mut standalone_node = TestNode::standalone(genesis).await.unwrap();

    let deploy = construct_deploy::basic_deploy_data(
        0,
        None,
        Some(standalone_node.genesis.shard_id.clone()),
    )
    .unwrap();
    let block = standalone_node
        .create_block_unsafe(&[deploy.clone()])
        .await
        .unwrap();
    let deploys: Vec<_> = block.body.deploys.iter().map(|pd| &pd.deploy).collect();
    let parents = proto_util::parent_hashes(&block);

    assert_eq!(parents.len(), 1);
    assert_eq!(parents[0], standalone_node.genesis.block_hash);
    assert_eq!(deploys.len(), 1);
    assert_eq!(deploys[0], &deploy);

    let data =
        rspace_util::get_data_at_public_channel_block(&block, 0, &standalone_node.runtime_manager)
            .await;
    assert_eq!(data, vec!["0"]);
}

#[tokio::test]
async fn multi_parent_casper_should_be_able_to_use_the_registry() {
    let genesis = get_genesis().await.clone();

    let mut standalone_node = TestNode::standalone(genesis.clone()).await.unwrap();

    let register_source = r#"
new uriCh, rr(`rho:registry:insertArbitrary`), hello in {
  contract hello(@name, return) = {
    return!("Hello, ${name}!" %% {"name" : name})
  } |
  rr!(bundle+{*hello}, *uriCh)
}
"#;

    fn call_source(registry_id: &str) -> String {
        format!(
            r#"
new out, rl(`rho:registry:lookup`), helloCh in {{
  rl!({}, *helloCh) |
  for(hello <- helloCh){{
    hello!("World", *out)
  }}
}}
"#,
            registry_id
        )
    }

    fn calculate_unforgeable_name(timestamp: i64) -> String {
        let secp256k1 = Secp256k1;
        let public_key = secp256k1.to_public(&construct_deploy::DEFAULT_SEC);
        hex::encode(RegistrySigGen::generate_unforgeable_name_id(
            &public_key,
            timestamp,
        ))
    }

    let register_deploy = construct_deploy::source_deploy_now(
        register_source.to_string(),
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let block0 = standalone_node
        .add_block_from_deploys(&[register_deploy.clone()])
        .await
        .unwrap();

    let registry_id = rspace_util::get_data_at_private_channel(
        &block0,
        &calculate_unforgeable_name(register_deploy.data.time_stamp),
        &standalone_node.runtime_manager,
    )
    .await;

    let call_deploy = construct_deploy::source_deploy_now(
        call_source(&registry_id[0]),
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let block1 = standalone_node
        .add_block_from_deploys(&[call_deploy.clone()])
        .await
        .unwrap();

    let data = rspace_util::get_data_at_private_channel(
        &block1,
        &calculate_unforgeable_name(call_deploy.data.time_stamp),
        &standalone_node.runtime_manager,
    )
    .await;

    assert_eq!(data, vec!["\"Hello, World!\""]);
}
