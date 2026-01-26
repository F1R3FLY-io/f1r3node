// See rholang/src/test/scala/coop/rchain/rholang/interpreter/DeployDataSpec.scala

use crypto::rust::public_key::PublicKey;
use models::rhoapi::{
    expr::ExprInstance, g_unforgeable::UnfInstance, BindPattern, Expr, GDeployId, GDeployerId,
    GUnforgeable, ListParWithRandom, Par, TaggedContinuation,
};
use rholang::rust::interpreter::{
    matcher::r#match::Matcher,
    rho_runtime::{create_rho_runtime, RhoRuntime},
    system_processes::DeployData,
};
use rspace_plus_plus::rspace::{
    rspace::RSpace,
    shared::{
        in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
    },
};
use std::sync::Arc;

#[tokio::test]
async fn rho_deploy_data_system_channel_should_return_timestamp_deployer_id_and_deploy_id() {
    let contract = r#"
        new deployData(`rho:deploy:data`) in {
          deployData!(0)
        }
    "#;

    let timestamp = 123i64;
    let key = PublicKey::from_bytes(&hex::decode("abcd").unwrap());
    let sig = hex::decode("1234").unwrap();

    let deploy_data = DeployData {
        timestamp,
        deployer_id: key.clone(),
        deploy_id: sig.clone(),
    };

    let expected = vec![
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GInt(timestamp)),
        }]),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                public_key: key.bytes.as_ref().to_vec(),
            })),
        }]),
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId { sig })),
        }]),
    ];

    TestDeployDataFixture::test(contract, deploy_data, expected).await;
}

struct TestDeployDataFixture;

impl TestDeployDataFixture {
    async fn test(contract: &str, deploy_data: DeployData, expected: Vec<Par>) {
        let mut kvm = InMemoryStoreManager::new();
        let store = kvm.r_space_stores().await.unwrap();
        let space: RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> =
            RSpace::create(store, Arc::new(Box::new(Matcher))).unwrap();

        let mut runtime = create_rho_runtime(space, Par::default(), true, &mut Vec::new()).await;

        runtime.set_deploy_data(deploy_data).await;

        runtime.evaluate_with_term(contract).await.unwrap();

        let channel = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GInt(0)),
        }]);

        let data = runtime.get_data(&channel);

        let result: Vec<Par> = if !data.is_empty() {
            data.into_iter().flat_map(|datum| datum.a.pars).collect()
        } else {
            Vec::new()
        };

        assert_eq!(result, expected);
    }
}
