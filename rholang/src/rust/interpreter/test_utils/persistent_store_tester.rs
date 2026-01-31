use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rspace_plus_plus::rspace::{
    rspace::RSpace,
    rspace_interface::ISpace,
    shared::{
        in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
    },
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::rust::interpreter::{
    accounting::{cost_accounting::CostAccounting, costs::Cost},
    matcher::r#match::Matcher,
    reduce::DebruijnInterpreter,
    rho_runtime::{create_runtime_from_kv_store, RhoISpace, RhoRuntimeImpl},
    system_processes::test_framework_contracts,
    external_services::ExternalServices,
};

pub async fn create_test_space<T>() -> (
    impl ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
    DebruijnInterpreter,
)
where
    T: ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
{
    let cost = CostAccounting::empty_cost();
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let space = RSpace::create(store, Arc::new(Box::new(Matcher))).unwrap();
    let rspace: RhoISpace = Arc::new(tokio::sync::Mutex::new(Box::new(space.clone())));

    let reducer = DebruijnInterpreter::new(
        rspace,
        Arc::new(HashMap::new()),
        Arc::new(std::sync::RwLock::new(HashSet::new())),
        Par::default(),
        cost.clone(),
    );

    cost.set(Cost::create(
        i64::MAX,
        "persistent_store_tester setup".to_string(),
    ));

    (space, reducer)
}

pub async fn create_test_runtime_with_genesis_contracts() -> RhoRuntimeImpl {
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let runtime = create_runtime_from_kv_store(
        store,
        Par::default(),
        true,
        &mut test_framework_contracts(),
        Arc::new(Box::new(Matcher)),
        ExternalServices::noop(),
    )
    .await;

    runtime
}
