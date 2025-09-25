use dashmap::DashSet;
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
    sync::{Arc, Mutex, RwLock},
};

use crate::rust::interpreter::{
    accounting::{cost_accounting::CostAccounting, costs::Cost},
    dispatch::ThreadSafeProcessContext,
    matcher::r#match::Matcher,
    openai_service::OpenAIService,
    reduce::DebruijnInterpreter,
    rho_runtime::{create_runtime_from_kv_store, RhoISpace, RhoRuntimeImpl},
    system_processes::{
        test_framework_contracts, test_framework_process_registry, BlockData, InvalidBlocks,
        SystemProcessRegistry,
    },
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

    // 1. Create empty registry (tests don't need system processes)
    let registry = Arc::new(SystemProcessRegistry::new());

    // 2. Create ThreadSafeProcessContext
    let process_context = ThreadSafeProcessContext::new(rspace.clone());

    // UPDATED: Use new method signature
    let reducer = DebruijnInterpreter::new(
        rspace,
        registry, // NEW
        HashMap::new(),
        Arc::new(DashSet::new()),
        Par::default(),
        process_context, // NEW
        cost.clone(),
    )
    .await;

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
        &mut test_framework_process_registry(),
        Arc::new(Box::new(Matcher)),
    )
    .await;

    runtime
}
