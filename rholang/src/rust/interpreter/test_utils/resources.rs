// See rholang/src/test/scala/coop/rchain/rholang/Resources.scala

use std::path::PathBuf;
use std::{future::Future, path::Path, sync::Arc};
use tempfile::Builder;

use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rspace_plus_plus::rspace::history::history_repository::HistoryRepository;
use rspace_plus_plus::rspace::rspace::{RSpace, RSpaceStore};

use crate::rust::interpreter::matcher::r#match::Matcher;
use crate::rust::interpreter::rho_runtime;
use crate::rust::interpreter::rho_runtime::{create_replay_rho_runtime, create_rho_runtime};
use crate::rust::interpreter::system_processes::Definition;
use crate::RhoRuntimeImpl;
use rspace_plus_plus::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager, lmdb_dir_store_manager::MB,
    rspace_store_manager::mk_rspace_store_manager,
};

pub fn mk_temp_dir(prefix: &str) -> PathBuf {
    let temp_dir = Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("Failed to create temp dir");
    temp_dir.keep()
}

pub fn with_temp_dir<F, R>(prefix: &str, f: F) -> R
where
    F: FnOnce(&Path) -> R,
{
    let temp_dir = Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("Failed to create temp dir");

    // Run the function with the temp_dir path
    let result = f(temp_dir.path());

    // TempDir will be dropped here and automatically cleaned up
    // unless we've decided to manually persist it
    result
}

pub async fn with_runtime<F, Fut, R>(prefix: &str, f: F) -> R
where
    F: FnOnce(RhoRuntimeImpl) -> Fut,
    Fut: Future<Output = R>,
{
    let temp_dir = Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("Failed to create temp dir");

    let mut store_manager = mk_rspace_store_manager(temp_dir.path().to_path_buf(), 100 * MB);
    let rspace_store = store_manager.r_space_stores().await.unwrap();
    let runtime = rho_runtime::create_runtime_from_kv_store(
        rspace_store,
        Par::default(),
        false,
        &mut Vec::new(),
        Arc::new(Box::new(Matcher)),
    )
    .await;

    f(runtime).await
}

pub async fn create_runtimes(
    stores: RSpaceStore,
    init_registry: bool,
    additional_system_processes: &mut Vec<Definition>,
) -> (
    RhoRuntimeImpl,
    RhoRuntimeImpl,
    Arc<Box<dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>>>,
) {
    let hrstores =
        RSpace::<Par, BindPattern, ListParWithRandom, TaggedContinuation>::create_with_replay(
            stores,
            Arc::new(Box::new(Matcher)),
        )
        .unwrap();

    let (space, replay) = hrstores;

    let rho_runtime = create_rho_runtime(
        space.clone(),
        Par::default(),
        init_registry,
        additional_system_processes,
    )
    .await;

    let replay_rho_runtime = create_replay_rho_runtime(
        replay,
        Par::default(),
        init_registry,
        additional_system_processes,
    )
    .await;
    (
        rho_runtime,
        replay_rho_runtime,
        space.history_repository.clone(),
    )
}
