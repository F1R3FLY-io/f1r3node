// See casper/src/test/scala/coop/rchain/casper/batch2/LmdbKeyValueStoreSpec.scala

use lazy_static::lazy_static;
use proptest::collection::hash_map;
use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;
use rspace_plus_plus::rspace::shared::lmdb_dir_store_manager::{Db, LmdbDirStoreManager, LmdbEnvConfig};
use std::collections::HashMap;

use crate::util::in_memory_key_value_store_spec::KeyValueStoreSut;
use crate::util::rholang::resources::{get_shared_lmdb_path, generate_scope_id};

// Optimization: proptest! macro generates sync functions but our tests are async.
// Creating a new Runtime for each test case is expensive (proptest runs 256 cases by default).
// Using a shared lazy_static Runtime is much more efficient.
lazy_static! {
    static ref RUNTIME: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
}

async fn with_sut<F, Fut>(f: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce(KeyValueStoreSut) -> Fut,
    Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error>>>,
{
    let scope_id = generate_scope_id();
    let scoped_db_id = format!("{}-test", scope_id);
    
    let db_config = LmdbEnvConfig::new("test-db".to_string(), 1024 * 1024 * 1024).with_max_dbs(10_000);
    let mut db_mappings = HashMap::new();
    db_mappings.insert(Db::new(scoped_db_id.clone(), None), db_config);

    let shared_path = get_shared_lmdb_path();
    let kvm = LmdbDirStoreManager::new(shared_path, db_mappings);
    
    let sut = KeyValueStoreSut::new_scoped(Box::new(kvm), scoped_db_id);

    f(sut).await?;

    Ok(())
}

fn gen_data() -> impl Strategy<Value = HashMap<i64, String>> {
    hash_map(any::<i64>(), any::<String>(), 0..2000)
}

// Note: LMDB is file-based storage with system resource limits (file descriptors, max environments).
// Proptest runs 256 test cases by default, which exhausts these resources causing:
// "Resource temporarily unavailable" (EAGAIN) errors.
// We limit test cases to 20 to stay within system limits while still getting property-based testing benefits.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]
    
    #[test]
    fn lmdb_key_value_store_should_put_and_get_data_from_the_store(expected in gen_data()) {
        RUNTIME.block_on(async {
            with_sut(|mut sut| async move {
                let result = sut.test_put_get(expected.clone()).await?;
                assert_eq!(result, expected);
                Ok(())
            })
            .await
            .expect("Test failed");
        });
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]
    
    #[test]
    fn lmdb_key_value_store_should_put_and_get_all_data_from_the_store(expected in gen_data()) {
        RUNTIME.block_on(async {
            with_sut(|mut sut| async move {
                let result = sut.test_put_iterate(expected.clone()).await?;
                assert_eq!(result, expected);
                Ok(())
            })
            .await
            .expect("Test failed");
        });
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]
    
    #[test]
    fn lmdb_key_value_store_should_not_have_deleted_keys_in_the_store(input in gen_data()) {
        RUNTIME.block_on(async {
            with_sut(|mut sut| async move {
                let all_keys: Vec<i64> = input.keys().copied().collect();

                let split_at = all_keys.len() / 2;
                let get_keys: Vec<i64> = all_keys[..split_at].to_vec();
                let delete_keys: Vec<i64> = all_keys[split_at..].to_vec();

                let expected: HashMap<i64, String> = get_keys
                    .iter()
                    .filter_map(|k| input.get(k).map(|v| (*k, v.clone())))
                    .collect();

                let result = sut.test_put_delete_get(input.clone(), delete_keys).await?;
                assert_eq!(result, expected);
                Ok(())
            })
            .await
            .expect("Test failed");
        });
    }
}
