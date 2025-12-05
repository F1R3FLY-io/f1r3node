// See casper/src/test/scala/coop/rchain/casper/batch2/LmdbKeyValueStoreSpec.scala

use lazy_static::lazy_static;
use proptest::collection::hash_map;
use proptest::prelude::*;
use proptest::string::string_regex;
use proptest::test_runner::Config as ProptestConfig;
use rholang::rust::interpreter::test_utils::resources::mk_temp_dir_guard;
use rspace_plus_plus::rspace::shared::lmdb_store_manager::LmdbStoreManager;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

use crate::util::in_memory_key_value_store_spec::KeyValueStoreSut;

lazy_static::lazy_static! {
    // TempDir guard kept in lazy_static for shared parent directory (cleaned up at program exit)
    static ref TEMP_DIR_GUARD: TempDir = mk_temp_dir_guard("lmdb-test-");
    static ref TEMP_PATH: PathBuf = TEMP_DIR_GUARD.path().to_path_buf();
}

// Optimization: proptest! macro generates sync functions but our tests are async.
// Using a shared lazy_static Runtime is much more efficient.
lazy_static! {
    static ref RUNTIME: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
}

const MAX_ENV_SIZE: usize = 1024 * 1024 * 1024;

async fn with_sut<F, Fut>(f: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce(KeyValueStoreSut) -> Fut,
    Fut: std::future::Future<Output = Result<KeyValueStoreSut, Box<dyn std::error::Error>>>,
{
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::SeqCst);
    let random_str = format!("test_{}", count);

    let temp_path = TEMP_PATH.join(random_str);

    let kvm = LmdbStoreManager::new(temp_path, MAX_ENV_SIZE);

    let sut = KeyValueStoreSut::new(kvm);

    // Closure takes ownership and must return the sut back for cleanup
    let mut sut = f(sut).await?;

    sut.shutdown().await?;

    Ok(())
}

fn gen_data() -> impl Strategy<Value = HashMap<i64, String>> {
    // Non-empty alphanumeric strings (at least 1 char)
    let non_empty_alphanum = string_regex("[a-zA-Z0-9]+").unwrap();
    // Non-empty map
    hash_map(any::<i64>(), non_empty_alphanum, 1..100)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn lmdb_key_value_store_should_put_and_get_data_from_the_store(expected in gen_data()) {
        RUNTIME.block_on(async {
            with_sut(|mut sut| async move {
                let result = sut.test_put_get(expected.clone()).await?;
                assert_eq!(result, expected);
                Ok(sut)
            })
            .await
            .expect("Test failed");
        });
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn lmdb_key_value_store_should_put_and_get_all_data_from_the_store(expected in gen_data()) {
        RUNTIME.block_on(async {
            with_sut(|mut sut| async move {
                let result = sut.test_put_iterate(expected.clone()).await?;
                assert_eq!(result, expected);
                Ok(sut)
            })
            .await
            .expect("Test failed");
        });
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

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
                Ok(sut)
            })
            .await
            .expect("Test failed");
        });
    }
}
