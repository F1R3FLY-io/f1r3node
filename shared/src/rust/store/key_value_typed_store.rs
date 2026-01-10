// See shared/src/main/scala/coop/rchain/store/KeyValueTypedStore.scala

use std::collections::HashMap;

use super::key_value_store::KvStoreError;

pub trait KeyValueTypedStore<K, V> {
    fn get(&self, keys: &Vec<K>) -> Result<Vec<Option<V>>, KvStoreError>;

    fn put(&self, kv_pairs: Vec<(K, V)>) -> Result<(), KvStoreError>;

    fn delete(&self, keys: Vec<K>) -> Result<(), KvStoreError>;

    fn contains(&self, keys: Vec<K>) -> Result<Vec<bool>, KvStoreError>;

    /**
     * Efficient way to iterate and filter the whole KV store
     *
     * @param pf Partial function to project and filter values
     */
    fn collect<F, T>(&self, f: F) -> Result<Vec<T>, KvStoreError>
    where
        F: FnMut((&K, &V)) -> Option<T>;

    fn to_map(&self) -> Result<HashMap<K, V>, KvStoreError>;
}
