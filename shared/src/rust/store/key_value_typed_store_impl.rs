// See shared/src/main/scala/coop/rchain/store/KeyValueTypedStoreCodec.scala

use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::rust::{store::key_value_store::KeyValueStore, BitVector};

use super::{key_value_store::KvStoreError, key_value_typed_store::KeyValueTypedStore};

#[derive(Clone)]
pub struct KeyValueTypedStoreImpl<K, V> {
    store: Arc<dyn KeyValueStore>,
    phantom_data: PhantomData<(K, V)>,
}

impl<K, V> KeyValueTypedStoreImpl<K, V>
where
    K: serde::Serialize
        + for<'a> serde::Deserialize<'a>
        + Clone
        + Eq
        + std::hash::Hash
        + std::fmt::Debug,
    V: serde::Serialize + for<'a> serde::Deserialize<'a> + Clone,
{
    pub fn new(store: Arc<dyn KeyValueStore>) -> Self {
        Self {
            store,
            phantom_data: PhantomData,
        }
    }

    pub fn encode_key(&self, key: &K) -> Result<BitVector, KvStoreError> {
        Ok(bincode::serialize(key)?)
    }

    pub fn decode_key(&self, encoded_key: &BitVector) -> Result<K, KvStoreError> {
        Ok(bincode::deserialize(encoded_key)?)
    }

    pub fn encode_value(&self, value: &V) -> Result<BitVector, KvStoreError> {
        Ok(bincode::serialize(value)?)
    }

    pub fn decode_value(&self, encoded_value: &BitVector) -> Result<V, KvStoreError> {
        Ok(bincode::deserialize(encoded_value)?)
    }

    // See shared/src/main/scala/coop/rchain/store/KeyValueTypedStoreSyntax.scala
    pub fn get_one(&self, key: &K) -> Result<Option<V>, KvStoreError> {
        let values = self.get(&vec![key.clone()])?;
        match values.split_first() {
            Some((first_value, _)) => Ok(first_value.clone()),
            None => Ok(None),
        }
    }

    pub fn get_batch(&self, keys: &Vec<K>) -> Result<Vec<V>, KvStoreError> {
        self.get(keys)?
            .into_iter()
            .zip(keys.into_iter())
            .map(|(value_opt, key)| {
                value_opt.ok_or(KvStoreError::KeyNotFound(format!(
                    "Error when reading from KeyValueStore: value for key {:?} not found.",
                    key
                )))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn get_unsafe(&self, key: &K) -> Result<V, KvStoreError> {
        self.get_one(&key)?.ok_or(KvStoreError::KeyNotFound(format!(
            "Error when reading from KeyValueStore: value for key {:?} not found.",
            key
        )))
    }

    pub fn put_one(&self, key: K, value: V) -> Result<(), KvStoreError> {
        self.put(vec![(key, value)])
    }

    pub fn put_if_absent(&self, kv_pairs: Vec<(K, V)>) -> Result<(), KvStoreError> {
        let keys: Vec<K> = kv_pairs.iter().map(|(k, _)| k.clone()).collect();
        let if_absent = self.contains(keys)?;
        let kv_if_absent: Vec<_> = kv_pairs.into_iter().zip(if_absent).collect();
        let kv_absent: Vec<_> = kv_if_absent
            .clone()
            .into_iter()
            .filter(|(_, is_present)| !is_present)
            .map(|(kv, _)| kv)
            .collect();

        self.put(kv_absent)
    }

    pub fn contains_key(&self, key: K) -> Result<bool, KvStoreError> {
        let results = self.contains(vec![key])?;
        Ok(*results.first().unwrap_or(&false))
    }

    pub fn get_or_else(&self, key: K, else_value: V) -> Result<V, KvStoreError> {
        match self.get_one(&key)? {
            Some(value) => Ok(value),
            None => Ok(else_value),
        }
    }
}

impl<K, V> KeyValueTypedStore<K, V> for KeyValueTypedStoreImpl<K, V>
where
    K: serde::Serialize
        + for<'a> serde::Deserialize<'a>
        + Clone
        + Eq
        + std::hash::Hash
        + std::fmt::Debug,
    V: serde::Serialize + for<'a> serde::Deserialize<'a> + Clone,
{
    fn get(&self, keys: &Vec<K>) -> Result<Vec<Option<V>>, KvStoreError> {
        let keys_bit_vector = keys
            .iter()
            .map(|key| self.encode_key(key))
            .collect::<Result<Vec<_>, _>>()?;
        let values_bytes = self.store.get(&keys_bit_vector)?;

        let values = values_bytes
            .iter()
            .map(|value_opt| {
                value_opt
                    .as_ref()
                    .map(|value| self.decode_value(value))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(values)
    }

    fn put(&self, kv_pairs: Vec<(K, V)>) -> Result<(), KvStoreError> {
        let pairs_bit_vector = kv_pairs
            .iter()
            .map(|(key, value)| {
                let encoded_key = self.encode_key(key)?;
                let encoded_value = self.encode_value(value)?;
                Ok((encoded_key, encoded_value))
            })
            .collect::<Result<Vec<(BitVector, BitVector)>, KvStoreError>>()?;

        self.store.put(pairs_bit_vector)?;
        Ok(())
    }

    fn delete(&self, keys: Vec<K>) -> Result<(), KvStoreError> {
        let keys_bit_vector = keys
            .iter()
            .map(|key| self.encode_key(key))
            .collect::<Result<Vec<_>, _>>()?;
        self.store.delete(keys_bit_vector)?;
        Ok(())
    }

    fn contains(&self, keys: Vec<K>) -> Result<Vec<bool>, KvStoreError> {
        let keys_bit_vector = keys
            .iter()
            .map(|key| self.encode_key(key))
            .collect::<Result<Vec<_>, _>>()?;

        let results = self.store.get(&keys_bit_vector)?;
        Ok(results.iter().map(|result| result.is_some()).collect())
    }

    fn collect<F, T>(&self, mut f: F) -> Result<Vec<T>, KvStoreError>
    where
        F: FnMut((&K, &V)) -> Option<T>,
    {
        let store_map = self.store.to_map()?;
        let mut result = Vec::new();

        for (key_bytes, value_bytes) in store_map {
            let key = self.decode_key(&key_bytes)?;
            let value = self.decode_value(&value_bytes)?;

            if let Some(item) = f((&key, &value)) {
                result.push(item);
            }
        }

        Ok(result)
    }

    fn to_map(&self) -> Result<HashMap<K, V>, KvStoreError> {
        let mut result = HashMap::new();
        let store_map = self.store.to_map()?;

        for (key_bytes, value_bytes) in store_map {
            let key = self.decode_key(&key_bytes)?;
            let value = self.decode_value(&value_bytes)?;
            result.insert(key, value);
        }

        Ok(result)
    }

    fn non_empty(&self) -> Result<bool, KvStoreError> {
        self.store.non_empty()
    }
}
