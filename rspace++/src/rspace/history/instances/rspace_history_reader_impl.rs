use std::marker::PhantomData;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_store::KeyValueStore;

use crate::rspace::errors::HistoryError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::hashing::stable_hash_provider::{hash, hash_from_vec};
use crate::rspace::history::cold_store::PersistedData;
use crate::rspace::history::history::History;
use crate::rspace::history::history_reader::{HistoryReader, HistoryReaderBase};
use crate::rspace::history::history_repository::{PREFIX_DATUM, PREFIX_JOINS, PREFIX_KONT};
use crate::rspace::history::history_repository_impl::prepend_bytes;
use crate::rspace::internal::{Datum, WaitingContinuation};
use crate::rspace::metrics_constants::{
    HISTORY_FETCH_DATA_CALLS_METRIC, HISTORY_FETCH_DATA_DESERIALIZE_NS_METRIC,
    HISTORY_FETCH_DATA_LEAF_GET_NS_METRIC, HISTORY_FETCH_DATA_LEGACY_FALLBACK_METRIC,
    HISTORY_FETCH_DATA_TIME_NS_METRIC, HISTORY_FETCH_DATA_TRIE_READ_NS_METRIC,
    RSPACE_METRICS_SOURCE,
};
use crate::rspace::serializers::serializers::{decode_continuations, decode_datums, decode_joins};

#[derive(Clone)]
pub struct RSpaceHistoryReaderImpl<C, P, A, K> {
    target_history: Arc<Box<dyn History>>,
    leaf_store: Arc<dyn KeyValueStore>,
    _marker: PhantomData<(C, P, A, K)>,
}

impl<C, P, A, K> RSpaceHistoryReaderImpl<C, P, A, K> {
    pub fn new(target_history: Box<dyn History>, leaf_store: Arc<dyn KeyValueStore>) -> Self {
        RSpaceHistoryReaderImpl {
            target_history: Arc::new(target_history),
            leaf_store,
            _marker: PhantomData,
        }
    }

    /** Fetch data on a hash pointer */
    fn fetch_data(
        &self,
        prefix: u8,
        key: &Blake2b256Hash,
    ) -> Result<Option<PersistedData>, HistoryError> {
        let __fetch_start = std::time::Instant::now();
        metrics::counter!(HISTORY_FETCH_DATA_CALLS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .increment(1);

        let __trie_start = std::time::Instant::now();
        let read_bytes = self
            .target_history
            .read(prepend_bytes(prefix, &key.bytes()))?;
        metrics::counter!(HISTORY_FETCH_DATA_TRIE_READ_NS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .increment(__trie_start.elapsed().as_nanos() as u64);

        let result = match read_bytes {
            Some(ref bytes) => {
                // Two leaf-store key shapes are supported: raw hash bytes
                // (modern path) and the bincode-serialized hash bytes (the
                // legacy_fallback below). Some staging-resident persisted
                // leaves still live under the bincode key; the fallback is
                // required to read them without `ConsumeFailed`.
                let __leaf_start = std::time::Instant::now();
                let mut get_opt = self.leaf_store.get_one(bytes)?;
                if get_opt.is_none() {
                    metrics::counter!(HISTORY_FETCH_DATA_LEGACY_FALLBACK_METRIC, "source" => RSPACE_METRICS_SOURCE)
                        .increment(1);
                    let serialized_read_hash = bincode::serialize(bytes).map_err(|e| {
                        HistoryError::ActionError(format!(
                            "RSpace History Reader Impl: Unable to serialize read hash bytes: {}",
                            e
                        ))
                    })?;
                    get_opt = self.leaf_store.get_one(&serialized_read_hash)?;
                }
                metrics::counter!(HISTORY_FETCH_DATA_LEAF_GET_NS_METRIC, "source" => RSPACE_METRICS_SOURCE)
                    .increment(__leaf_start.elapsed().as_nanos() as u64);

                match get_opt {
                    Some(store_value_bytes) => {
                        let __deser_start = std::time::Instant::now();
                        let decoded = bincode::deserialize(&store_value_bytes).map_err(|e| {
                            HistoryError::ActionError(format!(
                                "RSpace History Reader Impl: Failed to deserialize persisted \
                                 leaf: {}",
                                e
                            ))
                        })?;
                        metrics::counter!(HISTORY_FETCH_DATA_DESERIALIZE_NS_METRIC, "source" => RSPACE_METRICS_SOURCE)
                            .increment(__deser_start.elapsed().as_nanos() as u64);
                        Ok(Some(decoded))
                    }
                    None => Ok(None),
                }
            }
            None => Ok(None),
        };
        metrics::counter!(HISTORY_FETCH_DATA_TIME_NS_METRIC, "source" => RSPACE_METRICS_SOURCE)
            .increment(__fetch_start.elapsed().as_nanos() as u64);
        result
    }
}

impl<C, P, A, K> HistoryReader<Blake2b256Hash, C, P, A, K> for RSpaceHistoryReaderImpl<C, P, A, K>
where
    C: Clone + for<'a> Deserialize<'a> + Serialize + 'static + Sync + Send,
    P: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
    A: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
    K: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
{
    fn root(&self) -> Blake2b256Hash { self.target_history.root() }

    fn get_data_proj(&self, key: &Blake2b256Hash) -> Result<Vec<Datum<A>>, HistoryError> {
        match self.fetch_data(PREFIX_DATUM, key)? {
            Some(PersistedData::Data(data_leaf)) => Ok(decode_datums(&data_leaf.bytes)),
            Some(p) => {
                panic!(
                    "Found unexpected leaf while looking for data at key {:?}, data: {:?}",
                    key, p
                );
            }
            None => Ok(Vec::new()),
        }
    }

    fn get_data_proj_binary(&self, key: &Blake2b256Hash) -> Result<Vec<Vec<u8>>, HistoryError> {
        match self.fetch_data(PREFIX_DATUM, key)? {
            Some(PersistedData::Data(data_leaf)) => Ok(bincode::deserialize(&data_leaf.bytes)
                .expect("RSpace History Reader Impl: Failed to deserialize")),
            Some(p) => {
                panic!(
                    "Found unexpected leaf while looking for data at key {:?}, data: {:?}",
                    key, p
                );
            }
            None => Ok(Vec::new()),
        }
    }

    fn get_continuations_proj(
        &self,
        key: &Blake2b256Hash,
    ) -> Result<Vec<WaitingContinuation<P, K>>, HistoryError> {
        match self.fetch_data(PREFIX_KONT, key)? {
            Some(PersistedData::Continuations(continuation_leaf)) => {
                Ok(decode_continuations(&continuation_leaf.bytes))
            }
            Some(p) => {
                panic!(
                    "Found unexpected leaf while looking for continuations at key {:?}, data: {:?}",
                    key, p
                );
            }
            None => Ok(Vec::new()),
        }
    }

    fn get_continuations_proj_binary(
        &self,
        key: &Blake2b256Hash,
    ) -> Result<Vec<Vec<u8>>, HistoryError> {
        match self.fetch_data(PREFIX_KONT, key)? {
            Some(PersistedData::Continuations(continuation_leaf)) => {
                Ok(bincode::deserialize(&continuation_leaf.bytes)
                    .expect("RSpace History Reader Impl: Failed to deserialize"))
            }
            Some(p) => {
                panic!(
                    "Found unexpected leaf while looking for continuations at key {:?}, data: {:?}",
                    key, p
                );
            }
            None => Ok(Vec::new()),
        }
    }

    fn get_joins_proj(&self, key: &Blake2b256Hash) -> Result<Vec<Vec<C>>, HistoryError> {
        match self.fetch_data(PREFIX_JOINS, key)? {
            Some(PersistedData::Joins(joins_leaf)) => Ok(decode_joins(&joins_leaf.bytes)),
            Some(p) => {
                panic!(
                    "Found unexpected leaf while looking for joins at key {:?}, data: {:?}",
                    key, p
                );
            }
            None => Ok(Vec::new()),
        }
    }

    fn get_joins_proj_binary(&self, key: &Blake2b256Hash) -> Result<Vec<Vec<u8>>, HistoryError> {
        match self.fetch_data(PREFIX_JOINS, key)? {
            Some(PersistedData::Joins(joins_leaf)) => Ok(bincode::deserialize(&joins_leaf.bytes)
                .expect("RSpace History Reader Impl: Failed to deserialize")),
            Some(p) => {
                panic!(
                    "Found unexpected leaf while looking for joins at key {:?}, data: {:?}",
                    key, p
                );
            }
            None => Ok(Vec::new()),
        }
    }

    fn base(&self) -> Box<dyn HistoryReaderBase<C, P, A, K>> {
        struct HistoryReaderBaseImpl<C, P, A, K> {
            outer: Arc<RSpaceHistoryReaderImpl<C, P, A, K>>,
        }

        impl<C, P, A, K> HistoryReaderBase<C, P, A, K> for HistoryReaderBaseImpl<C, P, A, K>
        where
            C: Clone + for<'de> Deserialize<'de> + Serialize + 'static + Sync + Send,
            P: Clone + for<'de> Deserialize<'de> + 'static + Sync + Send,
            A: Clone + for<'de> Deserialize<'de> + 'static + Sync + Send,
            K: Clone + for<'de> Deserialize<'de> + 'static + Sync + Send,
        {
            fn get_data_proj(&self, key: &C) -> Vec<Datum<A>> {
                self.outer
                    .get_data_proj(&hash(key))
                    .expect("Failed to get data proj")
            }

            fn get_continuations_proj(&self, key: &Vec<C>) -> Vec<WaitingContinuation<P, K>> {
                self.outer
                    .get_continuations_proj(&hash_from_vec(key))
                    .expect("Failed to get continuations proj")
            }

            fn get_joins_proj(&self, key: &C) -> Vec<Vec<C>> {
                self.outer
                    .get_joins_proj(&hash(key))
                    .expect("Failed to get joins proj")
            }
        }

        let outer_arc = Arc::new(self.clone());
        Box::new(HistoryReaderBaseImpl { outer: outer_arc })
    }

    fn get_data_proj_generic(&self, key: &C) -> Vec<Datum<A>> {
        self.get_data_proj(&hash(key))
            .expect("Failed to get data proj")
    }

    fn get_continuations_proj_generic(&self, key: &Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.get_continuations_proj(&hash_from_vec(key))
            .expect("Failed to get continuations proj")
    }

    fn get_joins_proj_generic(&self, key: &C) -> Vec<Vec<C>> {
        self.get_joins_proj(&hash(key))
            .expect("Failed to get joins proj")
    }
}
