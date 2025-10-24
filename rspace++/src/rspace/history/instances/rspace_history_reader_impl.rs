use crate::rspace::{
    errors::HistoryError,
    hashing::{
        blake2b256_hash::Blake2b256Hash,
        stable_hash_provider::{hash, hash_from_vec},
    },
    history::{
        cold_store::PersistedData,
        history::History,
        history_reader::{HistoryReader, HistoryReaderBase},
        history_repository::{PREFIX_DATUM, PREFIX_JOINS, PREFIX_KONT},
        history_repository_impl::prepend_bytes,
    },
    internal::{Datum, WaitingContinuation},
    serializers::serializers::{decode_continuations, decode_datums, decode_joins},
};
use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_store::KeyValueStore;
use std::{marker::PhantomData, sync::Arc};

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
        // println!("\nhit fetch_data");
        let read_bytes = self
            .target_history
            .read(prepend_bytes(prefix, &key.bytes()))?;

        match read_bytes {
            Some(ref bytes) => {
                let read_hash = Blake2b256Hash::from_bytes(bytes.to_vec());

                let serialized_read_hash = bincode::serialize(&read_hash.bytes())
                    .expect("RSpace History Reader Impl: Unable to serialize");

                let mut get_opt = self.leaf_store.get_one(&serialized_read_hash)?;

                if get_opt.is_none() {
                    // Try fetch call for imported data. Ideally this should be removed.
                    get_opt = self.leaf_store.get_one(&bytes)?;
                }

                Ok(get_opt.map(|store_value_bytes| {
                    bincode::deserialize(&store_value_bytes)
                        .expect("RSpace History Reader Impl: Failed to deserialize")
                }))
            }
            None => Ok(None),
        }
    }
}

impl<C, P, A, K> HistoryReader<Blake2b256Hash, C, P, A, K> for RSpaceHistoryReaderImpl<C, P, A, K>
where
    C: Clone + for<'a> Deserialize<'a> + Serialize + 'static + Sync + Send,
    P: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
    A: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
    K: Clone + for<'a> Deserialize<'a> + 'static + Sync + Send,
{
    fn root(&self) -> Blake2b256Hash {
        self.target_history.root()
    }

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
