// See rspace/src/main/scala/coop/rchain/rspace/history/RootsStore.scala

use std::sync::Arc;

use crate::rspace::{errors::RootError, hashing::blake2b256_hash::Blake2b256Hash};
use shared::rust::{ByteBuffer, store::key_value_store::KeyValueStore};
pub trait RootsStore: Send + Sync {
    fn current_root(&self) -> Result<Option<Blake2b256Hash>, RootError>;

    fn validate_and_set_current_root(
        &self,
        key: Blake2b256Hash,
    ) -> Result<Option<Blake2b256Hash>, RootError>;

    fn record_root(&self, key: &Blake2b256Hash) -> Result<(), RootError>;
}

pub struct RootsStoreInstances;

impl RootsStoreInstances {
    pub fn roots_store(store: Arc<dyn KeyValueStore>) -> impl RootsStore {
        struct RootsStoreInstance {
            store: Arc<dyn KeyValueStore>,
        }

        impl RootsStore for RootsStoreInstance {
            fn current_root(&self) -> Result<Option<Blake2b256Hash>, RootError> {
                let current_root_name: ByteBuffer = "current-root".as_bytes().to_vec();

                let bytes = self.store.get_one(&current_root_name)?;

                let maybe_decoded = match bytes {
                    Some(b) => Some(Blake2b256Hash::from_bytes(b)),
                    None => None,
                };

                Ok(maybe_decoded)
            }

            fn validate_and_set_current_root(
                &self,
                key: Blake2b256Hash,
            ) -> Result<Option<Blake2b256Hash>, RootError> {
                let current_root_name: ByteBuffer = "current-root".as_bytes().to_vec();
                let key_bytes = key.bytes();

                match self.store.get_one(&key_bytes)? {
                    Some(_) => {
                        self.store.put_one(current_root_name, key_bytes)?;
                        Ok(Some(key))
                    }
                    None => Ok(None),
                }
            }

            fn record_root(&self, key: &Blake2b256Hash) -> Result<(), RootError> {
                let tag: ByteBuffer = "tag".as_bytes().to_vec();
                let current_root_name: ByteBuffer = "current-root".as_bytes().to_vec();
                let key_bytes = key.bytes();

                self.store.put_one(key_bytes.to_vec(), tag)?;
                self.store.put_one(current_root_name, key_bytes.to_vec())?;

                Ok(())
            }
        }

        RootsStoreInstance { store }
    }
}
