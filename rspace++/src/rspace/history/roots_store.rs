// See rspace/src/main/scala/coop/rchain/rspace/history/RootsStore.scala

use std::sync::{Arc, Mutex};

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
    pub fn roots_store(store: Arc<Mutex<Box<dyn KeyValueStore>>>) -> impl RootsStore {
        struct RootsStoreInstance {
            store: Arc<Mutex<Box<dyn KeyValueStore>>>,
        }

        impl RootsStore for RootsStoreInstance {
            fn current_root(&self) -> Result<Option<Blake2b256Hash>, RootError> {
                let current_root_name: ByteBuffer = "current-root".as_bytes().to_vec();

                let store_lock = self
                    .store
                    .lock()
                    .expect("Roots Store: Failed to acquire lock on store");

                let bytes = store_lock.get_one(&current_root_name)?;

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
                // println!("\nhit validate_and_set_current_root, key: {}", key);

                let current_root_name: ByteBuffer = "current-root".as_bytes().to_vec();
                let key_bytes = key.bytes();

                let mut store_lock = self
                    .store
                    .lock()
                    .expect("Roots Store: Failed to acquire lock on store");

                match store_lock.get_one(&key_bytes)? {
                    Some(_) => {
                        store_lock.put_one(current_root_name, key_bytes)?;
                        // println!("\nroots_store after validate_and_set_current_root: ");
                        // store_lock.print_store();
                        Ok(Some(key))
                    }
                    None => {
                        // println!("\nroots_store after validate_and_set_current_root: ");
                        // store_lock.print_store();
                        Ok(None)
                    }
                }
            }

            fn record_root(&self, key: &Blake2b256Hash) -> Result<(), RootError> {
                // println!("\nhit record_root, key: {}", key);

                let tag: ByteBuffer = "tag".as_bytes().to_vec();
                let current_root_name: ByteBuffer = "current-root".as_bytes().to_vec();
                let key_bytes = key.bytes();

                let mut store_lock = self
                    .store
                    .lock()
                    .expect("Roots Store: Failed to acquire lock on store");

                store_lock.put_one(key_bytes.to_vec(), tag)?;
                store_lock.put_one(current_root_name, key_bytes.to_vec())?;

                // println!("\nroots_store after record root: ");
                // store_lock.print_store();

                Ok(())
            }
        }

        RootsStoreInstance { store }
    }
}
