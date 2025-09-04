use super::instances::radix_history::RadixHistory;
use crate::rspace::{
    errors::RootError, hashing::blake2b256_hash::Blake2b256Hash, history::roots_store::RootsStore,
};

// See rspace/src/main/scala/coop/rchain/rspace/history/RootRepository.scala
pub struct RootRepository {
    pub roots_store: Box<dyn RootsStore>,
}

impl RootRepository {
    pub fn commit(&self, root: &Blake2b256Hash) -> Result<(), RootError> {
        self.roots_store.record_root(root)
    }

    pub fn current_root(&self) -> Result<Blake2b256Hash, RootError> {
        match self.roots_store.current_root()? {
            None => {
                let empty_root_hash = RadixHistory::empty_root_node_hash();
                self.roots_store.record_root(&empty_root_hash)?;
                Ok(empty_root_hash)
            }
            Some(root) => Ok(root),
        }
    }

    pub fn validate_and_set_current_root(&self, root: Blake2b256Hash) -> Result<(), RootError> {
        match self.roots_store.validate_and_set_current_root(root)? {
            Some(_) => Ok(()),
            None => Err(RootError::UnknownRootError("unknown root".to_string())),
        }
    }
}
