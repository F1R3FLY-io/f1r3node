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
        tracing::debug!("[RootRepository] commit {}", root);
        self.roots_store.record_root(root)
    }

    pub fn current_root(&self) -> Result<Blake2b256Hash, RootError> {
        match self.roots_store.current_root()? {
            None => {
                let empty_root_hash = RadixHistory::empty_root_node_hash();
                tracing::debug!(
                    "[RootRepository] currentRoot: empty store, recording {}",
                    empty_root_hash
                );
                self.roots_store.record_root(&empty_root_hash)?;
                Ok(empty_root_hash)
            }
            Some(root) => {
                tracing::debug!("[RootRepository] currentRoot: {}", root);
                Ok(root)
            }
        }
    }

    pub fn validate_and_set_current_root(&self, root: Blake2b256Hash) -> Result<(), RootError> {
        match self
            .roots_store
            .validate_and_set_current_root(root.clone())?
        {
            Some(_) => {
                tracing::debug!(
                    "[RootRepository] validateAndSetCurrentRoot OK: {}",
                    root
                );
                Ok(())
            }
            None => {
                tracing::error!(
                    "[RootRepository] validateAndSetCurrentRoot FAILED: {} not in roots store",
                    root
                );
                Err(RootError::UnknownRootError(format!(
                    "unknown root: {}",
                    root
                )))
            }
        }
    }
}
