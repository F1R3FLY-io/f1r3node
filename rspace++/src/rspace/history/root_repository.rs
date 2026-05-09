use super::instances::radix_history::RadixHistory;
use crate::rspace::errors::RootError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::history::roots_store::RootsStore;

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
                tracing::debug!("[RootRepository] validateAndSetCurrentRoot OK: {}", root);
                Ok(())
            }
            None => {
                tracing::error!(
                    "[RootRepository] validateAndSetCurrentRoot FAILED: {} not in roots store",
                    root
                );
                Err(RootError::UnknownRootError(format!("unknown root: {}", root)))
            }
        }
    }

    /// Pure lookup: returns true if the root is recorded in the store.
    /// Companion to `validate_and_set_current_root` without the side-effect
    /// of updating the current-root pointer.
    pub fn contains_root(&self, root: &Blake2b256Hash) -> Result<bool, RootError> {
        self.roots_store.contains_root(root)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;

    use super::*;

    /// Minimal in-memory RootsStore for testing the lookup-vs-set distinction.
    /// Tracks recorded roots and the current-root pointer separately so tests
    /// can observe whether `contains_root` updates the pointer (it must not).
    struct InmemRootsStore {
        roots: Mutex<HashSet<Blake2b256Hash>>,
        current: Mutex<Option<Blake2b256Hash>>,
    }

    impl RootsStore for InmemRootsStore {
        fn current_root(&self) -> Result<Option<Blake2b256Hash>, RootError> {
            Ok(self.current.lock().unwrap().clone())
        }

        fn validate_and_set_current_root(
            &self,
            key: Blake2b256Hash,
        ) -> Result<Option<Blake2b256Hash>, RootError> {
            if self.roots.lock().unwrap().contains(&key) {
                *self.current.lock().unwrap() = Some(key.clone());
                Ok(Some(key))
            } else {
                Ok(None)
            }
        }

        fn record_root(&self, key: &Blake2b256Hash) -> Result<(), RootError> {
            self.roots.lock().unwrap().insert(key.clone());
            *self.current.lock().unwrap() = Some(key.clone());
            Ok(())
        }

        fn contains_root(&self, key: &Blake2b256Hash) -> Result<bool, RootError> {
            Ok(self.roots.lock().unwrap().contains(key))
        }
    }

    fn make_repo() -> RootRepository {
        RootRepository {
            roots_store: Box::new(InmemRootsStore {
                roots: Mutex::new(HashSet::new()),
                current: Mutex::new(None),
            }),
        }
    }

    fn hash_for(seed: u8) -> Blake2b256Hash {
        let mut bytes = vec![0u8; 32];
        bytes[0] = seed;
        Blake2b256Hash::from_bytes(bytes)
    }

    #[test]
    fn contains_root_returns_true_for_present_root() {
        let repo = make_repo();
        let h = hash_for(1);
        repo.commit(&h).unwrap();
        assert!(repo.contains_root(&h).unwrap());
    }

    #[test]
    fn contains_root_returns_false_for_absent_root() {
        let repo = make_repo();
        assert!(!repo.contains_root(&hash_for(42)).unwrap());
    }

    /// The contract that distinguishes contains_root from
    /// validate_and_set_current_root: a positive lookup must NOT mutate the
    /// current-root pointer. This is the explicit reason the API was added —
    /// LFS forward-horizon sync calls contains_root in a hot loop and any
    /// pointer churn would interfere with concurrent reset/checkpoint flows.
    #[test]
    fn contains_root_does_not_update_current_root_pointer() {
        let repo = make_repo();
        let a = hash_for(1);
        let b = hash_for(2);
        // Record both; current-root advances to b after the second commit.
        repo.commit(&a).unwrap();
        repo.commit(&b).unwrap();
        assert_eq!(
            repo.roots_store.current_root().unwrap(),
            Some(b.clone()),
            "precondition: current is b"
        );
        // Look up a (present, but not current). Pointer must stay on b.
        assert!(repo.contains_root(&a).unwrap());
        assert_eq!(
            repo.roots_store.current_root().unwrap(),
            Some(b),
            "contains_root must not move the current-root pointer"
        );
    }
}
