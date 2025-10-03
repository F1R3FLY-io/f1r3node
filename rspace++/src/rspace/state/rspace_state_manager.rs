// See rspace++/src/main/scala/state/RSpacePlusPlusStateManager.scala
// See shared/src/main/scala/coop/rchain/state/StateManager.scala
// See rspace++/src/main/scala/state/RSpacePlusPlusStateManagerImpl.scala

use std::sync::{Arc, Mutex};

use crate::rspace::{
    errors::RootError,
    state::{rspace_exporter::RSpaceExporter, rspace_importer::RSpaceImporter},
};

pub struct RSpaceStateManager {
    pub exporter: Arc<Mutex<Box<dyn RSpaceExporter>>>,
    pub importer: Arc<Mutex<Box<dyn RSpaceImporter>>>,
}

impl RSpaceStateManager {
    pub fn new(
        exporter: Arc<Mutex<Box<dyn RSpaceExporter>>>,
        importer: Arc<Mutex<Box<dyn RSpaceImporter>>>,
    ) -> Self {
        Self { exporter, importer }
    }

    /// Returns true if the RSpace has no root (is empty), false otherwise.
    pub fn is_empty(&self) -> bool {
        self.has_root()
    }

    /// Returns true if the exporter can successfully get a root, false if there's no root.
    pub fn has_root(&self) -> bool {
        let exporter_lock = self
            .exporter
            .lock()
            .expect("Failed to acquire lock on exporter");

        match exporter_lock.get_root() {
            Ok(_) => true,
            Err(RootError::UnknownRootError(_)) => false,
            Err(_) => false,
        }
    }
}
