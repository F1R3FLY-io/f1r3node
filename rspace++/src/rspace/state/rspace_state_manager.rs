// See rspace++/src/main/scala/state/RSpacePlusPlusStateManager.scala
// See shared/src/main/scala/coop/rchain/state/StateManager.scala
// See rspace++/src/main/scala/state/RSpacePlusPlusStateManagerImpl.scala

use crate::rspace::{
    errors::RootError,
    state::{
        instances::{
            rspace_exporter_store::RSpaceExporterImpl, rspace_importer_store::RSpaceImporterImpl,
        },
        rspace_exporter::RSpaceExporter,
    },
};

// TODO: Don't need RSpaceExporter and RSpaceImporter traits

pub struct RSpaceStateManager {
    pub exporter: RSpaceExporterImpl,
    pub importer: RSpaceImporterImpl,
}

impl RSpaceStateManager {
    pub fn new(exporter: RSpaceExporterImpl, importer: RSpaceImporterImpl) -> Self {
        Self { exporter, importer }
    }

    /// Returns true if the RSpace has no root (is empty), false otherwise.
    pub fn is_empty(&self) -> bool {
        self.has_root()
    }

    /// Returns true if the exporter can successfully get a root, false if there's no root.
    pub fn has_root(&self) -> bool {
        match self.exporter.get_root() {
            Ok(_) => true,
            Err(RootError::UnknownRootError(_)) => false,
            Err(_) => false,
        }
    }
}
