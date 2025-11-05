// See node/src/main/scala/coop/rchain/node/state/instances/RNodeStateManagerImpl.scala

use casper::rust::{errors::CasperError, state::instances::BlockStateManager};
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;

pub struct RNodeStateManager {
    rspace_state_manager: RSpaceStateManager,
    block_state_manager: BlockStateManager,
}

impl RNodeStateManager {
    pub fn new(
        rspace_state_manager: RSpaceStateManager,
        block_state_manager: BlockStateManager,
    ) -> Self {
        Self {
            rspace_state_manager,
            block_state_manager,
        }
    }

    /// Checks if both RSpace and block state are empty
    pub fn is_empty(&self) -> Result<bool, CasperError> {
        let rspace_empty = self.rspace_state_manager.is_empty();
        let block_empty = self.block_state_manager.is_empty()?;
        Ok(rspace_empty && block_empty)
    }
}

