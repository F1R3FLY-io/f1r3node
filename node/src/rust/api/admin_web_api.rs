//! Admin Web API implementation for F1r3fly node
//! Ported from node/src/main/scala/coop/rchain/node/api/AdminWebApi.scala

use casper::rust::api::block_api::BlockAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::state::instances::proposer_state::ProposerState;
use casper::rust::ProposeFunction;
use eyre::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Admin Web API trait defining the interface for admin HTTP endpoints
#[async_trait::async_trait]
pub trait AdminWebApi {
    /// Trigger a block proposal
    async fn propose(&self) -> Result<String>;

    /// Get the result of the latest proposal
    async fn propose_result(&self) -> Result<String>;
}

/// Admin Web API implementation
pub struct AdminWebApiImpl {
    trigger_propose_f_opt: Option<Arc<ProposeFunction>>,
    proposer_state_ref_opt: Option<Arc<RwLock<ProposerState>>>,
    engine_cell: Arc<EngineCell>,
}

impl AdminWebApiImpl {
    pub fn new(
        trigger_propose_f_opt: Option<Arc<ProposeFunction>>,
        proposer_state_ref_opt: Option<Arc<RwLock<ProposerState>>>,
        engine_cell: Arc<EngineCell>,
    ) -> Self {
        Self {
            trigger_propose_f_opt,
            proposer_state_ref_opt,
            engine_cell,
        }
    }
}

#[async_trait::async_trait]
impl AdminWebApi for AdminWebApiImpl {
    async fn propose(&self) -> Result<String> {
        match &self.trigger_propose_f_opt {
            Some(trigger_propose_f) => {
                BlockAPI::create_block(&self.engine_cell, trigger_propose_f, false).await
            }
            None => Err(eyre::eyre!("Propose error: read-only node.")),
        }
    }

    async fn propose_result(&self) -> Result<String> {
        match &self.proposer_state_ref_opt {
            Some(proposer_state_ref) => {
                let mut proposer_state = proposer_state_ref.write().await;
                BlockAPI::get_propose_result(&mut proposer_state).await
            }
            None => Err(eyre::eyre!("Error: read-only node.")),
        }
    }
}
