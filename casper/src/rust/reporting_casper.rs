// See casper/src/main/scala/coop/rchain/casper/ReportingCasper.scala

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use block_storage::rust::{
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    key_value_block_store::KeyValueBlockStore,
};
use models::{
    rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation},
    rust::{
        block::state_hash::StateHash,
        block_hash::BlockHash,
        casper::protocol::casper_message::{
            BlockMessage, ProcessedDeploy, ProcessedSystemDeploy, SystemDeployData,
        },
        validator::Validator,
    },
};
use rholang::rust::interpreter::{
    accounting::_cost,
    reduce::DebruijnInterpreter,
    system_processes::{BlockData, InvalidBlocks},
};
use rspace_plus_plus::rspace::{
    reporting_rspace::{ReportingEvent, ReportingRspace},
    rspace::RSpaceStore,
};
use shared::rust::ByteString;

/// Deploy details + reporting events
#[derive(Clone, Debug)]
pub struct DeployReportResult {
    pub processed_deploy: ProcessedDeploy,
    pub events: Vec<Vec<ReportingEvent<Par, BindPattern, ListParWithRandom, TaggedContinuation>>>,
}

/// System deploy details + reporting events
#[derive(Clone, Debug)]
pub struct SystemDeployReportResult {
    pub processed_system_deploy: SystemDeployData,
    pub events: Vec<Vec<ReportingEvent<Par, BindPattern, ListParWithRandom, TaggedContinuation>>>,
}

/// Aggregated replay results
#[derive(Clone, Debug)]
pub struct ReplayResult {
    pub deploy_report_result: Vec<DeployReportResult>,
    pub system_deploy_report_result: Vec<SystemDeployReportResult>,
    pub post_state_hash: ByteString,
}

/// Main reporting interface
pub struct ReportingCasper;

type RhoReportingRspace = ReportingRspace<Par, BindPattern, ListParWithRandom, TaggedContinuation>;

impl ReportingCasper {
    pub async fn trace(&self, block: &BlockMessage) -> Result<ReplayResult, String> {
        todo!()
    }

    pub async fn replay_deploys(
        &self,
        runtime: &ReportingRuntime,
        start_hash: &StateHash,
        terms: Vec<ProcessedDeploy>,
        system_deploys: Vec<ProcessedSystemDeploy>,
        with_cost_accounting: bool,
        block_data: &BlockData,
        invalid_blocks: HashMap<BlockHash, Validator>,
    ) -> Result<ReplayResult, String> {
        todo!()
    }
}

pub fn rho_reporter(
    rspace_store: &RSpaceStore,
    block_store: &KeyValueBlockStore,
    block_dag_storage: &BlockDagKeyValueStorage,
) -> ReportingRuntime {
    todo!()
}

pub struct ReportingRuntime {
    pub reducer: DebruijnInterpreter,
    pub space: RhoReportingRspace,
    pub cost: _cost,
    pub block_data: Arc<tokio::sync::RwLock<BlockData>>,
    pub invalid_blocks_param: InvalidBlocks,
    pub merge_chs: Arc<std::sync::RwLock<HashSet<Par>>>,
}
