// See casper/src/main/scala/coop/rchain/casper/ReportingCasper.scala

use async_trait::async_trait;
use std::{collections::HashSet, sync::Arc};

use block_storage::rust::{
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    key_value_block_store::KeyValueBlockStore,
};
use models::{
    rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation},
    rust::casper::protocol::casper_message::{BlockMessage, ProcessedDeploy, SystemDeployData},
};
use rholang::rust::interpreter::{
    accounting::_cost,
    reduce::DebruijnInterpreter,
    system_processes::{BlockData, DeployData, InvalidBlocks},
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

type RhoReportingRspace = ReportingRspace<Par, BindPattern, ListParWithRandom, TaggedContinuation>;

/// Trait for reporting casper functionality
#[async_trait]
pub trait ReportingCasper: Send + Sync {
    async fn trace(&self, block: &BlockMessage) -> Result<ReplayResult, String>;
}

/// No-op implementation that returns empty results
pub struct NoopReportingCasper;

#[async_trait]
impl ReportingCasper for NoopReportingCasper {
    async fn trace(&self, _block: &BlockMessage) -> Result<ReplayResult, String> {
        Ok(ReplayResult {
            deploy_report_result: Vec::new(),
            system_deploy_report_result: Vec::new(),
            post_state_hash: ByteString::from("empty".as_bytes()),
        })
    }
}

/// Real implementation using RhoReporter
/*
At the moment, we have only one dead_code annotation in Casper, related to RhoReporterCasper.
In practice, this is not truly dead code, since the fields are stored and used within the rho_reporter function.
However, Rust treats fields that are stored but never read as dead code.
This annotation will no longer be needed once the trace logic is implemented,
which is currently marked as todo!("RhoReporter.trace implementation pending")
 */
#[allow(dead_code)]
pub struct RhoReporterCasper {
    rspace_store: RSpaceStore,
    block_store: KeyValueBlockStore,
    block_dag_storage: BlockDagKeyValueStorage,
}

#[async_trait]
impl ReportingCasper for RhoReporterCasper {
    async fn trace(&self, _block: &BlockMessage) -> Result<ReplayResult, String> {
        // Real implementation will involve:
        // 1. Creating reporting rspace from store
        // 2. Creating reporting runtime
        // 3. Replaying block deploys
        // 4. Collecting reporting events
        todo!("RhoReporter.trace implementation pending")
    }
}

/// Factory function to create noop reporting casper
pub fn noop() -> Arc<dyn ReportingCasper> {
    Arc::new(NoopReportingCasper)
}

/// Factory function to create rho reporter with real reporting capability
pub fn rho_reporter(
    rspace_store: &RSpaceStore,
    block_store: &KeyValueBlockStore,
    block_dag_storage: &BlockDagKeyValueStorage,
) -> Arc<dyn ReportingCasper> {
    Arc::new(RhoReporterCasper {
        rspace_store: rspace_store.clone(),
        block_store: block_store.clone(),
        block_dag_storage: block_dag_storage.clone(),
    })
}

/// Runtime for reporting with deploy data support (PR #165 backport)
///
/// # PR #165 Changes in Scala:
/// In the original Scala code, PR #165 added:
/// 1. `deployDataRef: Ref[F, DeployData]` parameter to ReportingRuntime class
/// 2. Updated constructor to pass deployDataRef to parent ReplayRhoRuntimeImpl
/// 3. Updated ReportingRuntime.apply() factory method to:
///    - Destructure deployRef from createRhoEnv (returns 4-tuple now)
///    - Pass deployRef when constructing new ReportingRuntime instance
///
/// # Current Rust Implementation:
/// We added the `deploy_data_ref` field to the struct definition.
/// However, the actual runtime construction code is not yet implemented (see `todo!` in trace() method).
/// When ReportingRuntime creation is eventually implemented, the deploy_data_ref field
/// will need to be properly initialized and passed, following the same pattern as
/// block_data and invalid_blocks_param.
pub struct ReportingRuntime {
    pub reducer: DebruijnInterpreter,
    pub space: RhoReportingRspace,
    pub cost: _cost,
    pub block_data: Arc<tokio::sync::RwLock<BlockData>>,
    pub invalid_blocks_param: InvalidBlocks,
    pub deploy_data_ref: Arc<tokio::sync::RwLock<DeployData>>, // Added in PR #165
    pub merge_chs: Arc<std::sync::RwLock<HashSet<Par>>>,
}
