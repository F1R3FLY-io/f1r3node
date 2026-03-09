// See casper/src/main/scala/coop/rchain/casper/api/BlockReportAPI.scala

use std::sync::Arc;

use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use dashmap::DashMap;
use models::casper::{
    BlockEventInfo, DeployInfoWithEventData, ReportProto, SingleReport,
    SystemDeployInfoWithEventData,
};
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{BlockMessage, SystemDeployData},
};
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::reporting_transformer::ReportingTransformer;
use shared::rust::{store::key_value_typed_store::KeyValueTypedStore, ByteString};
use tokio::sync::Semaphore;

use crate::rust::{
    api::block_api::BlockAPI, engine::engine_cell::EngineCell, report_store::ReportStore,
    reporting_casper::ReportingCasper, reporting_proto_transformer::ReportingProtoTransformer,
    safety_oracle::CliqueOracleImpl,
};

/// Domain-specific errors for BlockReportAPI operations
#[derive(Debug, thiserror::Error)]
pub enum BlockReportError {
    #[error("Casper instance not available")]
    CasperNotInitialized,
    #[error("Block report can only be executed on read-only RNode")]
    ReadOnlyRequired,
    #[error("Block {0:?} not found")]
    BlockNotFound(BlockHash),
    #[error("Failed to trace block: {0}")]
    ReplayFailed(String),
    #[error("Block info error: {0}")]
    BlockInfoError(String),
    #[error("Report store error: {0}")]
    StoreError(String),
    #[error("Failed to acquire semaphore: {0}")]
    SemaphoreError(String),
}

pub type ApiErr<T> = Result<T, BlockReportError>;

/// BlockReportAPI provides functionality to replay blocks and generate event reports
#[derive(Clone)]
pub struct BlockReportAPI {
    reporting_casper: Arc<dyn ReportingCasper>,
    report_store: ReportStore,
    engine_cell: EngineCell,
    #[allow(dead_code)] // Kept for API compatibility, but we use casper's block_store instead
    block_store: KeyValueBlockStore,
    #[allow(dead_code)] // Part of constructor signature matching Scala, not directly used
    oracle: CliqueOracleImpl,
    /// Thread-safe map of block hashes to semaphores for per-block locking
    /// Equivalent to Scala's `blockLockMap: TrieMap[BlockHash, MetricsSemaphore[F]]`
    block_lock_map: Arc<DashMap<BlockHash, Arc<Semaphore>>>,
    /// Transformer for converting reporting events to protobuf format
    report_transformer: Arc<ReportingProtoTransformer>,
    /// When true, allows block reports on validator nodes (bypasses read-only check)
    dev_mode: bool,
}

impl BlockReportAPI {
    /// Create a new BlockReportAPI
    pub fn new(
        reporting_casper: Arc<dyn ReportingCasper>,
        report_store: ReportStore,
        engine_cell: EngineCell,
        block_store: KeyValueBlockStore,
        oracle: CliqueOracleImpl,
        dev_mode: bool,
    ) -> Self {
        Self {
            reporting_casper,
            report_store,
            engine_cell,
            block_store,
            oracle,
            block_lock_map: Arc::new(DashMap::new()),
            report_transformer: Arc::new(ReportingProtoTransformer::new()),
            dev_mode,
        }
    }

    /// Replay a block and create BlockEventInfo
    async fn replay_block(
        &self,
        block: &BlockMessage,
        casper: &Arc<dyn crate::rust::casper::MultiParentCasper + Send + Sync>,
    ) -> ApiErr<BlockEventInfo> {
        let report_result = self
            .reporting_casper
            .trace(block)
            .await
            .map_err(|e| BlockReportError::ReplayFailed(e))?;

        let light_block = BlockAPI::get_light_block_info(casper.as_ref(), block)
            .await
            .map_err(|e| BlockReportError::BlockInfoError(e.to_string()))?;

        let deploys = self.create_deploy_report(&report_result.deploy_report_result);

        let sys_deploys =
            self.create_system_deploy_report(&report_result.system_deploy_report_result);

        let post_state_hash_bytes: Bytes = report_result.post_state_hash.into();
        Ok(BlockEventInfo {
            block_info: Some(light_block).into(),
            deploys,
            system_deploys: sys_deploys,
            post_state_hash: post_state_hash_bytes,
        })
    }

    /// Get block report with locking to prevent concurrent replays of the same block
    async fn block_report_within_lock(
        &self,
        force_replay: bool,
        block: &BlockMessage,
        casper: &Arc<dyn crate::rust::casper::MultiParentCasper + Send + Sync>,
    ) -> ApiErr<BlockEventInfo> {
        let block_hash = block.block_hash.clone();

        let semaphore = self
            .block_lock_map
            .entry(block_hash.clone())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone();

        metrics::gauge!("block_report.lock.queue_size", "source" => "casper").increment(1.0);
        let _permit = semaphore
            .acquire()
            .await
            .map_err(|e| BlockReportError::SemaphoreError(e.to_string()))?;
        metrics::gauge!("block_report.lock.queue_size", "source" => "casper").decrement(1.0);

        let result = self.block_report_inner(force_replay, block, casper).await;

        // Remove semaphore entry to prevent unbounded growth of the lock map
        self.block_lock_map.remove(&block_hash);

        result
    }

    /// Inner block report logic (separated to ensure lock map cleanup on all paths)
    async fn block_report_inner(
        &self,
        force_replay: bool,
        block: &BlockMessage,
        casper: &Arc<dyn crate::rust::casper::MultiParentCasper + Send + Sync>,
    ) -> ApiErr<BlockEventInfo> {
        let block_hash_bytes: ByteString = block.block_hash.to_vec().into();
        let cached = self
            .report_store
            .get(&vec![block_hash_bytes.clone()])
            .map_err(|e| BlockReportError::StoreError(e.to_string()))?;

        if let Some(Some(cached_report)) = cached.first() {
            if !force_replay {
                return Ok(cached_report.clone());
            }
        }

        let report = self.replay_block(block, casper).await?;

        self.report_store
            .put(vec![(block_hash_bytes, report.clone())])
            .map_err(|e| BlockReportError::StoreError(e.to_string()))?;

        Ok(report)
    }

    /// Get block report for a given block hash
    pub async fn block_report(
        &self,
        hash: BlockHash,
        force_replay: bool,
    ) -> ApiErr<BlockEventInfo> {
        let eng = self.engine_cell.get().await;
        let casper = eng
            .with_casper()
            .ok_or(BlockReportError::CasperNotInitialized)?;

        let validator_opt = casper.get_validator();
        if validator_opt.is_some() && !self.dev_mode {
            return Err(BlockReportError::ReadOnlyRequired);
        }

        let casper_block_store = casper.block_store();
        let block_opt = casper_block_store
            .get(&hash)
            .map_err(|e| BlockReportError::StoreError(e.to_string()))?;

        let block = block_opt.ok_or_else(|| BlockReportError::BlockNotFound(hash))?;

        self.block_report_within_lock(force_replay, &block, &casper)
            .await
    }

    /// Create system deploy report from replay results
    fn create_system_deploy_report(
        &self,
        result: &[crate::rust::reporting_casper::SystemDeployReportResult],
    ) -> Vec<SystemDeployInfoWithEventData> {
        result
            .iter()
            .map(|sd| {
                let system_deploy_proto =
                    SystemDeployData::to_proto(sd.processed_system_deploy.clone());

                let report: Vec<SingleReport> = sd
                    .events
                    .iter()
                    .map(|event_batch| {
                        let events: Vec<ReportProto> = event_batch
                            .iter()
                            .map(|event| {
                                ReportingTransformer::transform_event(
                                    self.report_transformer.as_ref(),
                                    event,
                                )
                            })
                            .collect();

                        SingleReport { events }
                    })
                    .collect();

                SystemDeployInfoWithEventData {
                    system_deploy: Some(system_deploy_proto).into(),
                    report,
                }
            })
            .collect()
    }

    /// Create deploy report from replay results
    fn create_deploy_report(
        &self,
        result: &[crate::rust::reporting_casper::DeployReportResult],
    ) -> Vec<DeployInfoWithEventData> {
        result
            .iter()
            .map(|p| {
                let deploy_info = p.processed_deploy.clone().to_deploy_info();

                let report: Vec<SingleReport> = p
                    .events
                    .iter()
                    .map(|event_batch| {
                        let events: Vec<ReportProto> = event_batch
                            .iter()
                            .map(|event| {
                                ReportingTransformer::transform_event(
                                    self.report_transformer.as_ref(),
                                    event,
                                )
                            })
                            .collect();

                        SingleReport { events }
                    })
                    .collect();

                DeployInfoWithEventData {
                    deploy_info: Some(deploy_info).into(),
                    report,
                }
            })
            .collect()
    }
}
