// See casper/src/main/scala/coop/rchain/casper/api/BlockReportAPI.scala

use std::sync::Arc;

use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use dashmap::DashMap;
use models::casper::{
    BlockEventInfo, DeployInfoWithEventData, ReportProto, SingleReport, SystemDeployInfoWithEventData,
};
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{BlockMessage, SystemDeployData},
};
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::reporting_transformer::ReportingTransformer;
use shared::rust::{
    ByteString,
    store::key_value_typed_store::KeyValueTypedStore,
};
use tokio::sync::Semaphore;

use crate::rust::{
    api::block_api::BlockAPI,
    engine::engine_cell::EngineCell,
    report_store::ReportStore,
    reporting_casper::ReportingCasper,
    reporting_proto_transformer::ReportingProtoTransformer,
    safety_oracle::CliqueOracleImpl,
};

pub type ApiErr<T> = eyre::Result<T>;

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
}

impl BlockReportAPI {
    /// Create a new BlockReportAPI
    /// 
    pub fn new(
        reporting_casper: Arc<dyn ReportingCasper>,
        report_store: ReportStore,
        engine_cell: EngineCell,
        block_store: KeyValueBlockStore,
        oracle: CliqueOracleImpl,
    ) -> Self {
        Self {
            reporting_casper,
            report_store,
            engine_cell,
            block_store,
            oracle,
            block_lock_map: Arc::new(DashMap::new()),
            report_transformer: Arc::new(ReportingProtoTransformer::new()),
        }
    }

    /// Replay a block and create BlockEventInfo
    async fn replay_block(
        &self,
        block: &BlockMessage,
    ) -> ApiErr<BlockEventInfo> {
        let eng = self.engine_cell.get().await;
        let casper = eng.with_casper().ok_or_else(|| {
            eyre::eyre!("Casper instance not available")
        })?;

        let report_result = self.reporting_casper.trace(block).await
            .map_err(|e| eyre::eyre!("Failed to trace block: {}", e))?;

        let light_block = BlockAPI::get_light_block_info(casper.as_ref(), block).await
            .map_err(|e| eyre::eyre!("Failed to get light block info: {}", e))?;

        let deploys = self.create_deploy_report(&report_result.deploy_report_result);

        let sys_deploys = self.create_system_deploy_report(&report_result.system_deploy_report_result);

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
    ) -> ApiErr<BlockEventInfo> {
        let block_hash = block.block_hash.clone();

        let semaphore = self.block_lock_map
            .entry(block_hash.clone())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone();

        let _permit = semaphore.acquire().await
            .map_err(|e| eyre::eyre!("Failed to acquire semaphore: {}", e))?;

        let block_hash_bytes: ByteString = block_hash.to_vec().into();
        let cached = self.report_store.get(&vec![block_hash_bytes.clone()])
            .map_err(|e| eyre::eyre!("Failed to get from report store: {}", e))?;

        if let Some(Some(cached_report)) = cached.first() {
            if !force_replay {
                return Ok(cached_report.clone());
            }
        }

        let report = self.replay_block(block).await?;
        
        self.report_store.put(vec![(block_hash_bytes, report.clone())])
            .map_err(|e| eyre::eyre!("Failed to put to report store: {}", e))?;

        Ok(report)
    }

    /// Get block report for a given block hash
    pub async fn block_report(
        &self,
        hash: BlockHash,
        force_replay: bool,
    ) -> ApiErr<BlockEventInfo> {
        let eng = self.engine_cell.get().await;
        let casper = eng.with_casper().ok_or_else(|| {
            eyre::eyre!("Could not get event data.")
        })?;

        let validator_opt = casper.get_validator();
        if validator_opt.is_some() {
            return Err(eyre::eyre!("Block report can only be executed on read-only RNode."));
        }

        // Use the casper's block_store instead of self.block_store to ensure we're using
        // the same store that has all the blocks (including parent blocks)
        let casper_block_store = casper.block_store();
        let block_opt = casper_block_store.get(&hash)
            .map_err(|e| eyre::eyre!("Failed to get block from store: {}", e))?;

        let block = block_opt.ok_or_else(|| {
            eyre::eyre!("Block {:?} not found", hash)
        })?;

        self.block_report_within_lock(force_replay, &block).await
    }

    /// Create system deploy report from replay results
    fn create_system_deploy_report(
        &self,
        result: &[crate::rust::reporting_casper::SystemDeployReportResult],
    ) -> Vec<SystemDeployInfoWithEventData> {
        result.iter().map(|sd| {
            let system_deploy_proto = SystemDeployData::to_proto(sd.processed_system_deploy.clone());
            
            let report: Vec<SingleReport> = sd.events.iter().map(|event_batch| {
                let events: Vec<ReportProto> = event_batch.iter().map(|event| {
                    ReportingTransformer::transform_event(self.report_transformer.as_ref(), event)
                }).collect();
                
                SingleReport { events }
            }).collect();

            SystemDeployInfoWithEventData {
                system_deploy: Some(system_deploy_proto).into(),
                report,
            }
        }).collect()
    }

    /// Create deploy report from replay results
    fn create_deploy_report(
        &self,
        result: &[crate::rust::reporting_casper::DeployReportResult],
    ) -> Vec<DeployInfoWithEventData> {
        result.iter().map(|p| {
            let deploy_info = p.processed_deploy.clone().to_deploy_info();
            
            let report: Vec<SingleReport> = p.events.iter().map(|event_batch| {
                let events: Vec<ReportProto> = event_batch.iter().map(|event| {
                    ReportingTransformer::transform_event(self.report_transformer.as_ref(), event)
                }).collect();
                
                SingleReport { events }
            }).collect();

            DeployInfoWithEventData {
                deploy_info: Some(deploy_info).into(),
                report,
            }
        }).collect()
    }
}
