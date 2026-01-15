// See casper/src/main/scala/coop/rchain/casper/ReportingCasper.scala

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use block_storage::rust::{
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    key_value_block_store::KeyValueBlockStore,
};
use models::{
    rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation},
    rust::casper::protocol::casper_message::{BlockMessage, ProcessedDeploy, ProcessedSystemDeploy, SystemDeployData},
};
use rholang::rust::interpreter::{
    rho_runtime::RhoRuntime,
    system_processes::{BlockData, Definition},
};
use rspace_plus_plus::rspace::{
    errors::RSpaceError,
    hashing::blake2b256_hash::Blake2b256Hash,
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
    async fn trace(&self, block: &BlockMessage) -> Result<ReplayResult, String> {
        use crate::rust::genesis::genesis::Genesis;
        use crate::rust::util::proto_util;

        let reporting_rspace = ReportingRuntime::create_reporting_rspace(self.rspace_store.clone())
            .map_err(|e| format!("Failed to create reporting rspace: {}", e))?;

        let mergeable_tag_name = Genesis::non_negative_mergeable_tag_name();
        let mut extra_system_processes = Vec::new();
        let mut reporting_runtime = ReportingRuntime::create_reporting_runtime(
            reporting_rspace,
            mergeable_tag_name,
            &mut extra_system_processes,
        )
        .await
        .map_err(|e| format!("Failed to create reporting runtime: {}", e))?;

        let mut dag = self.block_dag_storage.get_representation();

        let genesis = self
            .block_store
            .get_approved_block()
            .map_err(|e| format!("Failed to get approved block: {}", e))?;

        let is_genesis = genesis
            .as_ref()
            .map(|g| block.block_hash == g.candidate.block.block_hash)
            .unwrap_or(false);

        let invalid_blocks_set = dag.invalid_blocks();
        let invalid_blocks: HashMap<models::rust::block_hash::BlockHash, models::rust::validator::Validator> = invalid_blocks_set
            .iter()
            .map(|block_metadata| (block_metadata.block_hash.clone(), block_metadata.sender.clone()))
            .collect();

        let pre_state_hash_bytes = proto_util::pre_state_hash(block);
        let pre_state_hash = Blake2b256Hash::from_bytes_prost(&pre_state_hash_bytes);

        let block_data = BlockData::from_block(block);

        reporting_runtime.set_block_data(block_data.clone()).await;
        reporting_runtime.set_invalid_blocks(invalid_blocks.clone()).await;

        let unseen_blocks_set = proto_util::unseen_block_hashes(&mut dag, block)
            .map_err(|e| format!("Failed to get unseen block hashes: {}", e))?;

        let seen_invalid_blocks: HashMap<models::rust::block_hash::BlockHash, models::rust::validator::Validator> = invalid_blocks_set
            .iter()
            .filter(|block_metadata| !unseen_blocks_set.contains(&block_metadata.block_hash))
            .map(|block_metadata| (block_metadata.block_hash.clone(), block_metadata.sender.clone()))
            .collect();

        Self::replay_deploys(
            &mut reporting_runtime,
            &pre_state_hash,
            &block.body.deploys,
            &block.body.system_deploys,
            !is_genesis,
            &block_data,
            seen_invalid_blocks,
        )
        .await
    }
}

impl RhoReporterCasper {
    /// Replay deploys and collect reporting events
    async fn replay_deploys(
        runtime: &mut ReportingRuntime,
        start_hash: &Blake2b256Hash,
        terms: &[ProcessedDeploy],
        system_deploys: &[ProcessedSystemDeploy],
        with_cost_accounting: bool,
        block_data: &BlockData,
        invalid_blocks: HashMap<models::rust::block_hash::BlockHash, models::rust::validator::Validator>,
    ) -> Result<ReplayResult, String> {
        runtime.reset(start_hash);

        runtime.set_block_data(block_data.clone()).await;
        runtime.set_invalid_blocks(invalid_blocks).await;

        let mut deploy_results = Vec::new();
        for term in terms {
            let replay_result = runtime
                .replay_deploy_e(with_cost_accounting, term)
                .await;

            let events = match replay_result {
                Ok(_) => runtime.get_report().unwrap_or_default(),
                Err(_) => Vec::new(),
            };

            deploy_results.push(DeployReportResult {
                processed_deploy: term.clone(),
                events,
            });
        }

        let mut system_deploy_results = Vec::new();
        for system_deploy in system_deploys {
            let replay_result = runtime
                .replay_block_system_deploy(block_data, system_deploy)
                .await;

            let events = match replay_result {
                Ok(_) => runtime.get_report().unwrap_or_default(),
                Err(_) => Vec::new(),
            };

            let system_deploy_data = match system_deploy {
                ProcessedSystemDeploy::Succeeded { system_deploy, .. } => system_deploy.clone(),
                ProcessedSystemDeploy::Failed { .. } => SystemDeployData::Empty,
            };

            system_deploy_results.push(SystemDeployReportResult {
                processed_system_deploy: system_deploy_data,
                events,
            });
        }

        let checkpoint = runtime.create_checkpoint();
        let post_state_hash = ByteString::from(checkpoint.root.to_bytes_prost());

        Ok(ReplayResult {
            deploy_report_result: deploy_results,
            system_deploy_report_result: system_deploy_results,
            post_state_hash,
        })
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

/// ReportingRuntime wraps RhoRuntimeImpl with ReportingRspace to enable event collection
pub struct ReportingRuntime {
    runtime: rholang::rust::interpreter::rho_runtime::RhoRuntimeImpl,
    space: RhoReportingRspace,
}

impl ReportingRuntime {
    /// Get reporting events from the space
    pub fn get_report(&self) -> Result<Vec<Vec<ReportingEvent<Par, BindPattern, ListParWithRandom, TaggedContinuation>>>, RSpaceError> {
        self.space.get_report()
    }

    /// Reset the runtime to a specific state hash
    pub fn reset(&mut self, root: &Blake2b256Hash) {
        self.runtime.reset(root);
    }

    /// Set block data for the runtime
    pub async fn set_block_data(&self, block_data: BlockData) {
        RhoRuntime::set_block_data(&self.runtime, block_data).await;
    }

    /// Set invalid blocks for the runtime
    pub async fn set_invalid_blocks(&self, invalid_blocks: std::collections::HashMap<models::rust::block_hash::BlockHash, models::rust::validator::Validator>) {
        RhoRuntime::set_invalid_blocks(&self.runtime, invalid_blocks).await;
    }

    /// Create a checkpoint and return the root hash
    pub fn create_checkpoint(&mut self) -> rspace_plus_plus::rspace::checkpoint::Checkpoint {
        let _ = self.space.create_checkpoint();
        RhoRuntime::create_checkpoint(&mut self.runtime)
    }

    /// Replay a deploy and collect reporting events
    pub async fn replay_deploy_e(
        &mut self,
        with_cost_accounting: bool,
        processed_deploy: &ProcessedDeploy,
    ) -> Result<(), crate::rust::errors::CasperError> {
        use crate::rust::rholang::replay_runtime::ReplayRuntimeOps;
        
        let mut replay_ops = ReplayRuntimeOps::new_from_runtime(self.runtime.clone());
        
        replay_ops.replay_deploy_e(with_cost_accounting, processed_deploy).await?;
        
        self.runtime = replay_ops.runtime_ops.runtime;
        
        Ok(())
    }

    /// Replay a system deploy and collect reporting events
    pub async fn replay_block_system_deploy(
        &mut self,
        block_data: &BlockData,
        processed_system_deploy: &models::rust::casper::protocol::casper_message::ProcessedSystemDeploy,
    ) -> Result<(), crate::rust::errors::CasperError> {
        use crate::rust::rholang::replay_runtime::ReplayRuntimeOps;
        
        // Create ReplayRuntimeOps from the runtime
        let mut replay_ops = ReplayRuntimeOps::new_from_runtime(self.runtime.clone());
        
        // Replay the system deploy
        replay_ops.replay_block_system_deploy(block_data, processed_system_deploy).await?;
        
        // Update the runtime from replay_ops
        self.runtime = replay_ops.runtime_ops.runtime;
        
        Ok(())
    }
}

/// Factory functions for creating ReportingRuntime
impl ReportingRuntime {
    /// Create a ReportingRspace from RSpaceStore
    pub fn create_reporting_rspace(
        store: RSpaceStore,
    ) -> Result<RhoReportingRspace, RSpaceError> {
        use rholang::rust::interpreter::matcher::r#match::Matcher;
        use rspace_plus_plus::rspace::r#match::Match;
        
        let matcher: Arc<Box<dyn Match<BindPattern, ListParWithRandom>>> = 
            Arc::new(Box::new(Matcher));
        
        RhoReportingRspace::create(store, matcher)
    }

    /// Create a ReportingRuntime from a ReportingRspace
    pub async fn create_reporting_runtime(
        reporting_space: RhoReportingRspace,
        mergeable_tag_name: Par,
        extra_system_processes: &mut Vec<Definition>,
    ) -> Result<Self, String> {
        use rholang::rust::interpreter::rho_runtime::create_replay_rho_runtime;
        
        // Create the runtime using the reporting space
        // Now that ReportingRspace implements ISpace, we can use it directly
        let runtime = create_replay_rho_runtime(
            reporting_space.clone(),
            mergeable_tag_name,
            true, // init_registry - bootstrap registry for reporting
            extra_system_processes,
        )
        .await;
        
        rholang::rust::interpreter::rho_runtime::bootstrap_registry(&runtime).await;
        let mut runtime_mut = runtime;
        RhoRuntime::create_checkpoint(&mut runtime_mut);
        
        Ok(ReportingRuntime {
            runtime: runtime_mut,
            space: reporting_space,
        })
    }
}
