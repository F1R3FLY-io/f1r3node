// See casper/src/main/scala/coop/rchain/casper/Casper.scala

use async_trait::async_trait;
use comm::rust::transport::transport_layer::TransportLayer;
use dashmap::{DashMap, DashSet};
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use std::{
    collections::HashMap,
    fmt::{self, Display},
    sync::{Arc, Mutex},
};

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::{
        BlockDagKeyValueStorage, DeployId, KeyValueDagRepresentation,
    },
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use crypto::rust::signatures::signed::Signed;
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{BlockMessage, DeployData, Justification},
    validator::Validator,
};
use rspace_plus_plus::rspace::{history::Either, state::rspace_exporter::RSpaceExporter};

use crate::rust::{
    block_status::{BlockError, InvalidBlock, ValidBlock},
    engine::block_retriever::BlockRetriever,
    errors::CasperError,
    estimator::Estimator,
    multi_parent_casper_impl::MultiParentCasperImpl,
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeployError {
    ParsingError(String),
    MissingUser,
    UnknownSignatureAlgorithm(String),
    SignatureVerificationFailed,
}

impl DeployError {
    pub fn parsing_error(details: String) -> Self {
        DeployError::ParsingError(details)
    }

    pub fn missing_user() -> Self {
        DeployError::MissingUser
    }

    pub fn unknown_signature_algorithm(alg: String) -> Self {
        DeployError::UnknownSignatureAlgorithm(alg)
    }

    pub fn signature_verification_failed() -> Self {
        DeployError::SignatureVerificationFailed
    }
}

impl Display for DeployError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeployError::ParsingError(details) => write!(f, "Parsing error: {}", details),
            DeployError::MissingUser => write!(f, "Missing user"),
            DeployError::UnknownSignatureAlgorithm(alg) => {
                write!(f, "Unknown signature algorithm '{}'", alg)
            }
            DeployError::SignatureVerificationFailed => write!(f, "Signature verification failed"),
        }
    }
}

#[async_trait]
pub trait Casper {
    async fn get_snapshot(&self) -> Result<CasperSnapshot, CasperError>;

    fn contains(&self, hash: &BlockHash) -> bool;

    fn dag_contains(&self, hash: &BlockHash) -> bool;

    fn buffer_contains(&self, hash: &BlockHash) -> bool;

    fn get_approved_block(&self) -> Result<&BlockMessage, CasperError>;

    fn deploy(
        &self,
        deploy: Signed<DeployData>,
    ) -> Result<Either<DeployError, DeployId>, CasperError>;

    async fn estimator(
        &self,
        dag: &mut KeyValueDagRepresentation,
    ) -> Result<Vec<BlockHash>, CasperError>;

    fn get_version(&self) -> i64;

    async fn validate(
        &self,
        block: &BlockMessage,
        snapshot: &mut CasperSnapshot,
    ) -> Result<Either<BlockError, ValidBlock>, CasperError>;

    async fn handle_valid_block(
        &self,
        block: &BlockMessage,
    ) -> Result<KeyValueDagRepresentation, CasperError>;

    fn handle_invalid_block(
        &self,
        block: &BlockMessage,
        status: &InvalidBlock,
        dag: &KeyValueDagRepresentation,
    ) -> Result<KeyValueDagRepresentation, CasperError>;

    fn get_dependency_free_from_buffer(&self) -> Result<Vec<BlockMessage>, CasperError>;
}

#[async_trait]
pub trait MultiParentCasper: Casper + Send + Sync {
    async fn fetch_dependencies(&self) -> Result<(), CasperError>;

    // This is the weight of faults that have been accumulated so far.
    // We want the clique oracle to give us a fault tolerance that is greater than
    // this initial fault weight combined with our fault tolerance threshold t.
    fn normalized_initial_fault(
        &self,
        weights: HashMap<Validator, u64>,
    ) -> Result<f32, CasperError>;

    async fn last_finalized_block(&self) -> Result<BlockMessage, CasperError>;

    // Equivalent to Scala's blockDag: F[BlockDagRepresentation[F]]
    async fn block_dag(&self) -> Result<KeyValueDagRepresentation, CasperError>;

    fn block_store(&self) -> &KeyValueBlockStore;

    fn runtime_manager(&self) -> Arc<tokio::sync::Mutex<RuntimeManager>>;

    fn get_validator(&self) -> Option<ValidatorIdentity>;

    async fn get_history_exporter(&self) -> Arc<dyn RSpaceExporter>;
}

pub fn hash_set_casper<T: TransportLayer + Send + Sync>(
    block_retriever: BlockRetriever<T>,
    event_publisher: F1r3flyEvents,
    runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    estimator: Estimator,
    block_store: KeyValueBlockStore,
    block_dag_storage: BlockDagKeyValueStorage,
    deploy_storage: KeyValueDeployStorage,
    casper_buffer_storage: CasperBufferKeyValueStorage,
    validator_id: Option<ValidatorIdentity>,
    casper_shard_conf: CasperShardConf,
    approved_block: BlockMessage,
    heartbeat_signal_ref: crate::rust::heartbeat_signal::HeartbeatSignalRef,
) -> Result<MultiParentCasperImpl<T>, CasperError> {
    Ok(MultiParentCasperImpl {
        block_retriever,
        event_publisher,
        runtime_manager,
        estimator,
        block_store,
        block_dag_storage,
        deploy_storage: Arc::new(Mutex::new(deploy_storage)),
        casper_buffer_storage,
        validator_id,
        casper_shard_conf,
        approved_block,
        finalization_in_progress: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heartbeat_signal_ref,
    })
}

/**
 * Casper snapshot is a state that is changing in discrete manner with each new block added.
 * This class represents full information about the state. It is required for creating new blocks
 * as well as for validating blocks.
 */
#[derive(Clone)]
pub struct CasperSnapshot {
    pub dag: KeyValueDagRepresentation,
    pub last_finalized_block: BlockHash,
    pub lca: BlockHash,
    pub tips: Vec<BlockHash>,
    pub parents: Vec<BlockMessage>,
    pub justifications: DashSet<Justification>,
    pub invalid_blocks: HashMap<Validator, BlockHash>,
    pub deploys_in_scope: DashSet<Signed<DeployData>>,
    pub max_block_num: i64,
    pub max_seq_nums: DashMap<Validator, u64>,
    pub on_chain_state: OnChainCasperState,
}

impl CasperSnapshot {
    pub fn new(dag: KeyValueDagRepresentation) -> Self {
        Self {
            dag,
            last_finalized_block: BlockHash::default(),
            lca: BlockHash::default(),
            tips: vec![],
            parents: vec![],
            justifications: DashSet::new(),
            invalid_blocks: HashMap::new(),
            deploys_in_scope: DashSet::new(),
            max_block_num: 0,
            max_seq_nums: DashMap::new(),
            on_chain_state: OnChainCasperState::new(CasperShardConf::new()),
        }
    }
}

#[derive(Clone)]
pub struct OnChainCasperState {
    pub shard_conf: CasperShardConf,
    pub bonds_map: HashMap<Validator, i64>,
    pub active_validators: Vec<Validator>,
}

impl OnChainCasperState {
    pub fn new(shard_conf: CasperShardConf) -> Self {
        Self {
            shard_conf,
            bonds_map: HashMap::new(),
            active_validators: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct CasperShardConf {
    pub fault_tolerance_threshold: f32,
    pub shard_name: String,
    pub parent_shard_id: String,
    pub finalization_rate: i32,
    pub max_number_of_parents: i32,
    pub max_parent_depth: i32,
    pub synchrony_constraint_threshold: f32,
    pub height_constraint_threshold: i64,
    // Validators will try to put deploy in a block only for next `deployLifespan` blocks.
    // Required to enable protection from re-submitting duplicate deploys
    pub deploy_lifespan: i64,
    pub casper_version: i64,
    pub config_version: i64,
    pub bond_minimum: i64,
    pub bond_maximum: i64,
    pub epoch_length: i32,
    pub quarantine_length: i32,
    pub min_phlo_price: i64,
    /// Disable late block filtering in DagMerger (for testing or special configurations)
    pub disable_late_block_filtering: bool,
    /// Disable validator progress check (for standalone mode)
    pub disable_validator_progress_check: bool,
    /// Enable background garbage collection for mergeable channels.
    /// When enabled, uses safe reachability-based GC (required for multi-parent mode).
    /// When disabled (default), uses immediate deletion on finalization (legacy behavior).
    pub enable_mergeable_channel_gc: bool,
    /// Depth buffer for mergeable channels garbage collection.
    /// Additional safety margin beyond max-parent-depth before deleting data.
    pub mergeable_channels_gc_depth_buffer: i32,
}

impl CasperShardConf {
    pub fn new() -> Self {
        Self {
            fault_tolerance_threshold: 0.0,
            shard_name: "".to_string(),
            parent_shard_id: "".to_string(),
            finalization_rate: 0,
            max_number_of_parents: 0,
            max_parent_depth: 0,
            synchrony_constraint_threshold: 0.0,
            height_constraint_threshold: 0,
            deploy_lifespan: 0,
            casper_version: 0,
            config_version: 0,
            bond_minimum: 0,
            bond_maximum: 0,
            epoch_length: 0,
            quarantine_length: 0,
            min_phlo_price: 0,
            disable_late_block_filtering: true,
            disable_validator_progress_check: false,
            enable_mergeable_channel_gc: false,
            mergeable_channels_gc_depth_buffer: 10,
        }
    }
}

// TODO(#325): Move test_helpers to a #[cfg(test)] module or separate test-utils crate
// to avoid including test code in production binaries.
/// Test helpers for creating mock Casper implementations.
pub mod test_helpers {
    use super::*;
    use async_trait::async_trait;

    /// A test implementation of MultiParentCasper that returns a configurable snapshot and LFB.
    pub struct TestCasperWithSnapshot {
        snapshot: CasperSnapshot,
        lfb: BlockMessage,
    }

    impl TestCasperWithSnapshot {
        pub fn new(snapshot: CasperSnapshot, lfb: BlockMessage) -> Self {
            Self { snapshot, lfb }
        }

        /// Create an empty CasperSnapshot for testing.
        pub fn create_empty_snapshot() -> CasperSnapshot {
            use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
            use block_storage::rust::dag::block_metadata_store::BlockMetadataStore;
            use dashmap::{DashMap, DashSet};
            use rspace_plus_plus::rspace::shared::in_mem_key_value_store::InMemoryKeyValueStore;
            use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
            use std::collections::BTreeMap;
            use std::sync::{Arc, RwLock};

            let block_metadata_store =
                KeyValueTypedStoreImpl::new(Arc::new(InMemoryKeyValueStore::new()));
            let dag = KeyValueDagRepresentation {
                dag_set: Arc::new(DashSet::new()),
                latest_messages_map: Arc::new(DashMap::new()),
                child_map: Arc::new(DashMap::new()),
                height_map: Arc::new(RwLock::new(BTreeMap::new())),
                invalid_blocks_set: Arc::new(DashSet::new()),
                last_finalized_block_hash: BlockHash::new(),
                finalized_blocks_set: Arc::new(DashSet::new()),
                block_metadata_index: Arc::new(RwLock::new(BlockMetadataStore::new(
                    block_metadata_store,
                ))),
                deploy_index: Arc::new(RwLock::new(KeyValueTypedStoreImpl::new(Arc::new(
                    InMemoryKeyValueStore::new(),
                )))),
            };

            CasperSnapshot::new(dag)
        }
    }

    #[async_trait]
    impl Casper for TestCasperWithSnapshot {
        async fn get_snapshot(&self) -> Result<CasperSnapshot, CasperError> {
            Ok(self.snapshot.clone())
        }

        fn contains(&self, _hash: &BlockHash) -> bool {
            false
        }

        fn dag_contains(&self, _hash: &BlockHash) -> bool {
            false
        }

        fn buffer_contains(&self, _hash: &BlockHash) -> bool {
            false
        }

        fn get_approved_block(&self) -> Result<&BlockMessage, CasperError> {
            Err(CasperError::RuntimeError(
                "get_approved_block not implemented for TestCasperWithSnapshot".to_string(),
            ))
        }

        fn deploy(
            &self,
            _deploy: Signed<DeployData>,
        ) -> Result<Either<DeployError, DeployId>, CasperError> {
            Ok(Either::Right(DeployId::default()))
        }

        async fn estimator(
            &self,
            _dag: &mut KeyValueDagRepresentation,
        ) -> Result<Vec<BlockHash>, CasperError> {
            Ok(Vec::new())
        }

        fn get_version(&self) -> i64 {
            1
        }

        async fn validate(
            &self,
            _block: &BlockMessage,
            _snapshot: &mut CasperSnapshot,
        ) -> Result<Either<BlockError, ValidBlock>, CasperError> {
            Ok(Either::Right(ValidBlock::Valid))
        }

        async fn handle_valid_block(
            &self,
            _block: &BlockMessage,
        ) -> Result<KeyValueDagRepresentation, CasperError> {
            Ok(self.snapshot.dag.clone())
        }

        fn handle_invalid_block(
            &self,
            _block: &BlockMessage,
            _status: &InvalidBlock,
            dag: &KeyValueDagRepresentation,
        ) -> Result<KeyValueDagRepresentation, CasperError> {
            Ok(dag.clone())
        }

        fn get_dependency_free_from_buffer(&self) -> Result<Vec<BlockMessage>, CasperError> {
            Ok(Vec::new())
        }
    }

    #[async_trait]
    impl MultiParentCasper for TestCasperWithSnapshot {
        async fn fetch_dependencies(&self) -> Result<(), CasperError> {
            Ok(())
        }

        fn normalized_initial_fault(
            &self,
            _weights: HashMap<Validator, u64>,
        ) -> Result<f32, CasperError> {
            Ok(0.0)
        }

        async fn last_finalized_block(&self) -> Result<BlockMessage, CasperError> {
            Ok(self.lfb.clone())
        }

        async fn block_dag(&self) -> Result<KeyValueDagRepresentation, CasperError> {
            Ok(self.snapshot.dag.clone())
        }

        fn block_store(&self) -> &KeyValueBlockStore {
            unimplemented!("block_store not needed for heartbeat tests")
        }

        fn runtime_manager(&self) -> Arc<tokio::sync::Mutex<RuntimeManager>> {
            unimplemented!("runtime_manager not needed for heartbeat tests")
        }

        fn get_validator(&self) -> Option<ValidatorIdentity> {
            None
        }

        async fn get_history_exporter(&self) -> Arc<dyn RSpaceExporter> {
            unimplemented!("get_history_exporter not needed for heartbeat tests")
        }
    }
}
