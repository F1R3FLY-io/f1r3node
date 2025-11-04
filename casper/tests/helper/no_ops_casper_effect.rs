// See casper/src/test/scala/coop/rchain/casper/helper/NoOpsCasperEffect.scala

use crate::util::test_mocks::MockKeyValueStore;
use async_trait::async_trait;
use casper::rust::validator_identity::ValidatorIdentity;
use rspace_plus_plus::rspace::state::rspace_exporter::RSpaceExporter;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use block_storage::rust::{
    dag::block_dag_key_value_storage::{DeployId, KeyValueDagRepresentation},
    key_value_block_store::KeyValueBlockStore,
};
use casper::rust::{
    block_status::{BlockError, InvalidBlock, ValidBlock},
    casper::{Casper, CasperSnapshot, DeployError, MultiParentCasper},
    errors::CasperError,
    util::rholang::runtime_manager::RuntimeManager,
};
use crypto::rust::signatures::signed::Signed;
use models::rust::{
    block_hash::{BlockHash, BlockHashSerde},
    block_implicits::get_random_block_default,
    block_metadata::BlockMetadata,
    casper::protocol::casper_message::{BlockMessage, DeployData},
    validator::Validator,
};
use rspace_plus_plus::rspace::history::Either;

pub struct NoOpsCasperEffect {
    estimator_func: Vec<BlockHash>,
    pub runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    block_store: KeyValueBlockStore,
    // Shared data for block store to ensure clones can access the same blocks
    shared_block_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    shared_approved_block_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    block_dag_storage: KeyValueDagRepresentation,
}

unsafe impl Send for NoOpsCasperEffect {}
unsafe impl Sync for NoOpsCasperEffect {}

// For testing purposes, we'll implement Clone manually by creating stub instances
impl Clone for NoOpsCasperEffect {
    fn clone(&self) -> Self {
        // Create a clone that shares the same underlying storage so that blocks added to one instance
        // are visible to cloned instances (which is necessary for the engine tests to work)

        // Create new KeyValueBlockStore with shared underlying storage
        // Note: We need to share the underlying data between clones for tests to work
        let cloned_block_store = KeyValueBlockStore::new(
            Arc::new(MockKeyValueStore::with_shared_data(
                self.shared_block_data.clone(),
            )),
            Arc::new(MockKeyValueStore::with_shared_data(
                self.shared_approved_block_data.clone(),
            )),
        );

        Self {
            estimator_func: self.estimator_func.clone(),
            runtime_manager: self.runtime_manager.clone(), // Arc clone is cheap
            block_store: cloned_block_store,
            shared_block_data: self.shared_block_data.clone(),
            shared_approved_block_data: self.shared_approved_block_data.clone(),
            block_dag_storage: self.block_dag_storage.clone(),
        }
    }
}

// Using shared MockKeyValueStore from test_mocks module

impl NoOpsCasperEffect {
    pub fn new(
        _blocks: Option<HashMap<BlockHash, BlockMessage>>, // No longer used - blocks stored in actual KeyValueBlockStore
        estimator_func: Option<Vec<BlockHash>>,
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        _block_store: KeyValueBlockStore, // We'll ignore this and create our own with shared data
        block_dag_storage: KeyValueDagRepresentation,
    ) -> Self {
        // Create shared storage that will be used by all clones
        let shared_block_data = Arc::new(Mutex::new(HashMap::new()));
        let shared_approved_block_data = Arc::new(Mutex::new(HashMap::new()));

        // Create block store with shared underlying storage
        let block_store = KeyValueBlockStore::new(
            Arc::new(MockKeyValueStore::with_shared_data(
                shared_block_data.clone(),
            )),
            Arc::new(MockKeyValueStore::with_shared_data(
                shared_approved_block_data.clone(),
            )),
        );

        Self {
            estimator_func: estimator_func.unwrap_or_default(),
            runtime_manager,
            block_store,
            shared_block_data,
            shared_approved_block_data,
            block_dag_storage,
        }
    }

    /// Create NoOpsCasperEffect with externally provided shared kvm data
    /// This ensures all storages use the SAME kvm (like Scala's InMemoryStoreManager)
    pub fn new_with_shared_kvm(
        estimator_func: Option<Vec<BlockHash>>,
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        _block_store: KeyValueBlockStore, // We'll ignore this and create our own with shared data
        block_dag_storage: KeyValueDagRepresentation,
        shared_kvm_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    ) -> Self {
        // Use the provided shared kvm data for BOTH block store and approved block store
        // This matches Scala's behavior where all storages share one kvm
        let block_store = KeyValueBlockStore::new(
            Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
            Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
        );

        Self {
            estimator_func: estimator_func.unwrap_or_default(),
            runtime_manager,
            block_store,
            shared_block_data: shared_kvm_data.clone(),
            shared_approved_block_data: shared_kvm_data.clone(),
            block_dag_storage,
        }
    }
}

#[async_trait]
impl MultiParentCasper for NoOpsCasperEffect {
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
        Ok(get_random_block_default())
    }

    async fn block_dag(&self) -> Result<KeyValueDagRepresentation, CasperError> {
        Ok(self.block_dag_storage.clone())
    }

    fn block_store(&self) -> &KeyValueBlockStore {
        &self.block_store
    }

    fn get_validator(&self) -> Option<ValidatorIdentity> {
        None
    }

    async fn get_history_exporter(&self) -> Arc<dyn RSpaceExporter> {
        todo!()
    }

    fn runtime_manager(&self) -> Arc<tokio::sync::Mutex<RuntimeManager>> {
        self.runtime_manager.clone()
    }
}

#[async_trait]
impl Casper for NoOpsCasperEffect {
    async fn get_snapshot(&self) -> Result<CasperSnapshot, CasperError> {
        todo!()
    }

    fn contains(&self, hash: &BlockHash) -> bool {
        // Use actual KeyValueBlockStore instead of HashMap (preserving Scala test logic)
        match self.block_store.get(hash) {
            Ok(maybe_block) => maybe_block.is_some(),
            Err(_) => false,
        }
    }

    fn dag_contains(&self, _hash: &BlockHash) -> bool {
        false
    }

    fn buffer_contains(&self, _hash: &BlockHash) -> bool {
        false
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
        Ok(self.estimator_func.clone())
    }

    fn get_version(&self) -> i64 {
        1
    }

    async fn validate(
        &self,
        _block: &BlockMessage,
        _snapshot: &mut CasperSnapshot,
    ) -> Result<Either<BlockError, ValidBlock>, CasperError> {
        todo!()
    }

    async fn handle_valid_block(
        &self,
        _block: &BlockMessage,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        todo!()
    }

    fn handle_invalid_block(
        &self,
        _block: &BlockMessage,
        _status: &InvalidBlock,
        _dag: &KeyValueDagRepresentation,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        todo!()
    }

    fn get_dependency_free_from_buffer(&self) -> Result<Vec<BlockMessage>, CasperError> {
        todo!()
    }

    fn get_approved_block(&self) -> Result<&BlockMessage, CasperError> {
        // For test purposes, this is not used by our tests but we need to implement it
        // In a real implementation, this would return the actual approved block from storage
        Err(CasperError::RuntimeError(
            "get_approved_block not implemented for NoOpsCasperEffect test helper".to_string(),
        ))
    }
}

// Additional test-friendly methods for compatibility with the old MockCasper API
impl NoOpsCasperEffect {
    /// Add a block to the actual KeyValueBlockStore (preserving Scala test logic)
    pub fn add_block_to_store(&mut self, block: BlockMessage) {
        // Store in the KeyValueBlockStore (the actual block storage) - no HashMap fallback
        match self.block_store.put_block_message(&block) {
            Ok(_) => {
                log::debug!(
                    "Successfully stored block {} in KeyValueBlockStore",
                    hex::encode(&block.block_hash)
                );
            }
            Err(e) => log::error!("Failed to store block in KeyValueBlockStore: {:?}", e),
        }
    }

    /// Add block to DAG storage and update latest messages (like Scala blockDagStorage.insert)
    pub fn add_to_dag(&mut self, block_hash: BlockHash) {
        use shared::rust::store::key_value_typed_store::KeyValueTypedStore;

        // Get the block from actual KeyValueBlockStore to add to DAG (preserving Scala test logic)
        if let Ok(Some(block)) = self.block_store.get(&block_hash) {
            // Add to DAG set
            self.block_dag_storage.dag_set.insert(block_hash.clone());

            // Add block metadata to the metadata store
            let block_metadata = BlockMetadata::from_block(&block, false, None, None);
            let mut metadata_guard = self.block_dag_storage.block_metadata_index.write().unwrap();
            match metadata_guard.add(block_metadata) {
                Ok(_) => {
                    log::debug!(
                        "Successfully added block {} to DAG storage",
                        hex::encode(&block_hash)
                    );
                }
                Err(e) => log::error!("Failed to add block metadata to DAG storage: {:?}", e),
            }
            drop(metadata_guard);

            // Add deploy mappings
            let deploy_hashes: Vec<DeployId> = block
                .body
                .deploys
                .iter()
                .map(|deploy| deploy.deploy.sig.clone().into())
                .collect();
            let deploy_entries: Vec<(DeployId, BlockHashSerde)> = deploy_hashes
                .into_iter()
                .map(|deploy_id| (deploy_id, BlockHashSerde(block.block_hash.clone())))
                .collect();
            let deploy_index_guard = self.block_dag_storage.deploy_index.write().unwrap();
            if let Err(e) = deploy_index_guard.put(deploy_entries) {
                log::error!("Failed to add deploy mappings to DAG storage: {:?}", e);
            }
            drop(deploy_index_guard);

            // Update latest messages following BlockDagKeyValueStorage.insert logic

            // 1. Update latest message for block sender (if not empty)
            if !block.sender.is_empty() {
                self.block_dag_storage
                    .latest_messages_map
                    .insert(block.sender.clone().into(), block.block_hash.clone());
            }

            // 2. Handle newly bonded validators (matching Scala newLatestMessages logic)
            let bonded_validators: std::collections::HashSet<Validator> = block
                .body
                .state
                .bonds
                .iter()
                .map(|bond| bond.validator.clone().into())
                .collect();

            let justified_validators: std::collections::HashSet<Validator> = block
                .justifications
                .iter()
                .map(|justification| justification.validator.clone())
                .collect();

            let newly_bonded: std::collections::HashSet<_> = bonded_validators
                .difference(&justified_validators)
                .collect();

            // For newly bonded validators not already in latest messages, add current block
            for validator in newly_bonded {
                if !self
                    .block_dag_storage
                    .latest_messages_map
                    .contains_key(validator)
                {
                    self.block_dag_storage
                        .latest_messages_map
                        .insert(validator.clone(), block.block_hash.clone());
                }
            }
        } else {
            log::error!(
                "Cannot add block {} to DAG - block not found in store",
                hex::encode(&block_hash)
            );
        }
    }

    /// Insert block to both block store and DAG (matches Scala blockDagStorage.insert pattern)
    ///
    /// This method provides a more Scala-like API that combines storage and DAG operations
    /// matching the pattern: blockDagStorage.insert(block, approved)
    pub fn insert_block(&mut self, block: BlockMessage, approved: bool) {
        // First add to block store
        self.add_block_to_store(block.clone());

        // Then add to DAG
        self.add_to_dag(block.block_hash.clone());

        // If approved, also add to finalized blocks set
        if approved {
            self.block_dag_storage
                .finalized_blocks_set
                .insert(block.block_hash);
        }
    }
}
