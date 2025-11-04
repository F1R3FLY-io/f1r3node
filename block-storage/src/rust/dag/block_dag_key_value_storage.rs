// See block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagKeyValueStorage.scala

use dashmap::{DashMap, DashSet};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::{Arc, RwLock},
};

use models::rust::{
    block_hash::{self, BlockHash, BlockHashSerde},
    block_metadata::BlockMetadata,
    casper::{pretty_printer::PrettyPrinter, protocol::casper_message::BlockMessage},
    equivocation_record::{EquivocationRecord, SequenceNumber},
    validator::{self, Validator, ValidatorSerde},
};
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::store::{
    key_value_store::KvStoreError, key_value_typed_store::KeyValueTypedStore,
    key_value_typed_store_impl::KeyValueTypedStoreImpl,
};

use super::{
    block_metadata_store::BlockMetadataStore, equivocation_tracker_store::EquivocationTrackerStore,
};

pub type DeployId = shared::rust::ByteString;

#[derive(Clone)]
pub struct KeyValueDagRepresentation {
    pub dag_set: Arc<DashSet<BlockHash>>,
    pub latest_messages_map: Arc<DashMap<Validator, BlockHash>>,
    pub child_map: Arc<DashMap<BlockHash, Arc<DashSet<BlockHash>>>>,
    pub height_map: Arc<RwLock<BTreeMap<i64, DashSet<BlockHash>>>>,
    pub invalid_blocks_set: Arc<DashSet<BlockMetadata>>,
    pub last_finalized_block_hash: BlockHash,
    pub finalized_blocks_set: Arc<DashSet<BlockHash>>,
    pub block_metadata_index: Arc<RwLock<BlockMetadataStore>>,
    pub deploy_index: Arc<RwLock<KeyValueTypedStoreImpl<DeployId, BlockHashSerde>>>,
}

impl KeyValueDagRepresentation {
    pub fn lookup(&self, block_hash: &BlockHash) -> Result<Option<BlockMetadata>, KvStoreError> {
        if self.dag_set.contains(block_hash) {
            let block_metadata_index_guard = self.block_metadata_index.read().unwrap();
            block_metadata_index_guard.get(block_hash)
        } else {
            Ok(None)
        }
    }

    pub fn contains(&self, block_hash: &BlockHash) -> bool {
        block_hash.len() == block_hash::LENGTH && self.dag_set.contains(block_hash)
    }

    pub fn children(&self, block_hash: &BlockHash) -> Option<Arc<DashSet<BlockHash>>> {
        self.child_map.get(block_hash).map(|v| v.value().clone())
    }

    pub fn latest_message_hash(&self, validator: &Validator) -> Option<BlockHash> {
        self.latest_messages_map
            .get(validator)
            .map(|v| v.value().clone())
    }

    pub fn latest_message_hashes(&self) -> Arc<DashMap<Validator, BlockHash>> {
        self.latest_messages_map.clone()
    }

    pub fn invalid_blocks(&self) -> Arc<DashSet<BlockMetadata>> {
        self.invalid_blocks_set.clone()
    }

    pub fn last_finalized_block(&self) -> BlockHash {
        self.last_finalized_block_hash.clone()
    }

    // latestBlockNumber, topoSort and lookupByDeployId are only used in BlockAPI.
    // Do they need to be part of the DAG current state or they can be moved to DAG storage directly?

    pub fn get_max_height(&self) -> i64 {
        let height_map_guard = self.height_map.read().unwrap();
        if height_map_guard.is_empty() {
            0
        } else {
            height_map_guard
                .last_key_value()
                .expect("height_map is empty")
                .0
                + 1
        }
    }

    pub fn latest_block_number(&self) -> i64 {
        self.get_max_height()
    }

    pub fn is_finalized(&self, block_hash: &BlockHash) -> bool {
        self.finalized_blocks_set.contains(block_hash)
    }

    pub fn find(&self, truncated_hash: &str) -> Option<BlockHash> {
        if truncated_hash.len() % 2 == 0 {
            let truncated_bytes = hex::decode(truncated_hash).expect("invalid truncated hash");
            self.dag_set
                .iter()
                .find(|hash| hash.starts_with(&truncated_bytes))
                .map(|v| v.clone())
        } else {
            // if truncatedHash is odd length string we cannot convert it to ByteString with 8 bit resolution
            // because each symbol has 4 bit resolution. Need to make a string of even length by removing the last symbol,
            // then find all the matching hashes and choose one that matches the full truncatedHash string
            let truncated_bytes = hex::decode(&truncated_hash[..truncated_hash.len() - 1])
                .expect("invalid truncated hash");
            self.dag_set
                .iter()
                .filter(|hash| hash.starts_with(&truncated_bytes))
                .find(|hash| hex::encode(&**hash).starts_with(truncated_hash))
                .map(|v| v.clone())
        }
    }

    pub fn topo_sort(
        &self,
        start_block_number: i64,
        maybe_end_block_number: Option<i64>,
    ) -> Result<Vec<Vec<BlockHash>>, KvStoreError> {
        let max_number = self.get_max_height();
        let start_number = std::cmp::max(0, start_block_number);
        let end_number = maybe_end_block_number
            .map(|n| std::cmp::min(max_number, n))
            .unwrap_or(max_number);

        if start_number >= 0 && start_number <= end_number {
            Ok(self
                .height_map
                .read()
                .unwrap()
                .range(start_number..=end_number)
                .map(|(_, hashes)| hashes.iter().map(|hash_ref| hash_ref.clone()).collect())
                .collect())
        } else {
            Err(KvStoreError::InvalidArgument(format!(
                "Invalid start block number: {}, end block number: {}",
                start_number, end_number
            )))
        }
    }

    pub fn lookup_by_deploy_id(
        &self,
        deploy_id: &DeployId,
    ) -> Result<Option<BlockHash>, KvStoreError> {
        let deploy_index_guard = self.deploy_index.read().unwrap();
        deploy_index_guard
            .get_one(deploy_id)
            .map(|result| result.map(|block_hash_serde| block_hash_serde.into()))
    }

    // See block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockDagRepresentationSyntax.scala

    // Get block metadata, "unsafe" because method expects block already in the DAG.
    pub fn lookup_unsafe(&self, block_hash: &BlockHash) -> Result<BlockMetadata, KvStoreError> {
        match self.lookup(block_hash) {
            Ok(Some(metadata)) => Ok(metadata),
            _ => Err(KvStoreError::InvalidArgument(format!(
                "DAG storage is missing hash {}",
                PrettyPrinter::build_string_bytes(&block_hash)
            ))),
        }
    }

    pub fn lookups_unsafe(
        &self,
        hashes: Vec<BlockHash>,
    ) -> Result<Vec<BlockMetadata>, KvStoreError> {
        hashes.par_iter().map(|h| self.lookup_unsafe(h)).collect()
    }

    pub fn latest_message_hash_unsafe(
        &self,
        validator: &Validator,
    ) -> Result<BlockHash, KvStoreError> {
        match self.latest_message_hash(validator) {
            Some(hash) => Ok(hash),
            None => Err(KvStoreError::InvalidArgument(format!(
                "No latest message for validator {}",
                PrettyPrinter::build_string_bytes(&validator)
            ))),
        }
    }

    pub fn latest_message(
        &self,
        validator: &Validator,
    ) -> Result<Option<BlockMetadata>, KvStoreError> {
        match self.latest_message_hash(validator) {
            Some(hash) => self.lookup_unsafe(&hash).map(Some),
            None => Ok(None),
        }
    }

    pub fn latest_messages(&self) -> Result<HashMap<Validator, BlockMetadata>, KvStoreError> {
        let latest_messages = self.latest_message_hashes();

        let mut result = HashMap::new();
        for pair in latest_messages.iter() {
            let validator = pair.key().clone();
            let hash = pair.value();
            let metadata = self.lookup_unsafe(&hash)?;
            result.insert(validator, metadata);
        }

        Ok(result)
    }

    pub fn invalid_latest_messages(&self) -> Result<HashMap<Validator, BlockHash>, KvStoreError> {
        let latest_messages = self.latest_messages()?;
        let latest_message_hashes = latest_messages
            .into_iter()
            .map(|(validator, metadata)| (validator, metadata.block_hash))
            .collect();

        self.invalid_latest_messages_from_hashes(latest_message_hashes)
    }

    pub fn invalid_latest_messages_from_hashes(
        &self,
        latest_message_hashes: HashMap<Validator, BlockHash>,
    ) -> Result<HashMap<Validator, BlockHash>, KvStoreError> {
        let invalid_blocks = self.invalid_blocks();
        let invalid_block_hashes: HashSet<BlockHash> = invalid_blocks
            .iter()
            .map(|block| block.block_hash.clone())
            .collect();

        Ok(latest_message_hashes
            .into_iter()
            .filter(|(_, block_hash)| invalid_block_hashes.contains(block_hash))
            .collect())
    }

    pub fn invalid_blocks_map(&self) -> Result<HashMap<BlockHash, Validator>, KvStoreError> {
        let invalid_blocks = self.invalid_blocks();
        let invalid_block_hashes: HashMap<BlockHash, Validator> = invalid_blocks
            .iter()
            .map(|block| (block.block_hash.clone(), block.sender.clone()))
            .collect();

        Ok(invalid_block_hashes)
    }

    pub fn self_justification_chain(
        &self,
        block_hash: BlockHash,
    ) -> Result<Vec<BlockHash>, KvStoreError> {
        let mut result = Vec::new();
        let mut current_hash = block_hash;

        loop {
            let metadata = self.lookup_unsafe(&current_hash)?;
            let next_justification = metadata
                .justifications
                .iter()
                .find(|justification| justification.validator == metadata.sender);

            match next_justification {
                Some(justification) => {
                    let next_hash = justification.latest_block_hash.clone();
                    result.push(next_hash.clone());
                    current_hash = next_hash;
                }
                None => {
                    break;
                }
            }
        }

        Ok(result)
    }

    pub fn self_justification(
        &self,
        block_hash: &BlockHash,
    ) -> Result<Option<BlockHash>, KvStoreError> {
        self.self_justification_chain(block_hash.clone())
            .map(|chain| chain.into_iter().next())
    }

    pub fn main_parent_chain(
        &self,
        block_hash: BlockHash,
        stop_at_height: i64,
    ) -> Result<Vec<BlockHash>, KvStoreError> {
        let mut result = Vec::new();
        let mut current_hash = block_hash;

        loop {
            let metadata = self.lookup_unsafe(&current_hash)?;

            if metadata.block_number <= stop_at_height {
                break;
            }

            match metadata.parents.first() {
                Some(parent) => {
                    let parent_hash = parent.clone();
                    result.push(parent_hash.clone());
                    current_hash = parent_hash;
                }
                None => break,
            }
        }

        Ok(result)
    }

    pub fn is_in_main_chain(
        &self,
        ancestor: &BlockHash,
        descendant: &BlockHash,
    ) -> Result<bool, KvStoreError> {
        if ancestor == descendant {
            return Ok(true);
        }

        let metadata_ancestor = self.lookup_unsafe(ancestor)?;
        let height = metadata_ancestor.block_number;

        let main_chain = self.main_parent_chain(descendant.clone(), height)?;

        Ok(main_chain.contains(ancestor))
    }

    pub fn parents_unsafe(&self, block_hash: &BlockHash) -> Result<Vec<BlockHash>, KvStoreError> {
        let metadata = self.lookup_unsafe(block_hash)?;
        Ok(metadata.parents)
    }

    pub fn non_finalized_blocks(&self) -> Result<HashSet<BlockHash>, KvStoreError> {
        let mut result = HashSet::new();
        let mut tips = self
            .latest_messages()?
            .values()
            .map(|metadata| metadata.block_hash.clone())
            .collect::<Vec<_>>();

        while !tips.is_empty() {
            let mut next_level = Vec::new();

            for hash in &tips {
                if !self.is_finalized(hash) {
                    result.insert(hash.clone());

                    let metadata = self.lookup_unsafe(hash)?;
                    next_level.extend(metadata.parents.clone());
                }
            }

            tips = next_level;
        }

        Ok(result)
    }

    pub fn descendants(&self, block_hash: &BlockHash) -> Result<HashSet<BlockHash>, KvStoreError> {
        let mut result = HashSet::new();
        let mut current_level = vec![block_hash.clone()];

        while !current_level.is_empty() {
            let mut next_level = Vec::new();

            for hash in &current_level {
                if let Some(children) = self.children(hash) {
                    for child in children.iter() {
                        if result.insert(child.clone()) {
                            next_level.push(child.clone());
                        }
                    }
                }
            }

            current_level = next_level;
        }

        Ok(result)
    }

    pub fn ancestors(
        &self,
        block_hash: BlockHash,
        filter_f: impl Fn(&BlockHash) -> bool,
    ) -> Result<HashSet<BlockHash>, KvStoreError> {
        let mut result = HashSet::new();
        let mut current_level = vec![block_hash];

        while !current_level.is_empty() {
            let mut next_level = Vec::new();

            for hash in &current_level {
                let metadata = self.lookup_unsafe(hash)?;

                for parent in &metadata.parents {
                    if filter_f(parent) && !result.contains(parent) {
                        result.insert(parent.clone());
                        next_level.push(parent.clone());
                    }
                }
            }

            current_level = next_level;
        }

        Ok(result)
    }

    pub fn with_ancestors(
        &self,
        block_hash: BlockHash,
        filter_f: impl Fn(&BlockHash) -> bool,
    ) -> Result<HashSet<BlockHash>, KvStoreError> {
        let mut result = self.ancestors(block_hash.clone(), filter_f)?;
        result.insert(block_hash);
        Ok(result)
    }
}

#[derive(Clone)]
pub struct BlockDagKeyValueStorage {
    pub latest_messages_index: KeyValueTypedStoreImpl<ValidatorSerde, BlockHashSerde>,
    pub block_metadata_index: Arc<RwLock<BlockMetadataStore>>,
    pub deploy_index: Arc<RwLock<KeyValueTypedStoreImpl<DeployId, BlockHashSerde>>>,
    pub invalid_blocks_index: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata>,
    pub equivocation_tracker_index: EquivocationTrackerStore,
}

impl BlockDagKeyValueStorage {
    pub async fn new(kvm: &mut impl KeyValueStoreManager) -> Result<Self, KvStoreError> {
        let block_metadata_kv_store = kvm.store("block-metadata".to_string()).await?;
        let block_metadata_db: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata> =
            KeyValueTypedStoreImpl::new(block_metadata_kv_store);
        let block_metadata_store = BlockMetadataStore::new(block_metadata_db);

        let equivocation_tracker_kv_store = kvm.store("equivocation-tracker".to_string()).await?;
        let equivocation_tracker_db: KeyValueTypedStoreImpl<
            (ValidatorSerde, SequenceNumber),
            BTreeSet<BlockHashSerde>,
        > = KeyValueTypedStoreImpl::new(equivocation_tracker_kv_store);
        let equivocation_tracker_store = EquivocationTrackerStore::new(equivocation_tracker_db);

        let latest_messages_kv_store = kvm.store("latest-messages".to_string()).await?;
        let latest_messages_db: KeyValueTypedStoreImpl<ValidatorSerde, BlockHashSerde> =
            KeyValueTypedStoreImpl::new(latest_messages_kv_store);

        let invalid_blocks_kv_store = kvm.store("invalid-blocks".to_string()).await?;
        let invalid_blocks_db: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata> =
            KeyValueTypedStoreImpl::new(invalid_blocks_kv_store);

        let deploy_index_kv_store = kvm.store("deploy-index".to_string()).await?;
        let deploy_index_db: KeyValueTypedStoreImpl<DeployId, BlockHashSerde> =
            KeyValueTypedStoreImpl::new(deploy_index_kv_store);

        Ok(Self {
            block_metadata_index: Arc::new(RwLock::new(block_metadata_store)),
            deploy_index: Arc::new(RwLock::new(deploy_index_db)),
            invalid_blocks_index: invalid_blocks_db,
            equivocation_tracker_index: equivocation_tracker_store,
            latest_messages_index: latest_messages_db,
        })
    }

    pub fn equivocation_records(&self) -> Result<HashSet<EquivocationRecord>, KvStoreError> {
        self.equivocation_tracker_index.data()
    }

    pub fn insert_equivocation_record(
        &self,
        record: EquivocationRecord,
    ) -> Result<(), KvStoreError> {
        self.equivocation_tracker_index.add(record)
    }

    pub fn update_equivocation_record(
        &self,
        mut record: EquivocationRecord,
        block_hash: BlockHash,
    ) -> Result<(), KvStoreError> {
        self.equivocation_tracker_index.add({
            record.equivocation_detected_block_hashes.insert(block_hash);
            record
        })
    }

    pub fn get_representation(&self) -> KeyValueDagRepresentation {
        let latest_messages = self
            .latest_messages_index
            .to_map()
            .expect("Failed to convert latest_messages_index to map")
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let invalid_blocks = self
            .invalid_blocks_index
            .to_map()
            .expect("Failed to convert invalid_blocks_index to map")
            .into_iter()
            .map(|(_, v)| v)
            .collect();

        let block_metadata_index_guard = self.block_metadata_index.read().unwrap();
        let dag_set = block_metadata_index_guard.dag_set();
        let child_map = block_metadata_index_guard.child_map();
        let height_map = block_metadata_index_guard.height_map();
        let last_finalized_block = block_metadata_index_guard.last_finalized_block();
        let finalized_blocks = block_metadata_index_guard.finalized_block_set();

        KeyValueDagRepresentation {
            dag_set,
            latest_messages_map: Arc::new(latest_messages),
            child_map,
            height_map,
            invalid_blocks_set: Arc::new(invalid_blocks),
            last_finalized_block_hash: last_finalized_block,
            finalized_blocks_set: finalized_blocks,
            block_metadata_index: self.block_metadata_index.clone(),
            deploy_index: self.deploy_index.clone(),
        }
    }

    pub fn insert(
        &self,
        block: &BlockMessage,
        invalid: bool,
        approved: bool,
    ) -> Result<KeyValueDagRepresentation, KvStoreError> {
        let sender_is_empty = block.sender.is_empty();
        let sender_has_invalid_format =
            !sender_is_empty && (block.sender.len() != validator::LENGTH);
        let senders_new_lm = (block.sender.clone(), block.block_hash.clone());

        let log_already_stored = format!(
            "Block {} is already stored.",
            PrettyPrinter::build_string_block_message(&block, true)
        );
        let log_empty_sender = format!(
            "Block {} sender is empty.",
            PrettyPrinter::build_string_block_message(&block, true)
        );

        let new_latest_messages = || -> Result<HashMap<Validator, BlockHash>, KvStoreError> {
            let block_hash: BlockHash = block.block_hash.clone();

            let newly_bonded_set: HashSet<_> = block
                .body
                .state
                .bonds
                .iter()
                .map(|bond| &bond.validator)
                .collect();

            let justification_validators: HashSet<_> = block
                .justifications
                .iter()
                .map(|justification| &justification.validator)
                .collect();

            let mut result = HashMap::new();
            for validator in newly_bonded_set.difference(&justification_validators) {
                // This filter is required to enable adding blocks backward from higher height to lower
                if let Ok(false) = self
                    .latest_messages_index
                    .contains_key(ValidatorSerde((*validator).clone()))
                {
                    result.insert((*validator).clone(), block_hash.clone());
                }
            }

            Ok(result)
        };

        let block_exists = {
            let block_metadata_index_guard = self.block_metadata_index.read().unwrap();
            let exists = block_metadata_index_guard.contains(&block.block_hash);
            exists
        };

        if block_exists {
            log::warn!("{}", log_already_stored);
            Ok(self.get_representation())
        } else {
            let block_hash = block.block_hash.clone();
            let block_hash_is_invalid = !(block_hash.len() == block_hash::LENGTH);

            if sender_has_invalid_format {
                return Err(KvStoreError::InvalidArgument(format!(
                    "Block sender is malformed., Block: {:?}",
                    block
                )));
            }
            // TODO: should we have special error type for block hash error also?
            //  Should this be checked before calling insert? Is DAG storage responsible for that? - OLD
            if block_hash_is_invalid {
                return Err(KvStoreError::InvalidArgument(format!(
                    "Block hash {} is not correct length.",
                    PrettyPrinter::build_string_bytes(&block_hash)
                )));
            }

            if sender_is_empty {
                log::warn!("{}", log_empty_sender);
            }

            let block_metadata = BlockMetadata::from_block(&block, invalid, None, None);
            let mut block_metadata_guard = self.block_metadata_index.write().unwrap();
            block_metadata_guard.add(block_metadata.clone())?;
            drop(block_metadata_guard);

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
            let deploy_index_guard = self.deploy_index.write().unwrap();
            deploy_index_guard.put(deploy_entries)?;
            drop(deploy_index_guard);

            if invalid {
                self.invalid_blocks_index
                    .put_one(block_hash.clone().into(), block_metadata)?;
            }

            let new_latest_from_sender = if !sender_is_empty {
                // Add LM either if there is no existing message for the sender, or if sequence number advances
                // - assumes block sender is not valid hash
                if match self
                    .latest_messages_index
                    .get_one(&block.sender.clone().into())
                {
                    Ok(Some(latest_message_hash)) => {
                        let block_metadata_index_guard = self.block_metadata_index.read().unwrap();
                        match block_metadata_index_guard.get(&latest_message_hash.into()) {
                            Ok(Some(metadata)) => block.seq_num >= metadata.sequence_number,
                            _ => true,
                        }
                    }
                    _ => true,
                } {
                    HashMap::from([senders_new_lm])
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            };

            let mut new_latest_to_add = new_latest_messages()?;
            new_latest_to_add.extend(new_latest_from_sender);

            self.latest_messages_index.put(
                new_latest_to_add
                    .into_iter()
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect(),
            )?;

            if approved {
                let mut block_metadata_guard = self.block_metadata_index.write().unwrap();
                block_metadata_guard.record_finalized(block_hash, HashSet::new())?;
            }

            Ok(self.get_representation())
        }
    }

    pub fn access_equivocations_tracker<A>(
        &self,
        f: impl Fn(&EquivocationTrackerStore) -> Result<A, KvStoreError>,
    ) -> Result<A, KvStoreError> {
        f(&self.equivocation_tracker_index)
    }

    /** Record that some hash is directly finalized (detected by finalizer and becomes LFB). */
    pub async fn record_directly_finalized<F, Fut>(
        &self,
        directly_finalized_hash: BlockHash,
        mut finalization_effect: F,
    ) -> Result<(), KvStoreError>
    where
        F: FnMut(&HashSet<BlockHash>) -> Fut,
        Fut: std::future::Future<Output = Result<(), KvStoreError>>,
    {
        let dag = self.get_representation();
        if !dag.contains(&directly_finalized_hash) {
            return Err(KvStoreError::InvalidArgument(format!(
                "Attempting to finalize nonexistent hash {}",
                PrettyPrinter::build_string_bytes(&directly_finalized_hash)
            )));
        }

        let dag = self.get_representation();
        let indirectly_finalized = dag.ancestors(directly_finalized_hash.clone(), |hash| {
            !dag.is_finalized(&hash)
        })?;

        let mut all_finalized = indirectly_finalized.clone();
        all_finalized.insert(directly_finalized_hash.clone());

        finalization_effect(&all_finalized).await?;

        let mut block_metadata_index_guard = self.block_metadata_index.write().unwrap();
        block_metadata_index_guard
            .record_finalized(directly_finalized_hash, indirectly_finalized)?;

        Ok(())
    }
}
