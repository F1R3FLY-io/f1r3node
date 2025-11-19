// See casper/src/main/scala/coop/rchain/casper/MultiParentCasperImpl.scala

use async_trait::async_trait;
use rspace_plus_plus::rspace::state::rspace_exporter::RSpaceExporter;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::{
        BlockDagKeyValueStorage, DeployId, KeyValueDagRepresentation,
    },
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::transport::transport_layer::TransportLayer;
use crypto::rust::signatures::signed::Signed;
use models::rust::{
    block_hash::{BlockHash, BlockHashSerde},
    casper::{
        pretty_printer::PrettyPrinter,
        protocol::casper_message::{BlockMessage, DeployData, Justification},
    },
    equivocation_record::EquivocationRecord,
    normalizer_env::normalizer_env_from_deploy,
    validator::Validator,
};
use rspace_plus_plus::rspace::{hashing::blake2b256_hash::Blake2b256Hash, history::Either};
use shared::rust::{
    dag::dag_ops,
    shared::{f1r3fly_event::F1r3flyEvent, f1r3fly_events::F1r3flyEvents},
    store::{key_value_store::KvStoreError, key_value_typed_store::KeyValueTypedStore},
};

use crate::rust::{
    block_status::{BlockError, InvalidBlock, ValidBlock},
    casper::{
        Casper, CasperShardConf, CasperSnapshot, DeployError, MultiParentCasper, OnChainCasperState,
    },
    engine::block_retriever::{AdmitHashReason, BlockRetriever},
    equivocation_detector::EquivocationDetector,
    errors::CasperError,
    estimator::{Estimator, ForkChoice},
    finality::finalizer::Finalizer,
    util::{
        proto_util,
        rholang::{
            interpreter_util::{self, validate_block_checkpoint},
            runtime_manager::RuntimeManager,
        },
    },
    validate::Validate,
    validator_identity::ValidatorIdentity,
};

pub struct MultiParentCasperImpl<T: TransportLayer + Send + Sync> {
    pub block_retriever: BlockRetriever<T>,
    pub event_publisher: F1r3flyEvents,
    pub runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    pub estimator: Estimator,
    pub block_store: KeyValueBlockStore,
    pub block_dag_storage: BlockDagKeyValueStorage,
    pub deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    pub casper_buffer_storage: CasperBufferKeyValueStorage,
    pub validator_id: Option<ValidatorIdentity>,
    // TODO: this should be read from chain, for now read from startup options - OLD
    pub casper_shard_conf: CasperShardConf,
    pub approved_block: BlockMessage,
}

#[async_trait]
impl<T: TransportLayer + Send + Sync> Casper for MultiParentCasperImpl<T> {
    async fn get_snapshot(&self) -> Result<CasperSnapshot, CasperError> {
        let mut dag = self.block_dag_storage.get_representation();
        let ForkChoice { lca, tips } = self.estimator.tips(&mut dag, &self.approved_block).await?;

        // Before block merge, `EstimatorHelper.chooseNonConflicting` were used to filter parents, as we could not
        // have conflicting parents. With introducing block merge, all parents that share the same bonds map
        // should be parents. Parents that have different bond maps are only one that cannot be merged in any way.
        let parents = {
            // For now main parent bonds map taken as a reference, but might be we want to pick a subset with equal
            // bond maps that has biggest cumulative stake.
            let blocks = tips
                .iter()
                .map(|b| self.block_store.get(b).unwrap())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    CasperError::RuntimeError("Failed to get blocks from store".to_string())
                })?;

            let parents = blocks
                .iter()
                .filter(|b| {
                    if let Some(first_block) = blocks.first() {
                        b.body.state.bonds == first_block.body.state.bonds
                    } else {
                        false
                    }
                })
                .map(|b| b.clone())
                .collect::<Vec<_>>();

            parents
        };

        let on_chain_state = self
            .get_on_chain_state(
                parents
                    .first()
                    .expect("parents should never be empty after approved block"),
            )
            .await?;

        // We ensure that only the justifications given in the block are those
        // which are bonded validators in the chosen parent. This is safe because
        // any latest message not from a bonded validator will not change the
        // final fork-choice.
        let justifications = {
            let latest_messages = dag.latest_messages()?;
            let bonded_validators = &on_chain_state.bonds_map;

            latest_messages
                .into_iter()
                .filter(|(validator, _)| bonded_validators.contains_key(validator))
                .map(|(validator, block_metadata)| Justification {
                    validator,
                    latest_block_hash: block_metadata.block_hash,
                })
                .collect::<dashmap::DashSet<_>>()
        };

        let parent_hashes: Vec<BlockHash> = parents.iter().map(|b| b.block_hash.clone()).collect();
        let parent_metas = dag.lookups_unsafe(parent_hashes)?;
        let max_block_num = proto_util::max_block_number_metadata(&parent_metas);

        let max_seq_nums = {
            let latest_messages = dag.latest_messages()?;
            latest_messages
                .into_iter()
                .map(|(validator, block_metadata)| {
                    (validator, block_metadata.sequence_number as u64)
                })
                .collect::<dashmap::DashMap<_, _>>()
        };

        let deploys_in_scope = {
            let current_block_number = max_block_num + 1;
            let earliest_block_number =
                current_block_number - on_chain_state.shard_conf.deploy_lifespan;

            // Use bf_traverse to collect all deploys within the deploy lifespan
            let neighbor_fn = |block_metadata: &models::rust::block_metadata::BlockMetadata| -> Vec<models::rust::block_metadata::BlockMetadata> {
                match proto_util::get_parent_metadatas_above_block_number(block_metadata, earliest_block_number, &mut dag) {
                    Ok(parents) => parents,
                    Err(_) => vec![],
                }
            };

            let traversal_result = dag_ops::bf_traverse(parent_metas, neighbor_fn);

            let all_deploys = dashmap::DashSet::new();
            for block_metadata in traversal_result {
                let block = self.block_store.get(&block_metadata.block_hash)?.unwrap();
                let block_deploys = proto_util::deploys(&block);
                for processed_deploy in block_deploys {
                    all_deploys.insert(processed_deploy.deploy);
                }
            }
            all_deploys
        };

        let invalid_blocks = dag.invalid_blocks_map()?;
        let last_finalized_block = dag.last_finalized_block();

        Ok(CasperSnapshot {
            dag,
            last_finalized_block,
            lca,
            tips,
            parents,
            justifications,
            invalid_blocks,
            deploys_in_scope,
            max_block_num,
            max_seq_nums,
            on_chain_state,
        })
    }

    fn contains(&self, hash: &BlockHash) -> bool {
        self.buffer_contains(hash) || self.dag_contains(hash)
    }

    fn dag_contains(&self, hash: &BlockHash) -> bool {
        self.block_dag_storage.get_representation().contains(hash)
    }

    fn buffer_contains(&self, hash: &BlockHash) -> bool {
        let block_hash_serde = BlockHashSerde(hash.clone());
        self.casper_buffer_storage.contains(&block_hash_serde)
    }

    fn get_approved_block(&self) -> Result<&BlockMessage, CasperError> {
        Ok(&self.approved_block)
    }
    fn deploy(
        &self,
        deploy: Signed<DeployData>,
    ) -> Result<Either<DeployError, DeployId>, CasperError> {
        // Create normalizer environment from deploy
        let normalizer_env = normalizer_env_from_deploy(&deploy);

        // Try to parse the deploy term
        match interpreter_util::mk_term(&deploy.data.term, normalizer_env) {
            // Parse failed - return parsing error
            Err(interpreter_error) => Ok(Either::Left(DeployError::parsing_error(format!(
                "Error in parsing term: \n{}",
                interpreter_error
            )))),
            // Parse succeeded - call add_deploy
            Ok(_parsed_term) => Ok(Either::Right(self.add_deploy(deploy)?)),
        }
    }

    async fn estimator(
        &self,
        dag: &mut KeyValueDagRepresentation,
    ) -> Result<Vec<BlockHash>, CasperError> {
        let fork_choice = self.estimator.tips(dag, &self.approved_block).await?;
        Ok(fork_choice.tips)
    }

    fn get_version(&self) -> i64 {
        self.casper_shard_conf.casper_version
    }

    async fn validate(
        &self,
        block: &BlockMessage,
        snapshot: &mut CasperSnapshot,
    ) -> Result<Either<BlockError, ValidBlock>, CasperError> {
        log::info!(
            "Validating block {}",
            PrettyPrinter::build_string_block_message(block, true)
        );

        let start = std::time::Instant::now();
        let val_result = {
            let block_summary_result = Validate::block_summary(
                block,
                &self.approved_block,
                snapshot,
                &self.casper_shard_conf.shard_name,
                self.casper_shard_conf.deploy_lifespan as i32,
                &self.estimator,
                &self.block_store,
            )
            .await;

            tracing::debug!(target: "f1r3fly.casper", "post-validation-block-summary");
            if let Either::Left(block_error) = block_summary_result {
                return Ok(Either::Left(block_error));
            }

            let validate_block_checkpoint_result = validate_block_checkpoint(
                block,
                &self.block_store,
                snapshot,
                &mut *self.runtime_manager.lock().await,
            )
            .await?;

            tracing::debug!(target: "f1r3fly.casper", "transactions-validated");
            if let Either::Left(block_error) = validate_block_checkpoint_result {
                return Ok(Either::Left(block_error));
            }

            if let Either::Right(None) = validate_block_checkpoint_result {
                return Ok(Either::Left(BlockError::Invalid(
                    InvalidBlock::InvalidTransaction,
                )));
            }

            let bonds_cache_result =
                Validate::bonds_cache(block, &*self.runtime_manager.lock().await).await;
            tracing::debug!(target: "f1r3fly.casper", "bonds-cache-validated");
            if let Either::Left(block_error) = bonds_cache_result {
                return Ok(Either::Left(block_error));
            }

            let neglected_invalid_block_result = Validate::neglected_invalid_block(block, snapshot);
            tracing::debug!(target: "f1r3fly.casper", "neglected-invalid-block-validated");
            if let Either::Left(block_error) = neglected_invalid_block_result {
                return Ok(Either::Left(block_error));
            }

            let equivocation_detector_result =
                EquivocationDetector::check_neglected_equivocations_with_update(
                    block,
                    &snapshot.dag,
                    &self.block_store,
                    &self.approved_block,
                    &self.block_dag_storage,
                )
                .await?;

            tracing::debug!(target: "f1r3fly.casper", "neglected-equivocation-validated");
            if let Either::Left(block_error) = equivocation_detector_result {
                return Ok(Either::Left(block_error));
            }

            // This validation is only to punish validator which accepted lower price deploys.
            // And this can happen if not configured correctly.
            let phlo_price_result =
                Validate::phlo_price(block, self.casper_shard_conf.min_phlo_price);

            tracing::debug!(target: "f1r3fly.casper", "phlogiston-price-validated");
            if let Either::Left(_) = phlo_price_result {
                log::warn!(
                    "One or more deploys has phloPrice lower than {}",
                    self.casper_shard_conf.min_phlo_price
                );
            }

            let dep_dag = self.casper_buffer_storage.to_doubly_linked_dag();

            let equivocation_result = EquivocationDetector::check_equivocations(&dep_dag, block, &snapshot.dag).await?;
            tracing::debug!(target: "f1r3fly.casper", "equivocation-validated");
            equivocation_result
        };

        let elapsed = start.elapsed();

        if let Either::Right(ref status) = val_result {
            let block_info = PrettyPrinter::build_string_block_message(block, true);
            let deploy_count = block.body.deploys.len();
            log::info!(
                "Block replayed: {} ({}d) ({:?}) [{:?}]",
                block_info,
                deploy_count,
                status,
                elapsed
            );

            if self.casper_shard_conf.max_number_of_parents > 1 {
                let mergeable_chs = self.runtime_manager.lock().await.load_mergeable_channels(
                    &block.body.state.post_state_hash,
                    block.sender.clone(),
                    block.seq_num,
                )?;

                let _index_block = self
                    .runtime_manager
                    .lock()
                    .await
                    .get_or_compute_block_index(
                        &block.block_hash,
                        &block.body.deploys,
                        &block.body.system_deploys,
                        &Blake2b256Hash::from_bytes_prost(&block.body.state.pre_state_hash),
                        &Blake2b256Hash::from_bytes_prost(&block.body.state.post_state_hash),
                        &mergeable_chs,
                    )?;
            }
        }

        Ok(val_result)
    }

    async fn handle_valid_block(
        &self,
        block: &BlockMessage,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        // Insert block as valid into DAG storage
        let updated_dag = self.block_dag_storage.insert(block, false, false)?;

        // Remove block from casper buffer
        let block_hash_serde = BlockHashSerde(block.block_hash.clone());
        self.casper_buffer_storage.remove(block_hash_serde)?;

        // Update last finalized block if needed
        self.update_last_finalized_block(block).await?;

        Ok(updated_dag)
    }

    fn handle_invalid_block(
        &self,
        block: &BlockMessage,
        status: &InvalidBlock,
        dag: &KeyValueDagRepresentation,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        // Helper function to handle invalid block effect (logging + storage operations)
        let handle_invalid_block_effect =
            |block_dag_storage: &BlockDagKeyValueStorage,
             casper_buffer_storage: &CasperBufferKeyValueStorage,
             status: &InvalidBlock,
             block: &BlockMessage|
             -> Result<KeyValueDagRepresentation, CasperError> {
                log::warn!(
                    "Recording invalid block {} for {:?}.",
                    PrettyPrinter::build_string_bytes(&block.block_hash),
                    status
                );

                // TODO: should be nice to have this transition of a block from casper buffer to dag storage atomic - OLD
                let updated_dag = block_dag_storage.insert(block, true, false)?;
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                casper_buffer_storage.remove(block_hash_serde)?;
                Ok(updated_dag)
            };

        match status {
            InvalidBlock::AdmissibleEquivocation => {
                let base_equivocation_block_seq_num = block.seq_num - 1;

                // Check if equivocation record already exists for this validator and sequence number
                let equivocation_records = self.block_dag_storage.equivocation_records()?;
                let record_exists = equivocation_records.iter().any(|record| {
                    record.equivocator == block.sender
                        && record.equivocation_base_block_seq_num == base_equivocation_block_seq_num
                });

                if !record_exists {
                    // Create and insert new equivocation record
                    let new_equivocation_record = EquivocationRecord::new(
                        block.sender.clone(),
                        base_equivocation_block_seq_num,
                        BTreeSet::new(),
                    );
                    self.block_dag_storage
                        .insert_equivocation_record(new_equivocation_record)?;
                }

                // We can only treat admissible equivocations as invalid blocks if
                // casper is single threaded.
                handle_invalid_block_effect(
                    &self.block_dag_storage,
                    &self.casper_buffer_storage,
                    status,
                    block,
                )
            }

            InvalidBlock::IgnorableEquivocation => {
                /*
                 * We don't have to include these blocks to the equivocation tracker because if any validator
                 * will build off this side of the equivocation, we will get another attempt to add this block
                 * through the admissible equivocations.
                 */
                log::info!(
                    "Did not add block {} as that would add an equivocation to the BlockDAG",
                    PrettyPrinter::build_string_bytes(&block.block_hash)
                );
                Ok(dag.clone())
            }

            status if status.is_slashable() => {
                // TODO: Slash block for status except InvalidUnslashableBlock - OLD
                // This should implement actual slashing mechanism (reducing stake, etc.)
                handle_invalid_block_effect(
                    &self.block_dag_storage,
                    &self.casper_buffer_storage,
                    status,
                    block,
                )
            }

            _ => {
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                self.casper_buffer_storage.remove(block_hash_serde)?;
                log::warn!(
                    "Recording invalid block {} for {:?}.",
                    PrettyPrinter::build_string_bytes(&block.block_hash),
                    status
                );
                Ok(dag.clone())
            }
        }
    }

    fn get_dependency_free_from_buffer(&self) -> Result<Vec<BlockMessage>, CasperError> {
        // Get pendants from CasperBuffer
        let pendants = self.casper_buffer_storage.get_pendants();

        // Filter to pendants that exist in block store
        let mut pendants_stored = Vec::new();
        for pendant_serde in pendants.iter() {
            let pendant_hash = BlockHash::from(pendant_serde.0.clone());
            if self.block_store.get(&pendant_hash)?.is_some() {
                pendants_stored.push(pendant_hash);
            }
        }

        // Filter to dependency-free pendants
        let mut dep_free_pendants = Vec::new();
        for pendant_hash in pendants_stored {
            let block = self.block_store.get(&pendant_hash)?.unwrap();
            let justifications = &block.justifications;

            // Check if all justifications are in DAG
            // If even one justification is not in DAG - block is not dependency free
            let all_deps_in_dag = justifications
                .iter()
                .all(|j| self.dag_contains(&j.latest_block_hash));

            if all_deps_in_dag {
                dep_free_pendants.push(pendant_hash);
            }
        }

        // Get the actual BlockMessages
        let result = dep_free_pendants
            .into_iter()
            .map(|hash| self.block_store.get(&hash).unwrap())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                CasperError::RuntimeError("Failed to get blocks from store".to_string())
            })?;

        Ok(result)
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync> MultiParentCasper for MultiParentCasperImpl<T> {
    async fn fetch_dependencies(&self) -> Result<(), CasperError> {
        // Get pendants from CasperBuffer
        let pendants = self.casper_buffer_storage.get_pendants();

        // Filter to get unseen pendants (not in block store)
        let mut pendants_unseen = Vec::new();
        for pendant_serde in pendants.iter() {
            let pendant_hash = BlockHash::from(pendant_serde.0.clone());
            if self.block_store.get(&pendant_hash)?.is_none() {
                pendants_unseen.push(pendant_hash);
            }
        }

        // Log debug info about pendant count
        log::debug!(
            "Requesting CasperBuffer pendant hashes, {} items.",
            pendants_unseen.len()
        );

        // Send each unseen pendant to BlockRetriever
        for dependency in pendants_unseen {
            log::debug!(
                "Sending dependency {} to BlockRetriever",
                PrettyPrinter::build_string_bytes(&dependency)
            );

            self.block_retriever
                .admit_hash(
                    dependency,
                    None,
                    AdmitHashReason::MissingDependencyRequested,
                )
                .await?;
        }

        Ok(())
    }

    fn normalized_initial_fault(
        &self,
        weights: HashMap<Validator, u64>,
    ) -> Result<f32, CasperError> {
        // Access equivocations tracker to get equivocation records
        let equivocating_weight =
            self.block_dag_storage
                .access_equivocations_tracker(|tracker| {
                    let equivocation_records = tracker.data()?;

                    // Extract equivocators and sum their weights
                    let equivocating_weight: u64 = equivocation_records
                        .iter()
                        .map(|record| &record.equivocator)
                        .filter_map(|equivocator| weights.get(equivocator))
                        .sum();

                    Ok(equivocating_weight)
                })?;

        // Calculate total weight from the weights map
        let total_weight: u64 = weights.values().sum();

        // Return normalized fault (equivocating weight / total weight)
        if total_weight == 0 {
            Ok(0.0)
        } else {
            Ok(equivocating_weight as f32 / total_weight as f32)
        }
    }

    async fn last_finalized_block(&self) -> Result<BlockMessage, CasperError> {
        // Get current LFB hash and height
        let dag = self.block_dag_storage.get_representation();
        let last_finalized_block_hash = dag.last_finalized_block();
        let last_finalized_block_height =
            dag.lookup_unsafe(&last_finalized_block_hash)?.block_number;

        // Create references to avoid borrowing issues
        let block_store = &self.block_store;
        let deploy_storage = &self.deploy_storage;
        let runtime_manager = &self.runtime_manager;
        let block_dag_storage = &self.block_dag_storage;

        // Create simple finalization effect closure
        let new_lfb_found_effect = |new_lfb: BlockHash| async move {
            block_dag_storage
                .record_directly_finalized(new_lfb.clone(), |finalized_set: &HashSet<BlockHash>| {
                    let finalized_set = finalized_set.clone();
                    Box::pin(async move {
                        // process_finalized
                        for block_hash in &finalized_set {
                            let block = block_store.get(block_hash)?.unwrap();
                            let deploys: Vec<_> = block
                                .body
                                .deploys
                                .iter()
                                .map(|pd| pd.deploy.clone())
                                .collect();

                            // Remove block deploys from persistent store
                            let deploys_count = deploys.len();
                            deploy_storage
                                .lock()
                                .map_err(|_| {
                                    KvStoreError::LockError(
                                        "Failed to acquire deploy_storage lock".to_string(),
                                    )
                                })?
                                .remove(deploys)?;
                            let finalized_set_str = PrettyPrinter::build_string_hashes(
                                &finalized_set.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
                            );
                            let removed_deploy_msg = format!(
                                "Removed {} deploys from deploy history as we finalized block {}.",
                                deploys_count, finalized_set_str
                            );
                            log::info!("{}", removed_deploy_msg);

                            // Remove block index from cache
                            runtime_manager
                                .lock()
                                .await
                                .remove_block_index_cache(block_hash);

                            let state_hash =
                                Blake2b256Hash::from_bytes_prost(&block.body.state.post_state_hash);
                            runtime_manager
                                .lock()
                                .await
                                .mergeable_store
                                .delete(vec![state_hash.bytes()])?;
                        }
                        Ok(())
                    })
                })
                .await?;

            self.event_publisher
                .publish(F1r3flyEvent::block_finalised(hex::encode(new_lfb)))
                .map_err(|e| KvStoreError::IoError(e.to_string()))
        };

        // Run finalizer
        let new_finalized_hash_opt = Finalizer::run(
            &dag,
            self.casper_shard_conf.fault_tolerance_threshold,
            last_finalized_block_height,
            new_lfb_found_effect,
        )
        .await
        .map_err(|e| CasperError::KvStoreError(e))?;

        // Get the final LFB hash (either new or existing)
        let final_lfb_hash = new_finalized_hash_opt.unwrap_or(last_finalized_block_hash);

        // Return the finalized block
        let block_message = self.block_store.get(&final_lfb_hash)?.unwrap();
        Ok(block_message)
    }

    // Equivalent to Scala's def blockDag: F[BlockDagRepresentation[F]] = BlockDagStorage[F].getRepresentation
    async fn block_dag(&self) -> Result<KeyValueDagRepresentation, CasperError> {
        Ok(self.block_dag_storage.get_representation())
    }

    fn block_store(&self) -> &KeyValueBlockStore {
        &self.block_store
    }

    fn get_validator(&self) -> Option<ValidatorIdentity> {
        self.validator_id.clone()
    }

    async fn get_history_exporter(&self) -> Arc<dyn RSpaceExporter> {
        self.runtime_manager
            .lock()
            .await
            .get_history_repo()
            .exporter()
    }

    fn runtime_manager(&self) -> Arc<tokio::sync::Mutex<RuntimeManager>> {
        self.runtime_manager.clone()
    }
}

impl<T: TransportLayer + Send + Sync> MultiParentCasperImpl<T> {
    async fn update_last_finalized_block(
        &self,
        new_block: &BlockMessage,
    ) -> Result<(), CasperError> {
        if new_block.body.state.block_number % self.casper_shard_conf.finalization_rate as i64 == 0
        {
            // Using tracing events instead of spans for async context
            // Span[F].traceI("finalizer-run") equivalent from Scala
            tracing::info!(target: "f1r3fly.casper", "finalizer-run-started");
            self.last_finalized_block().await?;
            tracing::info!(target: "f1r3fly.casper", "finalizer-run-finished");
        }
        Ok(())
    }

    async fn get_on_chain_state(
        &self,
        block: &BlockMessage,
    ) -> Result<OnChainCasperState, CasperError> {
        let av = self
            .runtime_manager
            .lock()
            .await
            .get_active_validators(&block.body.state.post_state_hash)
            .await?;

        // bonds are available in block message, but please remember this is just a cache, source of truth is RSpace.
        let bm = &block.body.state.bonds;

        Ok(OnChainCasperState {
            shard_conf: self.casper_shard_conf.clone(),
            bonds_map: bm
                .iter()
                .map(|v| (v.validator.clone(), v.stake))
                .collect::<HashMap<_, _>>(),
            active_validators: av,
        })
    }

    fn add_deploy(&self, deploy: Signed<DeployData>) -> Result<DeployId, CasperError> {
        // Add deploy to storage
        self.deploy_storage
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError("Failed to acquire deploy_storage lock".to_string())
            })?
            .add(vec![deploy.clone()])?;

        // Log the received deploy
        let deploy_info = PrettyPrinter::build_string_signed_deploy_data(&deploy);
        log::info!("Received {}", deploy_info);

        // Return deploy signature as DeployId
        Ok(deploy.sig.to_vec())
    }
}

pub fn created_event(block: &BlockMessage) -> F1r3flyEvent {
    let block_hash = hex::encode(block.block_hash.clone());
    let parents_hashes = block
        .header
        .parents_hash_list
        .iter()
        .map(|h| hex::encode(h))
        .collect::<Vec<_>>();

    let justifications = block
        .justifications
        .iter()
        .map(|j| {
            (
                hex::encode(j.validator.clone()),
                hex::encode(j.latest_block_hash.clone()),
            )
        })
        .collect::<Vec<_>>();

    let deploy_ids = block
        .body
        .deploys
        .iter()
        .map(|d| hex::encode(d.deploy.sig.clone()))
        .collect::<Vec<_>>();

    let creator = hex::encode(block.sender.clone());
    let seq_num = block.seq_num;

    F1r3flyEvent::block_created(
        block_hash,
        parents_hashes,
        justifications,
        deploy_ids,
        creator,
        seq_num,
    )
}
