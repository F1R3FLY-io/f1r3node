// See casper/src/main/scala/coop/rchain/casper/blocks/BlockProcessor.scala

/*
 * ARCHITECTURAL CHOICE: Trait-based Dependency Injection
 *
 * This implementation uses trait-based dependency injection instead of functional closures
 * because Rust's ownership model and async system work better with traits than with complex
 * closure captures. Traits provide zero-cost abstractions, better testability, and seamless
 * async support while maintaining the same flexibility as the original Scala version.
 */

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::{
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::{
    block_hash::{BlockHash, BlockHashSerde},
    casper::pretty_printer::PrettyPrinter,
    casper::protocol::casper_message::{BlockMessage, CasperMessage},
};
use rspace_plus_plus::rspace::history::Either;

use crate::rust::block_status::BlockError;
use crate::rust::engine::block_retriever::{AdmitHashReason, BlockRetriever};
use crate::rust::{
    block_status::InvalidBlock,
    casper::{Casper, CasperSnapshot},
    errors::CasperError,
    util::proto_util,
    validate::Validate,
    ValidBlockProcessing,
};

/// Logic for processing incoming blocks
/// Blocks created by node itself are not held here, but in Proposer.
pub struct BlockProcessor<T: TransportLayer + Send + Sync> {
    dependencies: BlockProcessorDependencies<T>,
}

impl<T: TransportLayer + Send + Sync> BlockProcessor<T> {
    pub fn new(dependencies: BlockProcessorDependencies<T>) -> Self {
        Self { dependencies }
    }

    /// check if block should be processed
    pub async fn check_if_of_interest(
        &self,
        casper: &mut impl Casper,
        block: &BlockMessage,
    ) -> Result<bool, CasperError> {
        // TODO casper.dag_contains does not take into account equivocation tracker
        let already_processed =
            casper.dag_contains(&block.block_hash) || casper.buffer_contains(&block.block_hash);

        let shard_of_interest = casper.get_approved_block().map(|approved_block| {
            approved_block
                .shard_id
                .eq_ignore_ascii_case(&block.shard_id)
        })?;

        let version_of_interest = casper
            .get_approved_block()
            .map(|approved_block| Validate::version(block, approved_block.header.version))?;

        let old_block = casper.get_approved_block().map(|approved_block| {
            proto_util::block_number(block) < proto_util::block_number(approved_block)
        })?;

        Ok(!already_processed && shard_of_interest && version_of_interest && !old_block)
    }

    /// check block format and store if check passed
    pub async fn check_if_well_formed_and_store(
        &mut self,
        block: &BlockMessage,
    ) -> Result<bool, CasperError> {
        let valid_format = Validate::format_of_fields(block);
        let valid_sig = Validate::block_signature(block);
        let is_valid = valid_format && valid_sig;

        if is_valid {
            self.dependencies.store_block(block).await?;
        }

        Ok(is_valid)
    }

    /// check if block has all dependencies available and can be validated
    pub async fn check_dependencies_with_effects(
        &mut self,
        casper: &mut impl Casper,
        block: &BlockMessage,
    ) -> Result<bool, CasperError> {
        let (is_ready, deps_to_fetch, deps_in_buffer) = self
            .dependencies
            .get_non_validated_dependencies(casper, block)
            .await?;

        if is_ready {
            // store pendant block in buffer, it will be removed once block is validated and added to DAG
            self.dependencies.commit_to_buffer(block, None).await?;
        } else {
            // associate parents with new block in casper buffer
            let mut all_deps = deps_to_fetch.clone();
            all_deps.extend(deps_in_buffer.clone());
            self.dependencies
                .commit_to_buffer(block, Some(all_deps))
                .await?;
            self.dependencies
                .request_missing_dependencies(&deps_to_fetch)
                .await?;
            self.dependencies.ack_processed(block).await?;
        }

        Ok(is_ready)
    }

    /// validate block and invoke all effects required
    pub async fn validate_with_effects(
        &mut self,
        casper: &mut impl Casper,
        block: &BlockMessage,
        // this option is required for tests, as sometimes block without parents available are added, so
        // CasperSnapshot cannot be constructed
        snapshot_opt: Option<CasperSnapshot>,
    ) -> Result<ValidBlockProcessing, CasperError> {
        let mut snapshot = match snapshot_opt {
            Some(snapshot) => snapshot,
            None => self.dependencies.get_casper_state_snapshot(casper).await?,
        };

        let status = self
            .dependencies
            .validate_block(casper, &mut snapshot, block)
            .await?;

        let _ = match &status {
            Either::Right(_valid_block) => {
                self.dependencies
                    .effects_for_valid_block(casper, block)
                    .await
            }
            Either::Left(invalid_block) => {
                // this is to maintain backward compatibility with casper validate method.
                // as it returns not only InvalidBlock or ValidBlock
                match invalid_block {
                    BlockError::Invalid(i) => {
                        self.dependencies
                            .effects_for_invalid_block(casper, block, i, &snapshot)
                            .await
                    }
                    _ => {
                        // this should never happen
                        Ok(snapshot.dag.clone())
                    }
                }
            }
        }?;

        // once block is validated and effects are invoked, it should be removed from buffer
        self.dependencies.remove_from_buffer(block).await?;
        self.dependencies.ack_processed(block).await?;

        Ok(status)
    }
}

/// Unified dependencies structure - equivalent to Scala companion object approach
/// Contains all dependencies needed for block processing in one place
pub struct BlockProcessorDependencies<T: TransportLayer + Send + Sync> {
    block_store: Arc<Mutex<KeyValueBlockStore>>,
    casper_buffer: Arc<Mutex<CasperBufferKeyValueStorage>>,
    block_dag_storage: Arc<Mutex<BlockDagKeyValueStorage>>,
    block_retriever: Arc<Mutex<BlockRetriever<T>>>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
}

impl<T: TransportLayer + Send + Sync> BlockProcessorDependencies<T> {
    pub fn new(
        block_store: Arc<Mutex<KeyValueBlockStore>>,
        casper_buffer: Arc<Mutex<CasperBufferKeyValueStorage>>,
        block_dag_storage: Arc<Mutex<BlockDagKeyValueStorage>>,
        block_retriever: Arc<Mutex<BlockRetriever<T>>>,
        transport: Arc<T>,
        connections_cell: ConnectionsCell,
        conf: RPConf,
    ) -> Self {
        Self {
            block_store,
            casper_buffer,
            block_dag_storage,
            block_retriever,
            transport,
            connections_cell,
            conf,
        }
    }

    // Public getters for tests
    pub fn transport(&self) -> &Arc<T> {
        &self.transport
    }

    pub fn casper_buffer(&self) -> &Arc<Mutex<CasperBufferKeyValueStorage>> {
        &self.casper_buffer
    }

    /// Equivalent to Scala's: storeBlock = (b: BlockMessage) => BlockStore[F].put(b)
    pub async fn store_block(&mut self, block: &BlockMessage) -> Result<(), CasperError> {
        let mut store = self
            .block_store
            .lock()
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        store
            .put_block_message(block)
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        Ok(())
    }

    /// Equivalent to Scala's: getCasperStateSnapshot = (c: Casper[F]) => c.getSnapshot
    pub async fn get_casper_state_snapshot(
        &self,
        casper: &mut impl Casper,
    ) -> Result<CasperSnapshot, CasperError> {
        casper.get_snapshot().await
    }

    /// Equivalent to Scala's: getNonValidatedDependencies = (c: Casper[F], b: BlockMessage) => { ... }
    pub async fn get_non_validated_dependencies(
        &self,
        casper: &mut impl Casper,
        block: &BlockMessage,
    ) -> Result<(bool, HashSet<BlockHash>, HashSet<BlockHash>), CasperError> {
        let all_deps = proto_util::dependencies_hashes_of(block);

        // in addition, equivocation tracker has to be checked, as admissible equivocations are not stored in DAG
        let equivocation_hashes: HashSet<BlockHash> = {
            let block_dag_storage = self
                .block_dag_storage
                .lock()
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

            block_dag_storage
                .access_equivocations_tracker(|tracker| {
                    let equivocation_records = tracker.data()?;
                    // Use HashSet to ensure uniqueness and O(1) lookup, just like Scala's Set
                    let hashes: HashSet<BlockHash> = equivocation_records
                        .iter()
                        .flat_map(|record| record.equivocation_detected_block_hashes.iter())
                        .cloned()
                        .collect();
                    Ok(hashes)
                })
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?
        };

        let deps_in_buffer: Vec<BlockHash> = {
            let casper_buffer = self
                .casper_buffer
                .lock()
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

            all_deps
                .iter()
                .filter_map(|dep| {
                    let block_hash_serde = BlockHashSerde(dep.clone());
                    if casper_buffer.contains(&block_hash_serde)
                        || casper_buffer.is_pendant(&block_hash_serde)
                    {
                        Some(dep.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };

        let deps_in_dag: Vec<BlockHash> = all_deps
            .iter()
            .filter_map(|dep| {
                if casper.dag_contains(dep) {
                    Some(dep.clone())
                } else {
                    None
                }
            })
            .collect();

        let deps_in_eq_tracker: Vec<BlockHash> = all_deps
            .iter()
            .filter(|&dep| equivocation_hashes.contains(dep))
            .cloned()
            .collect();

        let mut deps_validated: Vec<BlockHash> = deps_in_dag.clone();
        deps_validated.extend(deps_in_eq_tracker.iter().cloned());

        let deps_to_fetch: Vec<BlockHash> = all_deps
            .iter()
            .filter(|&dep| !deps_in_buffer.contains(dep))
            .filter(|&dep| !deps_validated.contains(dep))
            .cloned()
            .collect();

        let ready = deps_to_fetch.is_empty() && deps_in_buffer.is_empty();

        if !ready {
            log::info!(
                "Block {} missing dependencies. To fetch: {}. In buffer: {}. Validated: {}.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                PrettyPrinter::build_string_hashes(
                    &deps_to_fetch
                        .iter()
                        .map(|h| h.as_ref().to_vec())
                        .collect::<Vec<_>>()
                ),
                PrettyPrinter::build_string_hashes(
                    &deps_in_buffer
                        .iter()
                        .map(|h| h.as_ref().to_vec())
                        .collect::<Vec<_>>()
                ),
                PrettyPrinter::build_string_hashes(
                    &deps_validated
                        .iter()
                        .map(|h| h.as_ref().to_vec())
                        .collect::<Vec<_>>()
                )
            );
        }

        Ok((
            ready,
            deps_to_fetch.into_iter().collect::<HashSet<BlockHash>>(),
            deps_in_buffer.into_iter().collect::<HashSet<BlockHash>>(),
        ))
    }

    /// Equivalent to Scala's: commitToBuffer = (b: BlockMessage, deps: Option[Set[BlockHash]]) => { ... }
    pub async fn commit_to_buffer(
        &mut self,
        block: &BlockMessage,
        deps: Option<HashSet<BlockHash>>,
    ) -> Result<(), CasperError> {
        let mut buffer = self
            .casper_buffer
            .lock()
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        match deps {
            None => {
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                buffer
                    .put_pendant(block_hash_serde)
                    .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
            }
            Some(dependencies) => {
                let block_hash_serde = BlockHashSerde(block.block_hash.clone());
                dependencies.iter().try_for_each(|dep| {
                    let dep_serde = BlockHashSerde(dep.clone());
                    buffer
                        .add_relation(dep_serde, block_hash_serde.clone())
                        .map_err(|e| CasperError::RuntimeError(e.to_string()))
                })?;
            }
        }

        Ok(())
    }

    /// Equivalent to Scala's: removeFromBuffer = (b: BlockMessage) => casperBuffer.remove(b.blockHash)
    pub async fn remove_from_buffer(&mut self, block: &BlockMessage) -> Result<(), CasperError> {
        let mut buffer = self
            .casper_buffer
            .lock()
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        let block_hash_serde = BlockHashSerde(block.block_hash.clone());
        buffer
            .remove(block_hash_serde)
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        Ok(())
    }

    /// Equivalent to Scala's: requestMissingDependencies = (deps: Set[BlockHash]) => { ... }
    pub async fn request_missing_dependencies(
        &mut self,
        deps: &HashSet<BlockHash>,
    ) -> Result<(), CasperError> {
        let retriever = self
            .block_retriever
            .lock()
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        for dep in deps {
            retriever
                .admit_hash(
                    dep.clone(),
                    None,
                    AdmitHashReason::MissingDependencyRequested,
                )
                .await
                .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        }

        Ok(())
    }

    /// Equivalent to Scala's: validateBlock = (c: Casper[F], s: CasperSnapshot[F], b: BlockMessage) => c.validate(b, s)
    pub async fn validate_block(
        &self,
        casper: &mut impl Casper,
        snapshot: &mut CasperSnapshot,
        block: &BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        casper.validate(block, snapshot).await
    }

    /// Equivalent to Scala's: ackProcessed = (b: BlockMessage) => BlockRetriever[F].ackInCasper(b.blockHash)
    pub async fn ack_processed(&mut self, block: &BlockMessage) -> Result<(), CasperError> {
        let retriever = self
            .block_retriever
            .lock()
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        retriever
            .ack_in_casper(block.block_hash.clone())
            .await
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        Ok(())
    }

    /// Equivalent to Scala's: effectsForInvalidBlock = (c: Casper[F], b: BlockMessage, r: InvalidBlock, s: CasperSnapshot[F]) => { ... }
    pub async fn effects_for_invalid_block(
        &mut self,
        casper: &mut impl Casper,
        block: &BlockMessage,
        invalid_block: &InvalidBlock,
        snapshot: &CasperSnapshot,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        let dag = casper.handle_invalid_block(block, invalid_block, &snapshot.dag)?;

        // Equivalent to Scala's: CommUtil[F].sendBlockHash(b.blockHash, b.sender)
        self.transport
            .send_block_hash(
                &self.connections_cell,
                &self.conf,
                &block.block_hash,
                &block.sender,
            )
            .await
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        Ok(dag)
    }

    /// Equivalent to Scala's: effectsForValidBlock = (c: Casper[F], b: BlockMessage) => { ... }
    pub async fn effects_for_valid_block(
        &mut self,
        casper: &mut impl Casper,
        block: &BlockMessage,
    ) -> Result<KeyValueDagRepresentation, CasperError> {
        let dag = casper.handle_valid_block(block).await?;

        // Equivalent to Scala's: CommUtil[F].sendBlockHash(b.blockHash, b.sender)
        self.transport
            .send_block_hash(
                &self.connections_cell,
                &self.conf,
                &block.block_hash,
                &block.sender,
            )
            .await
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

        Ok(dag)
    }
}

/// Constructor function equivalent to Scala's companion object apply method
/// Creates unified dependencies and BlockProcessor
pub fn new_block_processor<T: TransportLayer + Send + Sync>(
    block_store: Arc<Mutex<KeyValueBlockStore>>,
    casper_buffer: Arc<Mutex<CasperBufferKeyValueStorage>>,
    block_dag_storage: Arc<Mutex<BlockDagKeyValueStorage>>,
    block_retriever: Arc<Mutex<BlockRetriever<T>>>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
) -> BlockProcessor<T> {
    let dependencies = BlockProcessorDependencies::new(
        block_store,
        casper_buffer,
        block_dag_storage,
        block_retriever,
        transport,
        connections_cell,
        conf,
    );

    BlockProcessor::new(dependencies)
}
