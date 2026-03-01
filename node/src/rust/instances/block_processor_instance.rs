// See node/src/main/scala/coop/rchain/node/instances/BlockProcessorInstance.scala

use dashmap::DashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::mpsc;

use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::BlockMessage;

use casper::rust::blocks::block_processor::BlockProcessor;
use casper::rust::casper::MultiParentCasper;
use casper::rust::errors::CasperError;
use casper::rust::{ProposeFunction, ValidBlockProcessing};

use comm::rust::transport::transport_layer::TransportLayer;

const MAX_BLOCKS_IN_PROCESSING_DEFAULT: usize = 512;
const MAX_BLOCKS_IN_PROCESSING_ENV: &str = "F1R3_MAX_BLOCKS_IN_PROCESSING";
const BLOCK_PROCESSING_RESULT_QUEUE_CAPACITY: usize = 128;
static PROCESSED_BLOCKS: AtomicUsize = AtomicUsize::new(0);
static MALLOC_TRIM_EVERY_BLOCKS: OnceLock<usize> = OnceLock::new();
static MAX_BLOCKS_IN_PROCESSING: OnceLock<usize> = OnceLock::new();
static TRIGGER_PROPOSE_AFTER_BLOCK_PROCESSING: OnceLock<bool> = OnceLock::new();

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

fn malloc_trim_every_blocks() -> usize {
    *MALLOC_TRIM_EVERY_BLOCKS.get_or_init(|| {
        std::env::var("F1R3_MALLOC_TRIM_EVERY_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn max_blocks_in_processing() -> usize {
    *MAX_BLOCKS_IN_PROCESSING.get_or_init(|| {
        std::env::var(MAX_BLOCKS_IN_PROCESSING_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MAX_BLOCKS_IN_PROCESSING_DEFAULT)
    })
}

fn trigger_propose_after_block_processing_enabled() -> bool {
    *TRIGGER_PROPOSE_AFTER_BLOCK_PROCESSING.get_or_init(|| {
        std::env::var("F1R3_TRIGGER_PROPOSE_AFTER_BLOCK_PROCESSING")
            .ok()
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(false)
    })
}

fn maybe_trim_allocator_after_block() {
    let interval = malloc_trim_every_blocks();
    if interval == 0 {
        return;
    }

    let count = PROCESSED_BLOCKS.fetch_add(1, Ordering::Relaxed) + 1;
    if count % interval != 0 {
        return;
    }

    #[cfg(target_os = "linux")]
    unsafe {
        let _ = malloc_trim(0);
    }
}

/// Configuration for BlockProcessorInstance
pub struct BlockProcessorInstance<T: TransportLayer + Send + Sync + 'static> {
    pub blocks_queue_rx:
        mpsc::Receiver<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,

    pub block_queue_tx:
        mpsc::Sender<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,

    pub block_processor: Arc<BlockProcessor<T>>,

    pub blocks_in_processing: Arc<DashSet<BlockHash>>,

    pub trigger_propose_f: Option<Arc<ProposeFunction>>,

    pub max_parallel_blocks: usize,
}

impl<T: TransportLayer + Send + Sync + 'static> BlockProcessorInstance<T> {
    pub fn new(
        (blocks_queue_rx, block_queue_tx): (
            mpsc::Receiver<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
            mpsc::Sender<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
        ),
        block_processor: Arc<BlockProcessor<T>>,
        blocks_in_processing: Arc<DashSet<BlockHash>>,
        trigger_propose_f: Option<Arc<ProposeFunction>>,
        max_parallel_blocks: usize,
    ) -> Self {
        Self {
            blocks_queue_rx,
            block_queue_tx,
            block_processor,
            blocks_in_processing,
            trigger_propose_f,
            max_parallel_blocks,
        }
    }

    /// Create and start the block processor stream
    /// Returns a handle that can be used to await the processing task
    ///
    /// This is equivalent to Scala's `BlockProcessorInstance.create` method.
    /// It processes blocks with bounded parallelism.
    ///
    /// # Arguments
    ///
    /// * `blocks_queue_tx` - Sender to enqueue blocks for processing (for re-enqueuing buffer pendants)
    pub fn create(
        self,
    ) -> Result<mpsc::Receiver<(BlockMessage, ValidBlockProcessing)>, CasperError> {
        let (result_tx, result_rx) = mpsc::channel(BLOCK_PROCESSING_RESULT_QUEUE_CAPACITY);

        tokio::spawn(async move {
            let Self {
                mut blocks_queue_rx,
                block_queue_tx,
                block_processor,
                blocks_in_processing,
                trigger_propose_f,
                max_parallel_blocks,
            } = self;

            let semaphore = Arc::new(tokio::sync::Semaphore::new(max_parallel_blocks));

            while let Some((casper, block)) = blocks_queue_rx.recv().await {
                let block_processor = block_processor.clone();
                let blocks_in_processing = blocks_in_processing.clone();
                let trigger_propose_f = trigger_propose_f.clone();
                let block_queue_tx = block_queue_tx.clone();
                let casper = casper.clone();
                let result_tx = result_tx.clone();

                let permit = semaphore.clone().acquire_owned().await.unwrap();

                // Spawn task to process the block
                tokio::spawn(async move {
                    let block_str = PrettyPrinter::build_string_bytes(&block.block_hash);
                    if !blocks_in_processing.contains(&block.block_hash) {
                        // Fallback for legacy enqueue paths: mark before processing.
                        blocks_in_processing.insert(block.block_hash.clone());
                        let max_in_flight = max_blocks_in_processing();
                        if blocks_in_processing.len() > max_in_flight {
                            blocks_in_processing.remove(&block.block_hash);
                            if let Err(err) = block_processor.ack_processed(&block).await {
                                tracing::warn!(
                                    "Dropping block {} and cleanup failed: {}",
                                    block_str,
                                    err
                                );
                            }
                            tracing::warn!(
                                "Dropping block {} because in-flight block cap {} is reached",
                                block_str,
                                max_in_flight
                            );
                            return;
                        }
                    }

                    // Process the block with all its validation steps
                    let result = process_block_with_steps(
                        block_processor.clone(),
                        casper.clone(),
                        block.clone(),
                    )
                    .await;

                    match result {
                        Ok(res) => {
                            tracing::info!("Block {} processing finished.", block_str);
                            match result_tx.send(res).await {
                                Ok(_) => {}
                                Err(err) => tracing::error!(
                                    "Failed to send block processing result: {}",
                                    err
                                ),
                            }
                        }
                        Err(e) => {
                            match &e {
                                CasperError::Other(msg) if msg == "Missing dependencies" => {
                                    tracing::warn!(
                                        "Block {} delayed: missing dependencies.",
                                        block_str
                                    );
                                }
                                _ => {
                                    tracing::error!("Error processing block {}: {}", block_str, e);
                                }
                            }
                        }
                    }

                    // Step 6 (from Scala): Get dependency-free blocks from buffer and enqueue them
                    // Equivalent to: c.getDependencyFreeFromBuffer
                    // In Scala, if this fails, the stream short-circuits and triggerProposeF won't be called
                    match casper.get_dependency_free_from_buffer() {
                        Ok(buffer_pendants) => {
                            // Enqueue pendants if we can mark them as queued/in-processing first.
                            for pendant in &buffer_pendants {
                                let pendant_hash = BlockHash::from(pendant.block_hash.clone());
                                if blocks_in_processing.insert(pendant_hash.clone()) {
                                    let max_in_flight = max_blocks_in_processing();
                                    if blocks_in_processing.len() > max_in_flight {
                                        blocks_in_processing.remove(&pendant_hash);
                                        tracing::warn!(
                                            "Skipping dependency-free pendant {} enqueue because in-flight block cap {} is reached",
                                            PrettyPrinter::build_string_bytes(&pendant.block_hash),
                                            max_in_flight
                                        );
                                        continue;
                                    }
                                    if block_queue_tx
                                        .send((casper.clone(), pendant.clone()))
                                        .await
                                        .is_err()
                                    {
                                        blocks_in_processing.remove(&pendant_hash);
                                        tracing::warn!(
                                            "Dropping dependency-free pendant {} because block queue is closed",
                                            PrettyPrinter::build_string_bytes(&pendant.block_hash)
                                        );
                                    }
                                }
                            }

                            // Only call trigger_propose if get_dependency_free_from_buffer succeeded
                            // and this path is explicitly enabled. Heartbeat proposer is the
                            // default liveness path to avoid propose storms under heavy replay.
                            if trigger_propose_after_block_processing_enabled() {
                                if let Some(trigger_propose) = trigger_propose_f {
                                // Skip trigger if local validator is not currently bonded.
                                // This avoids repeated ReadOnlyMode propose attempts on non-bonded nodes.
                                let is_bonded_validator = if let Some(validator) =
                                    casper.get_validator()
                                {
                                    match casper.get_snapshot().await {
                                        Ok(snapshot) => snapshot
                                            .on_chain_state
                                            .active_validators
                                            .contains(&validator.public_key.bytes),
                                        Err(err) => {
                                            tracing::warn!(
                                                "Failed to get Casper snapshot for trigger-propose bond check: {}",
                                                err
                                            );
                                            false
                                        }
                                    }
                                } else {
                                    false
                                };

                                    if is_bonded_validator {
                                        // Clone the Arc and cast to trait object
                                        let casper_arc: Arc<dyn MultiParentCasper + Send + Sync> =
                                            Arc::clone(&casper)
                                                as Arc<dyn MultiParentCasper + Send + Sync>;
                                        match trigger_propose(casper_arc, true).await {
                                            Ok(_) => {}
                                            Err(err) => {
                                                tracing::error!("Failed to trigger propose: {}", err)
                                            }
                                        }
                                    } else {
                                        tracing::debug!(
                                            "Skipping trigger propose after block processing: validator is not bonded"
                                        );
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            tracing::error!(
                                "Failed to get dependency-free blocks from buffer: {}. Skipping trigger propose.",
                                err
                            );
                            // Don't call trigger_propose if get_dependency_free_from_buffer failed
                        }
                    }

                    // Clean up the hash from processing state
                    blocks_in_processing.remove(&block.block_hash);
                    maybe_trim_allocator_after_block();

                    drop(permit);
                });
            }

            tracing::info!("Block processing queue closed, stopping processor");

            Result::<(), CasperError>::Ok(())
        });

        Ok(result_rx)
    }
}

/// Process a block through all validation steps
///
/// This implements the Scala pipeline:
/// 1. checkIfOfInterest
/// 2. checkIfWellFormedAndStore
/// 3. checkDependenciesWithEffects
/// 4. validateWithEffects
/// 5. Enqueue dependency-free blocks from buffer
/// 6. Trigger propose if configured
async fn process_block_with_steps<T: TransportLayer + Send + Sync>(
    block_processor: Arc<BlockProcessor<T>>,
    casper: Arc<dyn MultiParentCasper + Send + Sync + 'static>,
    block: BlockMessage,
) -> Result<(BlockMessage, ValidBlockProcessing), CasperError> {
    let block_str = PrettyPrinter::build_string_bytes(&block.block_hash);

    // Step 1: Check if block is of interest
    // Equivalent to: blockProcessor.checkIfOfInterest(c, b)
    let is_of_interest = match block_processor.check_if_of_interest(casper.clone(), &block) {
        Ok(is_of_interest) => is_of_interest,
        Err(err) => {
            block_processor
                .ack_processed(&block)
                .await
                .map_err(|ack_err| {
                    CasperError::RuntimeError(format!(
                        "check_if_of_interest failed for {}, and cleanup failed: {}",
                        block_str, ack_err
                    ))
                })?;
            return Err(err);
        }
    };

    if !is_of_interest {
        tracing::info!("Block {} is not of interest. Dropped.", block_str);
        block_processor
            .purge_from_buffer_and_ack(&block)
            .await
            .map_err(|err| {
                CasperError::RuntimeError(format!(
                    "Block {} was not of interest, and purge+cleanup failed: {}",
                    block_str, err
                ))
            })?;
        return Err(CasperError::Other("Block not of interest".to_string()));
    }

    // Step 2: Check if well-formed and store
    // Equivalent to: blockProcessor.checkIfWellFormedAndStore(b)
    let is_well_formed = match block_processor.check_if_well_formed_and_store(&block).await {
        Ok(is_well_formed) => is_well_formed,
        Err(err) => {
            block_processor
                .ack_processed(&block)
                .await
                .map_err(|ack_err| {
                    CasperError::RuntimeError(format!(
                        "check_if_well_formed_and_store failed for {}, and cleanup failed: {}",
                        block_str, ack_err
                    ))
                })?;
            return Err(err);
        }
    };

    if !is_well_formed {
        tracing::info!("Block {} is malformed. Dropped.", block_str);
        block_processor
            .purge_from_buffer_and_ack(&block)
            .await
            .map_err(|err| {
                CasperError::RuntimeError(format!(
                    "Malformed block {} purge+cleanup failed: {}",
                    block_str, err
                ))
            })?;
        return Err(CasperError::Other("Block is malformed".to_string()));
    }

    // Step 3: Log started
    tracing::info!("Block {} processing started.", block_str);

    // Step 4: Check dependencies with effects
    // Equivalent to: blockProcessor.checkDependenciesWithEffects(c, b)
    let has_dependencies = match block_processor
        .check_dependencies_with_effects(casper.clone(), &block)
        .await
    {
        Ok(has_dependencies) => has_dependencies,
        Err(err) => {
            block_processor
                .ack_processed(&block)
                .await
                .map_err(|ack_err| {
                    CasperError::RuntimeError(format!(
                        "check_dependencies_with_effects failed for {}, and cleanup failed: {}",
                        block_str, ack_err
                    ))
                })?;
            return Err(err);
        }
    };

    if !has_dependencies {
        tracing::info!("Block {} missing dependencies.", block_str);
        // `check_dependencies_with_effects` already performs ack/cleanup for this path.
        return Err(CasperError::Other("Missing dependencies".to_string()));
    }

    // Step 5: Validate block with effects
    // Equivalent to: blockProcessor.validateWithEffects(c, b, None)
    let validation_result = match block_processor.validate_with_effects(casper.clone(), &block, None).await {
        Ok(validation_result) => validation_result,
        Err(err) => {
            // ensure this block is no longer tracked in the retriever even when validation fails
            block_processor
                .ack_processed(&block)
                .await
                .map_err(|ack_err| {
                    CasperError::RuntimeError(format!(
                        "validate_with_effects failed for {}, and cleanup failed: {}",
                        block_str, ack_err
                    ))
                })?;
            return Err(err);
        }
    };

    tracing::info!("Block {} validated {:?}.", block_str, validation_result);

    Ok((block, validation_result))
}
