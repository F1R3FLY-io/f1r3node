// See node/src/main/scala/coop/rchain/node/instances/BlockProcessorInstance.scala

use dashmap::DashSet;
use std::sync::Arc;
use tokio::sync::mpsc;

use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::BlockMessage;

use casper::rust::blocks::block_processor::BlockProcessor;
use casper::rust::casper::MultiParentCasper;
use casper::rust::errors::CasperError;
use casper::rust::{ProposeFunction, ValidBlockProcessing};

use comm::rust::transport::transport_layer::TransportLayer;

/// Configuration for BlockProcessorInstance
pub struct BlockProcessorInstance<T: TransportLayer + Send + Sync + 'static> {
    pub blocks_queue_rx: mpsc::UnboundedReceiver<BlockMessage>,

    pub block_queue_tx: mpsc::UnboundedSender<BlockMessage>, // Sender for the same channel as the blocks_queue_rx which is used to enqueue pendants that are not in processing

    pub block_processor: Arc<BlockProcessor<T>>,

    pub blocks_in_processing: Arc<DashSet<BlockHash>>,

    pub trigger_propose_f: Option<Arc<ProposeFunction>>,

    pub casper: Arc<dyn MultiParentCasper + Send + Sync + 'static>,

    pub max_parallel_blocks: usize,
}

impl<T: TransportLayer + Send + Sync + 'static> BlockProcessorInstance<T> {
    pub fn new(
        (blocks_queue_rx, block_queue_tx): (
            mpsc::UnboundedReceiver<BlockMessage>,
            mpsc::UnboundedSender<BlockMessage>,
        ),
        block_processor: Arc<BlockProcessor<T>>,
        blocks_in_processing: Arc<DashSet<BlockHash>>,
        trigger_propose_f: Option<Arc<ProposeFunction>>,
        casper: Arc<dyn MultiParentCasper + Send + Sync + 'static>,
        max_parallel_blocks: usize,
    ) -> Self {
        Self {
            casper,
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
    ) -> Result<mpsc::UnboundedReceiver<(BlockMessage, ValidBlockProcessing)>, CasperError> {
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            let Self {
                casper,
                mut blocks_queue_rx,
                block_queue_tx,
                block_processor,
                blocks_in_processing,
                trigger_propose_f,
                max_parallel_blocks,
            } = self;

            let semaphore = Arc::new(tokio::sync::Semaphore::new(max_parallel_blocks));

            while let Some(block) = blocks_queue_rx.recv().await {
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

                    // Check if block is already in processing and add it
                    let in_processing = {
                        let contains = blocks_in_processing.contains(&block.block_hash);
                        if contains {
                            true
                        } else {
                            blocks_in_processing.insert(block.block_hash.clone());
                            false
                        }
                    };

                    if in_processing {
                        tracing::info!("Block {} is already in processing. Dropped.", block_str);
                        return;
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
                            match result_tx.send(res) {
                                Ok(_) => {}
                                Err(err) => tracing::error!(
                                    "Failed to send block processing result: {}",
                                    err
                                ),
                            }
                        }
                        Err(e) => {
                            tracing::error!("Error processing block {}: {}", block_str, e);
                        }
                    }

                    // Step 6 (from Scala): Get dependency-free blocks from buffer and enqueue them
                    // Equivalent to: c.getDependencyFreeFromBuffer
                    // In Scala, if this fails, the stream short-circuits and triggerProposeF won't be called
                    match casper.get_dependency_free_from_buffer() {
                        Ok(buffer_pendants) => {
                            // Enqueue pendants that are not in processing
                            for pendant in &buffer_pendants {
                                if !blocks_in_processing
                                    .contains(&BlockHash::from(pendant.block_hash.clone()))
                                {
                                    let _ = block_queue_tx.send(pendant.clone());
                                }
                            }

                            // Only call trigger_propose if get_dependency_free_from_buffer succeeded
                            // This matches Scala behavior where evalTap failure prevents next evalTap
                            if let Some(trigger_propose) = trigger_propose_f {
                                // Clone the Arc and cast to trait object
                                let casper_arc: Arc<dyn MultiParentCasper + Send + Sync> =
                                    Arc::clone(&casper) as Arc<dyn MultiParentCasper + Send + Sync>;
                                match trigger_propose(casper_arc, true).await {
                                    Ok(_) => {}
                                    Err(err) => {
                                        tracing::error!("Failed to trigger propose: {}", err)
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
    let is_of_interest = block_processor.check_if_of_interest(casper.clone(), &block)?;

    if !is_of_interest {
        tracing::info!("Block {} is not of interest. Dropped.", block_str);
        return Err(CasperError::Other("Block not of interest".to_string()));
    }

    // Step 2: Check if well-formed and store
    // Equivalent to: blockProcessor.checkIfWellFormedAndStore(b)
    let is_well_formed = {
        block_processor
            .check_if_well_formed_and_store(&block)
            .await?
    };

    if !is_well_formed {
        tracing::info!("Block {} is malformed. Dropped.", block_str);
        return Err(CasperError::Other("Block is malformed".to_string()));
    }

    // Step 3: Log started
    tracing::info!("Block {} processing started.", block_str);

    // Step 4: Check dependencies with effects
    // Equivalent to: blockProcessor.checkDependenciesWithEffects(c, b)
    let has_dependencies = {
        // let casper = casper.lock().unwrap();
        block_processor
            .check_dependencies_with_effects(casper.clone(), &block)
            .await?
    };

    if !has_dependencies {
        tracing::info!("Block {} missing dependencies.", block_str);
        return Err(CasperError::Other("Missing dependencies".to_string()));
    }

    // Step 5: Validate block with effects
    // Equivalent to: blockProcessor.validateWithEffects(c, b, None)
    let validation_result = block_processor
        .validate_with_effects(casper.clone(), &block, None)
        .await?;

    tracing::info!("Block {} validated {:?}.", block_str, validation_result);

    Ok((block, validation_result))
}
