// See casper/src/main/scala/coop/rchain/casper/engine/LfsBlockRequester.scala

use async_stream::stream;
use async_trait::async_trait;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;

use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::{ApprovedBlock, BlockMessage};

use crate::rust::errors::CasperError;
use crate::rust::util::proto_util;

// Last Finalized State processor for receiving blocks.

/// Trait that abstracts the operations needed by the LFS block requester
#[async_trait]
pub trait BlockRequesterOps {
    async fn request_for_block(&self, block_hash: &BlockHash) -> Result<(), CasperError>;

    fn contains_block(&self, block_hash: &BlockHash) -> Result<bool, CasperError>;

    fn get_block_from_store(&self, block_hash: &BlockHash) -> BlockMessage;

    fn put_block_to_store(
        &mut self,
        block_hash: BlockHash,
        block: &BlockMessage,
    ) -> Result<(), CasperError>;

    fn validate_block(&self, block: &BlockMessage) -> bool;
}

/// Possible request statuses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqStatus {
    Init,
    Requested,
    Received,
}

/// Information about block reception status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReceiveInfo {
    pub requested: bool,
    pub latest: bool,
    pub lastlatest: bool,
}

impl ReceiveInfo {
    pub fn new(requested: bool, latest: bool, lastlatest: bool) -> Self {
        Self {
            requested,
            latest,
            lastlatest,
        }
    }
}

/// State to control processing of requests
#[derive(Debug, Clone)]
pub struct ST<Key: Hash + Eq + Clone> {
    pub d: HashMap<Key, ReqStatus>,
    pub latest: HashSet<Key>,
    pub lower_bound: i64,
    pub height_map: BTreeMap<i64, HashSet<Key>>,
    pub finished: HashSet<Key>,
}

impl<Key: Hash + Eq + Clone> ST<Key> {
    /// Create requests state with initial keys.
    pub fn new(
        initial: HashSet<Key>,
        latest: Option<HashSet<Key>>,
        lower_bound: Option<i64>,
    ) -> Self {
        let latest = latest.unwrap_or_default();
        let lower_bound = lower_bound.unwrap_or(0);

        // Set Init status for initial keys
        let d = initial
            .iter()
            .map(|k| (k.clone(), ReqStatus::Init))
            .collect();

        Self {
            d,
            latest,
            lower_bound,
            height_map: BTreeMap::new(),
            finished: HashSet::new(),
        }
    }

    /// Adds new keys to Init state, ready for processing. Existing keys are skipped.
    /// Returns a new ST instance with the added keys.
    pub fn add(&self, keys: HashSet<Key>) -> Self {
        // Filter keys that are in Done status or in request Map.
        let new_keys: HashSet<Key> = keys
            .difference(&self.finished)
            .filter(|k| !self.d.contains_key(k))
            .cloned()
            .collect();

        // Set Init status for new keys
        let mut new_d = self.d.clone();
        for key in new_keys {
            new_d.insert(key, ReqStatus::Init);
        }

        // Return new ST instance with updated request map
        Self {
            d: new_d,
            latest: self.latest.clone(),
            lower_bound: self.lower_bound,
            height_map: self.height_map.clone(),
            finished: self.finished.clone(),
        }
    }

    /// Get next keys not already requested or in case of resend together with Requested.
    /// Returns updated state with requested keys and the set of keys that were marked for request.
    pub fn get_next(&self, resend: bool) -> (Self, HashSet<Key>) {
        // Check if latest are requested
        let requests = if self.latest.is_empty() {
            self.d.clone()
        } else {
            // Add latest to Init if not already in d
            let mut requests = self.d.clone();
            for key in &self.latest {
                if !self.d.contains_key(key) {
                    requests.insert(key.clone(), ReqStatus::Init);
                }
            }
            requests
        };

        let mut new_requests = HashMap::new();
        let mut request_keys = HashSet::new();

        for (key, status) in &requests {
            // Select initialized or re-request if resending
            let check_for_request =
                *status == ReqStatus::Init || (resend && *status == ReqStatus::Requested);

            let should_request = if self.latest.is_empty() {
                // Latest are downloaded, no additional conditions
                check_for_request
            } else {
                // Only latest are requested first
                check_for_request && self.latest.contains(key)
            };

            if should_request {
                new_requests.insert(key.clone(), ReqStatus::Requested);
                request_keys.insert(key.clone());
            }
        }

        // Create new d map with updated requests
        let mut new_d = self.d.clone();
        new_d.extend(new_requests);

        let new_state = Self {
            d: new_d,
            latest: self.latest.clone(),
            lower_bound: self.lower_bound,
            height_map: self.height_map.clone(),
            finished: self.finished.clone(),
        };

        (new_state, request_keys)
    }

    /// Confirm key is Received if it was Requested.
    /// Returns updated state with the flags if Requested and last latest received.
    pub fn received(
        &self,
        k: Key,
        height: i64,
        latest_replacement: Option<Key>,
    ) -> (Self, ReceiveInfo) {
        let is_req = self.d.get(&k) == Some(&ReqStatus::Requested);

        if is_req {
            // Remove message from the set of latest messages (if exists)
            let mut adj_latest = self.latest.clone();
            let was_removed = adj_latest.remove(&k);
            let is_latest = was_removed;

            // Add replacement message if supplied
            let new_latest = if let Some(replacement) = latest_replacement {
                adj_latest.insert(replacement);
                adj_latest
            } else {
                adj_latest
            };

            let is_last_latest = is_latest && new_latest.is_empty();

            // Save in height map
            let mut new_height_map = self.height_map.clone();
            let height_keys = new_height_map.entry(height).or_insert_with(HashSet::new);
            height_keys.insert(k.clone());

            // Calculate new minimum height if latest message
            // - we need parents of latest message so it's `-1`
            let new_lower_bound = if is_latest {
                std::cmp::min(height - 1, self.lower_bound)
            } else {
                self.lower_bound
            };

            // Set new status to Received
            let mut new_d = self.d.clone();
            new_d.insert(k, ReqStatus::Received);

            // Set new minimum height and update latest
            let new_state = Self {
                d: new_d,
                latest: new_latest,
                lower_bound: new_lower_bound,
                height_map: new_height_map,
                finished: self.finished.clone(),
            };

            (
                new_state,
                ReceiveInfo::new(is_req, is_latest, is_last_latest),
            )
        } else {
            (self.clone(), ReceiveInfo::new(false, false, false))
        }
    }

    /// Mark key as finished (Done) with optionally set minimum lower bound.
    /// Returns a new ST instance with the key moved from requests to finished.
    pub fn done(&self, k: Key) -> Self {
        let is_received = self.d.get(&k) == Some(&ReqStatus::Received);

        if is_received {
            // If Received key, remove from request Map and add to Done Set
            let mut new_d = self.d.clone();
            new_d.remove(&k);

            let mut new_finished = self.finished.clone();
            new_finished.insert(k);

            Self {
                d: new_d,
                latest: self.latest.clone(),
                lower_bound: self.lower_bound,
                height_map: self.height_map.clone(),
                finished: new_finished,
            }
        } else {
            self.clone()
        }
    }

    /// Returns flag if all keys are marked as finished (Done).
    pub fn is_finished(&self) -> bool {
        self.latest.is_empty() && self.d.is_empty()
    }
}

struct StreamProcessor<'a, T: BlockRequesterOps> {
    requester: &'a mut T,
    st: Arc<Mutex<ST<BlockHash>>>,
    latest_messages: HashSet<BlockHash>,
    response_hash_sender: mpsc::UnboundedSender<BlockHash>,
}

impl<'a, T: BlockRequesterOps> StreamProcessor<'a, T> {
    fn new(
        requester: &'a mut T,
        st: Arc<Mutex<ST<BlockHash>>>,
        latest_messages: HashSet<BlockHash>,
        response_hash_sender: mpsc::UnboundedSender<BlockHash>,
    ) -> Self {
        Self {
            requester,
            st,
            latest_messages,
            response_hash_sender,
        }
    }

    async fn process_block(&mut self, block: &BlockMessage) -> Result<(), CasperError> {
        // Validate and mark received block
        let is_valid = self.validate_received_block(block).await?;

        // Save block to store
        if is_valid {
            self.save_block(block).await?;
        }
        // Trigger request queue (without resend of already requested)
        self.request_next(false).await?;
        Ok(())
    }

    /// Validate received block, check if it was requested and if block hash is correct.
    /// Following justifications from last finalized block gives us proof for all ancestor blocks.
    async fn validate_received_block(&self, block: &BlockMessage) -> Result<bool, CasperError> {
        let validation_start = std::time::Instant::now();
        let block_number = proto_util::block_number(block);
        let block_hash_str = format!("{:?}", block.block_hash);

        tracing::debug!(
            "Validating received block {} at height {}",
            block_hash_str,
            block_number
        );

        // Mark block as received and calculate minimum height (if latest)
        let received_result = {
            let mut state = self.st.lock().map_err(|_| {
                CasperError::StreamError("Failed to acquire state lock for validation".to_string())
            })?;

            // if message received is latest as per approved block - add its self justification
            // to target latest messages that has to be pulled
            let lm_replacement = if self.latest_messages.contains(&block.block_hash) {
                tracing::info!(
                    "Block {} is a latest message, checking for self-justification replacement",
                    block_hash_str
                );
                proto_util::creator_justification_block_message(block).map(|justification| {
                    tracing::debug!(
                        "Found self-justification replacement: {:?}",
                        justification.latest_block_hash
                    );
                    justification.latest_block_hash
                })
            } else {
                None
            };

            let (new_state, receive_info) =
                state.received(block.block_hash.clone(), block_number, lm_replacement);
            *state = new_state;
            receive_info
        };

        let ReceiveInfo {
            requested: is_received,
            latest: is_received_latest,
            lastlatest: is_last_latest,
        } = received_result;

        tracing::debug!(
            "Block {} validation status - requested: {}, latest: {}, last_latest: {}",
            block_hash_str,
            is_received,
            is_received_latest,
            is_last_latest
        );

        // Validate block hash if it was requested
        let hash_validation_start = std::time::Instant::now();
        let block_hash_is_valid = if is_received {
            let is_valid = self.requester.validate_block(block);
            let hash_validation_duration = hash_validation_start.elapsed();
            tracing::debug!(
                "Block {} hash validation completed in {:?}: {}",
                block_hash_str,
                hash_validation_duration,
                is_valid
            );
            is_valid
        } else {
            tracing::debug!(
                "Block {} was not requested, skipping hash validation",
                block_hash_str
            );
            false
        };

        // Log invalid block if block is requested but hash is invalid
        if is_received && !block_hash_is_valid {
            let invalid_block_msg = format!(
                "Received {} with invalid hash. Ignored block.",
                PrettyPrinter::build_string_block_message(block, false)
            );
            tracing::warn!("{}", invalid_block_msg);
        }

        // Try accept received block if it has valid hash
        let is_received_and_valid = if block_hash_is_valid {
            // Log minimum height when last latest block is received
            if is_last_latest {
                let minimum_height = {
                    let state = self.st.lock().map_err(|_| {
                        CasperError::StreamError(
                            "Failed to acquire state lock for minimum height".to_string(),
                        )
                    })?;
                    state.lower_bound
                };
                tracing::info!(
                    "Latest blocks downloaded. Minimum block height is {}.",
                    minimum_height
                );
            }

            // Accept block if it's requested and satisfy conditions
            // - received one of latest messages
            // - requested and block number is greater than minimum
            let block_is_accepted = {
                let state = self.st.lock().map_err(|_| {
                    CasperError::StreamError(
                        "Failed to acquire state lock for acceptance check".to_string(),
                    )
                })?;
                let accepted =
                    is_received_latest || (is_received && block_number >= state.lower_bound);
                tracing::debug!(
                    "Block {} acceptance check - latest: {}, height_ok: {} ({}>={}), accepted: {}",
                    block_hash_str,
                    is_received_latest,
                    block_number >= state.lower_bound,
                    block_number,
                    state.lower_bound,
                    accepted
                );
                accepted
            };

            if block_is_accepted {
                // Update dependencies for requesting
                let deps_start = std::time::Instant::now();
                let deps: HashSet<BlockHash> = proto_util::dependencies_hashes_of(block)
                    .into_iter()
                    .collect();
                let deps_duration = deps_start.elapsed();

                tracing::info!(
                    "Block {} accepted, adding {} dependencies (computed in {:?})",
                    block_hash_str,
                    deps.len(),
                    deps_duration
                );

                let mut state = self.st.lock().map_err(|_| {
                    CasperError::StreamError(
                        "Failed to acquire state lock for dependencies".to_string(),
                    )
                })?;
                *state = state.add(deps);
            } else {
                tracing::debug!(
                    "Block {} not accepted due to validation criteria",
                    block_hash_str
                );
            }

            is_received
        } else {
            tracing::debug!(
                "Block {} validation failed or not requested",
                block_hash_str
            );
            false
        };

        let validation_duration = validation_start.elapsed();
        tracing::debug!(
            "Block {} validation completed in {:?}: {}",
            block_hash_str,
            validation_duration,
            is_received_and_valid
        );

        Ok(is_received_and_valid)
    }

    async fn save_block(&mut self, block: &BlockMessage) -> Result<(), CasperError> {
        let save_start = std::time::Instant::now();
        let block_hash_str = format!("{:?}", block.block_hash);

        tracing::debug!("Saving block {} to store", block_hash_str);

        // Save block to the store
        let storage_start = std::time::Instant::now();
        let already_saved = self
            .requester
            .contains_block(&block.block_hash)
            .unwrap_or(false);

        if !already_saved {
            self.requester
                .put_block_to_store(block.block_hash.clone(), block)?;
            let storage_duration = storage_start.elapsed();
            tracing::info!(
                "Block {} saved to store in {:?}",
                block_hash_str,
                storage_duration
            );
        } else {
            tracing::debug!(
                "Block {} already exists in store, skipping save",
                block_hash_str
            );
        }

        // Mark block download as done
        let state_update_start = std::time::Instant::now();
        let mut state = self.st.lock().map_err(|_| {
            CasperError::StreamError("Failed to acquire state lock for done".to_string())
        })?;
        *state = state.done(block.block_hash.clone());
        let state_update_duration = state_update_start.elapsed();

        let total_save_duration = save_start.elapsed();
        tracing::debug!(
            "Block {} marked as done (state update: {:?}, total: {:?})",
            block_hash_str,
            state_update_duration,
            total_save_duration
        );

        Ok(())
    }

    /// Reads current state for next blocks and send the requests.
    async fn request_next(&self, resend: bool) -> Result<(), CasperError> {
        let start_time = std::time::Instant::now();

        // Check if stream is finished (no more requests)
        let (is_end, active_count, latest_count, finished_count, lower_bound) = {
            let state = self.st.lock().map_err(|_| {
                CasperError::StreamError("Failed to acquire state lock".to_string())
            })?;
            (
                state.is_finished(),
                state.d.len(),
                state.latest.len(),
                state.finished.len(),
                state.lower_bound,
            )
        };

        if resend {
            tracing::info!(
                "Processing resend request - active: {}, latest: {}, finished: {}, lower_bound: {}",
                active_count,
                latest_count,
                finished_count,
                lower_bound
            );
        } else {
            tracing::debug!(
                "Processing new request - active: {}, latest: {}, finished: {}, lower_bound: {}",
                active_count,
                latest_count,
                finished_count,
                lower_bound
            );
        }

        // Take next set of items to request (w/o duplicates)
        let hashes = {
            let mut state = self.st.lock().map_err(|_| {
                CasperError::StreamError("Failed to acquire state lock".to_string())
            })?;
            let (new_state, next_hashes) = state.get_next(resend);
            *state = new_state;
            next_hashes
        };

        if hashes.is_empty() {
            tracing::debug!("No new blocks to request (resend: {})", resend);
            return Ok(());
        }

        tracing::info!("Requesting {} blocks (resend: {})", hashes.len(), resend);

        // Check existing blocks
        let existing_check_start = std::time::Instant::now();
        let existing_hashes: Vec<BlockHash> = hashes
            .iter()
            .filter(|hash| self.requester.contains_block(hash).unwrap_or(false))
            .cloned()
            .collect();
        let existing_check_duration = existing_check_start.elapsed();

        tracing::debug!(
            "Block existence check completed in {:?} - {} of {} blocks already exist",
            existing_check_duration,
            existing_hashes.len(),
            hashes.len()
        );

        // Enqueue hashes of existing blocks
        if !existing_hashes.is_empty() {
            tracing::info!(
                "Found {} existing blocks in store, queueing for processing",
                existing_hashes.len()
            );
            for hash in existing_hashes.iter() {
                self.response_hash_sender.send(hash.clone()).map_err(|_| {
                    CasperError::StreamError("Failed to enqueue existing hash".to_string())
                })?;
            }
        }

        // Missing blocks not already in the block store
        let missing_blocks: HashSet<BlockHash> = hashes
            .difference(&existing_hashes.into_iter().collect())
            .cloned()
            .collect();

        // Send all requests in parallel for missing blocks
        if !is_end && !missing_blocks.is_empty() {
            let request_start = std::time::Instant::now();
            tracing::info!(
                "Broadcasting requests for {} missing blocks",
                missing_blocks.len()
            );

            // Create parallel futures for all missing block requests
            let request_futures: Vec<_> = missing_blocks
                .into_iter()
                .map(
                    |block_hash| async move { self.requester.request_for_block(&block_hash).await },
                )
                .collect();

            let request_count = request_futures.len();

            // Execute all requests in parallel, short-circuit on first error
            futures::future::try_join_all(request_futures).await?;

            let request_duration = request_start.elapsed();
            tracing::info!(
                "Completed broadcasting {} block requests in {:?}",
                request_count,
                request_duration
            );
        }

        let total_duration = start_time.elapsed();
        tracing::debug!(
            "request_next completed in {:?} (resend: {})",
            total_duration,
            resend
        );

        Ok(())
    }
}

/// Create a stream to receive blocks needed for Last Finalized State.
pub async fn stream<'a, T: BlockRequesterOps>(
    approved_block: &'a ApprovedBlock,
    initial_response_messages: &'a VecDeque<BlockMessage>,
    response_message_receiver: mpsc::UnboundedReceiver<BlockMessage>,
    initial_minimum_height: i64,
    request_timeout: Duration,
    block_ops: &'a mut T,
) -> Result<impl futures::stream::Stream<Item = ST<BlockHash>> + use<'a, T>, CasperError> {
    let block = &approved_block.candidate.block;

    // Active validators as per approved block state
    // - for approved state to be complete it is required to have block from each of them
    let latest_messages: HashSet<BlockHash> = block
        .justifications
        .iter()
        .map(|justification| justification.latest_block_hash.clone())
        .collect();

    let initial_hashes = {
        let mut set = HashSet::new();
        set.insert(block.block_hash.clone());
        set
    };

    // Requester state, fill with validators for required latest messages
    let st = Arc::new(Mutex::new(ST::new(
        initial_hashes,
        Some(latest_messages.clone()),
        Some(initial_minimum_height),
    )));

    // Queue to trigger processing of requests. `True` to resend requests.
    let (request_tx, request_rx) = mpsc::channel(2);
    // Response queue for existing blocks in the store.
    let (response_hash_tx, response_hash_rx) = mpsc::unbounded_channel();

    // Light the fire! / Starts the first request for block
    // - `false` means don't resend already requested blocks
    request_tx
        .send(false)
        .await
        .map_err(|_| CasperError::StreamError("Failed to send initial request".to_string()))?;

    let processor = StreamProcessor::new(block_ops, st.clone(), latest_messages, response_hash_tx);

    // Task 6.2: Enhanced resource cleanup with proper channel management
    tracing::info!("LFS Block Requester stream initialized - starting processing");
    tracing::debug!(
        "Initial messages queue size: {}",
        initial_response_messages.len()
    );
    tracing::debug!("Initial minimum height: {}", initial_minimum_height);
    tracing::debug!("Request timeout: {:?}", request_timeout);

    let stream_result = create_stream_with_processor(
        processor,
        request_rx,
        request_tx,
        response_hash_rx,
        &initial_response_messages,
        response_message_receiver,
        request_timeout,
    )
    .await;

    match &stream_result {
        Ok(_) => {
            tracing::info!("LFS Block Requester stream created successfully");
        }
        Err(e) => {
            tracing::error!("Failed to create LFS Block Requester stream: {:?}", e);
            // Cleanup is handled automatically by Drop implementations for channels
        }
    }

    stream_result
}

async fn create_stream_with_processor<'a, T: BlockRequesterOps>(
    mut processor: StreamProcessor<'a, T>,
    mut request_queue: mpsc::Receiver<bool>,
    request_queue_sender: mpsc::Sender<bool>,
    mut response_hash_queue: mpsc::UnboundedReceiver<BlockHash>,
    initial_response_messages: &'a VecDeque<BlockMessage>,
    mut response_message_receiver: mpsc::UnboundedReceiver<BlockMessage>,
    request_timeout: Duration,
) -> Result<impl futures::stream::Stream<Item = ST<BlockHash>> + use<'a, T>, CasperError> {
    let processor_count = num_cpus::get();
    tracing::info!(
        "LFS Block Requester using {} processor-bounded workers (parEvalMapProcBounded equivalent)",
        processor_count
    );

    // Create semaphore for bounded concurrency
    let response_semaphore = Arc::new(Semaphore::new(processor_count));

    // We reset this timeout every time there's activity (request/response processing)
    let mut idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

    let timeout_msg = format!(
        "No block responses for {:?}. Resending requests.",
        request_timeout
    );

    // Process all initial messages immediately and merge with regular message stream
    // This ensures proper ordering and efficient processing of pre-existing messages
    let (initial_messages_tx, mut initial_messages_rx) = mpsc::unbounded_channel();

    // Enqueue all initial messages for processing - this preserves order and integrates with normal flow
    if !initial_response_messages.is_empty() {
        tracing::info!(
            "Processing {} initial response messages",
            initial_response_messages.len()
        );
        for initial_message in initial_response_messages {
            if let Err(_) = initial_messages_tx.send(initial_message) {
                tracing::error!("Failed to enqueue initial message - channel closed");
                return Err(CasperError::StreamError(
                    "Failed to setup initial messages".to_string(),
                ));
            }
        }
    }
    // Close the sender to indicate no more initial messages
    drop(initial_messages_tx);

    // Create the actual stream using async_stream::stream!
    let stream = stream! {
        // Track consecutive errors for potential circuit breaking
        let mut consecutive_errors = 0;
        const MAX_CONSECUTIVE_ERRORS: u32 = 5;

        // Main stream processing loop
        loop {
            tokio::select! {
                Some(resend_flag) = request_queue.recv() => {
                    match processor.request_next(resend_flag).await {
                        Ok(()) => {
                            consecutive_errors = 0; // Reset error counter on success
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            tracing::error!("Failed to process request (attempt {}): {:?}", consecutive_errors, e);

                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                tracing::error!("Maximum consecutive errors reached ({}), terminating stream", MAX_CONSECUTIVE_ERRORS);
                                break;
                            }

                            // Continue processing other arms instead of immediate break
                            continue;
                        }
                    }

                    // Map to current state after requestNext
                    let current_state = {
                        match processor.st.lock() {
                            Ok(state) => state.clone(),
                            Err(e) => {
                                tracing::error!("Failed to acquire state lock for request processing: {:?}", e);
                                // Try to continue with other arms instead of breaking immediately
                                continue;
                            }
                        }
                    };

                    // Terminate when state is finished
                    if current_state.is_finished() {
                        tracing::info!("Request processing completed - all blocks downloaded");
                        yield current_state;
                        break;
                    }

                    // Reset idle timeout due to activity
                    idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

                    // Emit state to stream
                    yield current_state;
                }

                _ = &mut idle_timeout => {
                    tracing::warn!("{}", timeout_msg);

                    match request_queue_sender.try_send(true) {
                        Ok(()) => {
                            tracing::debug!("Timeout triggered - resend request enqueued successfully");
                            // Reset the timeout for next idle period
                            idle_timeout = Box::pin(tokio::time::sleep(request_timeout));
                        }
                        Err(e) => {
                            tracing::error!("Failed to enqueue resend request - channel error or full: {:?}", e);
                            tracing::warn!("Request queue channel appears closed or full, checking if stream should terminate");

                            // Check if we should terminate gracefully
                            let should_terminate = {
                                match processor.st.lock() {
                                    Ok(state) => state.is_finished(),
                                    Err(_) => {
                                        tracing::error!("Cannot acquire state lock to check termination condition");
                                        true // Assume termination if we can't check state
                                    }
                                }
                            };

                            if should_terminate {
                                tracing::info!("Stream terminating gracefully - processing appears complete");
                                break;
                            }
                            // Reset timeout even on error to continue monitoring
                            idle_timeout = Box::pin(tokio::time::sleep(request_timeout));
                        }
                    }
                }

                // Process initial messages with highest priority (before network messages)
                // This ensures initial messages are handled first, which is often more efficient
                Some(initial_block) = initial_messages_rx.recv() => {
                    let permit = match response_semaphore.clone().acquire_owned().await {
                        Ok(permit) => permit,
                        Err(e) => {
                            tracing::error!("Failed to acquire semaphore permit for initial response processing: {:?}", e);
                            // Continue with other arms instead of breaking
                            continue;
                        }
                    };

                    // Process with bounded concurrency - same as regular response processing
                    let _permit = permit; // Hold permit during processing

                    tracing::info!("Processing initial response message {}",
                        PrettyPrinter::build_string_block_message(initial_block, true));

                    // Process initial block with same logic as regular response messages
                    match processor.process_block(initial_block).await {
                        Ok(()) => {
                            consecutive_errors = 0; // Reset error counter on success
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            tracing::error!("Failed to process initial block message (attempt {}): {:?}", consecutive_errors, e);

                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                tracing::error!("Maximum consecutive errors reached while processing initial messages, terminating stream");
                                break;
                            }
                            continue;
                        }
                    }

                    // Emit current state after processing initial response
                    let current_state = {
                        match processor.st.lock() {
                            Ok(state) => state.clone(),
                            Err(e) => {
                                tracing::error!("Failed to acquire state lock for initial response processing: {:?}", e);
                                continue;
                            }
                        }
                    };

                    // Check termination condition after initial response processing
                    if current_state.is_finished() {
                        tracing::info!("Initial response processing completed - all blocks downloaded");
                        yield current_state;
                        break;
                    }

                    // Reset idle timeout due to activity
                    idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

                    // Emit state to stream
                    yield current_state;
                }

                Some(block_message) = response_message_receiver.recv() => {
                    let permit = match response_semaphore.clone().acquire_owned().await {
                        Ok(permit) => permit,
                        Err(e) => {
                            tracing::error!("Failed to acquire semaphore permit for response processing: {:?}", e);
                            continue;
                        }
                    };

                    // Process with bounded concurrency
                    let _permit = permit; // Hold permit during processing

                    match processor.process_block(&block_message).await {
                        Ok(()) => {
                            consecutive_errors = 0; // Reset error counter on success
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            tracing::error!("Failed to process block message (attempt {}): {:?}", consecutive_errors, e);

                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                tracing::error!("Maximum consecutive errors reached while processing block messages, terminating stream");
                                break;
                            }
                            continue;
                        }
                    }

                    // Emit current state after processing response
                    let current_state = {
                        match processor.st.lock() {
                            Ok(state) => state.clone(),
                            Err(e) => {
                                tracing::error!("Failed to acquire state lock for response processing: {:?}", e);
                                continue;
                            }
                        }
                    };

                    // Check termination condition after response processing
                    if current_state.is_finished() {
                        tracing::info!("Response processing completed - all blocks downloaded");
                        yield current_state;
                        break;
                    }

                    // Reset idle timeout due to activity
                    idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

                    // Emit state to stream
                    yield current_state;
                }

                Some(block_hash) = response_hash_queue.recv() => {
                    let permit = match response_semaphore.clone().acquire_owned().await {
                        Ok(permit) => permit,
                        Err(e) => {
                            tracing::error!("Failed to acquire semaphore permit for hash response processing: {:?}", e);
                            continue;
                        }
                    };

                    // Process with bounded concurrency
                    let _permit = permit; // Hold permit during processing

                    let block = processor.requester.get_block_from_store(&block_hash);

                    tracing::info!("Process existing {}", PrettyPrinter::build_string_block_message(&block, false));

                    match processor.process_block(&block).await {
                        Ok(()) => {
                            consecutive_errors = 0; // Reset error counter on success
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            tracing::error!("Failed to process existing block (attempt {}): {:?}", consecutive_errors, e);

                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                                tracing::error!("Maximum consecutive errors reached while processing existing blocks, terminating stream");
                                break;
                            }
                            continue;
                        }
                    }

                    // Emit current state after processing existing block
                    let current_state = {
                        match processor.st.lock() {
                            Ok(state) => state.clone(),
                            Err(e) => {
                                tracing::error!("Failed to acquire state lock for hash response processing: {:?}", e);
                                continue;
                            }
                        }
                    };

                    // Check termination condition after hash response processing
                    if current_state.is_finished() {
                        tracing::info!("Hash response processing completed - all blocks downloaded");
                        yield current_state;
                        break;
                    }

                    // Reset idle timeout due to activity
                    idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

                    // Emit state to stream
                    yield current_state;
                }
            }
        }

        // Integration testing and final state validation
        let final_state = {
            match processor.st.lock() {
                Ok(state) => {
                    let final_stats = (
                        state.d.len(),
                        state.latest.len(),
                        state.finished.len(),
                        state.lower_bound,
                        state.is_finished()
                    );
                    tracing::info!("LFS Block Requester stream completed - Final state: active: {}, latest: {}, finished: {}, lower_bound: {}, is_finished: {}",
                        final_stats.0, final_stats.1, final_stats.2, final_stats.3, final_stats.4);

                    if final_stats.4 {
                        tracing::info!("✅ LFS Block Requester completed successfully - all required blocks downloaded");
                    } else {
                        tracing::warn!("⚠️ LFS Block Requester terminated with incomplete state - some blocks may be missing");
                    }

                    Some(state.clone())
                }
                Err(e) => {
                    tracing::error!("Failed to acquire final state lock: {:?}", e);
                    None
                }
            }
        };

        // Emit final state if available for integration testing
        if let Some(final_state) = final_state {
            yield final_state;
        }

        tracing::info!("LFS Block Requester stream processing completed");
    };

    Ok(stream)
}
