// See casper/src/main/scala/coop/rchain/casper/engine/LfsTupleSpaceRequester.scala

use async_stream;
use async_trait::async_trait;
use futures::Stream;
use shared::rust::ByteVector;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

use models::rust::casper::protocol::casper_message::{ApprovedBlock, StoreItemsMessage};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, state::rspace_importer::RSpaceImporter,
};

use crate::rust::errors::CasperError;

// Last Finalized State processor for receiving Rholang state.

/// Possible request statuses
///
/// **Scala equivalent**: `trait ReqStatus` with case objects
/// - Init, Requested, Received, Done
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqStatus {
    Init,      // Scala: case object Init extends ReqStatus
    Requested, // Scala: case object Requested extends ReqStatus
    Received,  // Scala: case object Received extends ReqStatus
    Done,      // Scala: case object Done extends ReqStatus
}

/// Definition of RSpace state path (history tree).
///
/// **Scala equivalent**: `type StatePartPath = Seq[(Blake2b256Hash, Option[Byte])]`
/// Elements in the sequence represents nested levels, with 'byte' as index
/// in pointers if node is PointerBlock.
pub type StatePartPath = Vec<(Blake2b256Hash, Option<u8>)>;

/// Number of nodes in LFS sync data transfer
///
/// **Scala equivalent**: `val pageSize = 750`
pub const PAGE_SIZE: i32 = 750;

/// Trait that abstracts the network operations needed by the LFS tuple space requester
///
/// **Scala equivalent**: Function parameter `requestForStoreItem: (StatePartPath, Int) => F[Unit]`
/// in the stream method - we extract this into a trait for better Rust patterns
#[async_trait]
pub trait TupleSpaceRequesterOps {
    /// Send request for state chunk
    ///
    /// **Scala equivalent**: `requestForStoreItem(id, pageSize)` call in broadcastStreams
    async fn request_for_store_item(
        &self,
        path: &StatePartPath,
        page_size: i32,
    ) -> Result<(), CasperError>;

    fn validate_tuple_space_items(
        &self,
        history_items: Vec<(Blake2b256Hash, Vec<u8>)>,
        data_items: Vec<(Blake2b256Hash, Vec<u8>)>,
        start_path: StatePartPath,
        page_size: i32,
        skip: i32,
        get_from_history: impl Fn(Blake2b256Hash) -> Option<ByteVector> + Send + 'static,
    ) -> Result<(), CasperError>;
}

/// State to control processing of requests
///
/// **Scala equivalent**: `final case class ST[Key](private val d: Map[Key, ReqStatus])`
/// Note: Scala version is much simpler than block requester - no latest, lowerBound, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct ST<Key: Hash + Eq + Clone> {
    d: HashMap<Key, ReqStatus>, // Scala: private val d: Map[Key, ReqStatus]
}

impl<Key: Hash + Eq + Clone> ST<Key> {
    /// Create requests state with initial keys.
    ///
    /// **Scala equivalent**: `ST.apply[Key](initial: Seq[Key]): ST[Key]` in companion object
    pub fn new(initial: Vec<Key>) -> Self {
        let d = initial
            .into_iter()
            .map(|key| (key, ReqStatus::Init))
            .collect();

        Self { d }
    }

    /// Adds new keys to Init state, ready for processing. Existing keys are skipped.
    ///
    /// **Scala equivalent**: `def add(keys: Set[Key]): ST[Key]`
    pub fn add(&self, keys: std::collections::HashSet<Key>) -> Self {
        let mut new_d = self.d.clone();

        // Add keys that don't already exist in the map, setting them to Init
        for key in keys {
            if !self.d.contains_key(&key) {
                new_d.insert(key, ReqStatus::Init);
            }
        }

        Self { d: new_d }
    }

    /// Get next keys not already requested or in case of resend together with Requested.
    ///
    /// **Scala equivalent**: `def getNext(resend: Boolean): (ST[Key], Seq[Key])`
    pub fn get_next(&self, resend: bool) -> (Self, Vec<Key>) {
        let mut new_d = self.d.clone();
        let mut requested_keys = Vec::new();

        for (key, status) in &self.d {
            let should_request = match status {
                ReqStatus::Init => true,
                ReqStatus::Requested if resend => true,
                _ => false,
            };

            if should_request {
                new_d.insert(key.clone(), ReqStatus::Requested);
                requested_keys.push(key.clone());
            }
        }

        (Self { d: new_d }, requested_keys)
    }

    /// Confirm key is Received if it was Requested.
    ///
    /// **Scala equivalent**: `def received(k: Key): (ST[Key], Boolean)`
    pub fn received(&self, k: Key) -> (Self, bool) {
        let is_requested = self.d.get(&k) == Some(&ReqStatus::Requested);

        let new_d = if is_requested {
            let mut updated_d = self.d.clone();
            updated_d.insert(k, ReqStatus::Received);
            updated_d
        } else {
            self.d.clone()
        };

        (Self { d: new_d }, is_requested)
    }

    /// Mark key as finished (Done).
    ///
    /// **Scala equivalent**: `def done(k: Key): ST[Key]`
    pub fn done(&self, k: Key) -> Self {
        let is_received = self.d.get(&k) == Some(&ReqStatus::Received);

        if is_received {
            let mut new_d = self.d.clone();
            new_d.insert(k, ReqStatus::Done);
            Self { d: new_d }
        } else {
            self.clone()
        }
    }

    /// Returns flag if all keys are marked as finished (Done).
    ///
    /// **Scala equivalent**: `def isFinished: Boolean`
    pub fn is_finished(&self) -> bool {
        !self.d.values().any(|status| *status != ReqStatus::Done)
    }
}

/// Stream processor for tuple space requester operations
///
/// **Scala equivalent**: This corresponds to the collection of nested functions inside
/// `createStream(st, requestQueue, responseHashQueue)` in the Scala implementation:
/// - processBlock(block)
/// - validateReceivedBlock(block)  
/// - saveBlock(block)
/// - requestNext(resend)
/// We group these into a processor struct for better Rust organization
struct TupleSpaceStreamProcessor<T: TupleSpaceRequesterOps, R: RSpaceImporter + Clone + 'static> {
    request_ops: T,    // For network operations (broadcastStreams equivalent)
    state_importer: R, // Scala: stateImporter parameter
    st: Arc<Mutex<ST<StatePartPath>>>, // Scala: st: Ref[F, ST[StatePartPath]]
    request_tx: mpsc::Sender<bool>, // Scala: requestQueue for triggering new request cycles
}

impl<T: TupleSpaceRequesterOps, R: RSpaceImporter + Clone + 'static>
    TupleSpaceStreamProcessor<T, R>
{
    /// Create a new tuple space stream processor
    fn new(
        request_ops: T,
        state_importer: R,
        st: Arc<Mutex<ST<StatePartPath>>>,
        request_tx: mpsc::Sender<bool>,
    ) -> Self {
        Self {
            request_ops,
            state_importer,
            st,
            request_tx,
        }
    }

    /// Process incoming store items message from the network
    ///
    /// **Scala equivalent**: The main logic inside `responseStream` for handling `StoreItemsMessage`
    /// This combines the message destructuring and the validation/import workflow
    async fn process_store_items_message(
        &mut self,
        message: StoreItemsMessage,
    ) -> Result<(), CasperError> {
        // Extract message fields (Scala: StoreItemsMessage(startPath, lastPath, historyItems, dataItems) = msg)
        let StoreItemsMessage {
            start_path,
            last_path,
            history_items,
            data_items,
        } = message;

        // Convert Bytes to Vec<u8> for compatibility
        let history_items: Vec<(Blake2b256Hash, Vec<u8>)> = history_items
            .into_iter()
            .map(|(hash, bytes)| (hash, bytes.to_vec()))
            .collect();

        let data_items: Vec<(Blake2b256Hash, Vec<u8>)> = data_items
            .into_iter()
            .map(|(hash, bytes)| (hash, bytes.to_vec()))
            .collect();

        // Mark chunk as received (Scala: isReceived <- Stream.eval(st.modify(_.received(startPath))))
        let is_received = self.mark_chunk_received(start_path.clone()).await?;

        if is_received {
            // Add next paths for requesting (Scala: st.update(_.add(Set(lastPath))))
            let mut next_paths = std::collections::HashSet::new();
            next_paths.insert(last_path);
            self.add_next_paths(next_paths).await?;

            // Trigger request processing after adding next paths (Scala: requestQueue.enqueue1(false))
            if let Err(_) = self.request_tx.send(false).await {
                log::debug!("Failed to trigger request processing - channel may be closed");
            }

            // Validate and import chunk in parallel
            self.validate_and_import_chunk(start_path.clone(), history_items, data_items)
                .await?;

            // Mark chunk as done (Scala: st.update(_.done(startPath)))
            self.mark_chunk_done(start_path).await?;

            // Trigger request processing again after marking chunk as done (Scala: requestQueue.enqueue1(false))
            if let Err(_) = self.request_tx.send(false).await {
                log::debug!("Failed to trigger request processing - channel may be closed");
            }
        }

        Ok(())
    }

    /// Validate received chunk and import to RSpace in parallel
    ///
    /// **Scala equivalent**: The complex validation and import logic inside the responseStream:
    /// ```scala
    /// val validationProcess = Stream.eval { validateTupleSpaceItems(...) }
    /// val historySaveProcess = Stream.eval { stateImporter.setHistoryItems(...) }  
    /// val dataSaveProcess = Stream.eval { stateImporter.setDataItems(...) }
    /// Stream(validationProcess, historySaveProcess, dataSaveProcess).parJoinUnbounded.compile.drain
    /// ```
    async fn validate_and_import_chunk(
        &mut self,
        start_path: StatePartPath,
        history_items: Vec<(Blake2b256Hash, Vec<u8>)>,
        data_items: Vec<(Blake2b256Hash, Vec<u8>)>,
    ) -> Result<(), CasperError> {
        let total_validation_start = std::time::Instant::now();

        log::info!(
            "Validating and importing chunk for path (length: {}) - {} history items, {} data items",
            start_path.len(),
            history_items.len(),
            data_items.len()
        );

        // TODO: The three tasks below are not executed in parallel.
        // We should use tokio::try_join! to execute them in parallel if performance is an issue.
        let state_importer_cloned = self.state_importer.clone();

        // Create validation task (Scala: validateTupleSpaceItems(...))
        let validation_start = std::time::Instant::now();
        self.request_ops.validate_tuple_space_items(
            history_items.clone(),
            data_items.clone(),
            start_path,
            PAGE_SIZE,
            0, // offset
            move |hash: Blake2b256Hash| state_importer_cloned.get_history_item(hash),
        )?;
        let validation_duration = validation_start.elapsed();
        log::debug!("Validation completed in {:?}", validation_duration);

        // Create history import task (Scala: stateImporter.setHistoryItems(...))
        let history_start = std::time::Instant::now();
        let _ = self.state_importer.set_history_items(history_items);
        let history_duration = history_start.elapsed();
        log::debug!("History import completed in {:?}", history_duration);

        // Create data import task (Scala: stateImporter.setDataItems(...))
        let data_start = std::time::Instant::now();
        let _ = self.state_importer.set_data_items(data_items);
        let data_duration = data_start.elapsed();
        log::debug!("Data import completed in {:?}", data_duration);

        // Execute all in parallel (Scala: parJoinUnbounded.compile.drain)
        // let (_validation_result, _history_result, _data_result) =
        //     tokio::try_join!(validation_task, history_task, data_task)
        //         .map_err(|e| CasperError::StreamError(format!("Task join error: {}", e)))?;

        let total_duration = total_validation_start.elapsed();
        log::info!(
            "Chunk validation and import completed in {:?}",
            total_duration
        );

        Ok(())
    }

    /// Read current state for next state chunks and send the requests
    ///
    /// **Scala equivalent**: The `requestNext(resend: Boolean)` nested function inside createStream
    /// which calls `broadcastStreams(ids).parJoinUnbounded.compile.drain`
    async fn request_next(&self, resend: bool) -> Result<(), CasperError> {
        // Check if stream is finished (no more requests)
        let (is_end, active_count) = {
            let state = self.st.lock().map_err(|_| {
                CasperError::StreamError(
                    "Failed to acquire state lock for request_next".to_string(),
                )
            })?;
            (state.is_finished(), state.d.len())
        };

        if resend {
            log::info!(
                "Processing resend request - active: {}, is_finished: {}",
                active_count,
                is_end
            );
        } else {
            log::debug!(
                "Processing new request - active: {}, is_finished: {}",
                active_count,
                is_end
            );
        }

        // Take next set of paths to request
        let paths = {
            let mut state = self.st.lock().map_err(|_| {
                CasperError::StreamError("Failed to acquire state lock for get_next".to_string())
            })?;
            let (new_state, next_paths) = state.get_next(resend);
            *state = new_state;
            next_paths
        };

        if paths.is_empty() {
            log::debug!("No new state paths to request (resend: {})", resend);
            return Ok(());
        }

        log::info!(
            "Requesting {} state paths (resend: {})",
            paths.len(),
            resend
        );

        // Send all requests in parallel (Scala: broadcastStreams(ids).parJoinUnbounded)
        if !is_end {
            let request_futures: Vec<_> = paths
                .into_iter()
                .map(|path| async move {
                    self.request_ops
                        .request_for_store_item(&path, PAGE_SIZE)
                        .await
                })
                .collect();

            // Execute all requests in parallel, short-circuit on first error
            futures::future::try_join_all(request_futures).await?;

            log::debug!("Completed broadcasting state chunk requests");
        }

        Ok(())
    }

    /// Mark chunk as received if it was requested
    ///
    /// **Scala equivalent**: `st.modify(_.received(startPath))` call in responseStream
    async fn mark_chunk_received(&self, start_path: StatePartPath) -> Result<bool, CasperError> {
        let mut state = self.st.lock().map_err(|_| {
            CasperError::StreamError(
                "Failed to acquire state lock for mark_chunk_received".to_string(),
            )
        })?;

        let (new_state, is_requested) = state.received(start_path.clone());
        *state = new_state;

        log::debug!(
            "Marked chunk as received for path (length: {}), was_requested: {}",
            start_path.len(),
            is_requested
        );

        Ok(is_requested)
    }

    /// Mark chunk as done after successful import
    ///
    /// **Scala equivalent**: `st.update(_.done(startPath))` call in responseStream
    async fn mark_chunk_done(&self, start_path: StatePartPath) -> Result<(), CasperError> {
        let mut state = self.st.lock().map_err(|_| {
            CasperError::StreamError("Failed to acquire state lock for mark_chunk_done".to_string())
        })?;

        let new_state = state.done(start_path.clone());
        *state = new_state;

        log::debug!(
            "Marked chunk as done for path (length: {})",
            start_path.len()
        );

        Ok(())
    }

    /// Add new state paths for requesting (from lastPath in messages)
    ///
    /// **Scala equivalent**: `st.update(_.add(Set(lastPath)))` call in responseStream
    async fn add_next_paths(
        &self,
        paths: std::collections::HashSet<StatePartPath>,
    ) -> Result<(), CasperError> {
        if paths.is_empty() {
            return Ok(());
        }

        let mut state = self.st.lock().map_err(|_| {
            CasperError::StreamError("Failed to acquire state lock for add_next_paths".to_string())
        })?;

        let new_state = state.add(paths.clone());
        *state = new_state;

        log::debug!("Added {} new state paths for processing", paths.len());

        Ok(())
    }
}

/// Create a stream to receive tuple space needed for Last Finalized State.
///
/// **Scala equivalent**: The main `stream` method that creates the fs2.Stream[F, ST[StatePartPath]]
///
/// The Scala implementation structure:
/// 1. Extract stateHash from approvedBlock
/// 2. Create initial startRequest  
/// 3. Set root in stateImporter
/// 4. Create st: Ref[F, ST[StatePartPath]]
/// 5. Create requestQueue and responseHashQueue
/// 6. Return createStream(st, requestQueue, responseHashQueue) which has:
///    - requestStream (pulls and processes requests)
///    - responseStream (handles incoming messages)
///    - timeout handling with resendRequests
///    - final result with concurrency and termination
///
/// # Arguments
/// * `approved_block` - Last finalized block (Scala: approvedBlock: ApprovedBlock)
/// * `tuple_space_message_receiver` - Handler of tuple space messages (Scala: tupleSpaceMessageQueue)
/// * `request_timeout` - Time after request will be resent if not received (Scala: requestTimeout)
/// * `request_ops` - Network operations for requesting state chunks (Scala: requestForStoreItem function)
/// * `state_importer` - RSpace importer (Scala: stateImporter: RSpaceImporter[F])
///
/// # Returns
/// fs2.Stream processing all tuple space state (Scala: F[Stream[F, ST[StatePartPath]]])
pub async fn stream<T: TupleSpaceRequesterOps, R: RSpaceImporter + Clone + 'static>(
    approved_block: &ApprovedBlock,
    mut tuple_space_message_receiver: mpsc::UnboundedReceiver<StoreItemsMessage>,
    request_timeout: Duration,
    request_ops: T,
    state_importer: R,
) -> Result<impl Stream<Item = ST<StatePartPath>>, CasperError> {
    use crate::rust::util::proto_util;

    // 1. Extract state hash (Scala: Blake2b256Hash.fromByteString(ProtoUtil.postStateHash(approvedBlock.candidate.block)))
    let state_hash_bytes = proto_util::post_state_hash(&approved_block.candidate.block);
    let state_hash = Blake2b256Hash::from_bytes(state_hash_bytes.to_vec());

    // 2. Create start request (Scala: val startRequest: StatePartPath = Seq((stateHash, None)))
    let start_request: StatePartPath = vec![(state_hash.clone(), None)];

    // 3. Set root (Scala: stateImporter.setRoot(stateHash))
    state_importer.set_root(&state_hash);

    // 4. Create ST state (Scala: st <- Ref.of[F, ST[StatePartPath]](ST(Seq(startRequest))))
    let st = Arc::new(Mutex::new(ST::new(vec![start_request.clone()])));

    // 5. Create request queue (Scala: requestQueue <- Queue.bounded[F, Boolean](maxSize = 2))
    let (request_tx, mut request_rx) = mpsc::channel::<bool>(2);

    // 6. Enqueue initial request (Scala: requestQueue.enqueue1(false))
    request_tx
        .send(false)
        .await
        .map_err(|_| CasperError::StreamError("Failed to send initial request".to_string()))?;

    // Create the stream processor
    let mut processor =
        TupleSpaceStreamProcessor::new(request_ops, state_importer, st.clone(), request_tx.clone());

    log::info!("LFS Tuple Space Requester stream initialized - starting processing");

    // 7. Create and return the main stream (Scala: createStream equivalent)
    let stream = async_stream::stream! {
        // Timeout message (Scala: timeoutMsg)
        let timeout_msg = format!("No tuple space state responses for {:?}. Resending requests.", request_timeout);

        // Create timeout for resending requests (Scala: resendRequests)
        let mut idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

        // Main stream processing loop (Scala: requestStream.evalMap(_ => st.get).onIdle(...).terminateAfter(...) concurrently responseStream)
        loop {
            tokio::select! {
                // Request stream processing (Scala: requestStream.evalMap(_ => st.get))
                Some(resend_flag) = request_rx.recv() => {
                    match processor.request_next(resend_flag).await {
                        Ok(()) => {
                            log::debug!("Request processing completed (resend: {})", resend_flag);
                        }
                        Err(e) => {
                            log::error!("Failed to process request: {:?}", e);
                            // Continue processing other arms instead of breaking
                            continue;
                        }
                    }

                    // Get current state and emit to stream (Scala: .evalMap(_ => st.get))
                    let current_state = {
                        match st.lock() {
                            Ok(state) => state.clone(),
                            Err(e) => {
                                log::error!("Failed to acquire state lock: {:?}", e);
                                continue;
                            }
                        }
                    };

                    // Check termination condition (Scala: .terminateAfter(_.isFinished))
                    if current_state.is_finished() {
                        log::info!("Tuple space processing completed - all state downloaded");
                        yield current_state;
                        break;
                    }

                    // Reset idle timeout due to activity
                    idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

                    // Emit state to stream
                    yield current_state;
                }

                // Response stream processing (Scala: responseStream)
                Some(message) = tuple_space_message_receiver.recv() => {
                    match processor.process_store_items_message(message).await {
                        Ok(()) => {
                            log::debug!("Store items message processed successfully");
                        }
                        Err(e) => {
                            log::error!("Failed to process store items message: {:?}", e);
                            // On validation or processing error, terminate the stream
                            // Scala equivalent: Stream fails with validation error and terminates
                            log::error!("Stream terminating due to store items processing error");
                            break;
                        }
                    }

                    // Get current state after processing response
                    let current_state = {
                        match st.lock() {
                            Ok(state) => state.clone(),
                            Err(e) => {
                                log::error!("Failed to acquire state lock for response processing: {:?}", e);
                                continue;
                            }
                        }
                    };

                    // Check termination condition
                    if current_state.is_finished() {
                        log::info!("Response processing completed - all state downloaded");
                        yield current_state;
                        break;
                    }

                    // Reset idle timeout due to activity
                    idle_timeout = Box::pin(tokio::time::sleep(request_timeout));

                    // Emit state to stream
                    yield current_state;
                }

                // Timeout handling (Scala: .onIdle(requestTimeout, resendRequests))
                _ = &mut idle_timeout => {
                    log::warn!("{}", timeout_msg);

                    // Trigger resend request (Scala: resendRequests = requestQueue.enqueue1(true))
                    match request_tx.send(true).await {
                        Ok(()) => {
                            log::debug!("Timeout triggered - resend request enqueued successfully");
                            // Reset the timeout for next idle period
                            idle_timeout = Box::pin(tokio::time::sleep(request_timeout));
                        }
                        Err(e) => {
                            log::error!("Failed to enqueue resend request - channel error: {:?}", e);
                            log::warn!("Request queue channel appears closed, checking if stream should terminate");

                            // Check if we should terminate gracefully
                            let should_terminate = {
                                match st.lock() {
                                    Ok(state) => state.is_finished(),
                                    Err(_) => {
                                        log::error!("Cannot acquire state lock to check termination condition");
                                        true // Assume termination if we can't check state
                                    }
                                }
                            };

                            if should_terminate {
                                log::info!("Stream terminating gracefully - processing appears complete");
                                break;
                            }
                            // Reset timeout even on error to continue monitoring
                            idle_timeout = Box::pin(tokio::time::sleep(request_timeout));
                        }
                    }
                }
            }
        }

        // Final state emission (similar to block requester pattern)
        let final_state = {
            match st.lock() {
                Ok(state) => {
                    let final_stats = (
                        state.d.len(),
                        state.is_finished()
                    );
                    log::info!("LFS Tuple Space Requester stream completed - Final state: active: {}, is_finished: {}",
                        final_stats.0, final_stats.1);

                    if final_stats.1 {
                        log::info!("✅ LFS Tuple Space Requester completed successfully - all required state downloaded");
                    } else {
                        log::warn!("⚠️ LFS Tuple Space Requester terminated with incomplete state - some state may be missing");
                    }

                    Some(state.clone())
                }
                Err(e) => {
                    log::error!("Failed to acquire final state lock: {:?}", e);
                    None
                }
            }
        };

        // Emit final state if available
        if let Some(final_state) = final_state {
            yield final_state;
        }

        log::info!("LFS Tuple Space Requester stream processing completed");
    };

    Ok(stream)
}
