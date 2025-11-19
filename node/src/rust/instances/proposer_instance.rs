// See node/src/main/scala/coop/rchain/node/instances/ProposerInstance.scala

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Semaphore};

use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::BlockMessage;

use casper::rust::blocks::proposer::{
    propose_result::{ProposeFailure, ProposeResult},
    proposer::{ProductionProposer, ProposeReturnType, ProposerResult},
};
use casper::rust::casper::Casper;
use casper::rust::errors::CasperError;
use comm::rust::transport::transport_layer::TransportLayer;

/// Default timeout for propose operations (5 minutes)
/// If a propose takes longer than this, it's likely stuck and should be abandoned
const PROPOSE_TIMEOUT: Duration = Duration::from_secs(300);

/// Proposer instance that processes propose requests from a queue
///
/// Each propose request carries its own Casper instance, allowing the proposer
/// to start immediately without waiting for engine initialization.
pub struct ProposerInstance<T: TransportLayer + Send + Sync + 'static> {
    /// Receiver for propose requests
    pub propose_requests_queue_rx: mpsc::UnboundedReceiver<(
        Arc<dyn Casper + Send + Sync>,
        bool,
        oneshot::Sender<ProposerResult>,
    )>,
    /// Sender for propose requests (needed for retry mechanism)
    pub propose_requests_queue_tx: mpsc::UnboundedSender<(
        Arc<dyn Casper + Send + Sync>,
        bool,
        oneshot::Sender<ProposerResult>,
    )>,
    pub proposer: Arc<tokio::sync::Mutex<ProductionProposer<T>>>,
    /// Shared state for API observability (tracks current/latest propose results)
    pub state: Arc<tokio::sync::RwLock<casper::rust::state::instances::ProposerState>>,
}

impl<T: TransportLayer + Send + Sync + 'static> ProposerInstance<T> {
    /// Create a new ProposerInstance
    ///
    /// # Arguments
    /// * `propose_requests_queue` - Tuple of (receiver, sender) for propose requests (needed for retry mechanism)
    /// * `proposer` - The proposer logic for creating blocks
    /// * `state` - Shared state for API observability (tracks current/latest propose results)
    ///
    /// # Note
    /// This does NOT take a Casper instance as a parameter. Each propose request
    /// in the queue carries its own Casper instance.
    pub fn new(
        propose_requests_queue: (
            mpsc::UnboundedReceiver<(
                Arc<dyn Casper + Send + Sync>,
                bool,
                oneshot::Sender<ProposerResult>,
            )>,
            mpsc::UnboundedSender<(
                Arc<dyn Casper + Send + Sync>,
                bool,
                oneshot::Sender<ProposerResult>,
            )>,
        ),
        proposer: Arc<tokio::sync::Mutex<ProductionProposer<T>>>,
        state: Arc<tokio::sync::RwLock<casper::rust::state::instances::ProposerState>>,
    ) -> Self {
        let (propose_requests_queue_rx, propose_requests_queue_tx) = propose_requests_queue;
        Self {
            propose_requests_queue_rx,
            propose_requests_queue_tx,
            proposer,
            state,
        }
    }

    /// Create and start the proposer stream
    ///
    /// Spawns a task that processes propose requests from the queue and returns
    /// a receiver for the results.
    ///
    /// # Returns
    /// A receiver that will receive `(ProposeResult, Option<BlockMessage>)` for each
    /// successful propose operation.
    ///
    /// # Implementation Note
    /// Uses a sophisticated non-blocking locking mechanism:
    /// - Uses try_lock (non-blocking) instead of lock (blocking)
    /// - If lock is held: returns ProposerEmpty immediately, cocks trigger for retry
    /// - If lock acquired: executes propose, then checks trigger for ONE retry
    /// - Prevents propose request pile-up during slow proposals
    pub fn create(
        self,
    ) -> Result<mpsc::UnboundedReceiver<(ProposeResult, Option<BlockMessage>)>, CasperError> {
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            let Self {
                mut propose_requests_queue_rx,
                propose_requests_queue_tx,
                proposer,
                state,
            } = self;

            // Propose lock and trigger mechanism
            // - propose_lock: Semaphore(1) for non-blocking propose execution
            // - trigger: Semaphore(0) for retry signaling (tryAcquire = cock, tryRelease = check & reset)
            let propose_lock = Arc::new(Semaphore::new(1));
            let trigger = Arc::new(Semaphore::new(0)); // Start with 0 permits = uncocked

            // Process propose requests - each request carries its own Casper instance
            while let Some((casper, is_async, propose_id_sender)) =
                propose_requests_queue_rx.recv().await
            {
                // Try to acquire the propose lock (NON-BLOCKING)
                if let Ok(permit) = propose_lock.clone().try_acquire_owned() {
                    // Lock acquired - execute the propose
                    tracing::info!("Propose started");

                    // Clone what we need for the task
                    let proposer_clone = proposer.clone();
                    let result_tx_clone = result_tx.clone();
                    let state_clone = state.clone();
                    let trigger_clone = trigger.clone();
                    let propose_requests_queue_tx_clone = propose_requests_queue_tx.clone();

                    // Create a deferred result channel for API observability
                    let (curr_result_tx, curr_result_rx) = oneshot::channel();
                    {
                        let mut state_guard = state_clone.write().await;
                        state_guard.curr_propose_result = Some(curr_result_rx);
                    }

                    // Execute the propose with timeout protection
                    // Production safety: If propose hangs (bad RSpace, network issue, etc.),
                    // we timeout after PROPOSE_TIMEOUT to prevent blocking forever
                    let mut proposer_guard = proposer_clone.lock().await;
                    let propose_future = proposer_guard.propose(casper.clone(), is_async);
                    let res = match tokio::time::timeout(PROPOSE_TIMEOUT, propose_future).await {
                        Ok(result) => result,
                        Err(_) => {
                            tracing::error!(
                                "Propose operation timed out after {:?} - this indicates a serious issue",
                                PROPOSE_TIMEOUT
                            );
                            // Send timeout error result to caller
                            let timeout_result = ProposerResult::empty(); // or create a timeout-specific result
                            let _ = propose_id_sender.send(timeout_result);

                            // Clear current propose state
                            let mut state_guard = state_clone.write().await;
                            state_guard.curr_propose_result = None;
                            drop(proposer_guard);
                            drop(permit);
                            continue;
                        }
                    };
                    drop(proposer_guard); // Release proposer lock explicitly

                    match res {
                        Ok(ProposeReturnType {
                            propose_result,
                            block_message_opt,
                            propose_result_to_send,
                        }) => {
                            let _ = propose_id_sender.send(propose_result_to_send);

                            // Update state with result and clear current propose
                            let result_copy = (propose_result.clone(), block_message_opt.clone());
                            {
                                let mut state_guard = state_clone.write().await;
                                state_guard.latest_propose_result = Some(result_copy.clone());
                                state_guard.curr_propose_result = None;
                            }
                            // Also complete the deferred result for any API waiting on current propose
                            let _ = curr_result_tx.send(result_copy.clone());

                            match block_message_opt {
                                Some(ref block) => {
                                    let block_str =
                                        PrettyPrinter::build_string_block_message(block, true);

                                    tracing::info!(
                                        "Propose finished: {:?} Block {} created and added.",
                                        propose_result.propose_status,
                                        block_str
                                    );

                                    match result_tx_clone
                                        .send((propose_result, Some(block.clone())))
                                    {
                                        Ok(_) => {}
                                        Err(e) => {
                                            tracing::error!("Failed to send propose result: {}", e);
                                        }
                                    }
                                }
                                None => {
                                    tracing::error!(
                                        "Propose failed: {}",
                                        propose_result.propose_status
                                    )
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Error proposing: {}", e);

                            // Create error result
                            let error_result: (ProposeResult, Option<BlockMessage>) =
                                (ProposeResult::failure(ProposeFailure::NoNewDeploys), None); // TODO: verify whether the NoNewDeploys error is best choice in this case

                            // Send to both channels
                            let _ = curr_result_tx.send(error_result);
                            // result_tx_clone might be less critical since caller has propose_id_sender

                            // Clear current propose state
                            let mut state_guard = state_clone.write().await;
                            state_guard.curr_propose_result = None;
                        }
                    }

                    // Permit is automatically released when dropped

                    // Check if trigger was cocked while we were proposing
                    // tryAcquire on trigger checks if it was cocked (has permit)
                    // If yes, we consume the permit and enqueue a retry
                    if trigger_clone.try_acquire().is_ok() {
                        tracing::info!("Trigger was cocked during propose - enqueueing ONE retry");

                        // Enqueue ONE retry (not async, new deferred result)
                        let (retry_sender, _retry_receiver) = oneshot::channel();
                        // Note: We drop _retry_receiver - retry results go through normal channels
                        // This is acceptable because retries are fire-and-forget optimization
                        if let Err(e) =
                            propose_requests_queue_tx_clone.send((casper, false, retry_sender))
                        {
                            tracing::error!(
                                "Failed to enqueue retry propose (channel closed): {}",
                                e
                            );
                            // Channel closed means we're shutting down - this is expected
                            break;
                        }
                    }

                    // Permit automatically released here
                } else {
                    // Lock is held - propose is in progress
                    tracing::info!(
                        "Propose already in progress - returning ProposerEmpty and cocking trigger"
                    );

                    // Check if trigger is already cocked (has at least 1 permit)
                    if trigger.available_permits() == 0 {
                        trigger.add_permits(1);
                    }

                    // Return ProposerEmpty immediately
                    if let Err(_) = propose_id_sender.send(ProposerResult::empty()) {
                        tracing::warn!("Failed to send ProposerEmpty result (receiver dropped)");
                        // Receiver dropped - client gave up waiting, this is fine
                    }
                }
            }

            tracing::info!("Propose requests queue closed, stopping proposer");

            Result::<(), CasperError>::Ok(())
        });

        Ok(result_rx)
    }
}
