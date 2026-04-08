// See node/src/main/scala/coop/rchain/node/instances/ProposerInstance.scala

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Semaphore};

use models::rust::casper::pretty_printer::PrettyPrinter;
use models::rust::casper::protocol::casper_message::BlockMessage;

use casper::rust::blocks::proposer::{
    propose_result::{ProposeFailure, ProposeResult, ProposeStatus},
    proposer::{ProductionProposer, ProposeReturnType, ProposerResult},
};
use casper::rust::casper::Casper;
use casper::rust::errors::CasperError;
use casper::rust::metrics_constants::{
    PROPOSER_QUEUE_PENDING_METRIC, PROPOSER_QUEUE_REJECTED_TOTAL_METRIC, VALIDATOR_METRICS_SOURCE,
};
use comm::rust::transport::transport_layer::TransportLayer;

const PROPOSER_RESULT_QUEUE_CAPACITY: usize = 64;
const PROPOSER_MAX_IMMEDIATE_RETRIES: u8 = 2;
const PROPOSER_MIN_INTERVAL: Duration = Duration::from_millis(250);

type ProposeQueueEntry = (
    Arc<dyn Casper + Send + Sync>,
    bool,
    oneshot::Sender<ProposerResult>,
    u8,
);

fn should_retry_immediately_on_trigger(result: &ProposeResult, is_async: bool) -> bool {
    let _ = (result, is_async);
    false
}

fn proposer_min_interval() -> Duration {
    PROPOSER_MIN_INTERVAL
}

/// Proposer instance that processes propose requests from a queue
///
/// Each propose request carries its own Casper instance, allowing the proposer
/// to start immediately without waiting for engine initialization.
pub struct ProposerInstance<T: TransportLayer + Send + Sync + 'static> {
    /// Receiver for propose requests
    pub propose_requests_queue_rx: mpsc::Receiver<ProposeQueueEntry>,
    /// Sender for propose requests (needed for retry mechanism)
    pub propose_requests_queue_tx: mpsc::Sender<ProposeQueueEntry>,
    pub proposer: Arc<tokio::sync::Mutex<ProductionProposer<T>>>,
    /// Shared state for API observability (tracks current/latest propose results)
    pub state: Arc<tokio::sync::RwLock<casper::rust::state::instances::ProposerState>>,
    pub propose_queue_pending: Arc<AtomicUsize>,
    pub propose_queue_max_pending: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use casper::rust::blocks::proposer::propose_result::CheckProposeConstraintsFailure;

    #[test]
    fn should_not_retry_internal_deploy_error_immediately() {
        let result = ProposeResult::failure(ProposeFailure::InternalDeployError);
        assert!(
            !should_retry_immediately_on_trigger(&result, true),
            "InternalDeployError should not trigger immediate retry"
        );
    }

    #[test]
    fn should_not_retry_not_enough_new_blocks_for_async_propose() {
        let result = ProposeResult::failure(ProposeFailure::CheckConstraintsFailure(
            CheckProposeConstraintsFailure::NotEnoughNewBlocks,
        ));
        assert!(
            !should_retry_immediately_on_trigger(&result, true),
            "NotEnoughNewBlocks should not trigger immediate async retry"
        );
    }
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
            mpsc::Receiver<ProposeQueueEntry>,
            mpsc::Sender<ProposeQueueEntry>,
        ),
        proposer: Arc<tokio::sync::Mutex<ProductionProposer<T>>>,
        state: Arc<tokio::sync::RwLock<casper::rust::state::instances::ProposerState>>,
        propose_queue_pending: Arc<AtomicUsize>,
        propose_queue_max_pending: usize,
    ) -> Self {
        let (propose_requests_queue_rx, propose_requests_queue_tx) = propose_requests_queue;
        Self {
            propose_requests_queue_rx,
            propose_requests_queue_tx,
            proposer,
            state,
            propose_queue_pending,
            propose_queue_max_pending,
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
    ) -> Result<mpsc::Receiver<(ProposeResult, Option<BlockMessage>)>, CasperError> {
        let (result_tx, result_rx) = mpsc::channel(PROPOSER_RESULT_QUEUE_CAPACITY);

        tokio::spawn(async move {
            let Self {
                mut propose_requests_queue_rx,
                propose_requests_queue_tx,
                proposer,
                state,
                propose_queue_pending,
                propose_queue_max_pending,
            } = self;

            // Propose lock and trigger mechanism
            // - propose_lock: Semaphore(1) for non-blocking propose execution
            // - trigger: Semaphore(0) for retry signaling (tryAcquire = cock, tryRelease = check & reset)
            let propose_lock = Arc::new(Semaphore::new(1));
            let trigger = Arc::new(Semaphore::new(0)); // Start with 0 permits = uncocked
            let mut last_propose_started_at: Option<Instant> = None;

            // Process propose requests - each request carries its own Casper instance
            while let Some((casper, is_async, propose_id_sender, immediate_retry_count)) =
                propose_requests_queue_rx.recv().await
            {
                let _ = propose_queue_pending.fetch_update(
                    Ordering::AcqRel,
                    Ordering::Acquire,
                    |curr| Some(curr.saturating_sub(1)),
                );
                metrics::gauge!(
                    PROPOSER_QUEUE_PENDING_METRIC,
                    "source" => VALIDATOR_METRICS_SOURCE
                )
                .set(propose_queue_pending.load(Ordering::Relaxed) as f64);

                let min_interval = proposer_min_interval();
                // Allow one immediate follow-up after a trigger collision so deploy-driven
                // proposals are not penalized by the steady-state spacing gate.
                let effective_min_interval = if immediate_retry_count == 1 {
                    Duration::ZERO
                } else {
                    min_interval
                };
                if !effective_min_interval.is_zero() {
                    if let Some(last_started) = last_propose_started_at {
                        let elapsed = last_started.elapsed();
                        if elapsed < effective_min_interval {
                            tokio::time::sleep(effective_min_interval - elapsed).await;
                        }
                    }
                }

                // Try to acquire the propose lock (NON-BLOCKING)
                if let Ok(_permit) = propose_lock.clone().try_acquire_owned() {
                    last_propose_started_at = Some(Instant::now());
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

                    let mut proposer_guard = proposer_clone.lock().await;
                    let validator_public_key = proposer_guard.validator.public_key.bytes.clone();
                    // Await propose directly to avoid canceling in-flight block creation/replay
                    // through timeout-driven future drops.
                    let res = proposer_guard.propose(casper.clone(), is_async).await;
                    drop(proposer_guard); // Release proposer lock explicitly

                    let should_retry_on_result = match res {
                        Ok(ProposeReturnType {
                            propose_result,
                            block_message_opt,
                            propose_result_to_send,
                        }) => {
                            let _ = propose_id_sender.send(propose_result_to_send);

                            // Update state with result and clear current propose
                            let result_copy = (propose_result.clone(), block_message_opt.clone());
                            let should_retry_on_trigger =
                                should_retry_immediately_on_trigger(&propose_result, is_async);
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
                                        .await
                                    {
                                        Ok(_) => {}
                                        Err(e) => {
                                            tracing::error!("Failed to send propose result: {}", e);
                                        }
                                    }
                                }
                                None => {
                                    if propose_result.is_no_new_deploys() {
                                        tracing::info!(
                                            "Propose: {}",
                                            propose_result.propose_status
                                        )
                                    } else {
                                        tracing::error!(
                                            "Propose failed: {}",
                                            propose_result.propose_status
                                        )
                                    }
                                }
                            }

                            should_retry_on_trigger
                        }
                        Err(e) => {
                            tracing::error!("Error proposing: {}", e);

                            let failure_seq_number = match casper.get_snapshot().await {
                                Ok(snapshot) => snapshot
                                    .max_seq_nums
                                    .get(&validator_public_key)
                                    .map(|seq| *seq + 1)
                                    .unwrap_or(1)
                                    as i32,
                                Err(err) => {
                                    tracing::warn!(
                                        "Failed to get Casper snapshot for failure seq number: {}",
                                        err
                                    );
                                    -1
                                }
                            };

                            // Always resolve requester oneshot with a failure result.
                            // Dropping this sender causes "channel closed" at caller and
                            // unnecessarily breaks heartbeat liveness flow.
                            let _ = propose_id_sender.send(ProposerResult::failure(
                                ProposeStatus::Failure(ProposeFailure::BugError),
                                failure_seq_number,
                            ));

                            // Runtime propose errors are internal failures and should not be
                            // reported as NoNewDeploys / InternalDeployError.
                            let error_result: (ProposeResult, Option<BlockMessage>) =
                                (ProposeResult::failure(ProposeFailure::BugError), None);

                            // Send to both channels
                            let _ = curr_result_tx.send(error_result);
                            // result_tx_clone might be less critical since caller has propose_id_sender

                            // Clear current propose state
                            let mut state_guard = state_clone.write().await;
                            state_guard.curr_propose_result = None;
                            false
                        }
                    };

                    // Permit is automatically released when dropped

                    // Drain any left-over trigger and retry once after recoverable
                    // async failures so deploy-driven proposals are retried immediately.
                    let trigger_cocked = trigger_clone.try_acquire().is_ok();
                    let should_retry_on_trigger = should_retry_on_result;
                    let should_retry = should_retry_on_trigger || trigger_cocked;
                    let retry_budget_exhausted = should_retry_on_trigger
                        && immediate_retry_count >= PROPOSER_MAX_IMMEDIATE_RETRIES;
                    let should_enqueue_retry = should_retry && !retry_budget_exhausted;

                    if should_retry {
                        tracing::info!(
                                "Enqueueing retry after propose (result_retry:{}, has_retry_budget:{}, had_trigger:{})",
                                should_retry_on_trigger,
                                should_enqueue_retry,
                                trigger_cocked
                            );

                        if should_enqueue_retry {
                            // Enqueue retry request with bounded retry budget.
                            let (retry_sender, _retry_receiver) = oneshot::channel();
                            // Note: We drop _retry_receiver - retry results go through normal channels
                            // This is acceptable because retries are fire-and-forget optimization
                            let retry_reserved = propose_queue_pending
                                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |curr| {
                                    (curr < propose_queue_max_pending).then_some(curr + 1)
                                })
                                .is_ok();
                            if retry_reserved {
                                metrics::gauge!(
                                    PROPOSER_QUEUE_PENDING_METRIC,
                                    "source" => VALIDATOR_METRICS_SOURCE
                                )
                                .set(propose_queue_pending.load(Ordering::Relaxed) as f64);

                                if let Err(e) = propose_requests_queue_tx_clone
                                    .send((
                                        casper,
                                        is_async,
                                        retry_sender,
                                        immediate_retry_count.saturating_add(1),
                                    ))
                                    .await
                                {
                                    let _ = propose_queue_pending.fetch_update(
                                        Ordering::AcqRel,
                                        Ordering::Acquire,
                                        |curr| Some(curr.saturating_sub(1)),
                                    );
                                    metrics::gauge!(
                                        PROPOSER_QUEUE_PENDING_METRIC,
                                        "source" => VALIDATOR_METRICS_SOURCE
                                    )
                                    .set(propose_queue_pending.load(Ordering::Relaxed) as f64);
                                    tracing::error!(
                                        "Failed to enqueue retry propose (channel closed): {}",
                                        e
                                    );
                                    // Channel closed means we're shutting down - this is expected
                                    break;
                                }
                            } else {
                                metrics::counter!(
                                    PROPOSER_QUEUE_REJECTED_TOTAL_METRIC,
                                    "source" => VALIDATOR_METRICS_SOURCE
                                )
                                .increment(1);
                            }
                        } else {
                            metrics::counter!(
                                PROPOSER_QUEUE_REJECTED_TOTAL_METRIC,
                                "source" => VALIDATOR_METRICS_SOURCE
                            )
                            .increment(1);
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
