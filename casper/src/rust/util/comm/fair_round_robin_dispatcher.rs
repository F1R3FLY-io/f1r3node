// See casper/src/main/scala/coop/rchain/casper/util/comm/FairRoundRobinDispatcher.scala

use std::collections::{HashMap, VecDeque};
use std::fmt::Display;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

use crate::rust::errors::CasperError;

/// Dispatch decision returned by the filter function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// Process message through fair round-robin queue
    Handle,
    /// Bypass queue and handle immediately
    Pass,
    /// Drop message silently
    Drop,
}

/// Internal mutable state for the dispatcher.
struct DispatcherState<S, M> {
    /// Round-robin queue of sources
    queue: VecDeque<S>,
    /// Per-source message queues
    messages: HashMap<S, VecDeque<M>>,
    /// Retry counts per source
    retries: HashMap<S, usize>,
    /// Global skip counter
    skipped: usize,
}

impl<S, M> DispatcherState<S, M> {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            messages: HashMap::new(),
            retries: HashMap::new(),
            skipped: 0,
        }
    }
}

/// Immutable configuration for the dispatcher.
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    pub max_source_queue_size: usize,
    pub give_up_after_skipped: usize,
    pub drop_source_after_retries: usize,
}

impl DispatcherConfig {
    pub fn new(
        max_source_queue_size: usize,
        give_up_after_skipped: usize,
        drop_source_after_retries: usize,
    ) -> Self {
        assert!(max_source_queue_size > 0);

        Self {
            max_source_queue_size,
            give_up_after_skipped,
            drop_source_after_retries,
        }
    }
}

/// Fair round-robin dispatcher for processing messages from multiple sources.
pub struct FairRoundRobinDispatcher<S, M>
where
    S: Eq + Hash + Clone + Display + Send + Sync + 'static,
    M: Clone + PartialEq + Display + Send + Sync + 'static,
{
    /// Shared mutable state
    state: Arc<Mutex<DispatcherState<S, M>>>,
    /// Immutable configuration
    config: DispatcherConfig,
    /// Filter callback to determine dispatch action
    filter: Arc<dyn Fn(&M) -> Pin<Box<dyn Future<Output = Dispatch> + Send>> + Send + Sync>,
    /// Handle callback to process messages
    handle: Arc<dyn Fn(S, M) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
    /// Concurrency control lock
    lock: Arc<Semaphore>,
}

impl<S, M> FairRoundRobinDispatcher<S, M>
where
    S: Eq + Hash + Clone + Display + Send + Sync + 'static,
    M: Clone + PartialEq + Display + Send + Sync + 'static,
{
    /// Create a new FairRoundRobinDispatcher.
    pub fn new<FilterFn, HandleFn, FilterFut, HandleFut>(
        filter: FilterFn,
        handle: HandleFn,
        config: DispatcherConfig,
        lock: Arc<Semaphore>,
    ) -> Self
    where
        FilterFn: Fn(&M) -> FilterFut + Send + Sync + 'static,
        FilterFut: Future<Output = Dispatch> + Send + 'static,
        HandleFn: Fn(S, M) -> HandleFut + Send + Sync + 'static,
        HandleFut: Future<Output = ()> + Send + 'static,
    {
        let filter = Arc::new(
            move |m: &M| -> Pin<Box<dyn Future<Output = Dispatch> + Send>> { Box::pin(filter(m)) },
        );

        let handle = Arc::new(
            move |s: S, m: M| -> Pin<Box<dyn Future<Output = ()> + Send>> {
                Box::pin(handle(s, m))
            },
        );

        Self {
            state: Arc::new(Mutex::new(DispatcherState::new())),
            config,
            filter,
            handle,
            lock,
        }
    }

    /// Dispatch a message from a source.
    pub async fn dispatch(&self, source: S, message: M) -> Result<(), CasperError> {
        let dispatch_action = (self.filter)(&message).await;

        match dispatch_action {
            Dispatch::Handle => {
                let _permit = self.lock.acquire().await.map_err(|e| {
                    CasperError::LockError(format!("Failed to acquire semaphore: {}", e))
                })?;

                self.ensure_source_exists(&source).await?;

                if self.is_duplicate(&source, &message).await? {
                    tracing::info!("Dropped duplicate message {} from {}", message, source);
                    return Ok(());
                }

                self.enqueue_message(source.clone(), message).await?;
                self.handle_messages().await?;

                Ok(())
            }
            Dispatch::Pass => {
                (self.handle)(source, message).await;
                Ok(())
            }
            Dispatch::Drop => Ok(()),
        }
    }

    pub async fn ensure_source_exists(&self, source: &S) -> Result<(), CasperError> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

        if !state.messages.contains_key(source) {
            state.queue.push_front(source.clone());
            state.messages.insert(source.clone(), VecDeque::new());
            state.retries.insert(source.clone(), 0);

            tracing::info!("Added {} to the dispatch queue", source);
        }

        Ok(())
    }

    pub async fn is_duplicate(&self, source: &S, message: &M) -> Result<bool, CasperError> {
        let state = self
            .state
            .lock()
            .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

        Ok(state
            .messages
            .get(source)
            .map(|queue| queue.iter().any(|m| m == message))
            .unwrap_or(false))
    }

    pub async fn enqueue_message(&self, source: S, message: M) -> Result<(), CasperError> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

        let queue = state
            .messages
            .get_mut(&source)
            .ok_or_else(|| CasperError::RuntimeError(format!("Source {} not found", source)))?;

        if queue.len() < self.config.max_source_queue_size {
            queue.push_back(message.clone());
            tracing::info!(
                "Enqueued message {} from {} (queue length: {})",
                message,
                source,
                queue.len()
            );

            state.retries.insert(source, 0);
        } else {
            tracing::info!("Dropped message {} from {}", message, source);
        }

        Ok(())
    }

    async fn handle_messages(&self) -> Result<(), CasperError> {
        loop {
            let should_continue = self.handle_next_message_with_giveup().await?;
            if !should_continue {
                break;
            }
        }
        Ok(())
    }

    async fn handle_next_message_with_giveup(&self) -> Result<bool, CasperError> {
        let source = {
            let state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            match state.queue.front() {
                Some(s) => s.clone(),
                None => return Ok(false), // Stop processing if queue is empty
            }
        };

        let had_message = self.handle_message(&source).await?;

        if had_message {
            self.success(&source).await?;
            // After success, process subsequent messages without give-up logic
            self.handle_messages_without_giveup().await?;
            Ok(false) // Stop the outer loop
        } else {
            let gave_up = self.failure(&source).await?;
            Ok(gave_up)
        }
    }

    async fn handle_messages_without_giveup(&self) -> Result<(), CasperError> {
        loop {
            let source = {
                let state = self
                    .state
                    .lock()
                    .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

                match state.queue.front() {
                    Some(s) => s.clone(),
                    None => break, // Stop processing if queue is empty
                }
            };

            let had_message = self.handle_message(&source).await?;

            if had_message {
                self.success(&source).await?;
                // Continue processing next source
            } else {
                // Just stop, don't check give-up logic
                break;
            }
        }
        Ok(())
    }

    pub async fn handle_message(&self, source: &S) -> Result<bool, CasperError> {
        let message_opt = {
            let state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            state
                .messages
                .get(source)
                .and_then(|queue| queue.front().cloned())
        };

        if let Some(message) = message_opt {
            (self.handle)(source.clone(), message).await;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn rotate(&self) -> Result<(), CasperError> {
        let next_source = {
            let mut state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            if let Some(source) = state.queue.pop_front() {
                state.queue.push_back(source.clone());
                state.queue.front().cloned().unwrap_or(source)
            } else {
                return Ok(());
            }
        };

        tracing::info!("It's {} turn", next_source);
        Ok(())
    }

    pub async fn drop_source(&self, source: &S) -> Result<(), CasperError> {
        let next_source = {
            let mut state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            state.queue.retain(|s| s != source);
            state.messages.remove(source);
            state.retries.remove(source);

            tracing::info!("Dropped {} from the dispatch queue", source);

            state.queue.front().cloned()
        };

        if let Some(source) = next_source {
            tracing::info!("It's {} turn", source);
        }

        Ok(())
    }

    pub async fn give_up(&self, source: &S) -> Result<(), CasperError> {
        tracing::info!("Giving up on {}", source);

        let should_drop = {
            let mut state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            state.skipped = 0;

            let retry_count = state.retries.get(source).copied().unwrap_or(0) + 1;
            state.retries.insert(source.clone(), retry_count);

            retry_count > self.config.drop_source_after_retries
        };

        if should_drop {
            self.drop_source(source).await?;
        } else {
            self.rotate().await?;
        }

        Ok(())
    }

    pub async fn success(&self, source: &S) -> Result<(), CasperError> {
        let message = {
            let mut state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            let queue = state
                .messages
                .get_mut(source)
                .ok_or_else(|| CasperError::RuntimeError(format!("Source {} not found", source)))?;

            let message = queue.pop_front().ok_or_else(|| {
                CasperError::RuntimeError(format!("No message in queue for {}", source))
            })?;

            state.skipped = 0;

            message
        };

        tracing::info!("Dispatched message {} from {}", message, source);

        self.rotate().await?;

        Ok(())
    }

    pub async fn failure(&self, source: &S) -> Result<bool, CasperError> {
        tracing::info!("No message to dispatch for {}", source);

        let should_give_up = {
            let mut state = self
                .state
                .lock()
                .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

            state.skipped += 1;
            state.skipped >= self.config.give_up_after_skipped
        };

        if should_give_up {
            self.give_up(source).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_test_state(
        &self,
    ) -> Result<
        (
            VecDeque<S>,
            HashMap<S, VecDeque<M>>,
            HashMap<S, usize>,
            usize,
        ),
        CasperError,
    > {
        let state = self
            .state
            .lock()
            .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

        Ok((
            state.queue.clone(),
            state.messages.clone(),
            state.retries.clone(),
            state.skipped,
        ))
    }

    pub fn set_retries(&self, source: S, count: usize) -> Result<(), CasperError> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

        state.retries.insert(source, count);
        Ok(())
    }

    pub fn set_skipped(&self, count: usize) -> Result<(), CasperError> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| CasperError::LockError(format!("Failed to lock state: {}", e)))?;

        state.skipped = count;
        Ok(())
    }
}
