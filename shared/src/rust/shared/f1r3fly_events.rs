// F1r3flyEvents — event publishing with startup replay buffer.
// Ported from node/src/main/scala/coop/rchain/node/effects/RchainEvents.scala

use futures::stream::Stream;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

pub use super::f1r3fly_event::F1r3flyEvent;

/// Shared startup event buffer type.
/// `Some(vec)` during startup — events accumulate.
/// `None` after `seal_startup()` — no more buffering.
pub type StartupBuffer = Arc<Mutex<Option<Vec<F1r3flyEvent>>>>;

/// Structure to publish and consume F1r3flyEvents.
///
/// During startup, published events are buffered so that WebSocket clients
/// connecting after startup can receive a replay of events they missed.
/// Call `seal_startup()` once the node reaches Running state to stop
/// buffering and free the memory.
#[derive(Clone)]
pub struct F1r3flyEvents {
    sender: broadcast::Sender<F1r3flyEvent>,
    startup_sealed: Arc<AtomicBool>,
    startup_buffer: StartupBuffer,
}

impl F1r3flyEvents {
    /// Create a new F1r3flyEvents backed by a broadcast channel.
    /// The startup buffer is active from creation until `seal_startup()`.
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        F1r3flyEvents {
            sender,
            startup_sealed: Arc::new(AtomicBool::new(false)),
            startup_buffer: Arc::new(Mutex::new(Some(Vec::new()))),
        }
    }

    /// Publish an event to all active subscribers.
    /// If the startup buffer is still open, the event is also buffered
    /// for replay to late-connecting WebSocket clients.
    pub fn publish(&self, event: F1r3flyEvent) -> Result<(), String> {
        if !self.startup_sealed.load(Ordering::Acquire) {
            if let Ok(mut guard) = self.startup_buffer.lock() {
                if let Some(ref mut buf) = *guard {
                    buf.push(event.clone());
                }
            }
        }
        let receivers = self.sender.send(event).unwrap_or(0);
        tracing::trace!("Event published to {} receivers", receivers);
        Ok(())
    }

    /// Seal the startup buffer. After this call, no more events are
    /// buffered and `publish()` skips the lock entirely.
    /// Called once when the node transitions to Running state.
    pub fn seal_startup(&self) {
        self.startup_sealed.store(true, Ordering::Release);
        if let Ok(guard) = self.startup_buffer.lock() {
            let count = guard.as_ref().map_or(0, |v| v.len());
            tracing::info!("Startup event buffer sealed ({} events)", count);
        }
    }

    /// Get a shared reference to the startup buffer.
    /// Pass this to AppState so the WebSocket handler can replay events.
    pub fn startup_buffer(&self) -> StartupBuffer {
        self.startup_buffer.clone()
    }

    /// Publish a noop event
    pub fn noop(&self) -> Result<(), String> {
        Ok(())
    }

    /// Get a stream to consume events
    pub fn consume(&self) -> EventStream {
        EventStream {
            sender: self.sender.clone(),
            inner: BroadcastStream::new(self.sender.subscribe()),
        }
    }
}

/// Stream implementation for consuming events.
/// Uses BroadcastStream internally which properly handles async wakeups.
pub struct EventStream {
    sender: broadcast::Sender<F1r3flyEvent>, // required in order to create a new EventStream from current instance
    inner: BroadcastStream<F1r3flyEvent>,
}

impl EventStream {
    pub fn new_subscribe(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            inner: BroadcastStream::new(self.sender.subscribe()),
        }
    }
}

impl Stream for EventStream {
    type Item = F1r3flyEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Delegate to BroadcastStream which properly handles waker registration
        // BroadcastStream returns Option<Result<T, BroadcastStreamRecvError>>
        // We map errors to None (stream end) and unwrap Ok values
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => Poll::Ready(Some(item)),
            Poll::Ready(Some(Err(_))) => {
                // RecvError::Lagged means we missed some messages, but stream continues
                // We could log this, but for now just skip and poll again
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::super::f1r3fly_event::{BlockAdded, BlockCreated, BlockFinalised, DeployEvent};
    use super::*;
    use futures::StreamExt;

    fn create_test_deploy_event(id: &str) -> DeployEvent {
        DeployEvent::new(id.to_string(), 100, "deployer1".to_string(), false)
    }

    fn create_block_finalised_event() -> F1r3flyEvent {
        F1r3flyEvent::block_finalised(
            "hash123".to_string(),
            vec!["parent1".to_string()],
            vec![("j1".to_string(), "j2".to_string())],
            vec![create_test_deploy_event("deploy1")],
            "creator1".to_string(),
            1,
        )
    }

    fn create_block_created_event() -> F1r3flyEvent {
        F1r3flyEvent::block_created(
            "hash456".to_string(),
            vec!["parent1".to_string(), "parent2".to_string()],
            vec![("j1".to_string(), "j2".to_string())],
            vec![create_test_deploy_event("deploy1")],
            "creator1".to_string(),
            42,
        )
    }

    fn create_block_added_event() -> F1r3flyEvent {
        F1r3flyEvent::block_added(
            "hash789".to_string(),
            vec!["parent3".to_string()],
            vec![("j3".to_string(), "j4".to_string())],
            vec![
                create_test_deploy_event("deploy2"),
                create_test_deploy_event("deploy3"),
            ],
            "creator2".to_string(),
            43,
        )
    }

    #[tokio::test]
    async fn test_publish_and_consume() {
        // Test that stream properly wakes up when event is published AFTER subscription.
        // This tests the async wakeup path - the stream must wake immediately when
        // an event is published, not wait for a timeout or other external trigger.
        let start = std::time::Instant::now();

        let result = tokio::time::timeout(Duration::from_secs(2), async {
            let events = Arc::new(F1r3flyEvents::new());
            let mut stream = events.consume();

            // Publish event AFTER a delay (from another task)
            // This forces the stream to wait asynchronously for the event
            let events_clone = events.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                events_clone
                    .publish(create_block_finalised_event())
                    .expect("Failed to publish event");
            });

            // This should wake up when event arrives (~50ms)
            // If waker is broken, this hangs until timeout (2s)
            let received = stream.next().await.expect("No event available");

            match received {
                F1r3flyEvent::BlockFinalised(BlockFinalised { block_hash, .. }) => {
                    assert_eq!(block_hash, "hash123");
                }
                _ => panic!("Wrong event type received"),
            }
        })
        .await;

        let elapsed = start.elapsed();

        assert!(
            result.is_ok(),
            "Test timed out after {:?} - stream.next() never woke up (waker bug)",
            elapsed
        );

        // Should complete in ~50ms (the delay before publishing), not 2s
        // Allow up to 500ms for slow CI environments
        assert!(
            elapsed < Duration::from_millis(500),
            "Test took {:?} - expected ~50ms, waker may not be working correctly",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_circular_buffer_behavior() {
        let events = F1r3flyEvents::new();

        // Get stream first (before publishing)
        let mut stream = events.consume();

        // Publish three events
        events
            .publish(create_block_finalised_event())
            .expect("Failed to publish event");
        events
            .publish(create_block_created_event())
            .expect("Failed to publish event");
        events
            .publish(create_block_added_event())
            .expect("Failed to publish event");

        // Consume events from the stream
        let first_received = stream.next().await.expect("No event available");
        match first_received {
            F1r3flyEvent::BlockFinalised(BlockFinalised { block_hash, .. }) => {
                assert_eq!(block_hash, "hash123");
            }
            _ => panic!("Wrong event type received"),
        }

        let second_received = stream.next().await.expect("No event available");
        match second_received {
            F1r3flyEvent::BlockCreated(BlockCreated { block_hash, .. }) => {
                assert_eq!(block_hash, "hash456");
            }
            _ => panic!("Wrong event type received"),
        }

        let third_received = stream.next().await.expect("No event available");
        match third_received {
            F1r3flyEvent::BlockAdded(BlockAdded { block_hash, .. }) => {
                assert_eq!(block_hash, "hash789");
            }
            _ => panic!("Wrong event type received"),
        }
    }

    #[tokio::test]
    async fn test_new() {
        let events = F1r3flyEvents::new();

        // Get stream first (before publishing)
        let mut stream = events.consume();

        // Publish first event
        events
            .publish(create_block_finalised_event())
            .expect("Failed to publish event");

        // Publish second event - should not affect broadcast
        events
            .publish(create_block_created_event())
            .expect("Failed to publish event");

        // Consume events from the stream
        let first_received = stream.next().await.expect("No event available");
        match first_received {
            F1r3flyEvent::BlockFinalised(BlockFinalised { block_hash, .. }) => {
                assert_eq!(block_hash, "hash123");
            }
            _ => panic!("Wrong event type received"),
        }

        let second_received = stream.next().await.expect("No event available");
        match second_received {
            F1r3flyEvent::BlockCreated(BlockCreated { block_hash, .. }) => {
                assert_eq!(block_hash, "hash456");
            }
            _ => panic!("Wrong event type received"),
        }
    }

    #[tokio::test]
    async fn test_concurrent_publish_and_consume() {
        let events = Arc::new(F1r3flyEvents::new());
        let mut handles = vec![];

        // Spawn multiple consumers first
        for _ in 0..3 {
            let events = Arc::clone(&events);
            let handle = tokio::spawn(async move {
                let mut stream = events.consume();
                let mut count = 0;

                // Add a timeout to avoid waiting forever
                while let Some(_) = tokio::time::timeout(Duration::from_millis(100), stream.next())
                    .await
                    .unwrap_or(None)
                {
                    count += 1;
                    if count >= 5 {
                        break;
                    }
                }
                count
            });
            handles.push(handle);
        }

        // Give consumers time to start up
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Then publish events
        for _ in 0..5 {
            let event = create_block_created_event();
            events.publish(event).expect("Failed to publish event");
            // Add a small delay between publishing events
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Wait for all consumers to finish
        let results = futures::future::join_all(handles).await;
        let total_events: usize = results.into_iter().map(|r| r.expect("Task failed")).sum();

        // Each consumer should get all 5 events
        assert_eq!(total_events, 15);
    }

    #[tokio::test]
    async fn test_startup_buffer_captures_events() {
        // Events published before any subscriber should be buffered
        let events = F1r3flyEvents::new();

        events.publish(create_block_finalised_event()).unwrap();
        events.publish(create_block_created_event()).unwrap();

        // Buffer should contain both events
        let buf = events.startup_buffer();
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should be Some");
        assert_eq!(buffered.len(), 2);
    }

    #[tokio::test]
    async fn test_startup_buffer_survives_seal() {
        // After seal, buffer data stays available for replay
        let events = F1r3flyEvents::new();

        events.publish(create_block_finalised_event()).unwrap();
        events.publish(create_block_created_event()).unwrap();
        events.seal_startup();

        // Buffer should still contain the events (frozen, not cleared)
        let buf = events.startup_buffer();
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should be Some after seal");
        assert_eq!(buffered.len(), 2);
    }

    #[tokio::test]
    async fn test_startup_buffer_stops_appending_after_seal() {
        let events = F1r3flyEvents::new();

        events.publish(create_block_finalised_event()).unwrap();
        events.seal_startup();
        events.publish(create_block_created_event()).unwrap();

        // Only the event before seal should be in the buffer
        let buf = events.startup_buffer();
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should be Some after seal");
        assert_eq!(buffered.len(), 1);
    }

    #[tokio::test]
    async fn test_startup_replay_scenario() {
        // Simulates the full WebSocket startup replay scenario:
        // 1. Events published during startup (no subscribers)
        // 2. Subscriber connects and reads buffer
        // 3. More events arrive via live stream
        // 4. Seal happens
        // 5. Late subscriber connects and reads buffer + live
        let events = Arc::new(F1r3flyEvents::new());

        // Phase 1: Startup events (no subscribers yet)
        events.publish(create_block_finalised_event()).unwrap();

        // Phase 2: First subscriber connects
        let mut stream = events.consume();
        let buf = events.startup_buffer();

        // Read buffer (simulating WebSocket replay)
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should be Some");
        assert_eq!(buffered.len(), 1);
        drop(guard);

        // Phase 3: More events arrive — both buffered AND broadcast (pre-seal)
        events.publish(create_block_created_event()).unwrap();
        let live_event = stream.next().await.expect("Should receive live event");
        match live_event {
            F1r3flyEvent::BlockCreated(BlockCreated { block_hash, .. }) => {
                assert_eq!(block_hash, "hash456");
            }
            _ => panic!("Expected BlockCreated from live stream"),
        }

        // Buffer now has both events (all pre-seal events are buffered)
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should be Some");
        assert_eq!(buffered.len(), 2, "Both pre-seal events should be buffered");
        drop(guard);

        // Phase 4: Seal startup
        events.seal_startup();

        // Buffer still has both events (frozen, not cleared)
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should survive seal");
        assert_eq!(buffered.len(), 2);
        drop(guard);

        // Phase 5: Post-seal events only go to broadcast, not buffer
        events.publish(create_block_added_event()).unwrap();
        let post_seal = stream.next().await.expect("Should receive post-seal event");
        match post_seal {
            F1r3flyEvent::BlockAdded(BlockAdded { block_hash, .. }) => {
                assert_eq!(block_hash, "hash789");
            }
            _ => panic!("Expected BlockAdded from live stream"),
        }

        // Buffer still only has the 2 pre-seal events
        let guard = buf.lock().unwrap();
        let buffered = guard.as_ref().expect("Buffer should survive seal");
        assert_eq!(buffered.len(), 2, "Post-seal events should not be buffered");
    }
}
