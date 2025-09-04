// See comm/src/main/scala/coop/rchain/comm/transport/buffer/LimitedBuffer.scala
// See comm/src/main/scala/coop/rchain/comm/transport/buffer/LimitedBufferObservable.scala
// See comm/src/main/scala/coop/rchain/comm/transport/buffer/ConcurrentQueue.scala

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio_stream::Stream;

/// LimitedBuffer trait providing bounded buffering with overflow policy
pub trait LimitedBuffer<T> {
    /// Push the next element to the buffer
    /// Returns true if successfully enqueued, false if dropped due to overflow
    fn push_next(&self, elem: T) -> bool;

    /// Signal completion - no more elements will be pushed
    fn complete(&self);

    /// Check if the buffer is complete
    fn is_complete(&self) -> bool;
}

/// Observable interface for LimitedBuffer
pub trait LimitedBufferObservable<T>: LimitedBuffer<T> {
    type Subscription: Stream<Item = T> + Unpin;

    /// Subscribe to receive items from this buffer
    fn subscribe(&mut self) -> Option<Self::Subscription>;
}

/// FlumeLimitedBuffer: LimitedBuffer implementation with multi-consumer support
///
/// Uses a hybrid architecture:
/// - Single flume bounded channel for backpressure control
/// - tokio::broadcast for fan-out to multiple consumers
/// - Background task to pump messages from flume to broadcast
#[derive(Debug)]
pub struct FlumeLimitedBuffer<T> {
    sender: flume::Sender<T>,
    broadcast_tx: Arc<tokio::sync::broadcast::Sender<T>>,
    buffer_size: usize,
    // Completion state management
    complete: Arc<AtomicBool>,
    // Background task handle for the fan-out pump
    _pump_handle: Arc<tokio::task::JoinHandle<()>>,
}

impl<T: Clone + Send + 'static> FlumeLimitedBuffer<T> {
    /// Create a new FlumeLimitedBuffer with the specified buffer size
    pub fn drop_new(buffer_size: usize) -> Self {
        assert!(
            buffer_size > 0,
            "bufferSize must be a strictly positive number"
        );

        let (flume_tx, flume_rx) = flume::bounded(buffer_size);

        // Create broadcast channel with generous capacity for multiple consumers
        // Use 2x buffer_size to handle multiple subscribers without dropping
        let (broadcast_tx, _) = tokio::sync::broadcast::channel(buffer_size * 2);
        let broadcast_tx = Arc::new(broadcast_tx);

        let complete = Arc::new(AtomicBool::new(false));

        // Start background pump task to move messages from flume to broadcast
        let pump_handle = {
            let flume_rx = flume_rx;
            let broadcast_tx = broadcast_tx.clone();
            let complete = complete.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        // Try to receive from flume channel
                        result = flume_rx.recv_async() => {
                            match result {
                                Ok(item) => {
                                    // Broadcast to all subscribers
                                    // If no subscribers or all lagged, that's ok - we drop the message
                                    let _ = broadcast_tx.send(item);
                                }
                                Err(_) => {
                                    // Flume sender disconnected - end the pump
                                    log::debug!("FlumeLimitedBuffer pump: flume sender disconnected");
                                    break;
                                }
                            }
                        }
                        // Periodically check completion
                        _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {
                            if complete.load(Ordering::Acquire) && flume_rx.is_empty() {
                                log::debug!("FlumeLimitedBuffer pump: completed and empty");
                                break;
                            }
                        }
                    }
                }
                log::debug!("FlumeLimitedBuffer pump task ended");
            })
        };

        Self {
            sender: flume_tx,
            broadcast_tx,
            buffer_size,
            complete,
            _pump_handle: Arc::new(pump_handle),
        }
    }

    /// Get the buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Check if the sender is still active (not closed)
    pub fn is_active(&self) -> bool {
        !self.sender.is_disconnected()
    }
}

impl<T: Clone + Send + 'static> LimitedBuffer<T> for FlumeLimitedBuffer<T> {
    fn push_next(&self, elem: T) -> bool {
        if self.complete.load(Ordering::Acquire) {
            return false;
        }

        match self.sender.try_send(elem) {
            Ok(()) => true,
            Err(flume::TrySendError::Full(_)) => {
                // Buffer is full - implement "drop new" behavior
                false
            }
            Err(flume::TrySendError::Disconnected(_)) => {
                // Receiver has been dropped
                false
            }
        }
    }

    fn complete(&self) {
        self.complete.store(true, Ordering::Release);
        // The pump task will notice completion and stop
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

/// Subscription handle for FlumeLimitedBuffer using broadcast receiver
pub struct FlumeLimitedBufferSubscription<T> {
    receiver: tokio::sync::broadcast::Receiver<T>,
    complete: Arc<AtomicBool>,
}

impl<T: Clone + Send + 'static> Stream for FlumeLimitedBufferSubscription<T> {
    type Item = T;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Try to receive an item immediately
        match self.receiver.try_recv() {
            Ok(item) => std::task::Poll::Ready(Some(item)),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                // Buffer is empty - check if we're complete
                if self.complete.load(Ordering::Acquire) {
                    // Complete and empty - end the stream
                    std::task::Poll::Ready(None)
                } else {
                    // Not complete yet - register for future notification
                    let waker = cx.waker().clone();
                    let mut receiver = self.receiver.resubscribe();
                    let complete = self.complete.clone();

                    tokio::spawn(async move {
                        tokio::select! {
                            result = receiver.recv() => {
                                match result {
                                    Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                        // Got a message or lagged (both mean more data available)
                                        waker.wake();
                                    }
                                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                        // Channel closed
                                        waker.wake();
                                    }
                                }
                            }
                            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                                // Check completion status periodically
                                if complete.load(Ordering::Acquire) {
                                    waker.wake();
                                }
                            }
                        }
                    });

                    std::task::Poll::Pending
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                // Channel closed - end the stream
                std::task::Poll::Ready(None)
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                // We lagged behind - just continue and try to get the next message
                // This is acceptable behavior for overloaded consumers
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
        }
    }
}

impl<T: Clone + Send + 'static> LimitedBufferObservable<T> for FlumeLimitedBuffer<T> {
    type Subscription = FlumeLimitedBufferSubscription<T>;

    fn subscribe(&mut self) -> Option<Self::Subscription> {
        // Create a new broadcast receiver - each subscription gets its own independent stream
        let receiver = self.broadcast_tx.subscribe();

        Some(FlumeLimitedBufferSubscription {
            receiver,
            complete: self.complete.clone(),
        })
    }
}

/// Convenience constructor functions
impl<T: Clone + Send + 'static> FlumeLimitedBuffer<T> {
    /// Create a new drop-new limited buffer observable
    pub fn drop_new_observable(buffer_size: usize) -> Self {
        Self::drop_new(buffer_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_limited_buffer_push_next() {
        let buffer = FlumeLimitedBuffer::<i32>::drop_new(2);

        // Should accept first two items
        assert!(buffer.push_next(1));
        assert!(buffer.push_next(2));

        // Should reject third item (buffer full)
        assert!(!buffer.push_next(3));

        assert!(!buffer.is_complete());
    }

    #[tokio::test]
    async fn test_limited_buffer_completion() {
        let buffer = FlumeLimitedBuffer::<i32>::drop_new(10);

        assert!(!buffer.is_complete());

        buffer.complete();
        assert!(buffer.is_complete());

        // Should reject new items after completion
        assert!(!buffer.push_next(1));
    }

    #[tokio::test]
    async fn test_limited_buffer_subscription() {
        let mut buffer = FlumeLimitedBuffer::<i32>::drop_new(10);

        // Push some items
        assert!(buffer.push_next(1));
        assert!(buffer.push_next(2));
        assert!(buffer.push_next(3));

        // Subscribe and read items
        let mut subscription = buffer.subscribe().expect("Should get subscription");

        // Give the pump task time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let items: Vec<i32> = vec![
            subscription.next().await.unwrap(),
            subscription.next().await.unwrap(),
            subscription.next().await.unwrap(),
        ];

        assert_eq!(items, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_limited_buffer_completion_with_subscription() {
        let mut buffer = FlumeLimitedBuffer::<i32>::drop_new(10);

        let mut subscription = buffer.subscribe().expect("Should get subscription");

        // Push an item and complete
        assert!(buffer.push_next(42));
        buffer.complete();

        // Give the pump task time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Should receive the item
        assert_eq!(subscription.next().await, Some(42));

        // Should end the stream after completion
        assert_eq!(subscription.next().await, None);
    }

    #[test]
    fn test_buffer_size_validation() {
        // Should panic with zero buffer size
        std::panic::catch_unwind(|| {
            FlumeLimitedBuffer::<i32>::drop_new(0);
        })
        .expect_err("Should panic with zero buffer size");
    }

    #[tokio::test]
    async fn test_drop_new_behavior() {
        let mut buffer = FlumeLimitedBuffer::<String>::drop_new(2);

        // Fill buffer
        assert!(buffer.push_next("first".to_string()));
        assert!(buffer.push_next("second".to_string()));

        // These should be dropped
        assert!(!buffer.push_next("dropped1".to_string()));
        assert!(!buffer.push_next("dropped2".to_string()));

        // Subscribe and verify only first two items are present
        let mut subscription = buffer.subscribe().expect("Should get subscription");

        // Give the pump task time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        assert_eq!(subscription.next().await, Some("first".to_string()));
        assert_eq!(subscription.next().await, Some("second".to_string()));

        buffer.complete();

        // Give completion time to propagate
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        assert_eq!(subscription.next().await, None);
    }

    #[tokio::test]
    async fn test_multiple_subscriptions() {
        let mut buffer = FlumeLimitedBuffer::<i32>::drop_new(10);

        // Create multiple subscriptions
        let mut sub1 = buffer.subscribe().expect("Should get subscription 1");
        let mut sub2 = buffer.subscribe().expect("Should get subscription 2");
        let mut sub3 = buffer.subscribe().expect("Should get subscription 3");

        // Push some items
        assert!(buffer.push_next(100));
        assert!(buffer.push_next(200));
        assert!(buffer.push_next(300));

        // Give the pump task time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // All subscriptions should receive all items
        for sub in [&mut sub1, &mut sub2, &mut sub3] {
            assert_eq!(sub.next().await, Some(100));
            assert_eq!(sub.next().await, Some(200));
            assert_eq!(sub.next().await, Some(300));
        }
    }
}
