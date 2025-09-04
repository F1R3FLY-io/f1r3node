// See comm/src/main/scala/coop/rchain/comm/transport/StreamObservable.scala

use crate::rust::{
    errors::CommError,
    peer_node::PeerNode,
    transport::{
        limited_buffer::{FlumeLimitedBuffer, LimitedBuffer, LimitedBufferObservable},
        packet_ops::{PacketExt, StreamCache},
        transport_layer::Blob,
    },
};

/// Stream message containing a cache key and sender peer
#[derive(Debug, Clone)]
pub struct Stream {
    pub key: String,
    pub sender: PeerNode,
}

/// StreamObservable provides bounded buffering for streaming messages with overflow handling
#[derive(Debug)]
pub struct StreamObservable {
    peer: PeerNode,
    cache: StreamCache,
    subject: FlumeLimitedBuffer<Stream>,
}

impl StreamObservable {
    /// Create a new StreamObservable with the given peer, buffer size, and cache
    pub fn new(peer: PeerNode, buffer_size: usize, cache: StreamCache) -> Self {
        let subject = FlumeLimitedBuffer::drop_new_observable(buffer_size);

        Self {
            peer,
            cache,
            subject,
        }
    }

    /// Enqueue a blob for streaming
    pub async fn enque(&self, blob: &Blob) -> Result<(), CommError> {
        // Log stream information
        log::debug!(
            "Pushing message to {} stream message queue.",
            self.peer.endpoint.host
        );

        // Store blob packet in cache
        let store_result = blob.packet.store(&self.cache);

        match store_result {
            Ok(key) => {
                // Successfully stored

                // Create stream message
                let stream_msg = Stream {
                    key: key.clone(),
                    sender: blob.sender.clone(),
                };

                // Try to push to buffer
                let push_succeed = self.subject.push_next(stream_msg);

                if !push_succeed {
                    // Buffer is full
                    log::warn!(
                        "Client stream message queue for {} is full ({} items). Dropping message.",
                        self.peer.endpoint.host,
                        self.subject.buffer_size()
                    );
                    // Clean up cache
                    self.cache.remove(&key);
                }
            }
            Err(e) => {
                log::error!("Failed to store blob packet: {}", e);
            }
        }

        Ok(())
    }

    /// Complete the stream
    pub fn complete(&self) {
        self.subject.complete();
        log::debug!("Stream for {} marked as complete", self.peer.endpoint.host);
    }

    /// Check if the stream is complete
    pub fn is_complete(&self) -> bool {
        self.subject.is_complete()
    }

    /// Get a stream subscription
    pub fn subscribe(&mut self) -> Option<impl tokio_stream::Stream<Item = Stream> + Unpin> {
        self.subject.subscribe()
    }

    /// Get the peer this observable is associated with
    pub fn peer(&self) -> &PeerNode {
        &self.peer
    }

    /// Get the buffer size
    pub fn buffer_size(&self) -> usize {
        self.subject.buffer_size()
    }

    /// Check if the stream is still active (not closed)
    pub fn is_active(&self) -> bool {
        self.subject.is_active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::peer_node::{Endpoint, NodeIdentifier};
    use dashmap::DashMap;
    use models::routing::Packet;
    use prost::bytes::Bytes;
    use std::sync::Arc;
    use tokio_stream::StreamExt;

    fn create_test_peer() -> PeerNode {
        PeerNode {
            id: NodeIdentifier {
                key: Bytes::from("test_peer"),
            },
            endpoint: Endpoint::new("127.0.0.1".to_string(), 8080, 8080),
        }
    }

    fn create_test_cache() -> StreamCache {
        Arc::new(DashMap::new())
    }

    fn create_test_blob(sender: PeerNode, content: Vec<u8>) -> Blob {
        Blob {
            sender,
            packet: Packet {
                type_id: "TestPacket".to_string(),
                content: Bytes::from(content),
            },
        }
    }

    #[tokio::test]
    async fn test_stream_observable_creation() {
        let peer = create_test_peer();
        let cache = create_test_cache();
        let buffer_size = 10;

        let observable = StreamObservable::new(peer.clone(), buffer_size, cache);

        assert_eq!(observable.peer().id.key, peer.id.key);
        assert_eq!(observable.buffer_size(), buffer_size);
        assert!(observable.is_active());
        assert!(!observable.is_complete());
    }

    #[tokio::test]
    async fn test_enque_and_subscribe() {
        let peer = create_test_peer();
        let cache = create_test_cache();
        let mut observable = StreamObservable::new(peer.clone(), 10, cache.clone());

        // Get the stream first
        let mut subscription = observable.subscribe().expect("Should get subscription");

        // Create and enqueue a blob
        let sender = create_test_peer();
        let blob = create_test_blob(sender.clone(), vec![1, 2, 3, 4, 5]);

        let result = observable.enque(&blob).await;
        assert!(result.is_ok());

        // Read from stream
        if let Some(stream_msg) = subscription.next().await {
            assert!(stream_msg.key.starts_with("packet_receive/"));
            assert_eq!(stream_msg.sender.id.key, sender.id.key);

            // Verify packet was stored in cache
            assert!(cache.contains_key(&stream_msg.key));
        } else {
            panic!("Should receive a stream message");
        }
    }

    #[tokio::test]
    async fn test_completion_states() {
        let peer = create_test_peer();
        let cache = create_test_cache();
        let observable = StreamObservable::new(peer.clone(), 10, cache);

        // Initially not complete
        assert!(!observable.is_complete());

        // Complete the stream
        observable.complete();
        assert!(observable.is_complete());

        // Try to enqueue after completion - should still succeed but be dropped
        let sender = create_test_peer();
        let blob = create_test_blob(sender, vec![1, 2, 3]);
        let result = observable.enque(&blob).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_buffer_overflow_drop_new_behavior() {
        let peer = create_test_peer();
        let cache = create_test_cache();
        let buffer_size = 2; // Small buffer to test overflow
        let observable = StreamObservable::new(peer.clone(), buffer_size, cache.clone());

        // Don't subscribe yet - this will cause buffer to fill up
        let sender = create_test_peer();

        // Fill the buffer
        for i in 0..buffer_size {
            let blob = create_test_blob(sender.clone(), vec![i as u8; 10]);
            let result = observable.enque(&blob).await;
            assert!(result.is_ok(), "Enque should always return Ok");
        }

        // This should still return Ok(()) but drop the message
        let overflow_blob = create_test_blob(sender.clone(), vec![99; 10]);
        let result = observable.enque(&overflow_blob).await;
        assert!(result.is_ok(), "Always returns success even when dropping");

        // Only the successfully stored items should be in cache
        assert_eq!(cache.len(), buffer_size);
    }

    #[tokio::test]
    async fn test_multiple_enque_operations() {
        let peer = create_test_peer();
        let cache = create_test_cache();
        let mut observable = StreamObservable::new(peer.clone(), 10, cache.clone());

        let mut subscription = observable.subscribe().expect("Should get subscription");

        // Enqueue multiple blobs
        let sender = create_test_peer();
        for i in 0..3 {
            let blob = create_test_blob(sender.clone(), vec![i; 10]);
            let result = observable.enque(&blob).await;
            assert!(result.is_ok());
        }

        // Read all messages
        let mut received_count = 0;
        while let Some(_stream_msg) = subscription.next().await {
            received_count += 1;
            if received_count == 3 {
                break;
            }
        }

        assert_eq!(received_count, 3);
        assert_eq!(cache.len(), 3); // All packets should be in cache
    }

    #[tokio::test]
    async fn test_stream_message_properties() {
        let sender = create_test_peer();

        let stream_msg = Stream {
            key: "test_key".to_string(),
            sender: sender.clone(),
        };

        assert_eq!(stream_msg.key, "test_key");
        assert_eq!(stream_msg.sender.id.key, sender.id.key);
    }
}
