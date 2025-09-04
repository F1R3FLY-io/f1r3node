// See comm/src/test/scala/coop/rchain/comm/transport/StreamHandlerSpec.scala

use comm::rust::{
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    transport::{
        chunker::Chunker,
        stream_handler::{Circuit, StreamError, StreamHandler},
        transport_layer::Blob,
    },
};
use futures::stream;
use models::routing::{Chunk, Packet};
use prost::bytes::Bytes;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;
use tokio_stream::Stream;

const NETWORK_ID: &str = "test";

#[cfg(test)]
mod stream_handler_spec {
    use super::*;

    #[tokio::test]
    async fn should_contain_sender_and_message_type_information() {
        // given
        let stream = create_stream(None, None, None, Some("BlockMessageTest".to_string())).await;

        // when
        let msg = handle_stream(stream).await;

        // then
        assert_eq!(msg.sender, peer_node("sender"));
        assert_eq!(msg.type_id, "BlockMessageTest");
    }

    #[tokio::test]
    async fn should_contain_content_length_of_stored_file() {
        // given
        let message_size = 10 * 1024; // 10 kb
        let content_length = message_size * 3 + (message_size / 2);
        let stream = create_stream(Some(message_size), Some(content_length), None, None).await;

        // when
        let msg = handle_stream(stream).await;

        // then
        assert_eq!(msg.content_length, content_length as i32);
    }

    #[tokio::test]
    async fn should_stop_receiving_stream_if_circuit_broken() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let break_on_second_chunk =
            |_streamed: &comm::rust::transport::stream_handler::Streamed| {
                Circuit::opened(StreamError::circuit_opened()) // MaxSizeReached
            };
        let stream = create_stream(None, None, None, None).await;

        // when
        let err = handle_stream_err(stream, Some(break_on_second_chunk), Some(cache.clone())).await;

        // then
        assert!(matches!(err, StreamError::MaxSizeReached));
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn should_stop_processing_stream_if_missing_header() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let stream_without_header = create_stream_without_header().await;

        // when
        let err = handle_stream_err(stream_without_header, None, Some(cache.clone())).await;

        // then
        assert!(matches!(err, StreamError::NotFullMessage { .. }));
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn should_stop_processing_stream_if_incomplete_data() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let incomplete_stream = create_incomplete_stream().await;

        // when
        let err = handle_stream_err(incomplete_stream, None, Some(cache.clone())).await;

        // then
        assert!(matches!(err, StreamError::NotFullMessage { .. }));
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn should_handle_compressed_content() {
        // given
        let stream = create_compressed_stream().await;

        // when
        let msg = handle_stream(stream).await;

        // then
        assert_eq!(msg.sender, peer_node("sender"));
        assert_eq!(msg.type_id, "BlockMessageTest");
        assert!(msg.compressed); // Should be marked as compressed
    }

    #[tokio::test]
    async fn should_handle_zero_length_content() {
        // given
        let stream = create_stream(Some(1024), Some(0), None, None).await; // Zero content length

        // when
        let msg = handle_stream(stream).await;

        // then
        assert_eq!(msg.content_length, 0);
        assert_eq!(msg.sender, peer_node("sender"));
    }

    #[tokio::test]
    async fn should_fail_with_wrong_network_id() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let stream = create_stream_with_wrong_network_id().await;

        // when - This should still process but could be caught by circuit breaker
        // The network ID validation would typically be done by a circuit breaker
        let network_id_breaker = |streamed: &comm::rust::transport::stream_handler::Streamed| {
            if let Some(header) = &streamed.header {
                if header.network_id != NETWORK_ID {
                    return Circuit::opened(StreamError::wrong_network_id());
                }
            }
            Circuit::Closed
        };

        let err = handle_stream_err(stream, Some(network_id_breaker), Some(cache.clone())).await;

        // then
        assert!(matches!(err, StreamError::WrongNetworkId));
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn should_fail_on_empty_stream() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let empty_stream = stream::iter(Vec::<Chunk>::new()); // Completely empty stream

        // when
        let err = handle_stream_err(empty_stream, None, Some(cache.clone())).await;

        // then
        assert!(matches!(err, StreamError::NotFullMessage { .. }));
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn should_stop_on_circuit_breaker_opening_mid_stream() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());

        // Circuit breaker that opens when we have both header and some data
        let break_after_header_and_data =
            |streamed: &comm::rust::transport::stream_handler::Streamed| {
                if streamed.header.is_some() && streamed.read_so_far > 1000 {
                    Circuit::opened(StreamError::MaxSizeReached)
                } else {
                    Circuit::Closed
                }
            };

        let stream = create_stream(Some(1024), Some(5120), None, None).await; // Medium sized content

        // when
        let err = handle_stream_err(
            stream,
            Some(break_after_header_and_data),
            Some(cache.clone()),
        )
        .await;

        // then
        // The circuit breaker may cause either MaxSizeReached or NotFullMessage depending on timing
        match err {
            StreamError::MaxSizeReached => {
                // Direct circuit breaker error
            }
            StreamError::NotFullMessage { .. } => {
                // Circuit breaker stopped processing, resulting in incomplete message
                // This is also a valid outcome for mid-stream circuit breaking
            }
            other => {
                panic!(
                    "Expected MaxSizeReached or NotFullMessage, got: {:?}",
                    other
                );
            }
        }
        assert!(cache.is_empty());
    }

    #[tokio::test]
    async fn should_restore_blob_from_cache() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let test_data = b"Hello, World!".to_vec();
        let cache_key = "test_key".to_string();

        cache.insert(cache_key.clone(), test_data.clone());

        let stream_message = comm::rust::transport::messages::StreamMessage::new(
            peer_node("sender"),
            "TestType".to_string(),
            cache_key,
            false, // Not compressed
            test_data.len() as i32,
        );

        // when
        let result = StreamHandler::restore(&stream_message, &cache).await;

        // then
        assert!(result.is_ok());
        let blob = result.unwrap();
        assert_eq!(blob.sender, peer_node("sender"));
        assert_eq!(blob.packet.type_id, "TestType");
        assert_eq!(blob.packet.content.as_ref(), test_data);

        // Cache should be cleaned up
        assert!(!cache.contains_key(&stream_message.key));
    }

    #[tokio::test]
    async fn should_fail_restore_with_missing_cache_entry() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());
        let stream_message = comm::rust::transport::messages::StreamMessage::new(
            peer_node("sender"),
            "TestType".to_string(),
            "non_existent_key".to_string(),
            false,
            10,
        );

        // when
        let result = StreamHandler::restore(&stream_message, &cache).await;

        // then
        assert!(result.is_err());
        // Check the error type without unwrapping
        match result {
            Err(comm::rust::errors::CommError::InternalCommunicationError(_)) => {
                // Expected error type
            }
            _ => panic!("Expected InternalCommunicationError"),
        }
    }

    #[tokio::test]
    async fn should_handle_malformed_chunks() {
        // given
        let cache = Arc::new(dashmap::DashMap::new());

        // Create a stream with chunks that have no content
        let malformed_chunks = vec![
            Chunk { content: None }, // Malformed chunk
            Chunk { content: None }, // Another malformed chunk
        ];
        let malformed_stream = stream::iter(malformed_chunks);

        // when
        let err = handle_stream_err(malformed_stream, None, Some(cache.clone())).await;

        // then
        assert!(matches!(err, StreamError::NotFullMessage { .. }));
        assert!(cache.is_empty());
    }

    // Helper methods

    /// Handle a stream and expect success
    async fn handle_stream<S>(stream: S) -> comm::rust::transport::messages::StreamMessage
    where
        S: Stream<Item = Chunk> + Unpin,
    {
        let cache = Arc::new(dashmap::DashMap::new());
        StreamHandler::handle_stream(stream, never_break, &cache)
            .await
            .expect("Expected successful stream handling")
    }

    /// Handle a stream and expect an error
    async fn handle_stream_err<S>(
        stream: S,
        circuit_breaker: Option<comm::rust::transport::stream_handler::CircuitBreaker>,
        cache: Option<Arc<dashmap::DashMap<String, Vec<u8>>>>,
    ) -> StreamError
    where
        S: Stream<Item = Chunk> + Unpin,
    {
        let cache = cache.unwrap_or_else(|| Arc::new(dashmap::DashMap::new()));
        let breaker = circuit_breaker.unwrap_or(never_break);

        StreamHandler::handle_stream(stream, breaker, &cache)
            .await
            .expect_err("Expected stream handling to fail")
    }

    /// Create a test stream with configurable parameters
    async fn create_stream(
        message_size: Option<usize>,
        content_length: Option<usize>,
        sender: Option<String>,
        type_id: Option<String>,
    ) -> impl Stream<Item = Chunk> + Unpin {
        let message_size = message_size.unwrap_or(10 * 1024);
        let content_length = content_length.unwrap_or(30 * 1024);
        let sender = sender.unwrap_or_else(|| "sender".to_string());
        let type_id = type_id.unwrap_or_else(|| "BlockMessageTest".to_string());

        let chunks = create_stream_iterator(message_size, content_length, sender, type_id).await;
        stream::iter(chunks)
    }

    /// Create stream iterator using Chunker
    async fn create_stream_iterator(
        message_size: usize,
        content_length: usize,
        sender: String,
        type_id: String,
    ) -> Vec<Chunk> {
        // Create random content
        let mut rng = StdRng::seed_from_u64(42);
        let content: Vec<u8> = (0..content_length).map(|_| rng.random::<u8>()).collect();

        let packet = Packet {
            type_id,
            content: Bytes::from(content),
        };

        let peer = peer_node(&sender);
        let blob = Blob {
            sender: peer,
            packet,
        };

        // Use Chunker to create chunks
        Chunker::chunk_it(NETWORK_ID, &blob, message_size)
    }

    /// Create a stream without header (for testing missing header scenario)
    async fn create_stream_without_header() -> impl Stream<Item = Chunk> + Unpin {
        let chunks = create_stream_iterator(
            10 * 1024,
            30 * 1024,
            "sender".to_string(),
            "BlockMessageTest".to_string(),
        )
        .await;

        // Remove the header chunk (first chunk)
        let data_chunks: Vec<Chunk> = chunks.into_iter().skip(1).collect();
        stream::iter(data_chunks)
    }

    /// Create an incomplete stream (missing some data chunks)
    async fn create_incomplete_stream() -> impl Stream<Item = Chunk> + Unpin {
        let chunks = create_stream_iterator(
            10 * 1024,
            30 * 1024,
            "sender".to_string(),
            "BlockMessageTest".to_string(),
        )
        .await;

        // Take header and skip one data chunk (making it incomplete)
        let mut incomplete_chunks = Vec::new();
        let mut chunks_iter = chunks.into_iter();

        // Add header
        if let Some(header_chunk) = chunks_iter.next() {
            incomplete_chunks.push(header_chunk);
        }

        // Skip one data chunk
        chunks_iter.next();

        // Add remaining data chunks
        incomplete_chunks.extend(chunks_iter);

        stream::iter(incomplete_chunks)
    }

    /// Create a test PeerNode
    fn peer_node(name: &str) -> PeerNode {
        PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(name.as_bytes().to_vec()),
            },
            endpoint: Endpoint::new("".to_string(), 80, 80),
        }
    }

    /// Circuit breaker that never breaks
    fn never_break(_streamed: &comm::rust::transport::stream_handler::Streamed) -> Circuit {
        Circuit::Closed
    }

    /// Create a stream with compressed content
    async fn create_compressed_stream() -> impl Stream<Item = Chunk> + Unpin {
        // Create a large enough content to trigger compression (> 500KB)
        let large_content_size = 600 * 1024; // 600KB
        create_stream(Some(4096), Some(large_content_size), None, None).await
    }

    /// Create a stream with wrong network ID
    async fn create_stream_with_wrong_network_id() -> impl Stream<Item = Chunk> + Unpin {
        let content = b"test content".to_vec();
        let packet = Packet {
            type_id: "TestPacket".to_string(),
            content: Bytes::from(content.clone()),
        };

        let peer = peer_node("sender");
        let blob = Blob {
            sender: peer,
            packet,
        };

        // Create chunks with wrong network ID
        let mut chunks = Chunker::chunk_it("wrong_network", &blob, 4096);

        // Replace the network_id in the header chunk to simulate wrong network
        if let Some(first_chunk) = chunks.get_mut(0) {
            if let Some(models::routing::chunk::Content::Header(ref mut header)) =
                &mut first_chunk.content
            {
                header.network_id = "wrong_network".to_string();
            }
        }

        stream::iter(chunks)
    }
}
