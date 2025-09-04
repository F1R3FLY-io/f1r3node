// See comm/src/test/scala/coop/rchain/comm/transport/GrpcTransportSpec.scala

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use models::routing::{
    tl_response::Payload, Ack, Chunk, Header, InternalServerError, TlRequest, TlResponse,
};
use prost::bytes::Bytes;
use rand::Rng;
use tokio_stream::{Stream, StreamExt};
use tonic::Status;

use comm::rust::{
    errors::CommError,
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::protocol_helper,
    transport::{
        chunker::Chunker,
        grpc_transport::{GrpcTransport, TransportLayerClientTrait},
        transport_layer::Blob,
    },
};

const NETWORK_ID: &str = "test";

/// Creates a test peer node with random ID
fn create_peer_node() -> PeerNode {
    let mut rng = rand::rng();
    let id_bytes: Vec<u8> = (0..4).map(|_| rng.random()).collect();

    PeerNode {
        id: NodeIdentifier {
            key: Bytes::from(id_bytes),
        },
        endpoint: Endpoint::new("host".to_string(), 0, 0),
    }
}

/// Test Transport Layer
#[derive(Clone)]
pub struct TestTransportLayer {
    /// The response this transport will return
    response: Arc<Mutex<Result<TlResponse, Status>>>,
    /// Track all send requests
    send_messages: Arc<Mutex<VecDeque<TlRequest>>>,
    /// Track all stream requests as lists of chunks
    stream_messages: Arc<Mutex<VecDeque<Vec<Chunk>>>>,
}

impl TestTransportLayer {
    /// Create with a successful response
    pub fn new(response: TlResponse) -> Self {
        Self {
            response: Arc::new(Mutex::new(Ok(response))),
            send_messages: Arc::new(Mutex::new(VecDeque::new())),
            stream_messages: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Create with an error response  
    pub fn with_error(error: Status) -> Self {
        Self {
            response: Arc::new(Mutex::new(Err(error))),
            send_messages: Arc::new(Mutex::new(VecDeque::new())),
            stream_messages: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Get send message count
    pub fn send_messages_length(&self) -> usize {
        self.send_messages.lock().unwrap().len()
    }

    /// Get stream message count
    pub fn stream_messages_length(&self) -> usize {
        self.stream_messages.lock().unwrap().len()
    }

    /// Get first send message
    pub fn send_messages_head(&self) -> Option<TlRequest> {
        self.send_messages.lock().unwrap().front().cloned()
    }

    /// Get first stream message
    pub fn stream_messages_head(&self) -> Option<Vec<Chunk>> {
        self.stream_messages.lock().unwrap().front().cloned()
    }
}

/// Implement TransportLayerTrait for TestTransportLayer
#[async_trait]
impl TransportLayerClientTrait for TestTransportLayer {
    /// Simulate the send operation
    async fn send(&mut self, request: TlRequest) -> Result<TlResponse, Status> {
        // Track the request
        {
            let mut messages = self.send_messages.lock().unwrap();
            messages.push_back(request);
        }

        // Return configured response
        let response = self.response.lock().unwrap().clone();
        response
    }

    /// Simulate the stream operation
    async fn stream<S>(&mut self, input: S) -> Result<TlResponse, Status>
    where
        S: Stream<Item = Chunk> + Send + Unpin + 'static,
    {
        // Collect all chunks from the stream
        let chunks: Vec<Chunk> = input.collect().await;

        // Track the chunks
        {
            let mut messages = self.stream_messages.lock().unwrap();
            messages.push_back(chunks);
        }

        // Always return configured response
        let response = self.response.lock().unwrap().clone();
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test data setup
    fn create_test_setup() -> (PeerNode, PeerNode, models::routing::Protocol) {
        let peer_local = create_peer_node();
        let peer_remote = create_peer_node();
        let msg = protocol_helper::heartbeat(&peer_local, NETWORK_ID);
        (peer_local, peer_remote, msg)
    }

    fn ack_response() -> TlResponse {
        TlResponse {
            payload: Some(Payload::Ack(Ack {
                header: Some(Header {
                    sender: None,
                    network_id: NETWORK_ID.to_string(),
                }),
            })),
        }
    }

    fn internal_server_error_response(msg: &str) -> TlResponse {
        TlResponse {
            payload: Some(Payload::InternalServerError(InternalServerError {
                error: protocol_helper::to_protocol_bytes(msg).into(),
            })),
        }
    }

    fn unavailable_throwable() -> Status {
        Status::unavailable("Service unavailable")
    }

    fn timeout_throwable() -> Status {
        Status::deadline_exceeded("Request timeout")
    }

    fn test_throwable() -> Status {
        Status::internal("Test exception")
    }

    #[tokio::test]
    async fn test_send_everything_is_fine_should_send_and_receive_unit() {
        // given
        let (_, peer_remote, msg) = create_test_setup();
        let response = ack_response();
        let mut stub = TestTransportLayer::new(response);

        // when
        let result = GrpcTransport::send(&mut stub, &peer_remote, &msg).await;

        // then
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ());

        assert_eq!(stub.send_messages_length(), 1);
        assert_eq!(stub.send_messages_head().unwrap().protocol.unwrap(), msg);
        assert_eq!(stub.stream_messages_length(), 0);
    }

    #[tokio::test]
    async fn test_send_server_replies_with_internal_communication_error() {
        // given
        let (_, peer_remote, msg) = create_test_setup();
        let response = internal_server_error_response("Test error");
        let mut stub = TestTransportLayer::new(response);

        // when
        let result = GrpcTransport::send(&mut stub, &peer_remote, &msg).await;

        // then
        assert!(result.is_err());
        match result.unwrap_err() {
            CommError::InternalCommunicationError(error_msg) => {
                assert!(error_msg.contains("Got response: Test error"));
            }
            other => panic!("Expected InternalCommunicationError, got: {:?}", other),
        }

        assert_eq!(stub.send_messages_length(), 1);
        assert_eq!(stub.send_messages_head().unwrap().protocol.unwrap(), msg);
        assert_eq!(stub.stream_messages_length(), 0);
    }

    #[tokio::test]
    async fn test_send_server_is_unavailable_should_fail_with_peer_unavailable() {
        // given
        let (_, peer_remote, msg) = create_test_setup();
        let mut stub = TestTransportLayer::with_error(unavailable_throwable());

        // when
        let result = GrpcTransport::send(&mut stub, &peer_remote, &msg).await;

        // then
        assert!(result.is_err());
        match result.unwrap_err() {
            CommError::PeerUnavailable(host) => {
                assert_eq!(host, peer_remote.endpoint.host);
            }
            other => panic!("Expected PeerUnavailable, got: {:?}", other),
        }

        assert_eq!(stub.send_messages_length(), 1);
        assert_eq!(stub.send_messages_head().unwrap().protocol.unwrap(), msg);
        assert_eq!(stub.stream_messages_length(), 0);
    }

    #[tokio::test]
    async fn test_send_timeout_should_fail_with_timeout() {
        // given
        let (_, peer_remote, msg) = create_test_setup();
        let mut stub = TestTransportLayer::with_error(timeout_throwable());

        // when
        let result = GrpcTransport::send(&mut stub, &peer_remote, &msg).await;

        // then
        assert!(result.is_err());
        match result.unwrap_err() {
            CommError::TimeOut => {
                // Expected
            }
            other => panic!("Expected TimeOut, got: {:?}", other),
        }

        assert_eq!(stub.send_messages_length(), 1);
        assert_eq!(stub.send_messages_head().unwrap().protocol.unwrap(), msg);
        assert_eq!(stub.stream_messages_length(), 0);
    }

    #[tokio::test]
    async fn test_send_any_other_exception_should_fail_with_protocol_exception() {
        // given
        let (_, peer_remote, msg) = create_test_setup();
        let mut stub = TestTransportLayer::with_error(test_throwable());

        // when
        let result = GrpcTransport::send(&mut stub, &peer_remote, &msg).await;

        // then
        assert!(result.is_err());
        match result.unwrap_err() {
            CommError::ProtocolException(error_msg) => {
                assert!(error_msg.contains("Test exception"));
            }
            other => panic!("Expected ProtocolException, got: {:?}", other),
        }

        assert_eq!(stub.send_messages_length(), 1);
        assert_eq!(stub.send_messages_head().unwrap().protocol.unwrap(), msg);
        assert_eq!(stub.stream_messages_length(), 0);
    }

    #[tokio::test]
    async fn test_stream_streaming_successful_should_deliver_list_of_chunks() {
        // given
        let (peer_local, peer_remote, _) = create_test_setup();

        // Create big content
        let message_size = 4 * 1024 * 1024; // 4MB
        let big_content_size = (4 * message_size) + (message_size / 2);
        let big_content: Vec<u8> = vec![128u8; big_content_size];

        let packet = models::routing::Packet {
            type_id: "N/A".to_string(),
            content: Bytes::from(big_content),
        };

        let blob = Blob {
            sender: peer_local,
            packet,
        };

        // Create expected chunks
        let expected_chunks = Chunker::chunk_it(NETWORK_ID, &blob, message_size);

        let mut stub = TestTransportLayer::new(ack_response());

        // when
        let result =
            GrpcTransport::stream(&mut stub, &peer_remote, NETWORK_ID, &blob, message_size).await;

        // then
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ());

        assert_eq!(stub.stream_messages_length(), 1);
        assert_eq!(stub.stream_messages_head().unwrap(), expected_chunks);
        assert_eq!(stub.send_messages_length(), 0);
    }
}
