// Rust port of casper/src/test/scala/coop/rchain/casper/util/comm/TransportLayerTestImpl.scala

use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};

use comm::rust::{
    errors::CommError,
    peer_node::PeerNode,
    rp::protocol_helper,
    transport::{
        communication_response::CommunicationResponse,
        grpc_transport_server::{Cancelable, DispatchFn, HandleStreamedFn, TransportLayerServer},
        transport_layer::{Blob, TransportLayer},
    },
};
use models::routing::Protocol;

/// Test network module providing message queue management between peers
pub mod test_network {
    use super::*;

    /// Type alias for node message queues - maps each peer to its message queue
    pub type NodeMessageQueues = HashMap<PeerNode, VecDeque<Protocol>>;

    /// TestNetwork manages message queues between peers for testing
    #[derive(Clone)]
    pub struct TestNetwork {
        state: Arc<Mutex<NodeMessageQueues>>,
    }

    impl TestNetwork {
        /// Create an empty test network
        pub fn empty() -> Self {
            Self {
                state: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        /// Add a peer to the network with an empty message queue (only if not already present)
        pub fn add_peer(&self, peer: &PeerNode) -> Result<(), CommError> {
            let mut state = self.state.lock().map_err(|_| {
                CommError::InternalCommunicationError("Failed to acquire state lock".to_string())
            })?;
            // Only add if peer doesn't exist - don't replace existing queue!
            state.entry(peer.clone()).or_insert_with(VecDeque::new);
            Ok(())
        }

        /// Get the message queue for a specific peer
        pub fn peer_queue(&self, peer: &PeerNode) -> Result<VecDeque<Protocol>, CommError> {
            let state = self.state.lock().map_err(|_| {
                CommError::InternalCommunicationError("Failed to acquire state lock".to_string())
            })?;

            Ok(state.get(peer).cloned().unwrap_or_else(VecDeque::new))
        }

        /// Send a message to a peer by adding it to their queue
        pub fn send(&self, peer: &PeerNode, msg: &Protocol) -> Result<(), CommError> {
            let mut state = self.state.lock().map_err(|_| {
                CommError::InternalCommunicationError("Failed to acquire state lock".to_string())
            })?;

            // Get or create the peer's queue
            let queue = state.entry(peer.clone()).or_insert_with(VecDeque::new);
            queue.push_back(msg.clone());

            Ok(())
        }

        /// Clear all messages in a peer's queue
        pub fn clear(&self, peer: &PeerNode) -> Result<(), CommError> {
            let mut state = self.state.lock().map_err(|_| {
                CommError::InternalCommunicationError("Failed to acquire state lock".to_string())
            })?;

            state.insert(peer.clone(), VecDeque::new());
            Ok(())
        }

        /// Handle all queued messages for a peer using the provided dispatch function
        pub async fn handle_queue<F, Fut>(
            &self,
            dispatch: F,
            peer: &PeerNode,
        ) -> Result<(), CommError>
        where
            F: Fn(Protocol) -> Fut + Clone,
            Fut: Future<Output = Result<CommunicationResponse, CommError>>,
        {
            loop {
                // Get the next message from the queue
                let maybe_message = {
                    let mut state = self.state.lock().map_err(|_| {
                        CommError::InternalCommunicationError(
                            "Failed to acquire state lock".to_string(),
                        )
                    })?;

                    if let Some(queue) = state.get_mut(peer) {
                        queue.pop_front()
                    } else {
                        None
                    }
                };

                // Process the message or break if queue is empty
                match maybe_message {
                    Some(protocol) => {
                        // Dispatch the message
                        let _response = dispatch(protocol).await?;
                        // Continue processing remaining messages
                    }
                    None => {
                        // Queue is empty, we're done
                        break;
                    }
                }
            }

            Ok(())
        }
    }
}

/// Test implementation of TransportLayer for testing purposes
#[derive(Clone)]
pub struct TransportLayerTestImpl {
    test_network: test_network::TestNetwork,
}

impl TransportLayerTestImpl {
    /// Create a new test transport layer with the given test network
    pub fn new(test_network: test_network::TestNetwork) -> Self {
        Self { test_network }
    }

    /// Get access to the underlying test network for test setup
    pub fn test_network(&self) -> &test_network::TestNetwork {
        &self.test_network
    }
}

#[async_trait]
impl TransportLayer for TransportLayerTestImpl {
    async fn send(&self, peer: &PeerNode, msg: &Protocol) -> Result<(), CommError> {
        // Add the peer if it doesn't exist
        let _ = self.test_network.add_peer(peer);

        // Send the message to the peer's queue
        self.test_network.send(peer, msg)
    }

    async fn broadcast(&self, peers: &[PeerNode], msg: &Protocol) -> Result<(), CommError> {
        // Send to each peer individually
        for peer in peers {
            self.send(peer, msg).await?;
        }
        Ok(())
    }

    async fn stream(&self, peer: &PeerNode, blob: &Blob) -> Result<(), CommError> {
        self.stream_mult(&[peer.clone()], blob).await
    }

    async fn stream_mult(&self, peers: &[PeerNode], blob: &Blob) -> Result<(), CommError> {
        // Convert blob to protocol message using protocol_helper
        let protocol_msg = protocol_helper::packet(&blob.sender, "test", blob.packet.clone());

        // Broadcast the protocol message
        self.broadcast(peers, &protocol_msg).await
    }

    async fn disconnect(&self, _peer: &PeerNode) -> Result<(), CommError> {
        // Test implementation - do nothing
        Ok(())
    }

    async fn get_channeled_peers(&self) -> Result<std::collections::HashSet<PeerNode>, CommError> {
        // Test implementation - return empty set
        Ok(std::collections::HashSet::new())
    }
}

/// Test implementation of TransportLayerServer for testing purposes
pub struct TransportLayerServerTestImpl {
    identity: PeerNode,
    test_network: test_network::TestNetwork,
}

impl TransportLayerServerTestImpl {
    /// Create a new test transport layer server
    pub fn new(identity: PeerNode, test_network: test_network::TestNetwork) -> Self {
        Self {
            identity,
            test_network,
        }
    }
}

#[async_trait]
impl TransportLayerServer for TransportLayerServerTestImpl {
    async fn handle_receive(
        &self,
        dispatch: DispatchFn,
        _handle_streamed: HandleStreamedFn,
    ) -> Result<Cancelable, CommError> {
        // In tests, process all messages synchronously before returning
        let dispatch_fn = move |protocol: Protocol| {
            let dispatch = dispatch.clone();
            async move { dispatch(protocol).await }
        };

        // Process all messages in the queue for this peer synchronously
        self.test_network
            .handle_queue(dispatch_fn, &self.identity)
            .await?;

        // Return a no-op cancelable since we've already processed everything
        Ok(tokio::spawn(async {}))
    }
}
