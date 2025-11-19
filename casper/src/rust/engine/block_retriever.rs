// See casper/src/main/scala/coop/rchain/casper/engine/BlockRetriever.scala

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use comm::rust::{
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use log::{debug, info};
use models::rust::{block_hash::BlockHash, casper::pretty_printer::PrettyPrinter};

use crate::rust::errors::CasperError;

#[derive(Debug, Clone, PartialEq)]
pub enum AdmitHashReason {
    HasBlockMessageReceived,
    HashBroadcastReceived,
    MissingDependencyRequested,
    BlockReceived,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AdmitHashStatus {
    NewSourcePeerAddedToRequest,
    NewRequestAdded,
    Ignore,
}

#[derive(Debug, Clone)]
pub struct AdmitHashResult {
    pub status: AdmitHashStatus,
    pub broadcast_request: bool,
    pub request_block: bool,
}

#[derive(Debug, Clone)]
pub struct RequestState {
    pub timestamp: u64,
    pub peers: HashSet<PeerNode>,
    pub received: bool,
    pub in_casper_buffer: bool,
    pub waiting_list: Vec<PeerNode>,
}

// Scala: type RequestedBlocks[F[_]] = Ref[F, Map[BlockHash, RequestState]]
// In Rust, we use Arc<Mutex<...>> as shared mutable state (passed as implicit in Scala)
pub type RequestedBlocks = Arc<Mutex<HashMap<BlockHash, RequestState>>>;

#[derive(Debug, Clone, PartialEq)]
enum AckReceiveResult {
    AddedAsReceived,
    MarkedAsReceived,
}

/**
* BlockRetriever makes sure block is received once Casper request it.
* Block is in scope of BlockRetriever until it is added to CasperBuffer.
*
* Scala: BlockRetriever.of[F[_]: Monad: RequestedBlocks: ...]
* In Scala, RequestedBlocks is passed as an implicit parameter (type class constraint).
* In Rust, we explicitly pass it as a constructor parameter.
*/
#[derive(Debug, Clone)]
pub struct BlockRetriever<T: TransportLayer + Send + Sync> {
    requested_blocks: RequestedBlocks,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
}

impl<T: TransportLayer + Send + Sync> BlockRetriever<T> {
    /// Creates a new BlockRetriever with shared requested_blocks state.
    ///
    /// # Arguments
    /// * `requested_blocks` - Shared state for tracking block requests (equivalent to Scala implicit RequestedBlocks[F])
    /// * `transport` - Transport layer for network communication
    /// * `connections_cell` - Peer connections
    /// * `conf` - RP configuration
    pub fn new(
        requested_blocks: RequestedBlocks,
        transport: Arc<T>,
        connections_cell: ConnectionsCell,
        conf: RPConf,
    ) -> Self {
        Self {
            requested_blocks,
            transport,
            connections_cell,
            conf,
        }
    }

    /// Get access to the requested_blocks for testing purposes
    pub fn requested_blocks(&self) -> &RequestedBlocks {
        &self.requested_blocks
    }

    /// Helper method to add a source peer to an existing request
    fn add_source_peer_to_request(
        init_state: &mut HashMap<BlockHash, RequestState>,
        hash: &BlockHash,
        peer: &PeerNode,
    ) {
        if let Some(request_state) = init_state.get_mut(hash) {
            request_state.waiting_list.push(peer.clone());
        }
    }

    /// Helper method to add a new request
    fn add_new_request(
        init_state: &mut HashMap<BlockHash, RequestState>,
        hash: BlockHash,
        now: u64,
        mark_as_received: bool,
        source_peer: Option<&PeerNode>,
    ) -> bool {
        if init_state.contains_key(&hash) {
            false // Request already exists
        } else {
            let waiting_list = if let Some(peer) = source_peer {
                vec![peer.clone()]
            } else {
                Vec::new()
            };

            init_state.insert(
                hash,
                RequestState {
                    timestamp: now,
                    peers: HashSet::new(),
                    received: mark_as_received,
                    in_casper_buffer: false,
                    waiting_list,
                },
            );
            true
        }
    }

    /// Get current timestamp in milliseconds
    fn current_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub async fn admit_hash(
        &self,
        hash: BlockHash,
        peer: Option<PeerNode>,
        admit_hash_reason: AdmitHashReason,
    ) -> Result<AdmitHashResult, CasperError> {
        let now = Self::current_millis();

        // Lock the requested_blocks mutex and modify state atomically
        let result = {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            let unknown_hash = !state.contains_key(&hash);

            if unknown_hash {
                // Add new request
                Self::add_new_request(&mut state, hash.clone(), now, false, peer.as_ref());
                AdmitHashResult {
                    status: AdmitHashStatus::NewRequestAdded,
                    broadcast_request: peer.is_none(),
                    request_block: peer.is_some(),
                }
            } else if let Some(ref peer_node) = peer {
                // Hash exists, check if peer is already in waiting list
                let request_state = state.get(&hash).unwrap();

                if request_state.waiting_list.contains(peer_node) {
                    // Peer already in waiting list, ignore
                    AdmitHashResult {
                        status: AdmitHashStatus::Ignore,
                        broadcast_request: false,
                        request_block: false,
                    }
                } else {
                    // Add peer to waiting list
                    let was_empty = request_state.waiting_list.is_empty();
                    Self::add_source_peer_to_request(&mut state, &hash, peer_node);

                    AdmitHashResult {
                        status: AdmitHashStatus::NewSourcePeerAddedToRequest,
                        broadcast_request: false,
                        // Request block if this is the first peer in waiting list
                        request_block: was_empty,
                    }
                }
            } else {
                // Hash exists but no peer provided, ignore
                AdmitHashResult {
                    status: AdmitHashStatus::Ignore,
                    broadcast_request: false,
                    request_block: false,
                }
            }
        };

        // Log the result
        match result.status {
            AdmitHashStatus::NewSourcePeerAddedToRequest => {
                if let Some(ref peer_node) = peer {
                    debug!(
                        "Adding {} to waiting list of {} request. Reason: {:?}",
                        peer_node.endpoint.host,
                        PrettyPrinter::build_string_bytes(&hash),
                        admit_hash_reason
                    );
                }
            }
            AdmitHashStatus::NewRequestAdded => {
                info!(
                    "Adding {} hash to RequestedBlocks because of {:?}",
                    PrettyPrinter::build_string_bytes(&hash),
                    admit_hash_reason
                );
            }
            AdmitHashStatus::Ignore => {
                // No logging for ignore case
            }
        }

        // Handle broadcasting and requesting
        if result.broadcast_request {
            self.transport
                .broadcast_has_block_request(&self.connections_cell, &self.conf, &hash)
                .await?;
            debug!(
                "Broadcasted HasBlockRequest for {}",
                PrettyPrinter::build_string_bytes(&hash)
            );
        }

        if result.request_block {
            if let Some(ref peer_node) = peer {
                self.transport
                    .request_for_block(&self.conf, peer_node, hash.clone())
                    .await?;
                debug!(
                    "Requested block {} from {}",
                    PrettyPrinter::build_string_bytes(&hash),
                    peer_node.endpoint.host
                );
            }
        }

        Ok(result)
    }

    pub async fn request_all(&self, age_threshold: Duration) -> Result<(), CasperError> {
        let current_time = Self::current_millis();

        // Get all hashes that need processing
        let hashes_to_process: Vec<BlockHash> = {
            let state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            debug!(
                "Running BlockRetriever maintenance ({} items unexpired).",
                state.keys().len()
            );

            state.keys().cloned().collect()
        };

        // Process each hash
        for hash in hashes_to_process {
            // Get the current state for this hash
            let (expired, _received, sent_to_casper, should_rerequest) = {
                let state = self.requested_blocks.lock().map_err(|_| {
                    CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
                })?;

                if let Some(requested) = state.get(&hash) {
                    let expired =
                        current_time - requested.timestamp > age_threshold.as_millis() as u64;
                    let received = requested.received;
                    let sent_to_casper = requested.in_casper_buffer;

                    if !received {
                        debug!(
                            "Casper loop: checking if should re-request {}. Received: {}.",
                            PrettyPrinter::build_string_bytes(&hash),
                            received
                        );
                    }

                    (expired, received, sent_to_casper, !received && expired)
                } else {
                    continue; // Hash was removed, skip
                }
            };

            // Try to re-request if needed
            if should_rerequest {
                self.try_rerequest(&hash).await?;
            }

            // Remove expired entries that have been sent to Casper
            if sent_to_casper && expired {
                let mut state = self.requested_blocks.lock().map_err(|_| {
                    CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
                })?;
                state.remove(&hash);
            }
        }

        Ok(())
    }

    /// Helper method to try re-requesting a block from the next peer in waiting list
    async fn try_rerequest(&self, hash: &BlockHash) -> Result<(), CasperError> {
        // Get next peer from waiting list and update state
        let next_peer_opt = {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            if let Some(request_state) = state.get_mut(hash) {
                if !request_state.waiting_list.is_empty() {
                    let next_peer = request_state.waiting_list.remove(0);
                    let now = Self::current_millis();

                    // Update timestamp and add to peers set
                    request_state.timestamp = now;
                    request_state.peers.insert(next_peer.clone());

                    Some((next_peer, request_state.waiting_list.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        };

        match next_peer_opt {
            Some((next_peer, remaining_waiting)) => {
                debug!(
                    "Trying {} to query for {} block. Remain waiting: {}.",
                    next_peer.endpoint.host,
                    PrettyPrinter::build_string_bytes(hash),
                    remaining_waiting
                        .iter()
                        .map(|p| p.endpoint.host.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                // Request block from the peer
                self.transport
                    .request_for_block(&self.conf, &next_peer, hash.clone())
                    .await?;

                // If this was the last peer in the waiting list, also broadcast HasBlockRequest
                if remaining_waiting.is_empty() {
                    debug!(
                        "Last peer in waiting list for block {}. Broadcasting HasBlockRequest.",
                        PrettyPrinter::build_string_bytes(hash)
                    );

                    // Broadcast request for the block
                    self.transport
                        .broadcast_has_block_request(&self.connections_cell, &self.conf, hash)
                        .await?;
                }
            }
            None => {
                // No more peers in waiting list - do nothing
                // The HasBlockRequest broadcast is now only sent when exhausting the last peer,
                // not when the waiting list is already empty
                debug!(
                    "No peers in waiting list for block {}. No action taken.",
                    PrettyPrinter::build_string_bytes(hash)
                );
            }
        }

        Ok(())
    }

    pub async fn ack_receive(&self, hash: BlockHash) -> Result<(), CasperError> {
        let now = Self::current_millis();

        // Lock the requested_blocks mutex and modify state atomically
        let result = {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            match state.get(&hash) {
                // There might be blocks that are not maintained by RequestedBlocks, e.g. fork-choice tips
                None => {
                    Self::add_new_request(&mut state, hash.clone(), now, true, None);
                    AckReceiveResult::AddedAsReceived
                }
                Some(requested) => {
                    // Make Casper loop aware that the block has been received
                    let mut updated_request = requested.clone();
                    updated_request.received = true;
                    state.insert(hash.clone(), updated_request);
                    AckReceiveResult::MarkedAsReceived
                }
            }
        };

        // Log based on the result
        match result {
            AckReceiveResult::AddedAsReceived => {
                info!(
                    "Block {} is not in RequestedBlocks. Adding and marking received.",
                    PrettyPrinter::build_string_bytes(&hash)
                );
            }
            AckReceiveResult::MarkedAsReceived => {
                info!(
                    "Block {} marked as received.",
                    PrettyPrinter::build_string_bytes(&hash)
                );
            }
        }

        Ok(())
    }

    pub async fn ack_in_casper(&self, hash: BlockHash) -> Result<(), CasperError> {
        // Check if block is already received
        let is_received = self.is_received(hash.clone()).await?;

        // If not received, acknowledge receipt first
        if !is_received {
            self.ack_receive(hash.clone()).await?;
        }

        // Mark as in Casper buffer
        let mut state = self.requested_blocks.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
        })?;

        if let Some(request_state) = state.get_mut(&hash) {
            request_state.in_casper_buffer = true;
        }

        Ok(())
    }

    pub async fn is_received(&self, hash: BlockHash) -> Result<bool, CasperError> {
        let state = self.requested_blocks.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
        })?;

        match state.get(&hash) {
            Some(request_state) => Ok(request_state.received),
            None => Ok(false),
        }
    }

    /// Get the number of peers in the waiting list for a specific hash
    /// Returns 0 if the hash is not in requested blocks
    pub async fn get_waiting_list_size(&self, hash: &BlockHash) -> Result<usize, CasperError> {
        let state = self.requested_blocks.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
        })?;

        match state.get(hash) {
            Some(request_state) => Ok(request_state.waiting_list.len()),
            None => Ok(0),
        }
    }

    /// Get the total number of hashes being tracked in requested blocks
    pub async fn get_requested_blocks_count(&self) -> Result<usize, CasperError> {
        let state = self.requested_blocks.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
        })?;

        Ok(state.len())
    }

    /// Test-only helper methods for setting up specific test scenarios
    pub async fn set_request_state_for_test(
        &self,
        hash: BlockHash,
        request_state: RequestState,
    ) -> Result<(), CasperError> {
        let mut state = self.requested_blocks.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
        })?;

        state.insert(hash, request_state);
        Ok(())
    }

    /// Test-only helper to get request state for verification
    pub async fn get_request_state_for_test(
        &self,
        hash: &BlockHash,
    ) -> Result<Option<RequestState>, CasperError> {
        let state = self.requested_blocks.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
        })?;

        Ok(state.get(hash).cloned())
    }

    /// Test-only helper to create a timed out timestamp
    pub fn create_timed_out_timestamp(timeout: std::time::Duration) -> u64 {
        let now = Self::current_millis();
        now.saturating_sub((2 * timeout.as_millis()) as u64)
    }
}
