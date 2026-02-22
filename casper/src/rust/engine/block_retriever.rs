// See casper/src/main/scala/coop/rchain/casper/engine/BlockRetriever.scala

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use comm::rust::{
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::{block_hash::BlockHash, casper::pretty_printer::PrettyPrinter};
use tracing::{debug, info};

use crate::rust::errors::CasperError;
use crate::rust::metrics_constants::{
    BLOCK_RETRIEVER_BROADCAST_TRACKING_SIZE_METRIC,
    BLOCK_RETRIEVER_DEP_RECOVERY_TRACKING_SIZE_METRIC,
    BLOCK_RETRIEVER_PEERS_TOTAL_SIZE_METRIC,
    BLOCK_RETRIEVER_REQUESTED_BLOCKS_SIZE_METRIC,
    BLOCK_RETRIEVER_WAITING_LIST_TOTAL_SIZE_METRIC,
    BLOCK_DOWNLOAD_END_TO_END_TIME_METRIC, BLOCK_REQUESTS_RETRIES_METRIC,
    BLOCK_REQUESTS_RETRY_ACTION_METRIC,
    BLOCK_REQUESTS_STALE_EVICTIONS_METRIC,
    BLOCK_REQUESTS_TOTAL_METRIC, BLOCK_RETRIEVER_METRICS_SOURCE,
};

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
    pub initial_timestamp: u64,
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
    dependency_recovery_last_request: Arc<Mutex<HashMap<BlockHash, u64>>>,
    broadcast_retry_last_request: Arc<Mutex<HashMap<BlockHash, u64>>>,
    peer_requery_last_request: Arc<Mutex<HashMap<BlockHash, u64>>>,
    transport: Arc<T>,
    connections_cell: ConnectionsCell,
    conf: RPConf,
}

impl<T: TransportLayer + Send + Sync> BlockRetriever<T> {
    const MAX_REQUESTED_BLOCKS_ENTRIES: usize = 2048;
    const MAX_WAITING_LIST_PER_HASH: usize = 64;
    fn peer_requery_retry_cooldown_ms() -> u64 {
        static VALUE: OnceLock<u64> = OnceLock::new();
        *VALUE.get_or_init(|| {
            std::env::var("F1R3_BLOCK_RETRIEVER_PEER_REQUERY_COOLDOWN_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(1000)
        })
    }

    fn update_aux_tracking_metrics(&self) -> Result<(), CasperError> {
        let (requested_size, waiting_list_total_size, peers_total_size) = {
            let state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;
            let waiting_total = state.values().map(|r| r.waiting_list.len()).sum::<usize>();
            let peers_total = state.values().map(|r| r.peers.len()).sum::<usize>();
            (state.len(), waiting_total, peers_total)
        };
        let dep_size = {
            let last_requests = self.dependency_recovery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire dependency_recovery_last_request lock".to_string(),
                )
            })?;
            last_requests.len()
        };
        let broadcast_size = {
            let broadcast_last = self.broadcast_retry_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire broadcast_retry_last_request lock".to_string(),
                )
            })?;
            broadcast_last.len()
        };
        let peer_requery_size = {
            let peer_requery_last = self.peer_requery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire peer_requery_last_request lock".to_string(),
                )
            })?;
            peer_requery_last.len()
        };

        metrics::gauge!(BLOCK_RETRIEVER_REQUESTED_BLOCKS_SIZE_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
            .set(requested_size as f64);
        metrics::gauge!(BLOCK_RETRIEVER_WAITING_LIST_TOTAL_SIZE_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
            .set(waiting_list_total_size as f64);
        metrics::gauge!(BLOCK_RETRIEVER_PEERS_TOTAL_SIZE_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
            .set(peers_total_size as f64);
        metrics::gauge!(BLOCK_RETRIEVER_DEP_RECOVERY_TRACKING_SIZE_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
            .set(dep_size as f64);
        metrics::gauge!(BLOCK_RETRIEVER_BROADCAST_TRACKING_SIZE_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
            .set(broadcast_size as f64);
        // Reuse broadcast-tracking gauge as a proxy to include the requery cooldown map pressure.
        metrics::gauge!(BLOCK_RETRIEVER_BROADCAST_TRACKING_SIZE_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "kind" => "peer_requery")
            .set(peer_requery_size as f64);
        Ok(())
    }

    fn cleanup_aux_tracking_for_hash(&self, hash: &BlockHash) -> Result<(), CasperError> {
        {
            let mut last_requests = self.dependency_recovery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire dependency_recovery_last_request lock".to_string(),
                )
            })?;
            last_requests.remove(hash);
        }
        {
            let mut broadcast_last = self.broadcast_retry_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire broadcast_retry_last_request lock".to_string(),
                )
            })?;
            broadcast_last.remove(hash);
        }
        {
            let mut peer_requery_last = self.peer_requery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire peer_requery_last_request lock".to_string(),
                )
            })?;
            peer_requery_last.remove(hash);
        }
        Ok(())
    }

    fn sweep_orphaned_aux_tracking(&self) -> Result<(), CasperError> {
        let active_hashes: HashSet<BlockHash> = {
            let state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;
            state.keys().cloned().collect()
        };

        {
            let mut last_requests = self.dependency_recovery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire dependency_recovery_last_request lock".to_string(),
                )
            })?;
            last_requests.retain(|hash, _| active_hashes.contains(hash));
        }

        {
            let mut broadcast_last = self.broadcast_retry_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire broadcast_retry_last_request lock".to_string(),
                )
            })?;
            broadcast_last.retain(|hash, _| active_hashes.contains(hash));
        }
        {
            let mut peer_requery_last = self.peer_requery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire peer_requery_last_request lock".to_string(),
                )
            })?;
            peer_requery_last.retain(|hash, _| active_hashes.contains(hash));
        }

        Ok(())
    }

    fn enforce_requested_blocks_bound(&self) -> Result<usize, CasperError> {
        let hashes_to_evict: Vec<BlockHash> = {
            let state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            if state.len() <= Self::MAX_REQUESTED_BLOCKS_ENTRIES {
                Vec::new()
            } else {
                let mut candidates: Vec<(BlockHash, u64, bool)> = state
                    .iter()
                    .map(|(hash, req)| {
                        (
                            hash.clone(),
                            req.initial_timestamp,
                            !req.received && !req.in_casper_buffer,
                        )
                    })
                    .collect();

                // Prefer evicting oldest unresolved/non-buffered requests first.
                candidates.sort_by_key(|(_, ts, preferred)| (!*preferred, *ts));
                let to_remove = state.len().saturating_sub(Self::MAX_REQUESTED_BLOCKS_ENTRIES);
                candidates
                    .into_iter()
                    .take(to_remove)
                    .map(|(hash, _, _)| hash)
                    .collect()
            }
        };

        if hashes_to_evict.is_empty() {
            return Ok(0);
        }

        let evicted_count = {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;
            let mut count = 0usize;
            for hash in &hashes_to_evict {
                if state.remove(hash).is_some() {
                    count += 1;
                }
            }
            count
        };

        for hash in &hashes_to_evict {
            self.cleanup_aux_tracking_for_hash(hash)?;
        }

        if evicted_count > 0 {
            metrics::counter!(BLOCK_REQUESTS_STALE_EVICTIONS_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
                .increment(evicted_count as u64);
            debug!(
                "Evicted {} requested block entries to enforce max bound {}.",
                evicted_count,
                Self::MAX_REQUESTED_BLOCKS_ENTRIES
            );
        }

        Ok(evicted_count)
    }

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
            dependency_recovery_last_request: Arc::new(Mutex::new(HashMap::new())),
            broadcast_retry_last_request: Arc::new(Mutex::new(HashMap::new())),
            peer_requery_last_request: Arc::new(Mutex::new(HashMap::new())),
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
                    initial_timestamp: now,
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
                metrics::counter!(BLOCK_REQUESTS_TOTAL_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE).increment(1);
                Self::add_new_request(&mut state, hash.clone(), now, false, peer.as_ref());
                AdmitHashResult {
                    status: AdmitHashStatus::NewRequestAdded,
                    broadcast_request: peer.is_none(),
                    request_block: peer.is_some(),
                }
            } else if let Some(ref peer_node) = peer {
                // Hash exists, check if peer is already in waiting list
                let request_state = state.get(&hash).unwrap();

                let already_waiting = request_state.waiting_list.contains(peer_node);
                let already_queried = request_state.peers.contains(peer_node);
                let waiting_list_full =
                    request_state.waiting_list.len() >= Self::MAX_WAITING_LIST_PER_HASH;
                if already_waiting || already_queried {
                    // Peer is already queued or already queried for this hash, ignore.
                    AdmitHashResult {
                        status: AdmitHashStatus::Ignore,
                        broadcast_request: false,
                        request_block: false,
                    }
                } else if waiting_list_full {
                    debug!(
                        "Ignoring additional source peer for {}: waiting list already at cap {}.",
                        PrettyPrinter::build_string_bytes(&hash),
                        Self::MAX_WAITING_LIST_PER_HASH
                    );
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
        const STALE_REQUEST_LIFETIME_MULTIPLIER: u64 = 6;
        let stale_request_lifetime_ms =
            (age_threshold.as_millis() as u64).saturating_mul(STALE_REQUEST_LIFETIME_MULTIPLIER);

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
            let (expired, received, sent_to_casper, should_rerequest, should_evict_stale) = {
                let state = self.requested_blocks.lock().map_err(|_| {
                    CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
                })?;

                if let Some(requested) = state.get(&hash) {
                    let expired =
                        current_time - requested.timestamp > age_threshold.as_millis() as u64;
                    let received = requested.received;
                    let sent_to_casper = requested.in_casper_buffer;
                    let stale_lifetime = current_time.saturating_sub(requested.initial_timestamp);
                    let should_evict_stale =
                        !received && stale_lifetime > stale_request_lifetime_ms;

                    if !received {
                        debug!(
                            "Casper loop: checking if should re-request {}. Received: {}.",
                            PrettyPrinter::build_string_bytes(&hash),
                            received
                        );
                    }

                    (
                        expired,
                        received,
                        sent_to_casper,
                        !received && expired,
                        should_evict_stale,
                    )
                } else {
                    continue; // Hash was removed, skip
                }
            };

            // Try to re-request if needed
            if should_rerequest {
                let did_retry = self.try_rerequest(&hash).await?;
                if did_retry {
                    metrics::counter!(BLOCK_REQUESTS_RETRIES_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE).increment(1);
                }
            }

            // Remove expired entries that are already received.
            // Also evict very stale unresolved requests to prevent unbounded growth
            // when dependencies never resolve.
            if (received && expired) || should_evict_stale {
                let mut state = self.requested_blocks.lock().map_err(|_| {
                    CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
                })?;
                if state.remove(&hash).is_some() {
                    drop(state);
                    self.cleanup_aux_tracking_for_hash(&hash)?;
                    if should_evict_stale && !received {
                        metrics::counter!(BLOCK_REQUESTS_STALE_EVICTIONS_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE).increment(1);
                        debug!(
                            "Evicting stale unresolved block request {} after lifetime threshold.",
                            PrettyPrinter::build_string_bytes(&hash)
                        );
                    }
                    if received && expired && !sent_to_casper {
                        debug!(
                            "Evicting received/non-buffered block request {} after timeout.",
                            PrettyPrinter::build_string_bytes(&hash)
                        );
                    }
                }
            }
        }

        // Keep cooldown-tracking maps bounded to active requested hashes.
        self.enforce_requested_blocks_bound()?;
        self.sweep_orphaned_aux_tracking()?;
        self.update_aux_tracking_metrics()?;

        Ok(())
    }

    /// Force dependency recovery by reopening request state and rebroadcasting HasBlockRequest.
    /// This is used when the processor detects buffered dependency deadlocks.
    pub async fn recover_dependency(&self, hash: BlockHash) -> Result<(), CasperError> {
        let now = Self::current_millis();
        const DEPENDENCY_RECOVERY_REREQUEST_COOLDOWN_MS: u64 = 1000;

        {
            let mut last_requests = self.dependency_recovery_last_request.lock().map_err(|_| {
                CasperError::RuntimeError(
                    "Failed to acquire dependency_recovery_last_request lock".to_string(),
                )
            })?;

            if let Some(last_ts) = last_requests.get(&hash) {
                if now.saturating_sub(*last_ts) < DEPENDENCY_RECOVERY_REREQUEST_COOLDOWN_MS {
                    debug!(
                        "Skipping dependency recovery re-request for {} (cooldown {}ms)",
                        PrettyPrinter::build_string_bytes(&hash),
                        DEPENDENCY_RECOVERY_REREQUEST_COOLDOWN_MS
                    );
                    return Ok(());
                }
            }

            last_requests.insert(hash.clone(), now);
        }

        {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            match state.get_mut(&hash) {
                Some(request_state) => {
                    request_state.received = false;
                    request_state.in_casper_buffer = false;
                    request_state.timestamp = now;
                }
                None => {
                    Self::add_new_request(&mut state, hash.clone(), now, false, None);
                }
            }
        }

        self.transport
            .broadcast_has_block_request(&self.connections_cell, &self.conf, &hash)
            .await?;

        info!(
            "Recovery re-request issued for dependency {}",
            PrettyPrinter::build_string_bytes(&hash)
        );
        self.update_aux_tracking_metrics()?;

        Ok(())
    }

    /// Helper method to try re-requesting a block from the next peer in waiting list
    async fn try_rerequest(&self, hash: &BlockHash) -> Result<bool, CasperError> {
        enum RerequestAction {
            RequestPeer(PeerNode, Vec<PeerNode>),
            RequestKnownPeer(PeerNode),
            BroadcastOnly,
            None,
        }

        // Determine retry action and update request timestamp to enforce age-threshold backoff.
        let action = {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            if let Some(request_state) = state.get_mut(hash) {
                let now = Self::current_millis();
                request_state.timestamp = now;

                if !request_state.waiting_list.is_empty() {
                    let next_peer = request_state.waiting_list.remove(0);
                    request_state.peers.insert(next_peer.clone());

                    RerequestAction::RequestPeer(next_peer, request_state.waiting_list.clone())
                } else {
                    if let Some(known_peer) = request_state.peers.iter().next().cloned() {
                        RerequestAction::RequestKnownPeer(known_peer)
                    } else {
                        RerequestAction::BroadcastOnly
                    }
                }
            } else {
                RerequestAction::None
            }
        };

        match action {
            RerequestAction::RequestPeer(next_peer, remaining_waiting) => {
                metrics::counter!(BLOCK_REQUESTS_RETRY_ACTION_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "action" => "peer_request").increment(1);
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
                Ok(true)
            }
            RerequestAction::BroadcastOnly => {
                metrics::counter!(BLOCK_REQUESTS_RETRY_ACTION_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "action" => "broadcast_only").increment(1);
                const BROADCAST_ONLY_RETRY_COOLDOWN_MS: u64 = 1000;
                let now = Self::current_millis();
                let is_suppressed = {
                    let mut state = self.broadcast_retry_last_request.lock().map_err(|_| {
                        CasperError::RuntimeError(
                            "Failed to acquire broadcast_retry_last_request lock".to_string(),
                        )
                    })?;
                    if let Some(last) = state.get(hash) {
                        if now.saturating_sub(*last) < BROADCAST_ONLY_RETRY_COOLDOWN_MS {
                            true
                        } else {
                            state.insert(hash.clone(), now);
                            false
                        }
                    } else {
                        state.insert(hash.clone(), now);
                        false
                    }
                };

                if is_suppressed {
                    metrics::counter!(BLOCK_REQUESTS_RETRY_ACTION_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "action" => "broadcast_suppressed").increment(1);
                    debug!(
                        "Suppressing HasBlockRequest broadcast for {} due to cooldown {}ms.",
                        PrettyPrinter::build_string_bytes(hash),
                        BROADCAST_ONLY_RETRY_COOLDOWN_MS
                    );
                    return Ok(false);
                }

                debug!(
                    "No peers in waiting list for block {}. Broadcasting HasBlockRequest.",
                    PrettyPrinter::build_string_bytes(hash)
                );
                self.transport
                    .broadcast_has_block_request(&self.connections_cell, &self.conf, hash)
                    .await?;
                Ok(true)
            }
            RerequestAction::RequestKnownPeer(known_peer) => {
                let peer_requery_retry_cooldown_ms = Self::peer_requery_retry_cooldown_ms();
                let now = Self::current_millis();
                let is_suppressed = {
                    let mut state = self.peer_requery_last_request.lock().map_err(|_| {
                        CasperError::RuntimeError(
                            "Failed to acquire peer_requery_last_request lock".to_string(),
                        )
                    })?;
                    if let Some(last) = state.get(hash) {
                        if now.saturating_sub(*last) < peer_requery_retry_cooldown_ms {
                            true
                        } else {
                            state.insert(hash.clone(), now);
                            false
                        }
                    } else {
                        state.insert(hash.clone(), now);
                        false
                    }
                };

                if is_suppressed {
                    metrics::counter!(BLOCK_REQUESTS_RETRY_ACTION_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "action" => "peer_requery_suppressed").increment(1);
                    debug!(
                        "Suppressing peer requery for {} due to cooldown {}ms.",
                        PrettyPrinter::build_string_bytes(hash),
                        peer_requery_retry_cooldown_ms
                    );
                    return Ok(false);
                }

                metrics::counter!(BLOCK_REQUESTS_RETRY_ACTION_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "action" => "peer_requery").increment(1);
                debug!(
                    "Re-querying known peer {} for block {}.",
                    known_peer.endpoint.host,
                    PrettyPrinter::build_string_bytes(hash)
                );
                self.transport
                    .request_for_block(&self.conf, &known_peer, hash.clone())
                    .await?;
                Ok(true)
            }
            RerequestAction::None => {
                metrics::counter!(BLOCK_REQUESTS_RETRY_ACTION_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE, "action" => "none").increment(1);
                Ok(false)
            }
        }
    }

    pub async fn ack_receive(&self, hash: BlockHash) -> Result<(), CasperError> {
        let now = Self::current_millis();

        // Lock the requested_blocks mutex and modify state atomically
        let (result, request_timestamp) = {
            let mut state = self.requested_blocks.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire requested_blocks lock".to_string())
            })?;

            match state.get(&hash) {
                // There might be blocks that are not maintained by RequestedBlocks, e.g. fork-choice tips
                None => {
                    Self::add_new_request(&mut state, hash.clone(), now, true, None);
                    (AckReceiveResult::AddedAsReceived, None)
                }
                Some(requested) => {
                    let initial_timestamp = requested.initial_timestamp;
                    // Make Casper loop aware that the block has been received
                    let mut updated_request = requested.clone();
                    updated_request.received = true;
                    state.insert(hash.clone(), updated_request);
                    (AckReceiveResult::MarkedAsReceived, Some(initial_timestamp))
                }
            }
        };

        // Record block download end-to-end time if we have the original request timestamp
        if let Some(timestamp) = request_timestamp {
            let download_time_ms = now.saturating_sub(timestamp);
            let download_time_seconds = download_time_ms as f64 / 1000.0;
            metrics::histogram!(BLOCK_DOWNLOAD_END_TO_END_TIME_METRIC, "source" => BLOCK_RETRIEVER_METRICS_SOURCE)
                .record(download_time_seconds);
        }

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
