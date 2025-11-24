// See comm/src/main/scala/coop/rchain/comm/transport/StreamHandler.scala

use crate::rust::{
    errors::CommError,
    peer_node::PeerNode,
    rp::protocol_helper,
    transport::{
        messages::StreamMessage,
        packet_ops::{PacketOps, StreamCache},
        transport_layer::Blob,
    },
};
use futures::StreamExt;
use models::routing::{chunk::Content, Chunk, ChunkData, ChunkHeader};
use shared::rust::shared::compression::Compression;
use tokio_stream::Stream;
use tracing;

/// Type alias for circuit breaker function
/// Takes a Streamed state and returns a Circuit decision
pub type CircuitBreaker = fn(&Streamed) -> Circuit;

/// Header information for a streaming operation
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    /// The peer that sent the stream
    pub sender: PeerNode,
    /// Type identifier of the streamed packet
    pub type_id: String,
    /// Expected content length
    pub content_length: i32,
    /// Network ID for validation
    pub network_id: String,
    /// Whether the content is compressed
    pub compressed: bool,
}

impl Header {
    /// Create a new Header
    pub fn new(
        sender: PeerNode,
        type_id: String,
        content_length: i32,
        network_id: String,
        compressed: bool,
    ) -> Self {
        Self {
            sender,
            type_id,
            content_length,
            network_id,
            compressed,
        }
    }
}

/// Circuit breaker state for stream processing
#[derive(Debug, Clone, PartialEq)]
pub enum Circuit {
    /// Circuit is open (broken) due to an error
    Opened { error: StreamError },
    /// Circuit is closed (normal operation)
    Closed,
}

impl Circuit {
    /// Check if the circuit is broken
    pub fn broken(&self) -> bool {
        matches!(self, Circuit::Opened { .. })
    }

    /// Create an opened circuit with an error
    pub fn opened(error: StreamError) -> Self {
        Circuit::Opened { error }
    }

    /// Create a closed circuit
    pub fn closed() -> Self {
        Circuit::Closed
    }
}

/// Stream error types
#[derive(Debug, Clone, PartialEq)]
pub enum StreamError {
    /// Wrong network ID detected
    WrongNetworkId,
    /// Maximum size reached
    MaxSizeReached,
    /// Incomplete message received
    NotFullMessage { streamed: String },
    /// Unexpected error occurred
    Unexpected { error: String },
}

impl StreamError {
    /// Get error message string
    pub fn message(&self) -> String {
        match self {
            StreamError::WrongNetworkId => {
                "Could not receive stream! Wrong network id.".to_string()
            }
            StreamError::MaxSizeReached => "Max message size was reached.".to_string(),
            StreamError::NotFullMessage { streamed } => {
                format!(
                    "Received not full stream message, will not process. {}",
                    streamed
                )
            }
            StreamError::Unexpected { error } => {
                format!("Could not receive stream! {}", error)
            }
        }
    }

    /// Create a WrongNetworkId error
    pub fn wrong_network_id() -> Self {
        StreamError::WrongNetworkId
    }

    /// Create a MaxSizeReached error (circuit opened)
    pub fn circuit_opened() -> Self {
        StreamError::MaxSizeReached
    }

    /// Create a NotFullMessage error
    pub fn not_full_message(streamed: String) -> Self {
        StreamError::NotFullMessage { streamed }
    }

    /// Create an Unexpected error
    pub fn unexpected(error: String) -> Self {
        StreamError::Unexpected { error }
    }
}

/// State of an ongoing streaming operation
#[derive(Debug, Clone)]
pub struct Streamed {
    /// Header information (if received)
    pub header: Option<Header>,
    /// Number of bytes read so far
    pub read_so_far: u64,
    /// Circuit breaker state
    pub circuit: Circuit,
    /// Cache key for the stream
    pub key: String,
}

impl Streamed {
    /// Create a new Streamed with default values
    pub fn new(key: String) -> Self {
        Self {
            header: None,
            read_so_far: 0,
            circuit: Circuit::Closed,
            key,
        }
    }

    /// Update the header
    pub fn with_header(mut self, header: Header) -> Self {
        self.header = Some(header);
        self
    }

    /// Update the read count
    pub fn with_read_so_far(mut self, read_so_far: u64) -> Self {
        self.read_so_far = read_so_far;
        self
    }

    /// Update the circuit state
    pub fn with_circuit(mut self, circuit: Circuit) -> Self {
        self.circuit = circuit;
        self
    }
}

impl std::fmt::Display for Streamed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Streamed(header: {:?}, read_so_far: {}, circuit_broken: {}, key: {})",
            self.header.as_ref().map(|h| &h.type_id),
            self.read_so_far,
            self.circuit.broken(),
            self.key
        )
    }
}

/// StreamHandler provides functionality for processing streaming data
pub struct StreamHandler;

impl StreamHandler {
    /// Initialize a new streaming operation
    /// Creates a cache entry with "packet_send/" prefix and returns a new Streamed instance.
    pub fn init(cache: &StreamCache) -> Result<Streamed, CommError> {
        let key = PacketOps::create_cache_entry("packet_send", cache)?;
        Ok(Streamed::new(key))
    }

    /// Handle a stream of chunks with proper resource management
    ///
    /// This method processes a stream of chunks using the circuit breaker pattern
    /// and provides proper cleanup in all scenarios (success, failure, errors).
    pub async fn handle_stream<S>(
        stream: S,
        circuit_breaker: CircuitBreaker,
        cache: &StreamCache,
    ) -> Result<StreamMessage, StreamError>
    where
        S: Stream<Item = Chunk> + Unpin,
    {
        // Initialize streaming state
        let init_stmd = match Self::init(cache) {
            Ok(stmd) => stmd,
            Err(e) => {
                tracing::error!("Failed to initialize stream: {}", e);
                return Err(StreamError::unexpected(format!(
                    "Initialization failed: {}",
                    e
                )));
            }
        };

        // Process the stream with proper cleanup
        let result =
            Self::process_stream_with_cleanup(init_stmd, stream, circuit_breaker, cache).await;

        // Handle the final result
        match result {
            Ok(stream_message) => {
                tracing::debug!("Stream collected.");
                Ok(stream_message)
            }
            Err(error) => {
                tracing::warn!("Failed collecting stream.");
                Err(error)
            }
        }
    }

    /// Process stream with automatic cleanup (internal helper)
    async fn process_stream_with_cleanup<S>(
        init_stmd: Streamed,
        stream: S,
        circuit_breaker: CircuitBreaker,
        cache: &StreamCache,
    ) -> Result<StreamMessage, StreamError>
    where
        S: Stream<Item = Chunk> + Unpin,
    {
        let key = init_stmd.key.clone(); // Store key for cleanup

        // Process the stream
        let collect_result = Self::collect(init_stmd, stream, circuit_breaker, cache).await;

        // Handle collection result and cleanup
        match collect_result {
            Ok(streamed) => {
                // Successfully collected - convert to result
                match Self::to_result(&streamed) {
                    Ok(stream_message) => {
                        // Success - no cleanup needed for cache (handled by caller)
                        Ok(stream_message)
                    }
                    Err(error) => {
                        // Failed to convert to result - cleanup and return error
                        tracing::warn!("Failed collecting stream.");
                        cache.remove(&key);
                        Err(error)
                    }
                }
            }
            Err(error) => {
                // Failed during collection - cleanup and return error
                tracing::warn!("Failed collecting stream.");
                cache.remove(&key);
                Err(error)
            }
        }
    }

    /// Collect and process chunks from a stream
    ///
    /// Processes each chunk in the stream, building up the Streamed state:
    /// - Header chunks: Creates Header and updates Streamed state
    /// - Data chunks: Appends data to cache and updates read count
    /// - Unknown chunks: Sets an error
    pub async fn collect<S>(
        init: Streamed,
        mut stream: S,
        circuit_breaker: CircuitBreaker,
        cache: &StreamCache,
    ) -> Result<Streamed, StreamError>
    where
        S: Stream<Item = Chunk> + Unpin,
    {
        let mut current_state = init;

        // Process each chunk in the stream
        while let Some(chunk) = stream.next().await {
            // Process the chunk and update state
            current_state = match Self::process_chunk(current_state, chunk, cache)? {
                ChunkProcessResult::Continue(state) => state,
                ChunkProcessResult::Error(error) => {
                    return Err(error);
                }
            };

            // Apply circuit breaker
            let circuit = circuit_breaker(&current_state);
            current_state = current_state.with_circuit(circuit.clone());

            // If circuit is broken, stop processing
            if circuit.broken() {
                if let Circuit::Opened { error } = circuit {
                    return Err(error);
                }
            }
        }

        // Check final state - if circuit is opened, return the error
        match &current_state.circuit {
            Circuit::Opened { error } => Err(error.clone()),
            Circuit::Closed => Ok(current_state),
        }
    }

    /// Convert a Streamed state to a StreamMessage result
    ///
    /// Validates the streamed state and creates a StreamMessage if valid:
    /// - Must have a valid header
    /// - For uncompressed content, read_so_far must equal content_length
    pub fn to_result(streamed: &Streamed) -> Result<StreamMessage, StreamError> {
        // Create the "not full" error message for reuse
        let not_full_error = StreamError::not_full_message(streamed.to_string());

        match &streamed.header {
            Some(header) => {
                // Extract header fields
                let Header {
                    sender,
                    type_id: packet_type,
                    content_length,
                    compressed,
                    ..
                } = header;

                // Create the StreamMessage
                let stream_message = StreamMessage::new(
                    sender.clone(),
                    packet_type.clone(),
                    streamed.key.clone(),
                    *compressed,
                    *content_length,
                );

                // Validate content length for uncompressed data
                if !compressed && streamed.read_so_far != *content_length as u64 {
                    Err(not_full_error)
                } else {
                    Ok(stream_message)
                }
            }
            None => {
                // No header present - return error
                Err(not_full_error)
            }
        }
    }

    /// Restore a blob from cache using a StreamMessage
    ///
    /// Retrieves the cached data, decompresses if necessary, and creates a Blob.
    /// Cleans up the cache entry after processing.
    pub async fn restore(msg: &StreamMessage, cache: &StreamCache) -> Result<Blob, CommError> {
        // Read data from cache
        let content = match cache.get(&msg.key) {
            Some(entry) => entry.value().clone(),
            None => {
                let error = format!("Could not read streamed data from cache (key: {})", msg.key);
                tracing::error!("{}", error);
                return Err(CommError::InternalCommunicationError(error));
            }
        };

        // Decompress content if necessary
        let decompressed_content =
            match Self::decompress_content(content, msg.compressed, msg.content_length).await {
                Ok(data) => data,
                Err(e) => {
                    let error = format!("Could not decompress data (key: {}): {}", msg.key, e);
                    tracing::error!("{}", error);
                    return Err(e);
                }
            };

        // Create blob using protocol helper
        let blob = protocol_helper::blob(&msg.sender, &msg.type_id, &decompressed_content);

        // Clean up cache entry
        cache.remove(&msg.key);

        Ok(blob)
    }

    /// Decompress content if compressed, otherwise return as-is
    async fn decompress_content(
        raw: Vec<u8>,
        compressed: bool,
        content_length: i32,
    ) -> Result<Vec<u8>, CommError> {
        if compressed {
            // Decompress the data
            match Compression::decompress(&raw, content_length as usize) {
                Some(decompressed) => Ok(decompressed),
                None => {
                    let error = "Could not decompress data".to_string();
                    Err(CommError::InternalCommunicationError(error))
                }
            }
        } else {
            // Return raw data as-is
            Ok(raw)
        }
    }

    /// Process a single chunk and update the Streamed state
    fn process_chunk(
        streamed: Streamed,
        chunk: Chunk,
        cache: &StreamCache,
    ) -> Result<ChunkProcessResult, StreamError> {
        match chunk.content {
            Some(Content::Header(chunk_header)) => {
                // Process header chunk
                let ChunkHeader {
                    sender,
                    type_id,
                    compressed,
                    content_length,
                    network_id,
                } = chunk_header;

                // Convert sender Node to PeerNode
                let peer_sender = match sender {
                    Some(node) => protocol_helper::to_peer_node(&node),
                    None => {
                        return Ok(ChunkProcessResult::Error(StreamError::not_full_message(
                            "Header chunk missing sender".to_string(),
                        )));
                    }
                };

                // Create header
                let header =
                    Header::new(peer_sender, type_id, content_length, network_id, compressed);

                // Update streamed state with header
                let new_streamed = streamed.with_header(header);
                Ok(ChunkProcessResult::Continue(new_streamed))
            }
            Some(Content::Data(chunk_data)) => {
                // Process data chunk
                let ChunkData { content_data } = chunk_data;
                let received_bytes = content_data.to_vec();

                // Write data to cache
                let mut existing_data = cache
                    .get(&streamed.key)
                    .map(|entry| entry.value().clone())
                    .unwrap_or_else(Vec::new);

                existing_data.extend_from_slice(&received_bytes);
                cache.insert(streamed.key.clone(), existing_data);

                // Update read count
                let new_read_so_far = streamed.read_so_far + received_bytes.len() as u64;
                let new_streamed = streamed.with_read_so_far(new_read_so_far);

                Ok(ChunkProcessResult::Continue(new_streamed))
            }
            None => {
                // Unknown/invalid chunk type
                Ok(ChunkProcessResult::Error(StreamError::not_full_message(
                    "Not all data received".to_string(),
                )))
            }
        }
    }
}

/// Result of processing a single chunk
enum ChunkProcessResult {
    /// Continue processing with the updated state
    Continue(Streamed),
    /// Stop processing due to an error
    Error(StreamError),
}
