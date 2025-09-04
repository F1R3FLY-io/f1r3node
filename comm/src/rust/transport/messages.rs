use crate::rust::peer_node::PeerNode;
use models::routing::Protocol;

/// Server message types for transport layer communication
#[derive(Debug, Clone)]
pub enum ServerMessage {
    /// Send a protocol message
    Send(Send),
    /// Stream message containing metadata about a streamed blob
    StreamMessage(StreamMessage),
}

/// Send message containing a protocol to be transmitted
#[derive(Debug, Clone)]
pub struct Send {
    pub msg: Protocol,
}

impl Send {
    /// Create a new Send message
    pub fn new(msg: Protocol) -> Self {
        Self { msg }
    }
}

/// Stream message containing metadata about a streamed blob
///
/// This represents the result of processing a streaming operation,
/// containing all necessary information to restore the original blob.
#[derive(Debug, Clone)]
pub struct StreamMessage {
    /// The peer that sent the stream
    pub sender: PeerNode,
    /// Type identifier of the streamed packet
    pub type_id: String,
    /// Cache key where the streamed data is stored
    pub key: String,
    /// Whether the streamed data is compressed
    pub compressed: bool,
    /// Expected content length (for validation)
    pub content_length: i32,
}

impl StreamMessage {
    /// Create a new StreamMessage
    pub fn new(
        sender: PeerNode,
        type_id: String,
        key: String,
        compressed: bool,
        content_length: i32,
    ) -> Self {
        Self {
            sender,
            type_id,
            key,
            compressed,
            content_length,
        }
    }
}
