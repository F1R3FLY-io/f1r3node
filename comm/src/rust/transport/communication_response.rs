// See comm/src/main/scala/coop/rchain/comm/transport/CommunicationResponse.scala

use crate::rust::errors::CommError;
use models::routing::Protocol;

/// Communication response types for handling transport layer messages
#[derive(Debug, Clone)]
pub enum CommunicationResponse {
    /// Response containing a protocol message
    HandledWithMessage { pm: Protocol },
    /// Response indicating successful handling without a message
    HandledWitoutMessage,
    /// Response indicating the message was not handled, with an error
    NotHandled { error: CommError },
}

impl CommunicationResponse {
    /// Create a response with a protocol message
    pub fn handled_with_message(protocol: Protocol) -> CommunicationResponse {
        CommunicationResponse::HandledWithMessage { pm: protocol }
    }

    /// Create a response indicating successful handling without a message
    pub fn handled_without_message() -> CommunicationResponse {
        CommunicationResponse::HandledWitoutMessage
    }

    /// Create a response indicating the message was not handled
    pub fn not_handled(error: CommError) -> CommunicationResponse {
        CommunicationResponse::NotHandled { error }
    }
}
