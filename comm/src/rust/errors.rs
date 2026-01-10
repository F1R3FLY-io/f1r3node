// See comm/src/main/scala/coop/rchain/comm/errors.scala

use std::fmt;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CommError {
    UnknownCommError(String),
    DatagramSizeError(usize),
    DatagramFramingError(String),
    DatagramException(String),
    HeaderNotAvailable,
    ProtocolException(String),
    UnknownProtocolError(String),
    PublicKeyNotAvailable(String),
    ParseError(String),
    EncryptionHandshakeIncorrectlySigned,
    BootstrapNotProvided,
    PeerNodeNotFound(String),
    PeerUnavailable(String),
    WrongNetwork(String, String),
    MessageToLarge(String),
    MalformedMessage(String),
    CouldNotConnectToBootstrap,
    InternalCommunicationError(String),
    TimeOut,
    UpstreamNotAvailable,
    UnexpectedMessage(String),
    SenderNotAvailable,
    PongNotReceivedForPing(String),
    UnableToStorePacket(String, String),
    UnableToRestorePacket(String, String),
    ConfigError(String),
    CasperError(String),
}

impl fmt::Display for CommError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CommError::PeerUnavailable(_) => write!(f, "Peer is currently unavailable"),
            CommError::MessageToLarge(p) => {
                write!(f, "Message rejected by peer {} because it was too large", p)
            }
            CommError::PongNotReceivedForPing(_) => write!(
                f,
                "Peer is behind a firewall and can't be accessed from outside"
            ),
            CommError::CouldNotConnectToBootstrap => {
                write!(f, "Node could not connect to bootstrap node")
            }
            CommError::TimeOut => write!(f, "Timeout"),
            CommError::InternalCommunicationError(msg) => {
                write!(f, "Internal communication error. {}", msg)
            }
            CommError::UnknownProtocolError(msg) => write!(f, "Unknown protocol error. {}", msg),
            CommError::UnableToStorePacket(p, er) => {
                write!(f, "Could not serialize packet {}. Error message: {}", p, er)
            }
            CommError::UnableToRestorePacket(p, er) => write!(
                f,
                "Could not deserialize packet {}. Error message: {}",
                p, er
            ),
            CommError::ProtocolException(msg) => write!(f, "Protocol error. {}", msg),
            CommError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CommError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            CommError::CasperError(msg) => write!(f, "Casper error: {}", msg),
            _ => write!(f, "{:?}", self),
        }
    }
}

// Helper functions matching Scala's API
pub fn unknown_comm_error(msg: String) -> CommError {
    CommError::UnknownCommError(msg)
}

pub fn unknown_protocol(msg: String) -> CommError {
    CommError::UnknownProtocolError(msg)
}

pub fn parse_error(msg: String) -> CommError {
    CommError::ParseError(msg)
}

pub fn protocol_exception(msg: String) -> CommError {
    CommError::ProtocolException(msg)
}

pub fn header_not_available() -> CommError {
    CommError::HeaderNotAvailable
}

pub fn peer_node_not_found(peer: String) -> CommError {
    CommError::PeerNodeNotFound(peer)
}

pub fn peer_unavailable(peer: String) -> CommError {
    CommError::PeerUnavailable(peer)
}

pub fn wrong_network(peer: String, msg: String) -> CommError {
    CommError::WrongNetwork(peer, msg)
}

pub fn message_too_large(peer: String) -> CommError {
    CommError::MessageToLarge(peer)
}

pub fn public_key_not_available(peer: String) -> CommError {
    CommError::PublicKeyNotAvailable(peer)
}

pub fn could_not_connect_to_bootstrap() -> CommError {
    CommError::CouldNotConnectToBootstrap
}

pub fn internal_communication_error(msg: String) -> CommError {
    CommError::InternalCommunicationError(msg)
}

pub fn malformed_message(msg: String) -> CommError {
    CommError::MalformedMessage(msg)
}

pub fn upstream_not_available() -> CommError {
    CommError::UpstreamNotAvailable
}

pub fn unexpected_message(msg: String) -> CommError {
    CommError::UnexpectedMessage(msg)
}

pub fn sender_not_available() -> CommError {
    CommError::SenderNotAvailable
}

pub fn pong_not_received_for_ping(peer: String) -> CommError {
    CommError::PongNotReceivedForPing(peer)
}

pub fn timeout() -> CommError {
    CommError::TimeOut
}

pub fn unable_to_store_packet(packet: String, error: String) -> CommError {
    CommError::UnableToStorePacket(packet, error)
}

pub fn unable_to_restore_packet(key: String, error: String) -> CommError {
    CommError::UnableToRestorePacket(key, error)
}
