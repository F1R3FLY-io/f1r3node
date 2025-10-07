use std::error::Error;
use std::fmt;

/// Custom error types for the node-cli application
#[derive(Debug)]
pub enum NodeCliError {
    /// Network-related errors (HTTP requests, connection issues)
    Network(NetworkError),
    /// Cryptographic errors (key generation, validation, signing)
    Crypto(CryptoError),
    /// File system errors (reading/writing files)
    File(FileError),
    /// API errors (gRPC calls, response parsing)
    Api(ApiError),
    /// Configuration errors (invalid arguments, missing required fields)
    Config(ConfigError),
    /// General application errors
    General(String),
}

/// Network-related error types
#[derive(Debug)]
pub enum NetworkError {
    ConnectionFailed(String),
    HttpError(u16, String),
    Timeout(String),
    InvalidUrl(String),
    RequestFailed(String),
}

/// Cryptographic error types
#[derive(Debug)]
pub enum CryptoError {
    InvalidPrivateKey(String),
    InvalidPublicKey(String),
    KeyGenerationFailed(String),
    SigningFailed(String),
    AddressGenerationFailed(String),
    HexDecodeFailed(String),
}

/// File system error types
#[derive(Debug)]
pub enum FileError {
    ReadFailed(String, String),
    WriteFailed(String, String),
    NotFound(String),
    PermissionDenied(String),
    InvalidPath(String),
}

/// API error types
#[derive(Debug)]
pub enum ApiError {
    GrpcError(String),
    ParseError(String),
    ResponseError(String),
    InvalidResponse(String),
    ServiceUnavailable(String),
}

/// Configuration error types
#[derive(Debug)]
pub enum ConfigError {
    MissingRequired(String),
    InvalidValue(String, String),
    ConflictingOptions(String),
    InvalidFormat(String),
}

impl fmt::Display for NodeCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeCliError::Network(e) => write!(f, "Network error: {}", e),
            NodeCliError::Crypto(e) => write!(f, "Crypto error: {}", e),
            NodeCliError::File(e) => write!(f, "File error: {}", e),
            NodeCliError::Api(e) => write!(f, "API error: {}", e),
            NodeCliError::Config(e) => write!(f, "Configuration error: {}", e),
            NodeCliError::General(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            NetworkError::HttpError(code, msg) => write!(f, "HTTP {} error: {}", code, msg),
            NetworkError::Timeout(msg) => write!(f, "Request timed out: {}", msg),
            NetworkError::InvalidUrl(url) => write!(f, "Invalid URL: {}", url),
            NetworkError::RequestFailed(msg) => write!(f, "Request failed: {}", msg),
        }
    }
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidPrivateKey(msg) => write!(f, "Invalid private key: {}", msg),
            CryptoError::InvalidPublicKey(msg) => write!(f, "Invalid public key: {}", msg),
            CryptoError::KeyGenerationFailed(msg) => write!(f, "Key generation failed: {}", msg),
            CryptoError::SigningFailed(msg) => write!(f, "Signing failed: {}", msg),
            CryptoError::AddressGenerationFailed(msg) => {
                write!(f, "Address generation failed: {}", msg)
            }
            CryptoError::HexDecodeFailed(msg) => write!(f, "Hex decode failed: {}", msg),
        }
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::ReadFailed(path, msg) => {
                write!(f, "Failed to read file '{}': {}", path, msg)
            }
            FileError::WriteFailed(path, msg) => {
                write!(f, "Failed to write file '{}': {}", path, msg)
            }
            FileError::NotFound(path) => write!(f, "File not found: {}", path),
            FileError::PermissionDenied(path) => write!(f, "Permission denied: {}", path),
            FileError::InvalidPath(path) => write!(f, "Invalid path: {}", path),
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::GrpcError(msg) => write!(f, "gRPC error: {}", msg),
            ApiError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ApiError::ResponseError(msg) => write!(f, "Response error: {}", msg),
            ApiError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            ApiError::ServiceUnavailable(msg) => write!(f, "Service unavailable: {}", msg),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingRequired(field) => write!(f, "Missing required field: {}", field),
            ConfigError::InvalidValue(field, msg) => {
                write!(f, "Invalid value for '{}': {}", field, msg)
            }
            ConfigError::ConflictingOptions(msg) => write!(f, "Conflicting options: {}", msg),
            ConfigError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl Error for NodeCliError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            NodeCliError::Network(e) => Some(e),
            NodeCliError::Crypto(e) => Some(e),
            NodeCliError::File(e) => Some(e),
            NodeCliError::Api(e) => Some(e),
            NodeCliError::Config(e) => Some(e),
            NodeCliError::General(_) => None,
        }
    }
}

impl Error for NetworkError {}
impl Error for CryptoError {}
impl Error for FileError {}
impl Error for ApiError {}
impl Error for ConfigError {}

// Conversion from std::io::Error to NodeCliError
impl From<std::io::Error> for NodeCliError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => {
                NodeCliError::File(FileError::NotFound(err.to_string()))
            }
            std::io::ErrorKind::PermissionDenied => {
                NodeCliError::File(FileError::PermissionDenied(err.to_string()))
            }
            _ => NodeCliError::File(FileError::ReadFailed(
                "unknown".to_string(),
                err.to_string(),
            )),
        }
    }
}

// Conversion from reqwest::Error to NodeCliError
impl From<reqwest::Error> for NodeCliError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            NodeCliError::Network(NetworkError::Timeout(err.to_string()))
        } else if err.is_connect() {
            NodeCliError::Network(NetworkError::ConnectionFailed(err.to_string()))
        } else {
            NodeCliError::Network(NetworkError::RequestFailed(err.to_string()))
        }
    }
}

// Conversion from serde_json::Error to NodeCliError
impl From<serde_json::Error> for NodeCliError {
    fn from(err: serde_json::Error) -> Self {
        NodeCliError::Api(ApiError::ParseError(err.to_string()))
    }
}

// Conversion from secp256k1::Error to NodeCliError
impl From<secp256k1::Error> for NodeCliError {
    fn from(err: secp256k1::Error) -> Self {
        NodeCliError::Crypto(CryptoError::InvalidPrivateKey(err.to_string()))
    }
}

// Conversion from hex::FromHexError to NodeCliError
impl From<hex::FromHexError> for NodeCliError {
    fn from(err: hex::FromHexError) -> Self {
        NodeCliError::Crypto(CryptoError::HexDecodeFailed(err.to_string()))
    }
}

// Conversion from String to NodeCliError
impl From<String> for NodeCliError {
    fn from(err: String) -> Self {
        NodeCliError::General(err)
    }
}

// Conversion from &str to NodeCliError
impl From<&str> for NodeCliError {
    fn from(err: &str) -> Self {
        NodeCliError::General(err.to_string())
    }
}

// Conversion from Box<dyn Error> to NodeCliError
impl From<Box<dyn Error>> for NodeCliError {
    fn from(err: Box<dyn Error>) -> Self {
        NodeCliError::General(err.to_string())
    }
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, NodeCliError>;

/// Helper functions for creating specific error types
impl NodeCliError {
    pub fn network_connection_failed(msg: &str) -> Self {
        NodeCliError::Network(NetworkError::ConnectionFailed(msg.to_string()))
    }

    pub fn network_http_error(code: u16, msg: &str) -> Self {
        NodeCliError::Network(NetworkError::HttpError(code, msg.to_string()))
    }

    pub fn crypto_invalid_private_key(msg: &str) -> Self {
        NodeCliError::Crypto(CryptoError::InvalidPrivateKey(msg.to_string()))
    }

    pub fn crypto_invalid_public_key(msg: &str) -> Self {
        NodeCliError::Crypto(CryptoError::InvalidPublicKey(msg.to_string()))
    }

    pub fn file_read_failed(path: &str, msg: &str) -> Self {
        NodeCliError::File(FileError::ReadFailed(path.to_string(), msg.to_string()))
    }

    pub fn file_write_failed(path: &str, msg: &str) -> Self {
        NodeCliError::File(FileError::WriteFailed(path.to_string(), msg.to_string()))
    }

    pub fn config_missing_required(field: &str) -> Self {
        NodeCliError::Config(ConfigError::MissingRequired(field.to_string()))
    }

    pub fn config_invalid_value(field: &str, msg: &str) -> Self {
        NodeCliError::Config(ConfigError::InvalidValue(
            field.to_string(),
            msg.to_string(),
        ))
    }
}
