// RGB data structures for F1r3fly integration
// These structures define the interface between RhoLang contracts and RGB processing

use serde::{Deserialize, Serialize};

/// Request structure for RGB state transition creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbStateTransitionRequest {
    pub contract_type: String,        // "RGB20", "RGB21", etc.
    pub operation: String,            // "transfer", "issue", etc.
    pub inputs: Vec<RgbInput>,
    pub outputs: Vec<RgbOutput>,
    pub metadata: Option<RgbMetadata>,
}

/// RGB input specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbInput {
    pub utxo: String,                 // Outpoint as string (e.g., "txid:vout")
    pub asset_id: String,             // Contract ID
    pub amount: u64,
}

/// RGB output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbOutput {
    pub utxo: String,                 // Outpoint as string (e.g., "txid:vout")
    pub asset_id: String,             // Contract ID
    pub amount: u64,
}

/// Optional metadata for RGB operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbMetadata {
    pub description: Option<String>,
    pub ticker: Option<String>,
    pub precision: Option<u8>,
    pub custom_data: Option<serde_json::Value>,
}

/// Result structure returned to RGB participants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbStateTransitionResult {
    pub state_transition: RgbStateTransition,
    pub mpc_commitment_hash: String,   // 32-byte hex string
    pub consignment_id: String,        // ID of generated binary consignment
    pub binary_files: BinaryFileInfo,  // Information about generated binary files
}

/// RGB state transition data (simplified representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbStateTransition {
    pub transition_id: String,
    pub contract_id: String,
    pub operation_type: String,
    pub inputs: Vec<RgbInput>,
    pub outputs: Vec<RgbOutput>,
    pub metadata: Option<RgbMetadata>,
    pub created_at: u64,              // Unix timestamp
}

/// Information about generated binary RGB files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryFileInfo {
    pub consignment_file: String,      // Path to binary consignment file
    pub consignment_size: u64,         // Size of binary consignment in bytes
    pub contract_id: String,           // RGB contract ID
    pub created_at: u64,              // Unix timestamp
}

/// Instructions for rgb-cli usage with generated binary files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbCliInstructions {
    pub consignment_file: String,     // Path to binary consignment file
    pub import_command: String,       // rgb-cli command to import consignment
    pub transfer_command: String,     // rgb-cli command to create transfers
    pub validate_command: String,     // rgb-cli command to validate consignment
}

/// Error types for RGB processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RgbProcessingError {
    InvalidRequest(String),
    InvalidUtxo(String),
    InvalidAmount(String),
    UnsupportedOperation(String),
    UnsupportedContractType(String),
    StateTransitionFailed(String),
    CommitmentGenerationFailed(String),
    SerializationError(String),
}

impl std::fmt::Display for RgbProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RgbProcessingError::InvalidRequest(msg) => write!(f, "Invalid RGB request: {}", msg),
            RgbProcessingError::InvalidUtxo(msg) => write!(f, "Invalid UTXO: {}", msg),
            RgbProcessingError::InvalidAmount(msg) => write!(f, "Invalid amount: {}", msg),
            RgbProcessingError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            RgbProcessingError::UnsupportedContractType(msg) => write!(f, "Unsupported contract type: {}", msg),
            RgbProcessingError::StateTransitionFailed(msg) => write!(f, "State transition failed: {}", msg),
            RgbProcessingError::CommitmentGenerationFailed(msg) => write!(f, "Commitment generation failed: {}", msg),
            RgbProcessingError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for RgbProcessingError {}
