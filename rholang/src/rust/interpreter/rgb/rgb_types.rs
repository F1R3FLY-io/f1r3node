// Production RGB types for F1r3fly integration
// Uses official RGB library types - no custom implementations

// Re-export core RGB types for contract creation and management
pub use rgb::{
    Assignment,   // For creating owned states
    Consensus,    // Bitcoin, Liquid, etc.
    Contract,     // Main contract type
    ContractId,   // Contract identifier
    Contracts,    // Contract collection
    CreateParams, // Parameters for contract creation
    EitherSeal,   // Seal wrapper
    Issuer,       // Contract issuer
    NamedState,   // State with a name
};

// Use bp-core with correct naming convention - following rgb-cli patterns
pub use bp::seals::{Noise, TxoSeal, TxoSealExt};
pub use bp::{Outpoint, Txid, Vout};
pub use single_use_seals::SingleUseSeal;
pub use strict_encoding::{StrictDeserialize, StrictSerialize};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Minimal request for RGB20 contract issuance from F1r3fly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rgb20IssuanceRequest {
    pub ticker: String,
    pub total_supply: u64,
    pub precision: u8,
    pub genesis_utxo: String, // Will parse to Outpoint
    pub description: Option<String>,
}

/// Response containing real RGB contract data
#[derive(Debug, Clone)]
pub struct Rgb20IssuanceResponse {
    pub contract_id: ContractId,
    pub consignment_file: PathBuf,
    pub consignment_data: Vec<u8>,
}

/// Error types for RGB processing (keeping minimal set)
#[derive(Debug, Clone, thiserror::Error)]
pub enum RgbError {
    #[error("Invalid UTXO format: {0}")]
    InvalidUtxo(String),

    #[error("RGB contract creation failed: {0}")]
    ContractCreationFailed(String),

    #[error("Consignment creation failed: {0}")]
    ConsignmentFailed(String),

    #[error("Storage operation failed: {0}")]
    StorageFailed(String),

    #[error("RhoLang integration error: {0}")]
    RhoLangError(String),
}
