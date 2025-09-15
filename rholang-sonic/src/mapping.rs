//! RGB operation type mapping module
//!
//! This module defines the core types for mapping RGB operations to their semantic meanings,
//! including the operation type enum and the mapping structures used by RholangCodex.

use amplify::{bmap, confinement::TinyOrdMap, tiny_bmap};
use hypersonic::CallId;

/// RGB operation types supported by Rholang execution
#[derive(Debug, Clone, PartialEq)]
pub enum RgbOperationType {
    /// RGB-20 fungible token issuance/genesis
    Rgb20Issue,
    /// RGB-20 fungible token transfer
    Rgb20Transfer,
    /// RGB-20 fungible token burn
    Rgb20Burn,
    /// RGB-21 NFT mint/issuance
    Rgb21Mint,
    /// RGB-21 NFT transfer
    Rgb21Transfer,
    /// RGB-25 collectible operations
    Rgb25Operation,
    /// Custom contract operation (with method name)
    Custom(String),
}

/// RGB operation mapping for Rholang execution
/// Maps CallId values to specific RGB operations, similar to how Ultrasonic maps to AluVM scripts
#[derive(Debug, Clone)]
pub struct RgbOperationMap {
    /// Maps CallId to RGB operation semantics
    call_id_mappings: TinyOrdMap<CallId, RgbOperationType>,
    /// Human-readable contract type name
    contract_type: String,
}

impl RgbOperationMap {
    /// Create mapping for standard RGB-20 fungible token contract
    pub fn rgb20_standard() -> Self {
        Self {
            call_id_mappings: tiny_bmap! {
                0 => RgbOperationType::Rgb20Issue,
                1 => RgbOperationType::Rgb20Transfer,
                2 => RgbOperationType::Rgb20Burn,
            },
            contract_type: "RGB-20 Fungible Token".to_string(),
        }
    }

    /// Create mapping for standard RGB-21 NFT contract
    pub fn rgb21_standard() -> Self {
        Self {
            call_id_mappings: tiny_bmap! {
                0 => RgbOperationType::Rgb21Mint,
                1 => RgbOperationType::Rgb21Transfer,
            },
            contract_type: "RGB-21 Non-Fungible Token".to_string(),
        }
    }

    /// Create mapping for RGB-25 collectible contract
    pub fn rgb25_standard() -> Self {
        Self {
            call_id_mappings: tiny_bmap! {
                0 => RgbOperationType::Rgb25Operation,
                1 => RgbOperationType::Rgb25Operation,
            },
            contract_type: "RGB-25 Collectible".to_string(),
        }
    }

    /// Create custom mapping for specialized contracts
    pub fn custom(mappings: TinyOrdMap<CallId, RgbOperationType>, contract_type: String) -> Self {
        Self {
            call_id_mappings: mappings,
            contract_type,
        }
    }

    /// Get RGB operation type for a given CallId
    pub fn get_operation_type(&self, call_id: CallId) -> Option<&RgbOperationType> {
        self.call_id_mappings.get(&call_id)
    }

    /// Get contract type description
    pub fn contract_type(&self) -> &str {
        &self.contract_type
    }

    /// Check if a CallId is supported
    pub fn supports_call_id(&self, call_id: CallId) -> bool {
        self.call_id_mappings.contains_key(&call_id)
    }

    /// Get all supported CallIds for this mapping
    pub fn supported_call_ids(&self) -> Vec<CallId> {
        self.call_id_mappings.keys().copied().collect()
    }
}
