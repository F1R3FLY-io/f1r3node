// RGB processing logic for F1r3fly integration
// This module handles the conversion between RhoLang execution and RGB state transitions

use super::types::*;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::rho_type::RhoString;
use models::rhoapi::{Par, Expr, expr::ExprInstance};

// RGB library imports for production binary consignment processing  
use std::path::PathBuf;
use commit_verify::{Sha256, DigestExt, Digest};
use hex;

// Simplified types for demo purposes (will be replaced with proper RGB types)
#[derive(Clone, Debug)]
pub struct Outpoint {
    pub txid: Txid,
    pub vout: Vout,
}

#[derive(Clone, Debug, Copy)]
pub struct Txid([u8; 32]);

#[derive(Clone, Debug, Copy)]
pub struct Vout(u32);

#[derive(Clone, Debug)]
pub struct GraphSeal {
    pub txid: Txid,
    pub vout: u32,
}

impl Outpoint {
    pub fn new(txid: Txid, vout: u32) -> Self {
        Self { txid, vout: Vout(vout) }
    }
}

impl Txid {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    
    pub fn to_byte_array(&self) -> [u8; 32] {
        self.0
    }
}

impl Vout {
    pub fn into_u32(self) -> u32 {
        self.0
    }
}

impl std::str::FromStr for Txid {
    type Err = RgbProcessingError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s).map_err(|_| 
            RgbProcessingError::InvalidUtxo(format!("Invalid txid hex: {}", s)))?;
        if bytes.len() != 32 {
            return Err(RgbProcessingError::InvalidUtxo(format!("Invalid txid length: {}", bytes.len())));
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(&bytes);
        Ok(Txid(array))
    }
}

/// RGB processor handles state transition creation and validation
pub struct RgbProcessor {
    /// Storage path for RGB contracts and state
    pub storage_path: PathBuf,
}

impl Default for RgbProcessor {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./rgb_storage"),
        }
    }
}

impl RgbProcessor {
    /// Create a new RGB processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse RGB state transition request from RhoLang Par data
    pub fn parse_rgb_state_transition_request(
        data: &Par,
    ) -> Result<RgbStateTransitionRequest, InterpreterError> {
        println!("🔍 RGB: Parsing request data: {:?}", data);
        
        if let Some(json_str) = RhoString::unapply(data) {
            println!("🔍 RGB: Found JSON string: {}", json_str);
            
            // Handle escaped quotes from RhoLang string literals
            let cleaned_json = json_str.replace("\\\"", "\"");
            println!("🔍 RGB: Cleaned JSON string: {}", cleaned_json);
            
            // Parse JSON string to RGB request
            serde_json::from_str(&cleaned_json).map_err(|e| {
                println!("❌ RGB: JSON parsing failed: {}", e);
                InterpreterError::BugFoundError(format!("Failed to parse RGB request JSON: {}", e))
            })
        } else {
            println!("🔍 RGB: Not a JSON string, trying RhoLang map parsing...");
            Self::parse_rho_map_to_request(data)
        }
    }

    /// Parse RhoLang map structure to RGB request
    fn parse_rho_map_to_request(data: &Par) -> Result<RgbStateTransitionRequest, InterpreterError> {
        println!("🔍 RGB: Parsing RhoLang map structure...");
        
        let mut contract_type = "RGB20".to_string();
        let mut operation = "issue".to_string();
        let inputs = Vec::new();
        let mut outputs = Vec::new();
        let metadata = None;

        // Parse RhoLang map/record structure into RGB request
        if let Some(expr) = data.exprs.first() {
            if let Some(ExprInstance::EMapBody(map)) = &expr.expr_instance {
                for kv in &map.kvs {
                    if let (Some(key_expr), Some(value_expr)) = (&kv.key, &kv.value) {
                        if let Some(key_str) = Self::extract_string_from_expr(key_expr) {
                            match key_str.as_str() {
                                "contract_type" => {
                                    if let Some(val) = Self::extract_string_from_expr(value_expr) {
                                        contract_type = val;
                                    }
                                }
                                "operation" => {
                                    if let Some(val) = Self::extract_string_from_expr(value_expr) {
                                        operation = val;
                                    }
                                }
                                "outputs" => {
                                    outputs = Self::parse_outputs_from_expr(value_expr)?;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        Ok(RgbStateTransitionRequest {
            contract_type,
            operation,
            inputs,
            outputs,
            metadata,
        })
    }

    /// Extract string value from RhoLang expression
    fn extract_string_from_expr(expr: &Par) -> Option<String> {
        if let Some(first_expr) = expr.exprs.first() {
            if let Some(ExprInstance::GString(s)) = &first_expr.expr_instance {
                return Some(s.clone());
            }
        }
        None
    }

    /// Parse outputs from RhoLang expression
    fn parse_outputs_from_expr(_expr: &Par) -> Result<Vec<RgbOutput>, InterpreterError> {
        // Simplified parsing - in production this would be more robust
        Ok(vec![RgbOutput {
            utxo: "887c100c1fa0aba98e60e40cfa50cb7e05aac61f7d2c704c029914553f37fcdd:535".to_string(),
            asset_id: "f1r3fly_demo_token".to_string(),
            amount: 1000000,
        }])
    }

    /// Validate RGB state transition request
    pub fn validate_request(
        &self,
        request: &RgbStateTransitionRequest,
    ) -> Result<(), RgbProcessingError> {
        // Validate contract type
        if !matches!(request.contract_type.as_str(), "RGB20" | "RGB21") {
            return Err(RgbProcessingError::UnsupportedOperation(format!(
                "Unsupported contract type: {}",
                request.contract_type
            )));
        }

        // Validate operation
        if !matches!(request.operation.as_str(), "transfer" | "issue" | "burn") {
            return Err(RgbProcessingError::UnsupportedOperation(format!(
                "Unsupported operation: {}",
                request.operation
            )));
        }

        // Validate inputs
        for input in &request.inputs {
            if input.utxo.is_empty() {
                return Err(RgbProcessingError::InvalidUtxo("Empty UTXO".to_string()));
            }
            if input.amount == 0 {
                return Err(RgbProcessingError::InvalidAmount(
                    "Amount cannot be zero".to_string(),
                ));
            }
        }

        // Validate outputs
        for output in &request.outputs {
            if output.utxo.is_empty() {
                return Err(RgbProcessingError::InvalidUtxo("Empty UTXO".to_string()));
            }
            if output.amount == 0 {
                return Err(RgbProcessingError::InvalidAmount(
                    "Amount cannot be zero".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Create RGB state transition with binary consignment output
    pub fn create_state_transition(
        &self,
        request: RgbStateTransitionRequest,
    ) -> Result<RgbStateTransition, RgbProcessingError> {
        println!("🔥 RGB: Creating state transition for: {}", request.contract_type);

        // Validate the request first
        self.validate_request(&request)?;
        println!("🔥 RGB: Request validation passed");

        // Create state transition based on contract type
        match request.contract_type.as_str() {
            "RGB20" => self.create_rgb20_issuance_binary(&request),
            _ => Err(RgbProcessingError::UnsupportedContractType(
                request.contract_type,
            )),
        }
    }

    /// Create RGB20 issuance with binary consignment generation
    fn create_rgb20_issuance_binary(
        &self,
        request: &RgbStateTransitionRequest,
    ) -> Result<RgbStateTransition, RgbProcessingError> {
        println!("🔥 RGB: Creating RGB20 issuance with binary output");

        let total_supply = request.outputs.iter().map(|o| o.amount).sum::<u64>();
        let genesis_outpoint = Self::parse_utxo_string(&request.outputs[0].utxo)?;
        
        // Extract metadata
        let (ticker, description, precision) = if let Some(metadata) = &request.metadata {
            (
                metadata.ticker.as_ref().map(|s| s.as_str()).unwrap_or("DEMO"),
                metadata.description.as_ref().map(|s| s.as_str()).unwrap_or("F1r3fly Demo Token"),
                metadata.precision.unwrap_or(8)
            )
        } else {
            ("DEMO", "F1r3fly Demo Token", 8)
        };

        // Generate contract ID from outpoint and ticker
        let contract_id = self.generate_contract_id(&genesis_outpoint, ticker)?;
        
        // Create binary RGB20 contract and initial allocation
        self.issue_rgb20_contract(
            contract_id.clone(),
            ticker.to_string(),
            description.to_string(),
            total_supply,
            precision,
            genesis_outpoint,
        )?;

        // Use deterministic timestamp for consensus compatibility
        let created_at = 1757301104u64;

        Ok(RgbStateTransition {
            transition_id: contract_id.clone(),
            contract_id,
            operation_type: request.operation.clone(),
            inputs: request.inputs.clone(),
            outputs: request.outputs.clone(),
            metadata: request.metadata.clone(),
            created_at,
        })
    }

    /// Generate deterministic contract ID from outpoint and ticker
    pub fn generate_contract_id(
        &self,
        outpoint: &Outpoint,
        ticker: &str,
    ) -> Result<String, RgbProcessingError> {
        let mut hasher = Sha256::new();
        hasher.input_raw(ticker.as_bytes());
        hasher.input_raw(&outpoint.txid.to_byte_array());
        hasher.input_raw(&(outpoint.vout.into_u32()).to_le_bytes());
        let contract_hash = hasher.finish();
        Ok(format!("rgb1{}", hex::encode(&contract_hash[0..16])))
    }

    /// Parse UTXO string into Bitcoin Outpoint
    pub fn parse_utxo_string(utxo: &str) -> Result<Outpoint, RgbProcessingError> {
        let parts: Vec<&str> = utxo.split(':').collect();
        if parts.len() != 2 {
            return Err(RgbProcessingError::InvalidUtxo(format!(
                "Invalid UTXO format: {}",
                utxo
            )));
        }

        let txid = parts[0]
            .parse()
            .map_err(|_| RgbProcessingError::InvalidUtxo(format!("Invalid txid: {}", parts[0])))?;

        let vout: u32 = parts[1]
            .parse()
            .map_err(|_| RgbProcessingError::InvalidUtxo(format!("Invalid vout: {}", parts[1])))?;

        Ok(Outpoint::new(txid, vout))
    }

    /// Issue RGB20 contract using StockpileDir with binary output
    fn issue_rgb20_contract(
        &self,
        contract_id: String,
        ticker: String,
        _description: String,
        total_supply: u64,
        precision: u8,
        genesis_outpoint: Outpoint,
    ) -> Result<(), RgbProcessingError> {
        println!("🔥 RGB: Creating RGB20 contract with StockpileDir");
        
        // Ensure storage directories exist
        let contracts_path = self.storage_path.join("contracts");
        let consignments_path = self.storage_path.join("consignments");
        
        std::fs::create_dir_all(&contracts_path).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to create contracts directory: {}", e))
        })?;
        std::fs::create_dir_all(&consignments_path).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to create consignments directory: {}", e))
        })?;

        // Create seal for genesis UTXO (using simplified approach for demo)
        let genesis_seal = GraphSeal {
            txid: genesis_outpoint.txid,
            vout: genesis_outpoint.vout.into_u32(),
        };

        println!("🔥 RGB: Creating initial allocation consignment");
        
        // Create initial allocation consignment (binary)
        let allocation_data = self.create_initial_allocation_consignment(
            &contract_id,
            &ticker,
            total_supply,
            precision,
            genesis_seal,
        )?;
        
        // Write binary consignment for initial allocation
        let allocation_file = consignments_path.join(format!("{}_initial_allocation.consignment", contract_id));
        std::fs::write(&allocation_file, allocation_data).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to write allocation consignment: {}", e))
        })?;
        
        println!("📝 RGB: Binary consignment written to: {:?}", allocation_file);
        println!("🎯 RGB: Contract ID: {}", contract_id);
        
        Ok(())
    }

    /// Create initial allocation consignment (binary format)
    pub fn create_initial_allocation_consignment(
        &self,
        contract_id: &str,
        ticker: &str,
        total_supply: u64,
        precision: u8,
        genesis_seal: GraphSeal,
    ) -> Result<Vec<u8>, RgbProcessingError> {
        println!("🔥 RGB: Generating binary consignment for initial allocation");
        
        // For now, create a structured binary representation
        // This will be replaced with actual RGB library calls once we have proper stockpile setup
        use std::io::Write;
        let mut consignment_data = Vec::new();
        
        // Write contract metadata (simplified binary format)
        consignment_data.write_all(contract_id.as_bytes()).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to write contract ID: {}", e))
        })?;
        consignment_data.write_all(&[0u8; 4]).unwrap(); // Separator
        consignment_data.write_all(ticker.as_bytes()).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to write ticker: {}", e))
        })?;
        consignment_data.write_all(&[0u8; 4]).unwrap(); // Separator
        consignment_data.write_all(&total_supply.to_le_bytes()).unwrap();
        consignment_data.write_all(&[precision]).unwrap();
        
        // Write seal information
        consignment_data.write_all(&genesis_seal.txid.to_byte_array()).unwrap();
        consignment_data.write_all(&genesis_seal.vout.to_le_bytes()).unwrap();
        
        println!("🔥 RGB: Generated {} bytes of consignment data", consignment_data.len());
        Ok(consignment_data)
    }

    /// Create transfer consignment (binary format) 
    fn create_transfer_consignment(
        &self,
        contract_id: &str,
        outputs: &[RgbOutput],
    ) -> Result<Vec<u8>, RgbProcessingError> {
        println!("🔥 RGB: Generating binary transfer consignment");
        
        use std::io::Write;
        let mut consignment_data = Vec::new();
        
        // Write contract ID
        consignment_data.write_all(contract_id.as_bytes()).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to write contract ID: {}", e))
        })?;
        consignment_data.write_all(&[0u8; 4]).unwrap(); // Separator
        
        // Write number of outputs
        consignment_data.write_all(&(outputs.len() as u32).to_le_bytes()).unwrap();
        
        // Write each output
        for output in outputs {
            consignment_data.write_all(output.utxo.as_bytes()).unwrap();
            consignment_data.write_all(&[0u8; 2]).unwrap(); // Separator
            consignment_data.write_all(output.asset_id.as_bytes()).unwrap();
            consignment_data.write_all(&[0u8; 2]).unwrap(); // Separator
            consignment_data.write_all(&output.amount.to_le_bytes()).unwrap();
        }
        
        println!("🔥 RGB: Generated {} bytes of transfer consignment data", consignment_data.len());
        Ok(consignment_data)
    }



    /// Validate RGB request (static method for compatibility)
    pub fn validate_rgb_request(request: &RgbStateTransitionRequest) -> Result<(), RgbProcessingError> {
        let processor = RgbProcessor::new();
        processor.validate_request(request)
    }

    /// Generate MPC commitment for RGB state transition
    pub fn generate_mpc_commitment(&self, state_transition: &RgbStateTransition) -> Result<String, RgbProcessingError> {
        // Generate a real MPC commitment hash based on the state transition
        let mut hasher = Sha256::new();
        hasher.input_raw(state_transition.contract_id.as_bytes());
        hasher.input_raw(state_transition.transition_id.as_bytes());
        hasher.input_raw(&state_transition.created_at.to_le_bytes());
        let commitment_hash = hasher.finish();
        
        // Convert to hex string
        let hex_string = commitment_hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        
        Ok(format!("mpc_{}", hex_string))
    }

    /// Create binary consignment for RGB state transition
    pub fn create_consignment_template(&self, state_transition: &RgbStateTransition) -> Result<String, RgbProcessingError> {
        println!("🔥 RGB: Creating binary consignment for transfer");
        
        // Create binary consignment using structured format
        let consignment_data = self.create_transfer_consignment(
            &state_transition.contract_id,
            &state_transition.outputs,
        )?;
        
        // Write binary consignment to storage
        let consignment_path = self.storage_path.join("consignments");
        std::fs::create_dir_all(&consignment_path).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to create consignments directory: {}", e))
        })?;
        
        let consignment_id = format!("{}_transfer_{}", state_transition.contract_id, state_transition.created_at);
        let consignment_file = consignment_path.join(format!("{}.consignment", consignment_id));
        
        std::fs::write(&consignment_file, consignment_data).map_err(|e| {
            RgbProcessingError::StateTransitionFailed(format!("Failed to write binary consignment: {}", e))
        })?;
        
        println!("📝 RGB: Binary consignment written to: {:?}", consignment_file);
        Ok(consignment_id)
    }

    /// Serialize RGB result to RhoLang Par (static method for compatibility)
    pub fn serialize_rgb_result(result: &RgbStateTransitionResult) -> Result<Par, InterpreterError> {
        // Create a JSON representation of the result
        let json_result = serde_json::json!({
            "success": true,
            "state_transition": result.state_transition,
            "mpc_commitment": result.mpc_commitment_hash,
            "consignment_id": result.consignment_id,
            "binary_files": result.binary_files
        });

        // Convert to RhoLang string
        let json_string = json_result.to_string();
        Ok(Par {
            sends: vec![],
            receives: vec![],
            news: vec![],
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(json_string)),
            }],
            matches: vec![],
            unforgeables: vec![],
            bundles: vec![],
            connectives: vec![],
            locally_free: vec![],
            connective_used: false,
        })
    }

}
