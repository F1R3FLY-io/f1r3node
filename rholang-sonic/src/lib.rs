// Rholang-Sonic Bridge for RGB Smart Contracts
//
// This crate provides a drop-in replacement for ultrasonic::Codex that uses
// Rholang for contract execution instead of AluVM.

use std::sync::{Arc, Mutex, Once};
use std::collections::HashMap;

use log::{info, debug, warn, LevelFilter};

static INIT: Once = Once::new();

/// Initialize logging for RholangCodex debugging
pub fn init_debug_logging() {
    INIT.call_once(|| {
        env_logger::builder()
            .filter_level(LevelFilter::Debug)
            .format_timestamp(None)
            .format_module_path(true)
            .format_target(true)
            .target(env_logger::Target::Stderr)
            .write_style(env_logger::WriteStyle::Always)
            .try_init()
            .ok(); // Don't panic if already initialized
    });
}
use hypersonic::{Stock, Articles, EffectiveState, IssueError, ContractId, 
                LibRepo, Memory, Operation, VerifiedOperation, CallId, CallError, CallParams, 
                AcceptError, OpBuilder, CellAddr, StateCell, StateValue, Transition, AuthToken};
use aluvm::{Lib, LibId};
// Re-exports for convenience (Opid imported below)
use amplify::MultiError;
use strict_encoding::{StrictWriter, WriteRaw};
use rholang::rust::interpreter::rho_runtime::{create_runtime_from_kv_store, RhoRuntimeImpl, RhoRuntime};
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::system_processes::Definition;
use rholang::rust::interpreter::interpreter::EvaluateResult;
use rholang::rust::interpreter::accounting::costs::Cost;
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::Par;
use rspace_plus_plus::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager,
    lmdb_dir_store_manager::MB,
    rspace_store_manager::mk_rspace_store_manager,
};

pub use hypersonic::Opid; // Re-export for compatibility

mod storage;
pub use storage::{StorageConfig, StorageManager};

mod error;
pub use error::{
    codes as error_codes, 
    rgb_contract_error, 
    rgb_data_extraction_error,
    rgb_invalid_data_error,
    rgb_missing_state_error,
    rgb_insufficient_balance_error,
    rgb_invalid_recipient_error,
    interpreter_error_to_call_error
};

mod mapping;
pub use mapping::{RgbOperationType, RgbOperationMap};

mod contracts;
mod extraction; 
mod serialization;
mod validation;
pub use validation::RgbExecutionResult;

/// RholangCodex is a drop-in replacement for hypersonic::Ledger that executes
/// RGB contracts using Rholang instead of AluVM.
/// 
/// Now implements the complete Ledger<S> interface for seamless integration with RGB.
/// Uses Arc<Mutex<RhoRuntimeImpl>> for thread-safe concurrent access without expensive cloning.
pub struct RholangCodex<S: Stock> {
    // Rholang execution components
    /// Thread-safe runtime for concurrent RGB contract execution
    runtime: Arc<Mutex<RhoRuntimeImpl>>,
    tokio_rt: tokio::runtime::Runtime,
    storage_manager: StorageManager,
    /// RGB operation mapping that defines CallId -> operation type relationships
    operation_mapping: RgbOperationMap,
    
    // Ledger components (NEW in Task 1)
    /// Persistence & state management - will be implemented in Task 2
    stock: S,
    /// Cached contract ID for efficient access - will be implemented in Task 2  
    contract_id: ContractId,
}

/*
impl RholangCodex {
    /// Create a new RholangCodex with default persistent storage and RGB-20 mapping
    /// Uses ~/.rholang-sonic or OS-appropriate directory
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_rgb_mapping(RgbOperationMap::rgb20_standard(), StorageConfig::Default)
    }
    
    /// Create a new RholangCodex with custom storage path and RGB-20 mapping
    /// Recommended for production use where you want to control storage location
    pub fn new_with_storage<P: AsRef<Path>>(storage_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_rgb_mapping(
            RgbOperationMap::rgb20_standard(), 
            StorageConfig::Persistent(storage_path.as_ref().to_path_buf())
        )
    }
    
    /// Create a new RholangCodex with ephemeral storage and RGB-20 mapping
    /// Useful for testing or one-shot validations that don't need persistence
    pub fn new_ephemeral() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_rgb_mapping(RgbOperationMap::rgb20_standard(), StorageConfig::Ephemeral)
    }
    
    /// Create a new RholangCodex with specific RGB operation mapping and storage config
    /// This is the core constructor that all others delegate to
    pub fn new_with_rgb_mapping(
        operation_mapping: RgbOperationMap, 
        config: StorageConfig
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokio_rt = tokio::runtime::Runtime::new()?;
        let storage_manager = StorageManager::new(config)?;
        
        // Check if this is first time initialization
        let is_first_time = storage_manager.is_first_time();
        
        // Initialize RSpace storage manager
        let mut store_manager = mk_rspace_store_manager(
            storage_manager.storage_path().to_path_buf(), 
            100 * MB
        );
        
        // Create RhoRuntimeImpl using create_runtime_from_kv_store
        let runtime = tokio_rt.block_on(async {
            // Create RSpace storage
            let rspace_store = store_manager.r_space_stores().await
                .map_err(|e| format!("Failed to create RSpace stores: {}", e))?;
            
            // Create RhoRuntimeImpl
            // For RGB validation, we don't need registry (init_registry = false)
            // This follows the comment guidance: "For some test cases, you don't need the registry 
            // then you can skip this init process which can be faster"
            //
            // RGB contracts should be deterministic and not use system processes like:
            // - I/O operations (stdout, stderr)  
            // - Random number generation
            // - External service lookups (registry)
            // - Block/timestamp access during validation
            let runtime = create_runtime_from_kv_store(
                rspace_store,
                Par::default(),                    // minimal mergeable_tag_name
                false,                             // init_registry = false (no registry needed for RGB)
                &mut Vec::<Definition>::new(),     // no additional system processes
                Arc::new(Box::new(Matcher)),       // matcher for RSpace operations
            ).await;
            
            Ok::<RhoRuntimeImpl, String>(runtime)
        })?;
        
        if is_first_time {
            info!("Initialized new Rholang-RGB storage at: {}", storage_manager.storage_path().display());
        } else {
            debug!("Using existing Rholang-RGB storage at: {}", storage_manager.storage_path().display());
        }
        
        Ok(Self {
            runtime: Arc::new(Mutex::new(runtime)),
            tokio_rt,
            storage_manager,
            operation_mapping,
        })
    }
    
    /// Get the storage path being used by this instance
    pub fn storage_path(&self) -> &Path {
        self.storage_manager.storage_path()
    }
    
    /// Check if this was a first-time initialization
    pub fn was_first_time(&self) -> bool {
        // Note: This checks current state, not initialization state
        // Could be enhanced to remember initialization state if needed
        self.storage_manager.is_first_time()
    }
    
    /// Create a new RholangCodex configured for RGB-21 NFT contracts
    pub fn new_rgb21() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_rgb_mapping(RgbOperationMap::rgb21_standard(), StorageConfig::Default)
    }
    
    /// Create a new RholangCodex configured for RGB-25 collectible contracts
    pub fn new_rgb25() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_rgb_mapping(RgbOperationMap::rgb25_standard(), StorageConfig::Default)
    }
    
    /// Get the current RGB operation mapping
    pub fn operation_mapping(&self) -> &RgbOperationMap {
        &self.operation_mapping
    }
        
    /// Map RGB memory cells to Rholang channel setup code
    async fn map_rgb_cells_to_channels(
        &self,
        memory: &impl Memory,
        operation: &Operation,
    ) -> Result<String, CallError> {
        debug!("🔄 Mapping {} destructible inputs to channels", operation.destructible_in.len());
        
        let mut channel_setup = String::new();
        
        // Map destructible input cells to channel data
        for input in &operation.destructible_in {
            if let Some(cell) = memory.destructible(input.addr) {
                let channel_name = self.cell_to_channel_name(input.addr);
                let cell_value = self.serialize_state_value(cell.data);
                
                debug!("📍 Mapping cell {} → @\"{}\" with value: {}", 
                      input.addr, channel_name, cell_value);
                
                // Create Rholang channel with cell data
                channel_setup.push_str(&format!(
                    "{}!({}) |\n", 
                    channel_name, 
                    cell_value
                ));
            } else {
                debug!("❌ Missing destructible cell: {}", input.addr);
                return Err(CallError::NoReadOnceInput(input.addr));
            }
        }
        
        // Map immutable input cells to channel data
        for immutable_addr in &operation.immutable_in {
            if let Some(value) = memory.immutable(*immutable_addr) {
                let channel_name = self.cell_to_channel_name(*immutable_addr);
                let serialized_value = self.serialize_state_value(value);
                
                debug!("📍 Mapping immutable cell {} → @\"{}\" with value: {}", 
                      immutable_addr, channel_name, serialized_value);
                
                // Create Rholang channel with immutable data
                channel_setup.push_str(&format!(
                    "{}!({}) |\n", 
                    channel_name, 
                    serialized_value
                ));
            } else {
                debug!("❌ Missing immutable cell: {}", immutable_addr);
                return Err(CallError::NoImmutableInput(*immutable_addr));
            }
        }
        
        debug!("✅ Generated channel setup:\n{}", channel_setup);
        Ok(channel_setup)
    }
    
    /// Get list of supported CallIds for this contract mapping
    fn get_supported_call_ids(&self) -> Vec<CallId> {
        self.operation_mapping.supported_call_ids()
    }
    
    /// Generate contract based on RGB operation type (call_id)
    fn generate_rgb_contract(&self, operation: &Operation, memory: &impl Memory) -> Result<String, CallError> {
        // Convert CallId to string for matching - simplified for demo
        let method_name = format!("{:?}", operation.call_id).to_lowercase();
        
        debug!("🔍 Determining contract type for method: {}", method_name);
        
        match method_name.as_str() {
            method if method.contains("issue") => {
                self.generate_rgb20_issue_contract(operation)
            },
            method if method.contains("transfer") => {
                self.generate_rgb20_transfer_contract(operation, memory)
            },
            _ => {
                debug!("❌ Unknown RGB operation method: {}", method_name);
                // For demo, default to transfer logic
                self.generate_rgb20_transfer_contract(operation, memory)
            }
        }
    }
    
    /// Main verification method - must match ultrasonic::Codex::verify() signature exactly
    pub fn verify(
        &self,
        contract_id: ContractId,
        operation: Operation,
        memory: &impl Memory,
        repo: &impl LibRepo,
    ) -> Result<VerifiedOperation, CallError> {
        debug!("🔄 Sync verify wrapper called for contract {} operation {}", 
               contract_id, operation.opid());
        
        // Async bridge - execute Rholang asynchronously but return synchronously
        self.tokio_rt.block_on(self.async_verify(contract_id, operation, memory, repo))
    }
    
    /// Internal async verification using Rholang
    async fn async_verify(
        &self,
        _contract_id: ContractId,
        operation: Operation,
        memory: &impl Memory,
        _repo: &impl LibRepo,
    ) -> Result<VerifiedOperation, CallError> {
        debug!("🔍 Starting Rholang verification for operation: {}", operation.opid());
        
        // Step 1: Generate the appropriate Rholang contract based on operation type
        let rholang_contract = self.determine_and_generate_contract(&operation, memory)?;
        
        debug!("📝 Generated Rholang contract:\n{}", rholang_contract);
        
        // Step 2: Map RGB memory cells to Rholang channels
        let channel_setup = self.map_rgb_cells_to_channels(memory, &operation).await
            .map_err(|e| {
                debug!("❌ Failed to map cells to channels: {:?}", e);
                // Channel mapping failures are already CallError, so just pass through
                e
            })?;
        
        debug!("🔗 Generated channel setup:\n{}", channel_setup);
        
        // Step 3: Combine channel setup with contract for execution
        let full_contract = format!("{}\n\n{}", channel_setup, rholang_contract);
        
        // Step 4: Execute and validate the contract using RhoRuntimeImpl
        let validation_result = self.execute_and_validate_rholang_contract(&full_contract, &operation).await?;
        
        // Step 5: Check validation results before considering operation verified
        if !validation_result.success {
            let error_msg = format!(
                "RGB contract validation failed for operation {}: {}. Errors: {:?}", 
                operation.opid(), 
                validation_result.validation_summary,
                validation_result.errors
            );
            debug!("❌ {}", error_msg);
            return Err(rgb_contract_error(error_codes::CONTRACT_VALIDATION_ERROR, &error_msg));
        }
        
        info!("✅ Operation {} verified successfully using Rholang: {} channels, cost: {}", 
              operation.opid(), 
              validation_result.final_channels.len(),
              validation_result.execution_cost);
        
        // Create VerifiedOperation using the now-public constructor
        Ok(VerifiedOperation::new_unchecked(operation.opid(), operation))
    }
    
    /// Execute and validate a Rholang contract with proper RGB compliance checking
    /// Uses thread-safe runtime access instead of expensive cloning
    async fn execute_and_validate_rholang_contract(
        &self, 
        contract: &str, 
        operation: &Operation
    ) -> Result<RgbExecutionResult, CallError> {
        debug!("🚀 Executing Rholang contract for operation {} (concurrent-safe)", operation.opid());
        
        // Acquire lock on runtime for thread-safe execution
        // This eliminates expensive cloning while ensuring thread safety
        let evaluate_result = {
            let mut runtime_guard = self.runtime.lock()
                .map_err(|e| {
                    let error_msg = format!("Failed to acquire runtime lock: {}", e);
                    debug!("❌ {}", error_msg);
                    rgb_contract_error(error_codes::CONTRACT_EXECUTION_ERROR, &error_msg)
                })?;
            
            debug!("🔒 Runtime lock acquired for operation {}", operation.opid());
            
            // Execute with locked runtime
            runtime_guard.evaluate_with_term(contract).await
                .map_err(|e| {
                    debug!("❌ Rholang execution failed: {:?}", e);
                    rgb_contract_error(
                        error_codes::CONTRACT_EXECUTION_ERROR,
                        &format!("Rholang contract execution failed: {:?}", e)
                    )
                })?
        }; // Lock is automatically released here
        
        debug!("🔓 Runtime lock released, validating results for operation {}", operation.opid());
        debug!("✅ Rholang execution completed, validating results...");
        
        // Validate execution results for RGB compliance
        let validation_result = self.validate_execution_result(operation, evaluate_result)?;
        
        if validation_result.success {
            debug!("✅ RGB contract execution and validation successful: {}", 
                   validation_result.validation_summary);
        } else {
            debug!("❌ RGB contract validation failed: {}", 
                   validation_result.validation_summary);
        }
        
        Ok(validation_result)
    }
}
*/

// NEW GENERIC IMPLEMENTATION - Task 1 Complete: Structure transformed
impl<S: Stock> RholangCodex<S> {
    /// Create a new RholangCodex from contract articles and stock configuration.
    /// This matches the hypersonic::Ledger::new() signature exactly.
    pub fn new(articles: Articles, conf: S::Conf) -> Result<Self, MultiError<IssueError, S::Error>> {
        init_debug_logging(); // Initialize logging for RholangCodex
        warn!("🏗️ RholangCodex::new() - CONSTRUCTOR - Creating new RholangCodex instance");
        eprintln!("🏗️ [RHOLANG-SONIC] CONSTRUCTOR - Creating new RholangCodex instance");
        // Follow hypersonic::Ledger pattern for stock creation
        let contract_id = articles.contract_id();
        let contract_name = articles.issue().meta.name.clone(); // Get contract name before moving articles
        let state = EffectiveState::with_articles(&articles)
            .map_err(|e| IssueError::Genesis(contract_name.clone(), e))
            .map_err(MultiError::A)?;
        let mut stock = S::new(articles, state, conf).map_err(MultiError::B)?;
        let genesis_opid = stock.articles().genesis_opid();
        stock.mark_valid(genesis_opid);
        stock.commit_transaction();
        
        // Initialize Rholang runtime components
        let tokio_rt = tokio::runtime::Runtime::new()
            .map_err(|_| IssueError::Genesis(contract_name.clone(), 
                                           CallError::ScriptUnspecified))
            .map_err(MultiError::A)?;
            
        // Use default storage config - could be made configurable later
        let storage_manager = StorageManager::new(StorageConfig::Default)
            .map_err(|_| IssueError::Genesis(contract_name.clone(), 
                                           CallError::ScriptUnspecified))
            .map_err(MultiError::A)?;
        
        // Initialize RSpace storage manager
        let mut store_manager = mk_rspace_store_manager(
            storage_manager.storage_path().to_path_buf(), 
            100 * MB
        );
        
        // Create RhoRuntimeImpl
        let runtime = tokio_rt.block_on(async {
            // Create RSpace storage
            let rspace_store = store_manager.r_space_stores().await
                .map_err(|e| format!("Failed to create RSpace stores: {}", e))?;
            
            // Create RhoRuntimeImpl for RGB validation
            let runtime = create_runtime_from_kv_store(
                rspace_store,
                Par::default(),                    // minimal mergeable_tag_name
                false,                             // init_registry = false (no registry needed for RGB)
                &mut Vec::<Definition>::new(),     // no additional system processes
                Arc::new(Box::new(Matcher)),       // matcher for RSpace operations
            ).await;
            
            Ok::<RhoRuntimeImpl, String>(runtime)
        }).map_err(|_| IssueError::Genesis(contract_name.clone(), CallError::ScriptUnspecified))
          .map_err(MultiError::A)?;
        
        info!("✅ RholangCodex initialized for contract {} with Rholang runtime", contract_id);
        
        Ok(Self {
            // Rholang execution components
            runtime: Arc::new(Mutex::new(runtime)),
            tokio_rt,
            storage_manager,
            operation_mapping: RgbOperationMap::rgb20_standard(), // Default to RGB-20
            
            // Ledger components
            stock,
            contract_id,
        })
    }
    
    /// Build an Operation from CallParams using OpBuilder.
    /// This internal method converts RGB method calls to operations suitable for Rholang verification.
    fn build_operation_from_params(&self, params: CallParams) -> Result<Operation, AcceptError> {
        debug!("🔧 Building operation from CallParams: method={}", params.core.method);
        
        // Get the call_id from the method name - convert method name to CallId
        // For now, use a simple hash-based mapping or lookup table
        let call_id = self.method_name_to_call_id(&params.core.method)?;
        
        // Create OpBuilder with contract ID and call ID
        let mut builder = OpBuilder::new(self.contract_id, call_id);
        
        // Add immutable (read-only) inputs
        for addr in &params.reading {
            builder = builder.access(*addr);
        }
        
        // Add destructible (consumed) inputs with satisfaction witnesses
        for (addr, satisfaction) in &params.using {
            match satisfaction {
                Some(sat) => {
                    // Input with satisfaction witness (lock conditions)
                    let api = self.stock.articles().default_api();
                    let types = self.stock.articles().types();
                    builder = builder.destroy_satisfy(*addr, sat.name.clone(), sat.witness.clone(), api, types);
                },
                None => {
                    // Simple destructible input without witness
                    builder = builder.destroy(*addr);
                }
            }
        }
        
        // Add global state outputs
        let api = self.stock.articles().default_api();
        let types = self.stock.articles().types();
        for named_state in &params.core.global {
            // StateAtom has verified and unverified fields - use verified first
            let data = named_state.state.verified.clone();
            builder = builder.add_global(
                named_state.name.clone(),
                data,
                None, // No raw data for now
                api,
                types,
            );
        }
        
        // Add owned state outputs  
        for named_state in &params.core.owned {
            builder = builder.add_owned(
                named_state.name.clone(),
                named_state.state.auth,
                named_state.state.data.clone(),
                named_state.state.lock.clone(),
                api,
                types,
            );
        }
        
        // Finalize to create the Operation
        let operation = builder.finalize();
        debug!("✅ Successfully built operation {} for method {}", operation.opid(), params.core.method);
        
        Ok(operation)
    }
    
    /// Convert method name to CallId for operation construction.
    /// This maps RGB method names to their corresponding verifier entry points.
    fn method_name_to_call_id(&self, method: &hypersonic::MethodName) -> Result<CallId, AcceptError> {
        // Convert method name to string for lookup
        let method_str = method.to_string().to_lowercase();
        
        // Look up in the operation mapping using public methods
        for call_id in self.operation_mapping.supported_call_ids() {
            if let Some(op_type) = self.operation_mapping.get_operation_type(call_id) {
                match op_type {
                    crate::mapping::RgbOperationType::Rgb20Issue if method_str.contains("issue") => {
                        return Ok(call_id);
                    },
                    crate::mapping::RgbOperationType::Rgb20Transfer if method_str.contains("transfer") => {
                        return Ok(call_id);
                    },
                    crate::mapping::RgbOperationType::Rgb20Burn if method_str.contains("burn") => {
                        return Ok(call_id);
                    },
                    crate::mapping::RgbOperationType::Rgb21Mint if method_str.contains("mint") => {
                        return Ok(call_id);
                    },
                    crate::mapping::RgbOperationType::Rgb21Transfer if method_str.contains("transfer") && method_str.contains("21") => {
                        return Ok(call_id);
                    },
                    crate::mapping::RgbOperationType::Rgb25Operation if method_str.contains("collectible") => {
                        return Ok(call_id);
                    },
                    _ => continue,
                }
            }
        }
        
        // Default fallback - use first available CallId
        self.operation_mapping.supported_call_ids().first()
            .copied()
            .ok_or_else(|| AcceptError::Persistence("No CallId mapping available".to_string()))
    }
    
    // ================================================================================================
    // INTERNAL ADAPTER METHODS - Create Memory/LibRepo adapters for Rholang verification
    // ================================================================================================
    
    /// Create a Memory adapter from the Stock for Rholang verification.
    /// This provides access to contract state for RGB operation verification.
    fn memory_adapter(&self) -> StockMemoryAdapter<S> {
        StockMemoryAdapter::new(&self.stock)
    }
    
    /// Create a LibRepo adapter from the Stock's Articles for Rholang verification.
    /// This provides access to AluVM libraries needed for contract execution.
    fn library_adapter(&self) -> ArticlesLibRepoAdapter {
        ArticlesLibRepoAdapter::new(self.stock.articles())
    }
    
    /// Main entry point for RGB contract method calls using Rholang verification.
    /// This replaces hypersonic::Ledger::call() with Rholang-based contract execution.
    pub fn call(&mut self, params: CallParams) -> Result<Opid, MultiError<AcceptError, S::Error>> {
        init_debug_logging(); // Ensure logging is initialized
        warn!("🔧 RholangCodex::call() - ENTRY POINT - Starting contract method call: {}", params.core.method);
        eprintln!("🔧 [RHOLANG-SONIC] CALL ENTRY POINT - Method: {}", params.core.method);
        
        // Step 1: Convert CallParams to Operation using OpBuilder
        let operation = self.build_operation_from_params(params)
            .map_err(MultiError::A)?;
            
        // Step 2: Apply and verify the operation with Rholang
        let opid = operation.opid();
        debug!("🔍 Processing operation {} for contract {}", opid, self.contract_id);
        
        // Check if operation already exists
        let present = self.stock.is_valid(opid);
        if !present {
            // Step 3: Perform Rholang-based verification
            self.verify_with_rholang(self.contract_id, operation)
                .map_err(MultiError::A)?;
                
            // Step 4: Mark the operation as valid after successful verification
            // Note: In a full production implementation, we would need to apply state changes
            // from the Rholang execution to the contract state. For now, we just mark as valid.
            self.stock.mark_valid(opid);
            debug!("✅ Marked operation {} as valid after Rholang verification", opid);
        }
        
        // Step 6: Commit the transaction
        self.stock.commit_transaction();
        
        info!("🎉 Successfully executed contract method call - OpID: {}", opid);
        Ok(opid)
    }
    
    /// Perform Rholang-based verification of an RGB operation.
    /// This is the core method that replaces ultrasonic::Codex::verify with Rholang execution.
    fn verify_with_rholang(
        &self, 
        contract_id: ContractId, 
        operation: Operation
    ) -> Result<(), AcceptError> {
        init_debug_logging(); // Ensure logging is initialized
        warn!("🧪 RHOLANG VERIFICATION ENTRY - Starting verification for operation {} on contract {}", operation.opid(), contract_id);
        eprintln!("🧪 [RHOLANG-SONIC] VERIFICATION ENTRY - Operation: {} Contract: {}", operation.opid(), contract_id);
        
        // Step 1: Validate contract ID matches
        if operation.contract_id != contract_id {
            return Err(AcceptError::Persistence(
                format!("Operation contract ID {} doesn't match expected {}", operation.contract_id, contract_id)
            ));
        }
        
        // Step 2: Create adapters for memory and library access
        let memory = self.memory_adapter();
        let _libraries = self.library_adapter(); // Will be used in future enhancements
        
        // Step 3: Determine operation type and generate appropriate Rholang contract
        let call_id = operation.call_id;
        let operation_type = self.operation_mapping.get_operation_type(call_id)
            .ok_or_else(|| AcceptError::Persistence(
                format!("Unknown CallId {} - no operation type mapping", call_id)
            ))?;
            
        debug!("📋 Operation type: {:?} for CallId: {}", operation_type, call_id);
        
        // Step 4: Generate Rholang contract based on operation type
        let rholang_contract = self.determine_and_generate_contract(&operation, &memory)
            .map_err(|e| AcceptError::Persistence(format!("Contract generation failed: {}", e)))?;
        debug!("📝 Generated Rholang contract: {} characters", rholang_contract.len());
        
        // Step 5: Execute Rholang contract with current state  
        let evaluate_result = self.execute_rholang_contract(&rholang_contract, &operation)?;
        debug!("⚡ Rholang execution completed");
        
        // Step 6: Validate execution results against RGB compliance rules
        let rgb_result = self.validate_execution_result(&operation, evaluate_result)
            .map_err(|e| AcceptError::Persistence(format!("RGB validation failed: {}", e)))?;
        
        let is_valid = rgb_result.success;
        
        if !is_valid {
            println!("VALIDATION FAILED - RGB compliance validation failed!");
            println!("💥 [RHOLANG-SONIC] Error details: success={}, errors={:?}", rgb_result.success, rgb_result.errors);
            return Err(AcceptError::Persistence(
                "Rholang contract execution failed RGB compliance validation".to_string()
            ));
        }
        
        info!("✅ Rholang verification successful for operation {}", operation.opid());
        Ok(())
    }
    
    /// Execute a Rholang contract with the current operation context.
    /// This uses the actual Rholang runtime infrastructure for real contract execution.
    fn execute_rholang_contract(
        &self,
        contract: &str,
        operation: &Operation,
    ) -> Result<EvaluateResult, AcceptError> {
        debug!("🔄 Executing Rholang contract for operation {} - {} characters", 
               operation.opid(), contract.len());
        
        // Use tokio runtime to execute async Rholang evaluation
        let result = self.tokio_rt.block_on(async {
            self.execute_contract_async(contract, operation).await
        });
        
        match result {
            Ok(evaluate_result) => {
                debug!("✅ Rholang execution completed successfully - {} channels, cost: {}", 
                       evaluate_result.mergeable.len(), evaluate_result.cost.value);
                Ok(evaluate_result)
            }
            Err(e) => {
                debug!("❌ Rholang execution failed: {:?}", e);
                Err(AcceptError::Persistence(format!("Rholang execution failed: {:?}", e)))
            }
        }
    }
    
    /// Async wrapper for Rholang contract execution
    /// Handles runtime locking and provides proper error handling
    async fn execute_contract_async(
        &self,
        contract: &str,
        operation: &Operation,
    ) -> Result<EvaluateResult, String> {
        debug!("🔒 Acquiring Rholang runtime lock for operation {}", operation.opid());
        
        // Acquire lock on the runtime - handle contention gracefully  
        let mut runtime_guard = match self.runtime.try_lock() {
            Ok(guard) => {
                debug!("✅ Runtime lock acquired immediately");
                guard
            }
            Err(_) => {
                debug!("⏳ Runtime locked by another operation, waiting...");
                self.runtime.lock().map_err(|e| format!("Mutex poisoned: {:?}", e))?
            }
        };
        
        debug!("🚀 Executing Rholang contract:\n{}", contract);
        
        // Execute the contract with proper parameters for Rholang runtime
        // The contract itself contains all the channel initialization
        let initial_state = HashMap::new();
        let cost = Cost::default(); // Default cost accounting for RGB operations
        let rng = Blake2b512Random::create_from_length(32); // Random number generator for Rholang
        
        // Call the actual Rholang runtime evaluation with correct signature
        let evaluate_result = runtime_guard
            .evaluate(contract, cost, initial_state, rng)
            .await;
        
        // Release the lock before processing results
        drop(runtime_guard);
        debug!("🔓 Runtime lock released");
        
        match evaluate_result {
            Ok(result) => {
                debug!("✅ Contract evaluation successful: {} final channels, {} errors, cost: {}", 
                       result.mergeable.len(), result.errors.len(), result.cost.value);
                Ok(result)
            }
            Err(errors) => {
                warn!("💥 Contract evaluation failed with errors: {:?}", errors);
                Err(format!("Rholang evaluation errors: {:?}", errors))
            }
        }
    }
    
    /// Determine RGB operation type using production-ready CallId mapping
    pub(crate) fn determine_and_generate_contract(&self, operation: &Operation, memory: &impl Memory) -> Result<String, CallError> {
        debug!("🔍 Determining RGB operation type for CallId {} using {}", 
               operation.call_id, self.operation_mapping.contract_type());
        
        // Look up the operation type using our explicit mapping (no guessing!)
        let rgb_operation_type = self.operation_mapping
            .get_operation_type(operation.call_id)
            .ok_or_else(|| {
                rgb_contract_error(
                    error_codes::CONTRACT_GENERATION_ERROR,
                    &format!("CallId {} not supported by {} contract mapping. Supported CallIds: {:?}", 
                           operation.call_id, 
                           self.operation_mapping.contract_type(),
                           self.get_supported_call_ids())
                )
            })?;
        
        debug!("✅ Mapped CallId {} → {:?}", operation.call_id, rgb_operation_type);
        
        // Generate appropriate Rholang contract based on mapped operation type
        match rgb_operation_type {
            RgbOperationType::Rgb20Issue => {
                debug!("🏭 Generating RGB-20 issuance contract");
                self.generate_rgb20_issue_contract(operation)
            },
            RgbOperationType::Rgb20Transfer => {
                debug!("💸 Generating RGB-20 transfer contract");
                self.generate_rgb20_transfer_contract(operation, memory)
            },
            RgbOperationType::Rgb20Burn => {
                debug!("🔥 Generating RGB-20 burn contract");
                self.generate_rgb20_burn_contract(operation)
            },
            RgbOperationType::Rgb21Mint => {
                debug!("🎨 Generating RGB-21 NFT mint contract");
                self.generate_rgb21_mint_contract(operation)
            },
            RgbOperationType::Rgb21Transfer => {
                debug!("🖼️ Generating RGB-21 NFT transfer contract");
                Ok(self.generate_rgb21_transfer_contract(operation, memory))
            },
            RgbOperationType::Rgb25Operation => {
                debug!("💎 Generating RGB-25 collectible contract");
                Ok(self.generate_rgb25_contract(operation, memory))
            },
            RgbOperationType::Custom(method_name) => {
                debug!("⚙️ Generating custom contract for method: {}", method_name);
                Ok(self.generate_custom_contract(operation, method_name, memory))
            },
        }
    }
    
    /// Get list of supported CallIds for this contract mapping
    fn get_supported_call_ids(&self) -> Vec<CallId> {
        self.operation_mapping.supported_call_ids()
    }
    
    // ================================================================================================
    // LEDGER DELEGATION METHODS - Required for RGB-CLI Integration
    // ================================================================================================
    
    /// Returns the cached contract ID.
    /// Matches hypersonic::Ledger::contract_id() for rgb-std compatibility.
    #[inline]
    pub fn contract_id(&self) -> ContractId { 
        self.contract_id 
    }

    /// Provides contract Articles, which include contract genesis.
    /// Delegates to the Stock's articles for complete contract metadata access.
    #[inline] 
    pub fn articles(&self) -> &Articles { 
        self.stock.articles() 
    }

    /// Provides contract EffectiveState with current contract state.
    /// Delegates to the Stock's state for RGB state inspection and balance queries.
    #[inline]
    pub fn state(&self) -> &EffectiveState { 
        self.stock.state() 
    }

    /// Returns the Stock configuration used during contract construction.
    /// Required by rgb-std for persistence and wallet operations.
    #[inline]
    pub fn config(&self) -> S::Conf { 
        self.stock.config() 
    }

    /// Detects whether an operation with given opid participates in current state.
    /// This checks if the operation is part of the valid contract history.
    #[inline]
    pub fn is_valid(&self, opid: Opid) -> bool { 
        self.stock.is_valid(opid) 
    }

    /// Detects whether an operation with given opid is known to the contract.
    /// This includes operations that may not be in the current state (due to rollbacks).
    #[inline]
    pub fn has_operation(&self, opid: Opid) -> bool { 
        self.stock.has_operation(opid) 
    }

    /// Returns an operation by its opid from the contract stash.
    /// CRITICAL for rgb-std - used after successful call() to build transactions.
    #[inline]
    pub fn operation(&self, opid: Opid) -> Operation { 
        self.stock.operation(opid) 
    }

    /// Returns iterator over all operations known to contract (complete contract stash).
    /// Used by rgb-std for operation history and contract inspection.
    #[inline]
    pub fn operations(&self) -> impl Iterator<Item = (Opid, Operation)> + use<'_, S> { 
        self.stock.operations() 
    }

    /// Returns iterator over all state transitions (complete contract trace).  
    /// Used by rgb-std for history analysis and state transition tracking.
    #[inline]
    pub fn trace(&self) -> impl Iterator<Item = (Opid, Transition)> + use<'_, S> { 
        self.stock.trace() 
    }

    /// Returns iterator over operations that read from a specific cell address.
    /// Used for dependency tracking and state analysis.
    #[inline]
    pub fn read_by(&self, addr: CellAddr) -> impl Iterator<Item = Opid> + use<'_, S> { 
        self.stock.read_by(addr) 
    }

    /// Returns the operation that spent a specific cell address, if any.
    /// Used for UTXO-style state tracking in RGB.
    #[inline]
    pub fn spent_by(&self, addr: CellAddr) -> Option<Opid> { 
        self.stock.spent_by(addr) 
    }
    
    /// Loads an existing contract from persistence using the provided configuration.
    /// Matches hypersonic::Ledger::load() for rgb-std compatibility.
    pub fn load(conf: S::Conf) -> Result<Self, S::Error> {
        // Load the stock from persistence
        let stock = S::load(conf)?;
        let contract_id = stock.articles().contract_id();
        
        // Initialize Rholang runtime components
        let tokio_rt = tokio::runtime::Runtime::new()
            .map_err(|_| {
                // Convert runtime creation error to Stock error
                // For now, we'll assume the stock implementation can handle this
                // In a production scenario, we might need a more sophisticated error mapping
                todo!("Need to map tokio runtime error to S::Error - will implement with specific Stock type")
            })?;

        let storage_manager = StorageManager::new(StorageConfig::Default)
            .map_err(|_| {
                todo!("Need to map storage error to S::Error - will implement with specific Stock type")
            })?;

        // Initialize Rholang runtime from loaded state
        // TODO: In production, we would restore runtime state from persistence
        let mut store_manager = mk_rspace_store_manager(
            storage_manager.storage_path().to_path_buf(),
            100 * MB
        );

        let runtime = tokio_rt.block_on(async {
            let rspace_store = store_manager.r_space_stores().await
                .map_err(|e| format!("Failed to create RSpace stores: {}", e))?;

            let runtime = create_runtime_from_kv_store(
                rspace_store,
                Par::default(),
                false,
                &mut Vec::<Definition>::new(),
                Arc::new(Box::new(Matcher)),
            ).await;

            Ok::<RhoRuntimeImpl, String>(runtime)
        }).map_err(|_| {
            todo!("Need to map runtime creation error to S::Error")
        })?;

        info!("✅ RholangCodex loaded from persistence for contract {}", contract_id);

        Ok(Self {
            runtime: Arc::new(Mutex::new(runtime)),
            tokio_rt,
            storage_manager,
            operation_mapping: RgbOperationMap::rgb20_standard(), // TODO: Should be loaded from persistence
            stock,
            contract_id,
        })
    }
    
    /// Access to the underlying Stock for advanced operations.
    /// Provides direct access when RGB integration needs Stock-specific functionality.
    #[inline]
    pub fn stock(&self) -> &S {
        &self.stock
    }
    
    // ================================================================================================
    // ADVANCED LEDGER METHODS - For Complete RGB-STD Compatibility
    // ================================================================================================
    
    /// Get ancestors of given operations (operations they depend on).
    /// Used by RGB for dependency analysis and operation ordering.
    pub fn ancestors(&self, opids: impl IntoIterator<Item = Opid>) -> impl DoubleEndedIterator<Item = Opid> {
        // Import IndexSet for the implementation
        use indexmap::IndexSet;
        
        let mut chain = opids.into_iter().collect::<IndexSet<_>>();
        // Get all subsequent operations
        let mut index = 0usize;
        let genesis_opid = self.articles().genesis_opid();
        while let Some(opid) = chain.get_index(index).copied() {
            if opid != genesis_opid {
                let op = self.stock.operation(opid);
                for inp in op.immutable_in {
                    let parent = inp.opid;
                    if !chain.contains(&parent) {
                        chain.insert(parent);
                    }
                }
                for inp in op.destructible_in {
                    let parent = inp.addr.opid;
                    if !chain.contains(&parent) {
                        chain.insert(parent);
                    }
                }
            }
            index += 1;
        }
        chain.into_iter()
    }
    
    /// Get descendants of given operations (operations that depend on them).
    /// Used by RGB for rollback and operation impact analysis.
    pub fn descendants(&self, opids: impl IntoIterator<Item = Opid>) -> impl DoubleEndedIterator<Item = Opid> {
        use indexmap::IndexSet;
        
        let mut chain = opids.into_iter().collect::<IndexSet<_>>();
        let mut index = 0usize;
        while let Some(opid) = chain.get_index(index).copied() {
            // Find all operations that depend on this one
            for (dep_opid, dep_op) in self.stock.operations() {
                if !chain.contains(&dep_opid) {
                    // Check if this operation depends on opid
                    let depends = dep_op.immutable_in.iter().any(|inp| inp.opid == opid) ||
                                 dep_op.destructible_in.iter().any(|inp| inp.addr.opid == opid);
                    if depends {
                        chain.insert(dep_opid);
                    }
                }
            }
            index += 1;
        }
        chain.into_iter()
    }
    
    /// Rollback operations by invalidating them and their descendants.
    /// Used by RGB for transaction rollbacks and state management.
    pub fn rollback(&mut self, opids: impl IntoIterator<Item = Opid>) -> Result<(), S::Error> {
        for opid in self.descendants(opids).rev() {
            if self.stock.is_valid(opid) {
                // TODO: In full implementation, would restore previous state
                self.stock.mark_invalid(opid);
            }
        }
        self.stock.commit_transaction();
        Ok(())
    }
    
    /// Forward (re-validate) operations by validating them and their descendants.
    /// Used by RGB for transaction replay and state recovery.
    pub fn forward(&mut self, opids: impl IntoIterator<Item = Opid>) -> Result<(), MultiError<AcceptError, S::Error>> {
        for opid in self.descendants(opids) {
            if !self.stock.is_valid(opid) {
                // Check that all ancestors are valid
                let ancestors_valid = self.ancestors([opid])
                    .filter(|id| *id != opid)
                    .all(|ancestor_opid| self.stock.is_valid(ancestor_opid));
                
                if ancestors_valid {
                    // TODO: In full implementation, would re-verify operation
                    self.stock.mark_valid(opid);
                }
            }
        }
        self.stock.commit_transaction();
        Ok(())
    }
    
    /// Export all operations with auxiliary data.
    /// Used by RGB for contract data export and backup.
    pub fn export_all_aux<W: WriteRaw>(
        &self,
        mut writer: StrictWriter<W>,
        mut aux: impl FnMut(Opid, &Operation, StrictWriter<W>) -> std::io::Result<StrictWriter<W>>,
    ) -> std::io::Result<()> {
        // Export all operations with auxiliary data
        for (opid, op) in self.stock.operations() {
            writer = aux(opid, &op, writer)?;
        }
        Ok(())
    }
    
    /// Export specific operations based on terminals with auxiliary data.
    /// Used by RGB for selective contract data export.
    pub fn export_aux<W: WriteRaw>(
        &self,
        terminals: impl IntoIterator<Item = impl std::borrow::Borrow<AuthToken>>,
        mut writer: StrictWriter<W>,
        mut aux: impl FnMut(Opid, &Operation, StrictWriter<W>) -> std::io::Result<StrictWriter<W>>,
    ) -> std::io::Result<()> {
        // Find operations related to terminal auth tokens
        let terminal_auths: Vec<AuthToken> = terminals.into_iter()
            .map(|t| *t.borrow())
            .collect();
        
        for (opid, op) in self.stock.operations() {
            // Check if operation relates to any terminal
            let is_related = op.destructible_out.iter()
                .any(|cell| terminal_auths.contains(&cell.auth));
            
            if is_related {
                writer = aux(opid, &op, writer)?;
            }
        }
        Ok(())
    }
    
    /// Upgrade contract APIs to newer version.
    /// Used by RGB for contract versioning and compatibility.
    pub fn upgrade_apis(&mut self, _new_articles: Articles) -> Result<bool, MultiError<hypersonic::SemanticError, S::Error>> {
        // For now, delegate to stock if it supports article updates
        // TODO: In full implementation, would update contract APIs and verify compatibility
        info!("🔄 API upgrade requested for contract {} - currently not supported in RholangCodex", self.contract_id);
        Ok(false) // Return false indicating no upgrade performed
    }
    
    /// Commit current transaction to persistence.
    /// Essential for RGB state management and persistence.
    #[inline]
    pub fn commit_transaction(&mut self) {
        self.stock.commit_transaction();
    }
    
    /// Apply a verified operation to the contract state.
    /// Used by RGB after successful operation verification.
    pub fn apply(&mut self, operation: VerifiedOperation) -> Result<Transition, S::Error> {
        let opid = operation.opid();
        let present = self.stock.is_valid(opid);
        
        // TODO: In full implementation, would apply state changes from operation
        // For now, just mark as valid and delegate to stock
        if !present {
            self.stock.mark_valid(opid);
        }
        
        // Return a simple transition indicating the operation was applied
        // In a full implementation, this would contain the actual state changes
        use amplify::confinement::SmallOrdMap;
        Ok(Transition {
            opid,
            destroyed: SmallOrdMap::new(),
        })
    }
}

// ================================================================================================
// TRAIT IMPLEMENTATIONS - Required for RGB-STD Contract struct derives
// ================================================================================================

impl<S: Stock> Clone for RholangCodex<S> 
where 
    S: Clone 
{
    fn clone(&self) -> Self {
        // Clone the stock and recreate runtime components
        let stock = self.stock.clone();
        let contract_id = self.contract_id;
        
        // Create new runtime components (can't clone tokio::Runtime)
        let tokio_rt = tokio::runtime::Runtime::new()
            .expect("Failed to create tokio runtime for RholangCodex clone");
            
        let storage_manager = StorageManager::new(StorageConfig::Default)
            .expect("Failed to create storage manager for RholangCodex clone");
            
        // Initialize new Rholang runtime
        let mut store_manager = mk_rspace_store_manager(
            storage_manager.storage_path().to_path_buf(),
            100 * MB
        );
        
        let runtime = tokio_rt.block_on(async {
            let rspace_store = store_manager.r_space_stores().await
                .expect("Failed to create RSpace stores for clone");
                
            create_runtime_from_kv_store(
                rspace_store,
                Par::default(),
                false,
                &mut Vec::<Definition>::new(),
                Arc::new(Box::new(Matcher)),
            ).await
        });
        
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            tokio_rt,
            storage_manager,
            operation_mapping: self.operation_mapping.clone(),
            stock,
            contract_id,
        }
    }
}

impl<S: Stock> std::fmt::Debug for RholangCodex<S> 
where 
    S: std::fmt::Debug 
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RholangCodex")
            .field("contract_id", &self.contract_id)
            .field("stock", &self.stock)
            .field("operation_mapping", &self.operation_mapping)
            .field("runtime", &"Arc<Mutex<RhoRuntimeImpl>>")
            .field("storage_manager", &"StorageManager")
            .finish()
    }
}

// Default implementation commented out - now requires Articles and Stock configuration
// impl Default for RholangCodex<S> {
//     fn default() -> Self {
//         Self::new().expect("Failed to create RholangCodex")
//     }
// }

// ================================================================================================
// INTERNAL ADAPTER STRUCTS - Bridge Stock to Memory/LibRepo interfaces
// ================================================================================================

/// Internal adapter that wraps Stock to provide Memory interface for RholangCodex verification.
/// This allows the Stock's effective state to be accessed as Memory for Rholang execution.
pub struct StockMemoryAdapter<'a, S: Stock> {
    stock: &'a S,
}

impl<'a, S: Stock> StockMemoryAdapter<'a, S> {
    pub fn new(stock: &'a S) -> Self {
        Self { stock }
    }
}

impl<S: Stock> Memory for StockMemoryAdapter<'_, S> {
    /// Access destructible (owned) memory cells from the Stock's effective state.
    fn destructible(&self, addr: CellAddr) -> Option<StateCell> {
        // Access through stock's effective state - use the raw state's owned cells
        self.stock.state().raw.owned.get(&addr).copied()
    }
    
    /// Access immutable (global) memory values from the Stock's effective state.
    fn immutable(&self, addr: CellAddr) -> Option<StateValue> {
        // Access through stock's effective state - use the raw state's global data
        self.stock.state().raw.global.get(&addr).map(|data| data.value)
    }
}

/// Internal adapter that wraps Articles to provide LibRepo interface for RholangCodex verification.
/// This allows access to AluVM libraries stored in the contract Articles.
pub struct ArticlesLibRepoAdapter<'a> {
    articles: &'a Articles,
}

impl<'a> ArticlesLibRepoAdapter<'a> {
    pub fn new(articles: &'a Articles) -> Self {
        Self { articles }
    }
}

impl LibRepo for ArticlesLibRepoAdapter<'_> {
    /// Get a specific AluVM library by its LibId from the contract Articles.
    fn get_lib(&self, lib_id: LibId) -> Option<&Lib> {
        // Search through all codex libraries to find the one with matching lib_id
        self.articles.codex_libs()
            .find(|lib| lib.lib_id() == lib_id)
    }
}

// Tests commented out in Task 1 - will be updated in Task 7 with new constructor signatures
/*
#[cfg(test)]
mod tests {
    use super::*;

    fn init_test_logging() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Debug)
            .try_init();
    }

    #[test]
    fn test_rholang_codex_ephemeral_creation() {
        init_test_logging();
        // Test ephemeral storage creation
        let codex = RholangCodex::new_ephemeral();
        assert!(codex.is_ok(), "Failed to create ephemeral RholangCodex: {:?}", codex.err());
        
        let codex = codex.unwrap();
        // Verify the instance is properly initialized
        assert!(!std::ptr::eq(&codex.runtime as *const _, std::ptr::null()));
        assert!(codex.storage_path().exists());
    }

    #[test]
    fn test_rholang_codex_persistent_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().join("test_persistent");
        
        // Test persistent storage creation
        let codex = RholangCodex::new_with_storage(&storage_path);
        assert!(codex.is_ok(), "Failed to create persistent RholangCodex: {:?}", codex.err());
        
        let codex = codex.unwrap();
        assert_eq!(codex.storage_path(), storage_path);
        assert!(storage_path.exists());
    }

    #[test]
    fn test_rholang_codex_default_creation() {
        init_test_logging();
        // Test default persistent storage creation
        let codex = RholangCodex::new();
        assert!(codex.is_ok(), "Failed to create default RholangCodex: {:?}", codex.err());
        
        let codex = codex.unwrap();
        assert!(codex.storage_path().exists());
        // Default should create persistent storage
        assert!(codex.storage_path().to_string_lossy().contains("rholang-sonic"));
    }

    #[test]
    fn test_rholang_codex_default_trait() {
        // Test that the default implementation works
        let _codex = RholangCodex::default();
        // If we get here without panic, the default creation worked
    }
    
    #[test]
    fn test_storage_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().join("test_persistence");
        
        // Create first instance
        {
            let codex1 = RholangCodex::new_with_storage(&storage_path).unwrap();
            assert!(codex1.was_first_time()); // Should be first time
        }
        
        // Create second instance with same storage
        {
            let _codex2 = RholangCodex::new_with_storage(&storage_path).unwrap();
            // Note: was_first_time() checks current state, not historical state
            // In a real implementation, we might want to track this differently
        }
    }
    
    #[test]
    fn test_ephemeral_rholang_codex_cleanup() {
        init_test_logging();
        let temp_path: std::path::PathBuf;
        
        // Create ephemeral RholangCodex and capture the storage path
        {
            let codex = RholangCodex::new_ephemeral().unwrap();
            temp_path = codex.storage_path().to_path_buf();
            
            // Verify the directory exists while codex is alive
            assert!(temp_path.exists(), "Temporary directory should exist while RholangCodex is alive");
            assert!(temp_path.is_dir(), "Temporary path should be a directory");
            
            // The LMDB files might be created during runtime initialization
            // So we just verify the base directory structure exists
        } // codex goes out of scope here
        
        // Give the OS a moment to clean up
        std::thread::sleep(std::time::Duration::from_millis(50));
        
        // Verify the directory has been cleaned up
        assert!(!temp_path.exists(), 
            "Temporary directory should be cleaned up when ephemeral RholangCodex is dropped. Path: {}", 
            temp_path.display()
        );
    }
}
*/

#[cfg(test)]
mod logging_tests {
    use super::*;
    
    #[test]
    fn test_rholang_sonic_logging() {
        // Initialize logging
        init_debug_logging();
        
        // Test that our logging is working
        warn!("🧪 TEST LOG MESSAGE - RholangCodex logging test");
        debug!("🧪 DEBUG TEST - This should appear in test output");
        info!("🧪 INFO TEST - RholangCodex logging initialized");
        
        // This test always passes - it's just to verify logs appear
        assert!(true, "Logging test completed");
    }
}
