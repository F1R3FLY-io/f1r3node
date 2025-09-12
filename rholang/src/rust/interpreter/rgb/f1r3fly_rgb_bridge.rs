// F1r3fly RhoLang bridge for clean RGB integration
// Minimal bridge between RhoLang system processes and production RGB processor

use super::rgb20_processor::Rgb20Processor;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;

use std::path::PathBuf;

/// Bridge for integrating RGB with F1r3fly RhoLang system
pub struct F1r3flyRgbBridge {
    rgb_processor: Rgb20Processor,
}

impl F1r3flyRgbBridge {
    /// Initialize bridge with RGB processor
    pub fn new(storage_path: PathBuf) -> Result<Self, InterpreterError> {
        let rgb_processor = Rgb20Processor::new(storage_path)
            .map_err(|e| InterpreterError::BugFoundError(format!("RGB processor init failed: {}", e)))?;
        
        Ok(Self { rgb_processor })
    }
    
    /// Process RGB request from RhoLang system_processes
    pub fn process_rgb_request(&mut self, request_data: &Par) -> Result<Par, InterpreterError> {
        println!("🌉 F1r3fly RGB Bridge: Processing request");
        
        // Parse request from RhoLang
        let request = Rgb20Processor::parse_request_from_rholang(request_data)
            .map_err(|e| InterpreterError::BugFoundError(format!("Failed to parse request: {}", e)))?;
        
        println!("🌉 Bridge: Parsed RGB20 issuance request for: {}", request.ticker);
        
        // Process with RGB processor
        let response = self.rgb_processor.issue_rgb20(request)
            .map_err(|e| InterpreterError::BugFoundError(format!("RGB20 issuance failed: {}", e)))?;
        
        println!("🌉 Bridge: RGB20 contract created: {}", response.contract_id);
        
        // Serialize response back to RhoLang
        let rholang_response = Rgb20Processor::serialize_response_to_rholang(&response)
            .map_err(|e| InterpreterError::BugFoundError(format!("Failed to serialize response: {}", e)))?;
        
        println!("✅ F1r3fly RGB Bridge: Request processed successfully");
        
        Ok(rholang_response)
    }
    
    /// Get the RGB processor (for direct access if needed)
    pub fn rgb_processor(&mut self) -> &mut Rgb20Processor {
        &mut self.rgb_processor
    }
}

/// Helper functions for system_processes integration
impl F1r3flyRgbBridge {
    /// Create bridge instance for system processes
    pub fn create_for_system_processes() -> Result<Self, InterpreterError> {
        let storage_path = PathBuf::from("./rgb_storage");
        Self::new(storage_path)
    }
    
    /// Validate RGB request before processing
    pub fn validate_request(request_data: &Par) -> Result<(), InterpreterError> {
        let _request = Rgb20Processor::parse_request_from_rholang(request_data)
            .map_err(|e| InterpreterError::BugFoundError(format!("Invalid RGB request: {}", e)))?;
        
        Ok(())
    }
}
