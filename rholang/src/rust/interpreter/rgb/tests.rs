#[cfg(test)]
mod tests {
    use crate::rust::interpreter::rgb::{
        RgbProcessor, RgbStateTransitionRequest, RgbOutput, RgbMetadata
    };
    use crate::rust::interpreter::rgb::processor::{GraphSeal, Txid};
    use models::rhoapi::{Par, Expr, expr::ExprInstance};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Helper function to create a test RgbProcessor with temp directory
    fn create_test_processor_with_temp_dir() -> (RgbProcessor, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let mut processor = RgbProcessor::new();
        processor.storage_path = temp_dir.path().to_path_buf();
        (processor, temp_dir)
    }

    // Helper function to create a test RgbProcessor
    fn create_test_processor() -> RgbProcessor {
        RgbProcessor::new()
    }

    // Helper function to create a test Par with string content
    fn create_string_par(content: &str) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(content.to_string())),
            }],
            sends: vec![],
            receives: vec![],
            news: vec![],
            matches: vec![],
            unforgeables: vec![],
            bundles: vec![],
            connectives: vec![],
            locally_free: vec![],
            connective_used: false,
        }
    }

    // Create test RGB20 issuance request
    fn create_test_rgb20_request() -> RgbStateTransitionRequest {
        RgbStateTransitionRequest {
            contract_type: "RGB20".to_string(),
            operation: "issue".to_string(),
            inputs: vec![],
            outputs: vec![RgbOutput {
                utxo: "887c100c1fa0aba98e60e40cfa50cb7e05aac61f7d2c704c029914553f37fcdd:535".to_string(),
                asset_id: "f1r3fly_demo_token".to_string(),
                amount: 1000000,
            }],
            metadata: Some(RgbMetadata {
                description: Some("F1r3fly Demo Token".to_string()),
                ticker: Some("DEMO".to_string()),
                precision: Some(8),
                custom_data: None,
            }),
        }
    }

    #[test]
    fn test_rgb_processor_creation() {
        let processor = RgbProcessor::default();
        assert_eq!(processor.storage_path, PathBuf::from("./rgb_storage"));
        
        let processor_new = create_test_processor();
        assert_eq!(processor_new.storage_path, PathBuf::from("./rgb_storage"));
    }

    #[test]
    fn test_parse_json_request_valid() {
        let json_data = r#"{"contract_type": "RGB20", "operation": "issue", "inputs": [], "outputs": [{"utxo": "887c100c1fa0aba98e60e40cfa50cb7e05aac61f7d2c704c029914553f37fcdd:535", "asset_id": "f1r3fly_demo_token", "amount": 1000000}], "metadata": {"description": "F1r3fly Demo Token", "ticker": "DEMO", "precision": 8}}"#;
        let par = create_string_par(json_data);
        
        let result = RgbProcessor::parse_rgb_state_transition_request(&par);
        assert!(result.is_ok());
        
        let request = result.unwrap();
        assert_eq!(request.contract_type, "RGB20");
        assert_eq!(request.operation, "issue");
        assert_eq!(request.outputs.len(), 1);
        assert_eq!(request.outputs[0].asset_id, "f1r3fly_demo_token");
        assert_eq!(request.outputs[0].amount, 1000000);
    }

    #[test]
    fn test_parse_json_request_invalid() {
        let invalid_json = r#"{"invalid": "json"#;
        let par = create_string_par(invalid_json);
        
        let result = RgbProcessor::parse_rgb_state_transition_request(&par);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rgb_request() {
        let request = create_test_rgb20_request();
        let result = RgbProcessor::validate_rgb_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_invalid_contract_type() {
        let mut request = create_test_rgb20_request();
        request.contract_type = "INVALID".to_string();
        
        let result = RgbProcessor::validate_rgb_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_operation() {
        let mut request = create_test_rgb20_request();
        request.operation = "invalid_op".to_string();
        
        let result = RgbProcessor::validate_rgb_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_empty_utxo() {
        let mut request = create_test_rgb20_request();
        request.outputs[0].utxo = "".to_string();
        
        let result = RgbProcessor::validate_rgb_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_zero_amount() {
        let mut request = create_test_rgb20_request();
        request.outputs[0].amount = 0;
        
        let result = RgbProcessor::validate_rgb_request(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_contract_id() {
        let processor = create_test_processor();
        let outpoint = RgbProcessor::parse_utxo_string("887c100c1fa0aba98e60e40cfa50cb7e05aac61f7d2c704c029914553f37fcdd:535").unwrap();
        
        let contract_id_1 = processor.generate_contract_id(&outpoint, "DEMO").unwrap();
        let contract_id_2 = processor.generate_contract_id(&outpoint, "DEMO").unwrap();
        
        // Same inputs should produce same contract ID
        assert_eq!(contract_id_1, contract_id_2);
        assert!(contract_id_1.starts_with("rgb1"));
        
        // Different ticker should produce different contract ID
        let contract_id_3 = processor.generate_contract_id(&outpoint, "OTHER").unwrap();
        assert_ne!(contract_id_1, contract_id_3);
    }

    #[test]
    fn test_binary_consignment_creation() {
        let (processor, _temp_dir) = create_test_processor_with_temp_dir();
        let request = create_test_rgb20_request();
        
        // Create state transition which should generate binary files
        let result = processor.create_state_transition(request);
        assert!(result.is_ok());
        
        let state_transition = result.unwrap();
        
        // Check that consignment directory was created
        let consignments_dir = processor.storage_path.join("consignments");
        assert!(consignments_dir.exists());
        
        // Check that initial allocation consignment was created
        let allocation_file = consignments_dir.join(format!("{}_initial_allocation.consignment", state_transition.contract_id));
        assert!(allocation_file.exists());
        
        // Check that the file contains binary data (not JSON)
        let file_contents = fs::read(&allocation_file).expect("Failed to read allocation file");
        assert!(!file_contents.is_empty());
        
        // Should not be valid JSON (since it's binary)
        let json_parse_result = serde_json::from_slice::<serde_json::Value>(&file_contents);
        assert!(json_parse_result.is_err());
        
        println!("✅ Binary consignment created at: {:?}", allocation_file);
        println!("✅ File size: {} bytes", file_contents.len());
    }

    #[test]
    fn test_transfer_consignment_creation() {
        let (processor, _temp_dir) = create_test_processor_with_temp_dir();
        let request = create_test_rgb20_request();
        
        // Create state transition
        let state_transition = processor.create_state_transition(request).unwrap();
        
        // Create transfer consignment
        let consignment_id = processor.create_consignment_template(&state_transition).unwrap();
        
        // Check that transfer consignment was created
        let consignment_file = processor.storage_path
            .join("consignments")
            .join(format!("{}.consignment", consignment_id));
        assert!(consignment_file.exists());
        
        // Check that the file contains binary data
        let file_contents = fs::read(&consignment_file).expect("Failed to read consignment file");
        assert!(!file_contents.is_empty());
        
        // Should not be valid JSON (since it's binary)
        let json_parse_result = serde_json::from_slice::<serde_json::Value>(&file_contents);
        assert!(json_parse_result.is_err());
        
        println!("✅ Transfer consignment created at: {:?}", consignment_file);
        println!("✅ File size: {} bytes", file_contents.len());
    }

    #[test]
    fn test_mpc_commitment_generation() {
        let processor = create_test_processor();
        let request = create_test_rgb20_request();
        let state_transition = processor.create_state_transition(request).unwrap();
        
        let mpc_commitment = processor.generate_mpc_commitment(&state_transition).unwrap();
        
        assert!(mpc_commitment.starts_with("mpc_"));
        assert!(mpc_commitment.len() > 4); // More than just "mpc_"
        
        // Same input should produce same commitment
        let mpc_commitment_2 = processor.generate_mpc_commitment(&state_transition).unwrap();
        assert_eq!(mpc_commitment, mpc_commitment_2);
        
        println!("✅ MPC commitment: {}", mpc_commitment);
    }

    #[test]
    fn test_complete_rgb20_issuance_flow() {
        let (processor, temp_dir) = create_test_processor_with_temp_dir();
        let request = create_test_rgb20_request();
        
        // 1. Validate request
        let validation_result = RgbProcessor::validate_rgb_request(&request);
        assert!(validation_result.is_ok());
        println!("✅ Request validation passed");

        // 2. Create state transition (should generate binary files)
        let state_transition_result = processor.create_state_transition(request);
        assert!(state_transition_result.is_ok());
        let state_transition = state_transition_result.unwrap();
        println!("✅ State transition created: {}", state_transition.contract_id);
        
        // 3. Generate MPC commitment
        let mpc_result = processor.generate_mpc_commitment(&state_transition);
        assert!(mpc_result.is_ok());
        let mpc_commitment = mpc_result.unwrap();
        println!("✅ MPC commitment generated: {}", mpc_commitment);
        
        // 4. Create transfer consignment
        let template_result = processor.create_consignment_template(&state_transition);
        assert!(template_result.is_ok());
        let consignment_id = template_result.unwrap();
        println!("✅ Transfer consignment created: {}", consignment_id);
        
        // 5. Verify all files exist and are binary
        let base_path = temp_dir.path();
        let contracts_dir = base_path.join("consignments");
        assert!(contracts_dir.exists());
        
        // Check initial allocation file
        let allocation_file = contracts_dir.join(format!("{}_initial_allocation.consignment", state_transition.contract_id));
        assert!(allocation_file.exists());
        let allocation_contents = fs::read(&allocation_file).unwrap();
        assert!(!allocation_contents.is_empty());
        
        // Check transfer consignment file  
        let transfer_file = contracts_dir.join(format!("{}.consignment", consignment_id));
        assert!(transfer_file.exists());
        let transfer_contents = fs::read(&transfer_file).unwrap();
        assert!(!transfer_contents.is_empty());
        
        // Verify files are binary (not JSON)
        assert!(serde_json::from_slice::<serde_json::Value>(&allocation_contents).is_err());
        assert!(serde_json::from_slice::<serde_json::Value>(&transfer_contents).is_err());
        
        println!("✅ All binary files created successfully");
        println!("   - Initial allocation: {} bytes", allocation_contents.len());
        println!("   - Transfer consignment: {} bytes", transfer_contents.len());
        println!("✅ Complete RGB20 issuance flow test passed");
    }

    #[test]
    fn test_binary_file_structure() {
        let (processor, _temp_dir) = create_test_processor_with_temp_dir();
        let _request = create_test_rgb20_request();
        
        // Create binary consignment
        let binary_data = processor.create_initial_allocation_consignment(
            "rgb1abc123def456",
            "DEMO", 
            1000000,
            8,
            GraphSeal {
                txid: Txid::new([0u8; 32]),
                vout: 535,
            }
        ).unwrap();
        
        assert!(!binary_data.is_empty());
        assert!(binary_data.len() > 50); // Should have substantial data
        
        // Check that it contains our expected data patterns
        let data_str = String::from_utf8_lossy(&binary_data);
        assert!(data_str.contains("rgb1abc123def456")); // Contract ID
        assert!(data_str.contains("DEMO")); // Ticker
        
        println!("✅ Binary structure test passed - {} bytes generated", binary_data.len());
    }
}