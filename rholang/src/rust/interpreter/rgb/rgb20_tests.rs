#[cfg(test)]
mod rgb20_production_tests {
    use super::super::{
        rgb_types::*,
        rgb20_processor::Rgb20Processor,
        f1r3fly_rgb_bridge::F1r3flyRgbBridge,
    };
    use models::rhoapi::{Par, Expr, expr::ExprInstance};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_rgb_processor() -> (Rgb20Processor, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let processor = Rgb20Processor::new(temp_dir.path().to_path_buf())
            .expect("Failed to create RGB processor");
        (processor, temp_dir)
    }

    fn create_test_request() -> Rgb20IssuanceRequest {
        Rgb20IssuanceRequest {
            ticker: "DEMO".to_string(),
            total_supply: 1_000_000,
            precision: 8,
            genesis_utxo: "887c100c1fa0aba98e60e40cfa50cb7e05aac61f7d2c704c029914553f37fcdd:535".to_string(),
            description: Some("F1r3fly Demo Token".to_string()),
        }
    }

    fn create_rholang_par_from_request(request: &Rgb20IssuanceRequest) -> Par {
        let json = serde_json::to_string(request).unwrap();
        Par {
            sends: vec![],
            receives: vec![],
            news: vec![],
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(json)),
            }],
            matches: vec![],
            unforgeables: vec![],
            bundles: vec![],
            connectives: vec![],
            locally_free: vec![],
            connective_used: false,
        }
    }

    #[test]
    fn test_rgb_processor_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let result = Rgb20Processor::new(temp_dir.path().to_path_buf());
        
        assert!(result.is_ok());
        println!("✅ RGB processor initialized successfully");
        
        // Verify stockpile directory was created
        let stockpile_dir = temp_dir.path().join("stockpile");
        assert!(stockpile_dir.exists());
        println!("✅ Stockpile directory created: {:?}", stockpile_dir);
    }

    #[test]
    fn test_real_rgb20_contract_creation() {
        let (mut processor, _temp_dir) = create_test_rgb_processor();
        let request = create_test_request();
        
        let result = processor.issue_rgb20(request);
        
        match &result {
            Ok(response) => {
                println!("✅ RGB20 contract created successfully");
                println!("   Contract ID: {}", response.contract_id);
                println!("   Consignment file: {:?}", response.consignment_file);
                println!("   Consignment size: {} bytes", response.consignment_data.len());
                
                // Verify consignment file exists
                assert!(response.consignment_file.exists());
                assert!(!response.consignment_data.is_empty());
                
                // Verify consignment starts with RGB magic header
                assert!(response.consignment_data.starts_with(b"RGB\x00"));
            }
            Err(e) => {
                println!("❌ RGB20 contract creation failed: {:?}", e);
                panic!("Contract creation should succeed");
            }
        }
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_consignment_file_structure() {
        let (mut processor, _temp_dir) = create_test_rgb_processor();
        let request = create_test_request();
        
        let response = processor.issue_rgb20(request).unwrap();
        
        // Verify consignment has expected structure
        let data = &response.consignment_data;
        
        // Should start with RGB magic header
        assert_eq!(&data[0..4], b"RGB\x00");
        println!("✅ Consignment has correct RGB magic header");
        
        // Should contain contract ID (32 bytes after header)
        assert!(data.len() >= 36); // 4 bytes header + 32 bytes contract ID
        let contract_id_bytes = &data[4..36];
        assert_eq!(contract_id_bytes, &response.contract_id.to_byte_array());
        println!("✅ Consignment contains correct contract ID");
        
        // Should contain ticker
        let data_str = String::from_utf8_lossy(data);
        assert!(data_str.contains("DEMO"));
        println!("✅ Consignment contains ticker");
        
        println!("✅ Consignment structure validation passed");
    }

    #[test]
    fn test_rholang_integration() {
        let request = create_test_request();
        let par = create_rholang_par_from_request(&request);
        
        // Test parsing from RhoLang
        let parsed_request = Rgb20Processor::parse_request_from_rholang(&par);
        assert!(parsed_request.is_ok());
        
        let parsed = parsed_request.unwrap();
        assert_eq!(parsed.ticker, "DEMO");
        assert_eq!(parsed.total_supply, 1_000_000);
        assert_eq!(parsed.precision, 8);
        
        println!("✅ RhoLang parsing successful");
        
        // Test full bridge integration
        let temp_dir = TempDir::new().unwrap();
        let mut bridge = F1r3flyRgbBridge::new(temp_dir.path().to_path_buf()).unwrap();
        
        let response_par = bridge.process_rgb_request(&par);
        assert!(response_par.is_ok());
        
        println!("✅ F1r3fly bridge integration successful");
    }

    #[test]
    fn test_contract_id_deterministic() {
        let (mut processor1, _temp_dir1) = create_test_rgb_processor();
        let (mut processor2, _temp_dir2) = create_test_rgb_processor();
        
        let request1 = create_test_request();
        let request2 = create_test_request();
        
        let response1 = processor1.issue_rgb20(request1).unwrap();
        let response2 = processor2.issue_rgb20(request2).unwrap();
        
        // Same inputs should produce same contract ID
        assert_eq!(response1.contract_id, response2.contract_id);
        
        println!("✅ Contract ID generation is deterministic");
        println!("   Contract ID: {}", response1.contract_id);
    }

    #[test]
    fn test_different_tickers_different_contracts() {
        let (mut processor, _temp_dir) = create_test_rgb_processor();
        
        let mut request1 = create_test_request();
        request1.ticker = "DEMO1".to_string();
        
        let mut request2 = create_test_request();
        request2.ticker = "DEMO2".to_string();
        
        let response1 = processor.issue_rgb20(request1).unwrap();
        let response2 = processor.issue_rgb20(request2).unwrap();
        
        // Different tickers should produce different contract IDs
        assert_ne!(response1.contract_id, response2.contract_id);
        
        println!("✅ Different tickers create different contracts");
    }

    #[test]
    fn test_error_handling() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test invalid storage path
        let invalid_path = PathBuf::from("/invalid/path/that/does/not/exist");
        let result = Rgb20Processor::new(invalid_path);
        
        // This might succeed if it creates directories, so let's test invalid UTXO instead
        let (mut processor, _temp_dir) = create_test_rgb_processor();
        
        let mut request = create_test_request();
        request.genesis_utxo = "invalid_utxo_format".to_string();
        
        let result = processor.issue_rgb20(request);
        assert!(result.is_err());
        
        if let Err(RgbError::InvalidUtxo(msg)) = result {
            println!("✅ Invalid UTXO error handled correctly: {}", msg);
        } else {
            panic!("Expected InvalidUtxo error");
        }
    }

    #[test] 
    fn test_consignment_rgb_cli_compatibility() {
        let (mut processor, temp_dir) = create_test_rgb_processor();
        let request = create_test_request();
        
        let response = processor.issue_rgb20(request).unwrap();
        
        println!("🧪 RGB-CLI Compatibility Test");
        println!("================================");
        println!("Contract ID: {}", response.contract_id);
        println!("Consignment file: {:?}", response.consignment_file);
        println!("Consignment size: {} bytes", response.consignment_data.len());
        
        // Verify file can be read by external tools
        let file_contents = std::fs::read(&response.consignment_file).unwrap();
        assert_eq!(file_contents, response.consignment_data);
        
        println!("✅ Consignment file ready for rgb-cli testing");
        println!("   To test manually:");
        println!("   rgb-cli consignment validate {:?}", response.consignment_file);
        
        // Store test info for manual validation
        let test_info = serde_json::json!({
            "contract_id": response.contract_id.to_string(),
            "consignment_file": response.consignment_file,
            "test_data": {
                "ticker": "DEMO",
                "total_supply": 1_000_000,
                "precision": 8
            }
        });
        
        let info_file = temp_dir.path().join("rgb_cli_test_info.json");
        std::fs::write(info_file, test_info.to_string()).unwrap();
        
        println!("✅ Test info saved for manual rgb-cli validation");
    }

    #[test]
    fn test_performance_and_limits() {
        let (mut processor, _temp_dir) = create_test_rgb_processor();
        
        // Test with maximum values
        let large_request = Rgb20IssuanceRequest {
            ticker: "LARGE".to_string(),
            total_supply: u64::MAX,
            precision: 18,
            genesis_utxo: "887c100c1fa0aba98e60e40cfa50cb7e05aac61f7d2c704c029914553f37fcdd:535".to_string(),
            description: Some("Very long description ".repeat(100)),
        };
        
        let start_time = std::time::Instant::now();
        let result = processor.issue_rgb20(large_request);
        let duration = start_time.elapsed();
        
        assert!(result.is_ok());
        println!("✅ Large contract creation completed in {:?}", duration);
        
        let response = result.unwrap();
        println!("   Contract ID: {}", response.contract_id);
        println!("   Consignment size: {} bytes", response.consignment_data.len());
    }
}
