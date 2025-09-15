//! RGB execution result validation module
//!
//! This module handles parsing and validation of Rholang execution results,
//! ensuring RGB contracts executed successfully with proper state transitions.

use crate::RholangCodex;
use log::{debug, warn};
use models::rhoapi::Par;
use rholang::rust::interpreter::interpreter::EvaluateResult;
use std::collections::HashSet;
use hypersonic::{CallError, Operation};

/// RGB contract execution result with validation details
#[derive(Debug, Clone)]
pub struct RgbExecutionResult {
    /// Whether the contract executed successfully
    pub success: bool,
    /// Final channel states after execution  
    pub final_channels: HashSet<Par>,
    /// Execution cost consumed
    pub execution_cost: i64,
    /// Validation errors if any
    pub errors: Vec<String>,
    /// Expected vs actual result comparison
    pub validation_summary: String,
}

impl<S: hypersonic::Stock> RholangCodex<S> {
    /// Parse and validate Rholang execution results for RGB compliance
    pub(crate) fn validate_execution_result(
        &self,
        operation: &Operation,
        evaluate_result: EvaluateResult,
    ) -> Result<RgbExecutionResult, CallError> {
        debug!(
            "🔍 Validating execution result for operation {}",
            operation.opid()
        );

        // Step 1: Check for interpreter errors
        if !evaluate_result.errors.is_empty() {
            let error_messages: Vec<String> = evaluate_result
                .errors
                .iter()
                .map(|e| format!("{:?}", e))
                .collect();

            warn!("❌ Rholang execution errors: {:?}", error_messages);

            return Ok(RgbExecutionResult {
                success: false,
                final_channels: evaluate_result.mergeable,
                execution_cost: evaluate_result.cost.value,
                errors: error_messages,
                validation_summary: "Execution failed with interpreter errors".to_string(),
            });
        }

        // Step 2: Validate expected channels exist in final state
        let validation_result =
            self.validate_rgb_postconditions(operation, &evaluate_result.mergeable)?;

        // Step 3: Check execution cost is reasonable
        let cost_valid = self.validate_execution_cost(&evaluate_result.cost.value, operation);

        let success = validation_result.success && cost_valid;
        let mut errors = validation_result.errors;

        if !cost_valid {
            errors.push(format!(
                "Execution cost {} exceeds reasonable limits for operation type",
                evaluate_result.cost.value
            ));
        }

        debug!(
            "✅ Execution validation complete: success={}, channels={}, cost={}",
            success,
            evaluate_result.mergeable.len(),
            evaluate_result.cost.value
        );

        let validation_summary = if success {
            format!("RGB operation {} validated successfully", operation.opid())
        } else {
            format!(
                "RGB operation {} validation failed: {}",
                operation.opid(),
                errors.join("; ")
            )
        };

        Ok(RgbExecutionResult {
            success,
            final_channels: evaluate_result.mergeable,
            execution_cost: evaluate_result.cost.value,
            errors,
            validation_summary,
        })
    }

    /// Validate RGB contract postconditions by checking expected channel states
    fn validate_rgb_postconditions(
        &self,
        operation: &Operation,
        final_channels: &HashSet<Par>,
    ) -> Result<ValidationResult, CallError> {
        debug!(
            "🔍 Validating RGB postconditions for {} channels",
            final_channels.len()
        );

        // For now, basic validation - in production this would be more sophisticated
        let mut errors = Vec::new();

        // Check that we have some final channel states (non-empty execution)
        if final_channels.is_empty() {
            errors.push("No final channel states found - possible execution failure".to_string());
        }

        // Look for result channel indicating success/failure
        let has_result_indicator = self.check_for_result_channels(final_channels);
        if !has_result_indicator {
            debug!("⚠️ No explicit result channel found, relying on state presence");
        }

        // Validate expected RGB channels exist
        let expected_channels = self.get_expected_rgb_channels(operation);
        let missing_channels = self.find_missing_channels(&expected_channels, final_channels);

        if !missing_channels.is_empty() {
            errors.push(format!(
                "Missing expected RGB channels: {:?}",
                missing_channels
            ));
        }

        // Check for RGB-specific error indicators in channel data
        let rgb_errors = self.check_for_rgb_error_indicators(final_channels);
        errors.extend(rgb_errors);

        let success = errors.is_empty();

        Ok(ValidationResult {
            success,
            errors,
            validated_channels: final_channels.len(),
        })
    }

    /// Check if execution cost is reasonable for the operation type
    fn validate_execution_cost(&self, cost: &i64, operation: &Operation) -> bool {
        // Define reasonable cost limits per operation type
        let max_cost = match self.operation_mapping.get_operation_type(operation.call_id) {
            Some(op_type) => {
                use crate::RgbOperationType;
                match op_type {
                    RgbOperationType::Rgb20Issue => 1_000_000i64, // Higher cost for issuance
                    RgbOperationType::Rgb20Transfer => 500_000i64, // Medium cost for transfers
                    RgbOperationType::Rgb20Burn => 300_000i64,    // Lower cost for burns
                    RgbOperationType::Rgb21Mint => 800_000i64,    // Higher cost for NFT operations
                    RgbOperationType::Rgb21Transfer => 400_000i64,
                    RgbOperationType::Rgb25Operation => 600_000i64, // Medium cost for collectibles
                    RgbOperationType::Custom(_) => 2_000_000i64, // Higher limit for custom operations
                }
            }
            None => 1_000_000i64, // Default reasonable limit
        };

        let valid = *cost <= max_cost;
        if !valid {
            warn!(
                "⚠️ Execution cost {} exceeds limit {} for operation {}",
                cost,
                max_cost,
                operation.opid()
            );
        }

        valid
    }

    /// Check for result channels indicating contract completion
    fn check_for_result_channels(&self, final_channels: &HashSet<Par>) -> bool {
        // Look for common result channel patterns in Rholang contracts
        // This is a simplified check - in production, we'd have more sophisticated detection
        final_channels.len() > 0 // Basic check that execution produced some state
    }

    /// Get expected channel names for RGB operation verification
    fn get_expected_rgb_channels(&self, operation: &Operation) -> Vec<String> {
        let mut expected = Vec::new();

        // Add operation-specific expected channels
        match self.operation_mapping.get_operation_type(operation.call_id) {
            Some(op_type) => {
                use crate::RgbOperationType;
                match op_type {
                    RgbOperationType::Rgb20Issue => {
                        expected.push("result".to_string());
                        expected.push("balances".to_string());
                    }
                    RgbOperationType::Rgb20Transfer => {
                        expected.push("result".to_string());
                        expected.push("balances".to_string());
                    }
                    RgbOperationType::Rgb20Burn => {
                        expected.push("result".to_string());
                    }
                    _ => {
                        expected.push("result".to_string()); // All operations should have result
                    }
                }
            }
            None => {
                expected.push("result".to_string()); // Fallback expectation
            }
        }

        expected
    }

    /// Find channels that are expected but missing from final state
    fn find_missing_channels(
        &self,
        expected: &[String],
        final_channels: &HashSet<Par>,
    ) -> Vec<String> {
        // For now, this is a simplified check
        // In production, we'd parse Par objects to extract actual channel names
        let mut missing = Vec::new();

        // This is placeholder logic - actual implementation would need to parse Par objects
        // to extract channel names and compare with expected channels
        if final_channels.is_empty() && !expected.is_empty() {
            missing.extend(expected.iter().cloned());
        }

        missing
    }

    /// Check for RGB-specific error indicators in channel states
    fn check_for_rgb_error_indicators(&self, final_channels: &HashSet<Par>) -> Vec<String> {
        let mut errors = Vec::new();

        // This would check for specific error patterns in channel data
        // For now, just basic validation that channels exist
        if final_channels.is_empty() {
            errors.push("Empty final state may indicate execution failure".to_string());
        }

        // TODO: Parse Par objects to look for error strings like "insufficient_balance", "invalid_operation", etc.

        errors
    }
}

/// Internal validation result structure
#[derive(Debug)]
struct ValidationResult {
    success: bool,
    errors: Vec<String>,
    validated_channels: usize,
}
