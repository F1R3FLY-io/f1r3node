//! RGB data extraction module
//!
//! This module contains all methods for extracting relevant data from RGB operations,
//! including amounts, recipients, token IDs, and other operation-specific information.

use amplify_num::hex::ToHex;
use log::debug;
use hypersonic::{fe256, CallError, Memory, Operation, StateCell, StateValue};

use crate::RholangCodex;

impl<S: hypersonic::Stock> RholangCodex<S> {
    /// Extract recipient and amount from RGB-20 token issuance operation
    pub(crate) fn extract_rgb20_issue_data(
        &self,
        operation: &Operation,
    ) -> Result<(String, u64), CallError> {
        debug!(
            "🔍 Extracting RGB-20 issue data from operation {}",
            operation.opid()
        );

        // For RGB-20 issuance, the recipient and amount are typically in destructible_out
        // The first output usually contains the recipient's balance
        if let Some(first_output) = operation.destructible_out.first() {
            let amount = self.extract_amount_from_state_value(first_output.data)?;
            let recipient = self.extract_recipient_from_state_cell(first_output)?;

            debug!(
                "✅ Extracted issue data: recipient={}, amount={}",
                recipient, amount
            );
            Ok((recipient, amount))
        } else {
            debug!("❌ No destructible_out found for issue operation");
            Err(CallError::Script(fe256::from(1001u32))) // No data found
        }
    }

    /// Extract sender, recipient, and amount from RGB-20 token transfer operation  
    pub(crate) fn extract_rgb20_transfer_data(
        &self,
        operation: &Operation,
        memory: &impl Memory,
    ) -> Result<(String, String, u64), CallError> {
        debug!(
            "🔍 Extracting RGB-20 transfer data from operation {}",
            operation.opid()
        );

        // For RGB-20 transfers, we need to look at both inputs and outputs
        // Inputs contain the sender's balance being consumed
        // Outputs contain the new balances for sender (change) and recipient

        let mut sender = String::new();
        let mut recipient = String::new();
        let mut amount = 0u64;

        // Extract sender from inputs (who is spending)
        if let Some(first_input) = operation.destructible_in.first() {
            // Look up the state cell being consumed
            if let Some(consumed_cell) = memory.destructible(first_input.addr) {
                sender = self.extract_recipient_from_state_cell(&consumed_cell)?;
                amount = self.extract_amount_from_state_value(consumed_cell.data)?;
            }
        }

        // Extract recipient from outputs (who is receiving)
        if let Some(first_output) = operation.destructible_out.first() {
            recipient = self.extract_recipient_from_state_cell(first_output)?;
            // For transfers, amount might be in output as well
            let output_amount = self.extract_amount_from_state_value(first_output.data)?;
            if amount == 0 {
                amount = output_amount;
            }
        }

        // RGB-20 transfers must have valid sender data
        if sender.is_empty() {
            return Err(crate::rgb_missing_state_error(
                "RGB-20 transfer requires valid sender information from operation inputs"
            ));
        }

        debug!(
            "✅ Extracted transfer data: sender={}, recipient={}, amount={}",
            sender, recipient, amount
        );
        Ok((sender, recipient, amount))
    }

    /// Extract burner and amount from RGB-20 token burn operation
    pub(crate) fn extract_rgb20_burn_data(
        &self,
        operation: &Operation,
    ) -> Result<(String, u64), CallError> {
        debug!(
            "🔍 Extracting RGB-20 burn data from operation {}",
            operation.opid()
        );

        // For RGB-20 burn, the burner and amount are typically in destructible_in (consuming the tokens)
        if let Some(first_input) = operation.destructible_in.first() {
            let amount = self.extract_amount_from_state_value(first_input.witness)?;
            let burner = format!(
                "addr_{}_{}",
                first_input.addr.opid.to_hex(),
                first_input.addr.pos
            );

            debug!(
                "✅ Extracted burn data: burner={}, amount={}",
                burner, amount
            );
            Ok((burner, amount))
        } else {
            debug!("⚠️ No destructible_in found for burn operation, checking outputs");
            // Fallback: check outputs
            if let Some(first_output) = operation.destructible_out.first() {
                let amount = self.extract_amount_from_state_value(first_output.data)?;
                let burner = self.extract_recipient_from_state_cell(first_output)?;

                debug!(
                    "✅ Extracted burn data from outputs: burner={}, amount={}",
                    burner, amount
                );
                Ok((burner, amount))
            } else {
                debug!("❌ No inputs or outputs found for burn operation");
                Err(CallError::Script(fe256::from(1001u32))) // No data found
            }
        }
    }

    /// Extract recipient and token ID from RGB-21 NFT mint operation
    pub(crate) fn extract_rgb21_mint_data(
        &self,
        operation: &Operation,
    ) -> Result<(String, String), CallError> {
        debug!(
            "🔍 Extracting RGB-21 mint data from operation {}",
            operation.opid()
        );

        // For RGB-21 mint, recipient and token info are in destructible_out
        if let Some(first_output) = operation.destructible_out.first() {
            let recipient = self.extract_recipient_from_state_cell(first_output)?;
            // For NFTs, the token ID is derived from the auth token
            let token_id = format!("token_{}", first_output.auth.to_fe256().to_u256().to_hex());

            debug!(
                "✅ Extracted mint data: recipient={}, token_id={}",
                recipient, token_id
            );
            Ok((recipient, token_id))
        } else {
            debug!("❌ No destructible_out found for NFT mint operation");
            Err(CallError::Script(fe256::from(1001u32))) // No data found
        }
    }

    /// Extract sender, recipient, and token ID from RGB-21 NFT transfer operation
    pub(crate) fn extract_rgb21_transfer_data(
        &self,
        operation: &Operation,
        _memory: &impl Memory,
    ) -> Result<(String, String, String), CallError> {
        debug!(
            "🔍 Extracting RGB-21 transfer data from operation {}",
            operation.opid()
        );

        // For NFT transfer: sender from inputs, recipient from outputs
        let sender = if let Some(first_input) = operation.destructible_in.first() {
            format!(
                "addr_{}_{}",
                first_input.addr.opid.to_hex(),
                first_input.addr.pos
            )
        } else {
            "unknown_sender".to_string()
        };

        let (recipient, token_id) = if let Some(first_output) = operation.destructible_out.first() {
            let recipient = self.extract_recipient_from_state_cell(first_output)?;
            let token_id = format!("token_{}", first_output.auth.to_fe256().to_u256().to_hex());
            (recipient, token_id)
        } else {
            ("unknown_recipient".to_string(), "unknown_token".to_string())
        };

        debug!(
            "✅ Extracted NFT transfer data: sender={}, recipient={}, token_id={}",
            sender, recipient, token_id
        );
        Ok((sender, recipient, token_id))
    }

    /// Extract sender, recipient, and item ID from RGB-25 collectible operation
    pub(crate) fn extract_rgb25_data(
        &self,
        operation: &Operation,
        _memory: &impl Memory,
    ) -> Result<(String, String, String), CallError> {
        debug!(
            "🔍 Extracting RGB-25 collectible data from operation {}",
            operation.opid()
        );

        // RGB-25 collectibles have more complex data structures
        let sender = if let Some(first_input) = operation.destructible_in.first() {
            format!(
                "addr_{}_{}",
                first_input.addr.opid.to_hex(),
                first_input.addr.pos
            )
        } else {
            "unknown_sender".to_string()
        };

        let (recipient, item_id) = if let Some(first_output) = operation.destructible_out.first() {
            let recipient = self.extract_recipient_from_state_cell(first_output)?;
            let item_id = format!(
                "collectible_{}",
                first_output.auth.to_fe256().to_u256().to_hex()
            );
            (recipient, item_id)
        } else {
            ("unknown_recipient".to_string(), "unknown_item".to_string())
        };

        debug!(
            "✅ Extracted RGB-25 data: sender={}, recipient={}, item_id={}",
            sender, recipient, item_id
        );
        Ok((sender, recipient, item_id))
    }

    /// Extract amount (as u64) from StateValue
    pub(crate) fn extract_amount_from_state_value(
        &self,
        state_value: StateValue,
    ) -> Result<u64, CallError> {
        match state_value {
            StateValue::None => Ok(0),
            StateValue::Single { first } => {
                // Convert fe256 to u64 (this may truncate for very large values)
                self.convert_fe256_to_u64(first)
            }
            StateValue::Double { first, second: _ } => {
                // For RGB-20, amount is typically in the first field
                self.convert_fe256_to_u64(first)
            }
            StateValue::Triple {
                first: _,
                second: _,
                third,
            } => {
                // For some contracts, amount might be in the third field
                self.convert_fe256_to_u64(third)
            }
            StateValue::Quadruple {
                first,
                second: _,
                third: _,
                fourth: _,
            } => {
                // For complex contracts, amount might be in the first field
                self.convert_fe256_to_u64(first)
            }
        }
    }

    /// Convert fe256 to u64 with overflow checking
    pub(crate) fn convert_fe256_to_u64(&self, value: fe256) -> Result<u64, CallError> {
        let u256_value = value.to_u256();

        // Convert u256 to u64, checking for overflow
        // u256 is 256 bits, u64 is 64 bits, so we need to check if the value fits
        if u256_value > u64::MAX.into() {
            // Value too large for u64
            Err(CallError::Script(fe256::from(1003u32)))
        } else {
            // Safe to convert - take the low 64 bits
            let bytes = u256_value.to_le_bytes();
            let u64_bytes = [
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ];
            Ok(u64::from_le_bytes(u64_bytes))
        }
    }

    /// Extract recipient identifier from StateCell
    /// This parses the RGB contract state data structure to identify the recipient
    pub(crate) fn extract_recipient_from_state_cell(
        &self,
        state_cell: &StateCell,
    ) -> Result<String, CallError> {
        // Parse recipient information from state data fields
        // RGB contracts store recipient identifiers in structured state data
        match state_cell.data {
            StateValue::Double { first: _, second } => {
                // Assume second field contains recipient identifier
                Ok(format!("recipient_{}", second.to_u256()))
            }
            StateValue::Triple {
                first: _,
                second,
                third: _,
            } => {
                // Assume second field contains recipient identifier
                Ok(format!("recipient_{}", second.to_u256()))
            }
            _ => {
                // Fallback: create identifier from auth token hash
                Ok(format!(
                    "recipient_{}",
                    state_cell.auth.to_fe256().to_u256()
                ))
            }
        }
    }
}
