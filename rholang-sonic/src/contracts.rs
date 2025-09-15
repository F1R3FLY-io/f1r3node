//! RGB contract generation module
//!
//! This module contains all Rholang contract generation logic for different RGB operation types.
//! Each RGB standard (RGB-20, RGB-21, RGB-25) has dedicated contract generators.

use log::debug;
use hypersonic::{Memory, Operation};

use crate::RholangCodex;

impl<S: hypersonic::Stock> RholangCodex<S> {
    /// Generate Rholang contract for RGB-20 token issuance
    pub(crate) fn generate_rgb20_issue_contract(&self, operation: &Operation) -> Result<String, hypersonic::CallError> {
        debug!(
            "🏭 Generating RGB-20 issue contract for operation {}",
            operation.opid()
        );

        // Extract recipient and amount from operation - required for RGB-20 issuance
        let (recipient, amount) = self.extract_rgb20_issue_data(operation)
            .map_err(|e| {
                debug!("❌ Failed to extract RGB-20 issue data: {:?}", e);
                crate::rgb_data_extraction_error(
                    &format!("RGB-20 issuance requires valid recipient and amount data: {:?}", e)
                )
            })?;

        let contract = format!(
            r#"
new rgb20_issue, result, balances in {{
  contract rgb20_issue(@recipient, @amount) = {{
    balances!(recipient, amount) |
    result!("issue_success")
  }} |
  
  // Execute the issue operation with extracted values
  rgb20_issue!("{}", {}) |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)
  }}
}}
        "#,
            recipient, amount
        );

        debug!(
            "✅ Generated RGB-20 issue contract for recipient={}, amount={}:\n{}",
            recipient, amount, contract
        );
        Ok(contract)
    }

    /// Generate Rholang contract for RGB-20 token transfer
    pub(crate) fn generate_rgb20_transfer_contract(
        &self,
        operation: &Operation,
        memory: &impl Memory,
    ) -> Result<String, hypersonic::CallError> {
        debug!(
            "💸 Generating RGB-20 transfer contract for operation {}",
            operation.opid()
        );

        // Extract transfer data from operation - required for RGB-20 transfer
        let (sender, recipient, amount) = self.extract_rgb20_transfer_data(operation, memory)
            .map_err(|e| {
                debug!("❌ Failed to extract RGB-20 transfer data: {:?}", e);
                crate::rgb_data_extraction_error(
                    &format!("RGB-20 transfer requires valid sender, recipient, and amount data: {:?}", e)
                )
            })?;

        let contract = format!(
            r#"
new rgb20_transfer, result, balances in {{
  contract rgb20_transfer(@sender, @recipient, @amount) = {{
    new current_balance_ch in {{
      // Check sender's current balance (simplified validation)
      for (@current_balance <- balances) {{
        match current_balance >= amount {{
          true => {{
            // Sufficient balance: execute transfer
            balances!(sender, current_balance - amount) |
            balances!(recipient, amount) |
            result!("transfer_success")
          }}
          false => {{
            // Insufficient balance: reject transfer
            balances!(sender, current_balance) |
            result!("insufficient_balance")
          }}
        }}
      }}
    }}
  }} |
  
  // Execute the transfer operation with extracted values
  rgb20_transfer!("{}", "{}", {}) |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)  
  }}
}}
        "#,
            sender, recipient, amount
        );

        debug!(
            "✅ Generated RGB-20 transfer contract for sender={}, recipient={}, amount={}:\n{}",
            sender, recipient, amount, contract
        );
        Ok(contract)
    }

    /// Generate Rholang contract for RGB-20 token burn
    pub(crate) fn generate_rgb20_burn_contract(&self, operation: &Operation) -> Result<String, hypersonic::CallError> {
        debug!(
            "🔥 Generating RGB-20 burn contract for operation {}",
            operation.opid()
        );

        // Extract burn amount from operation - required for RGB-20 burn
        let (burner, amount) = self.extract_rgb20_burn_data(operation)
            .map_err(|e| {
                debug!("❌ Failed to extract RGB-20 burn data: {:?}", e);
                crate::rgb_data_extraction_error(
                    &format!("RGB-20 burn requires valid burner and amount data: {:?}", e)
                )
            })?;

        let contract = format!(
            r#"
new rgb20_burn, result, balances in {{
  contract rgb20_burn(@burner, @amount) = {{
    new current_balance_ch in {{
      // Check burner's current balance
      for (@current_balance <- balances) {{
        match current_balance >= amount {{
          true => {{
            // Sufficient balance: execute burn (reduce total supply)
            balances!(burner, current_balance - amount) |
            result!("burn_success")
          }}
          false => {{
            // Insufficient balance: reject burn
            balances!(burner, current_balance) |
            result!("insufficient_balance")
          }}
        }}
      }}
    }}
  }} |
  
  // Execute the burn operation with extracted values
  rgb20_burn!("{}", {}) |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)  
  }}
}}
        "#,
            burner, amount
        );

        debug!(
            "✅ Generated RGB-20 burn contract for burner={}, amount={}:\n{}",
            burner, amount, contract
        );
        Ok(contract)
    }

    /// Generate Rholang contract for RGB-21 NFT mint
    pub(crate) fn generate_rgb21_mint_contract(&self, operation: &Operation) -> Result<String, hypersonic::CallError> {
        debug!(
            "🎨 Generating RGB-21 NFT mint contract for operation {}",
            operation.opid()
        );

        // Extract mint data from operation
        let (recipient, token_id) = match self.extract_rgb21_mint_data(operation) {
            Ok(data) => data,
            Err(e) => {
                debug!(
                    "❌ Failed to extract mint data, using fallback values: {:?}",
                    e
                );
                (
                    "fallback_recipient".to_string(),
                    "fallback_token".to_string(),
                )
            }
        };

        let contract = format!(
            r#"
new rgb21_mint, result, tokens in {{
  contract rgb21_mint(@recipient, @tokenId) = {{
    new token_exists_ch in {{
      // Check if token already exists (should not for mint)
      for (@existing_owner <- tokens) {{
        match existing_owner == "none" {{
          true => {{
            // Token doesn't exist: mint it
            tokens!(recipient) |
            result!("mint_success")
          }}
          false => {{
            // Token already exists: reject mint
            tokens!(existing_owner) |
            result!("token_already_exists")
          }}
        }}
      }}
    }}
  }} |
  
  // Execute the mint operation with extracted values
  rgb21_mint!("{}", "{}") |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)  
  }}
}}
        "#,
            recipient, token_id
        );

        debug!(
            "✅ Generated RGB-21 mint contract for recipient={}, token={}:\n{}",
            recipient, token_id, contract
        );
        Ok(contract)
    }

    /// Generate Rholang contract for RGB-21 NFT transfer
    pub(crate) fn generate_rgb21_transfer_contract(
        &self,
        operation: &Operation,
        memory: &impl Memory,
    ) -> String {
        debug!(
            "🖼️ Generating RGB-21 NFT transfer contract for operation {}",
            operation.opid()
        );

        // Extract transfer data from operation
        let (sender, recipient, token_id) =
            match self.extract_rgb21_transfer_data(operation, memory) {
                Ok(data) => data,
                Err(e) => {
                    debug!(
                        "❌ Failed to extract NFT transfer data, using fallback values: {:?}",
                        e
                    );
                    (
                        "fallback_sender".to_string(),
                        "fallback_recipient".to_string(),
                        "fallback_token".to_string(),
                    )
                }
            };

        let contract = format!(
            r#"
new rgb21_transfer, result, tokens in {{
  contract rgb21_transfer(@sender, @recipient, @tokenId) = {{
    new token_owner_ch in {{
      // Verify sender owns the token (simplified ownership check)
      for (@current_owner <- tokens) {{
        match current_owner == sender {{
          true => {{
            // Sender owns token: transfer it
            tokens!(recipient) |
            result!("nft_transfer_success")
          }}
          false => {{
            // Sender doesn't own token: reject transfer  
            tokens!(current_owner) |
            result!("unauthorized_nft_transfer")
          }}
        }}
      }}
    }}
  }} |
  
  // Execute the NFT transfer operation with extracted values
  rgb21_transfer!("{}", "{}", "{}") |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)  
  }}
}}
        "#,
            sender, recipient, token_id
        );

        debug!(
            "✅ Generated RGB-21 NFT transfer contract for sender={}, recipient={}, token={}:\n{}",
            sender, recipient, token_id, contract
        );
        contract
    }

    /// Generate Rholang contract for RGB-25 collectible operations
    pub(crate) fn generate_rgb25_contract(
        &self,
        operation: &Operation,
        memory: &impl Memory,
    ) -> String {
        debug!(
            "💎 Generating RGB-25 collectible contract for operation {}",
            operation.opid()
        );

        // RGB-25 operations are more complex and context-dependent
        // For now, implement basic collectible transfer logic
        let (sender, recipient, item_id) = match self.extract_rgb25_data(operation, memory) {
            Ok(data) => data,
            Err(e) => {
                debug!(
                    "❌ Failed to extract RGB-25 data, using fallback values: {:?}",
                    e
                );
                (
                    "fallback_sender".to_string(),
                    "fallback_recipient".to_string(),
                    "fallback_item".to_string(),
                )
            }
        };

        let contract = format!(
            r#"
new rgb25_operation, result, collectibles in {{
  contract rgb25_operation(@sender, @recipient, @itemId) = {{
    new item_owner_ch in {{
      // Check current owner of collectible
      for (@current_owner <- collectibles) {{
        match current_owner == sender {{
          true => {{
            // Sender owns collectible: transfer it
            collectibles!(recipient) |
            result!("collectible_transfer_success")
          }}
          false => {{
            // Sender doesn't own collectible: reject transfer
            collectibles!(current_owner) |
            result!("unauthorized_collectible_transfer")
          }}
        }}
      }}
    }}
  }} |
  
  // Execute the collectible operation with extracted values
  rgb25_operation!("{}", "{}", "{}") |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)  
  }}
}}
        "#,
            sender, recipient, item_id
        );

        debug!(
            "✅ Generated RGB-25 collectible contract for sender={}, recipient={}, item={}:\n{}",
            sender, recipient, item_id, contract
        );
        contract
    }

    /// Generate Rholang contract for custom RGB operations
    pub(crate) fn generate_custom_contract(
        &self,
        operation: &Operation,
        method_name: &str,
        _memory: &impl Memory,
    ) -> String {
        debug!(
            "⚙️ Generating custom contract for method '{}', operation {}",
            method_name,
            operation.opid()
        );

        // Generate a basic custom contract template
        let contract = format!(
            r#"
new custom_operation, result in {{
  contract custom_operation(@method, @data) = {{
    // Custom operation logic goes here
    // This is a placeholder template for method: {}
    result!("custom_operation_executed")
  }} |
  
  // Execute the custom operation
  custom_operation!("{}", "custom_data") |
  
  // Wait for completion
  for (@status <- result) {{
    result!(status)  
  }}
}}
        "#,
            method_name, method_name
        );

        debug!(
            "✅ Generated custom contract for method '{}'\n{}",
            method_name, contract
        );
        contract
    }
}
