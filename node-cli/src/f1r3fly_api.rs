use blake2::{Blake2b, Digest};
use models::casper::v1::deploy_response::Message as DeployResponseMessage;
use models::casper::v1::deploy_service_client::DeployServiceClient;
use models::casper::v1::exploratory_deploy_response::Message as ExploratoryDeployResponseMessage;
use models::casper::v1::is_finalized_response::Message as IsFinalizedResponseMessage;
use models::casper::v1::propose_response::Message as ProposeResponseMessage;
use models::casper::v1::propose_service_client::ProposeServiceClient;
use models::casper::{
    BlocksQuery, DeployDataProto, ExploratoryDeployQuery, IsFinalizedQuery, LightBlockInfo,
    ProposeQuery,
};
use models::rhoapi::Par;
use prost::Message;
use secp256k1::{Message as Secp256k1Message, Secp256k1, SecretKey};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use typenum::U32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployInfo {
    pub deploy_id: String,
    pub block_hash: Option<String>,
    pub sender: Option<String>,
    pub seq_num: Option<u64>,
    pub sig: Option<String>,
    pub sig_algorithm: Option<String>,
    pub shard_id: Option<String>,
    pub version: Option<u64>,
    pub timestamp: Option<u64>,
    pub status: DeployStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeployStatus {
    Pending,       // Deploy submitted but not yet in a block
    Included,      // Deploy included in a block
    NotFound,      // Deploy ID not found
    Error(String), // Error occurred
}

/// Client for interacting with the F1r3fly API
pub struct F1r3flyApi<'a> {
    signing_key: SecretKey,
    node_host: &'a str,
    grpc_port: u16,
}

impl<'a> F1r3flyApi<'a> {
    /// Creates a new F1r3fly API client
    ///
    /// # Arguments
    ///
    /// * `signing_key` - Hex-encoded private key for signing deploys
    /// * `node_host` - Hostname or IP address of the F1r3fly node
    /// * `grpc_port` - gRPC port for the node's API service
    ///
    /// # Returns
    ///
    /// A new `F1r3flyApi` instance
    pub fn new(signing_key: &str, node_host: &'a str, grpc_port: u16) -> Self {
        F1r3flyApi {
            signing_key: SecretKey::from_slice(&hex::decode(signing_key).unwrap()).unwrap(),
            node_host,
            grpc_port,
        }
    }

    /// Deploys Rholang code to the F1r3fly node
    ///
    /// # Arguments
    ///
    /// * `rho_code` - Rholang source code to deploy
    /// * `use_bigger_phlo_price` - Whether to use a larger phlo limit
    /// * `language` - Language of the deploy (typically "rholang")
    ///
    /// # Returns
    ///
    /// The deploy ID if successful, otherwise an error
    pub async fn deploy(
        &self,
        rho_code: &str,
        use_bigger_phlo_price: bool,
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let phlo_limit: i64 = if use_bigger_phlo_price {
            5_000_000_000
        } else {
            50_000
        };

        // Get current block number for VABN (solves Block 50 issue)
        let current_block = match self.get_current_block_number().await {
            Ok(block_num) => {
                println!("🔢 Current block: {}", block_num);
                println!(
                    "✅ Setting validity window: blocks {} to {} (50-block window)",
                    block_num,
                    block_num + 50
                );
                block_num
            }
            Err(e) => {
                println!(
                    "⚠️  Warning: Could not get current block number ({}), using VABN=0",
                    e
                );
                println!("⚠️  This may cause Block 50 issues if blockchain has > 50 blocks");
                0
            }
        };

        // Build and sign the deployment
        let deployment = self.build_deploy_msg(
            rho_code.to_string(),
            phlo_limit,
            language.to_string(),
            current_block,
        );

        // Connect to the F1r3fly node
        let mut deploy_service_client =
            DeployServiceClient::connect(format!("http://{}:{}/", self.node_host, self.grpc_port))
                .await?;

        // Send the deploy
        let deploy_response = deploy_service_client.do_deploy(deployment).await?;

        // Process the response
        let deploy_message = deploy_response
            .get_ref()
            .message
            .as_ref()
            .ok_or("Deploy result not found")?;

        match deploy_message {
            DeployResponseMessage::Error(service_error) => Err(service_error.clone().into()),
            DeployResponseMessage::Result(result) => {
                // Extract the deploy ID from the response - handle various formats
                let cleaned_result = result.trim();

                // Try different possible prefixes and formats
                if let Some(deploy_id) = cleaned_result.strip_prefix("Success! DeployId is: ") {
                    Ok(deploy_id.trim().to_string())
                } else if let Some(deploy_id) =
                    cleaned_result.strip_prefix("Success!\nDeployId is: ")
                {
                    Ok(deploy_id.trim().to_string())
                } else if cleaned_result.starts_with("Success!") {
                    // Look for any long hex string in the response
                    let lines: Vec<&str> = cleaned_result.lines().collect();
                    for line in lines {
                        let trimmed = line.trim();
                        if trimmed.len() > 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
                            return Ok(trimmed.to_string());
                        }
                    }
                    Err(format!("Could not extract deploy ID from response: {}", result).into())
                } else {
                    // Assume it's already just the deploy ID
                    Ok(cleaned_result.to_string())
                }
            }
        }
    }

    /// Executes Rholang code without committing to the blockchain (exploratory deployment)
    ///
    /// # Arguments
    ///
    /// * `rho_code` - Rholang source code to execute
    /// * `block_hash` - Optional block hash to use as reference
    /// * `use_pre_state_hash` - Whether to use pre-state hash instead of post-state hash
    ///
    /// # Returns
    ///
    /// A tuple of (result data as JSON string, block info) if successful, otherwise an error
    pub async fn exploratory_deploy(
        &self,
        rho_code: &str,
        block_hash: Option<&str>,
        use_pre_state_hash: bool,
    ) -> Result<(String, String), Box<dyn std::error::Error>> {
        // Connect to the F1r3fly node
        let mut deploy_service_client =
            DeployServiceClient::connect(format!("http://{}:{}/", self.node_host, self.grpc_port))
                .await?;

        // Build the exploratory deploy query
        let query = ExploratoryDeployQuery {
            term: rho_code.to_string(),
            block_hash: block_hash.unwrap_or("").to_string(),
            use_pre_state_hash,
        };

        // Send the exploratory deploy
        let response = deploy_service_client.exploratory_deploy(query).await?;

        // Process the response
        let message = response
            .get_ref()
            .message
            .as_ref()
            .ok_or("Exploratory deploy result not found")?;

        match message {
            ExploratoryDeployResponseMessage::Error(service_error) => {
                Err(service_error.clone().into())
            }
            ExploratoryDeployResponseMessage::Result(result) => {
                // Format the Par data structure to a readable string
                let data = {
                    let mut result_str = String::new();

                    // Process the data
                    if !result.post_block_data.is_empty() {
                        for (i, par) in result.post_block_data.iter().enumerate() {
                            if i > 0 {
                                result_str.push_str("\n");
                            }
                            // We're using a simplified representation of the Par data
                            // A more sophisticated approach would be to recursively traverse the structure
                            match extract_par_data(par) {
                                Some(data) => result_str.push_str(&data),
                                None => result_str
                                    .push_str(&format!("Result {}: Complex data structure", i + 1)),
                            }
                        }
                    } else {
                        result_str = "No data returned".to_string();
                    }

                    result_str
                };

                // Format the block info to a readable string
                let block_info = {
                    if let Some(block) = &result.block {
                        format!(
                            "Block hash: {}, Block number: {}",
                            block.block_hash, block.block_number
                        )
                    } else {
                        "No block info".to_string()
                    }
                };

                Ok((data, block_info))
            }
        }
    }

    /// Sends a proposal to the network to create a new block
    ///
    /// # Returns
    ///
    /// The block hash of the proposed block if successful, otherwise an error
    pub async fn propose(&self) -> Result<String, Box<dyn std::error::Error>> {
        // Connect to the F1r3fly node's propose service
        let mut propose_client =
            ProposeServiceClient::connect(format!("http://{}:{}/", self.node_host, self.grpc_port))
                .await?;

        // Send the propose request
        let propose_response = propose_client
            .propose(ProposeQuery { is_async: false })
            .await?
            .into_inner();

        // Process the response
        let message = propose_response.message.ok_or("Missing propose response")?;

        match message {
            ProposeResponseMessage::Result(block_hash) => {
                // Extract the block hash from the response
                if let Some(hash) = block_hash
                    .strip_prefix("Success! Block ")
                    .and_then(|s| s.strip_suffix(" created and added."))
                {
                    Ok(hash.to_string())
                } else {
                    Ok(block_hash) // Return the full message if we can't extract the hash
                }
            }
            ProposeResponseMessage::Error(error) => {
                Err(format!("Propose error: {:?}", error).into())
            }
        }
    }

    /// Performs a full deployment cycle: deploy and propose
    ///
    /// # Arguments
    ///
    /// * `rho_code` - Rholang source code to deploy
    /// * `use_bigger_phlo_price` - Whether to use a larger phlo limit
    /// * `language` - Language of the deploy (typically "rholang")
    ///
    /// # Returns
    ///
    /// The block hash if successful, otherwise an error
    pub async fn full_deploy(
        &self,
        rho_code: &str,
        use_bigger_phlo_price: bool,
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // First deploy the code
        self.deploy(rho_code, use_bigger_phlo_price, language)
            .await?;

        // Then propose a block
        self.propose().await
    }

    /// Checks if a block is finalized, with retry logic
    ///
    /// # Arguments
    ///
    /// * `block_hash` - The hash of the block to check
    /// * `max_attempts` - Maximum number of retry attempts (default: 12)
    /// * `retry_delay_sec` - Delay between retries in seconds (default: 5)
    ///
    /// # Returns
    ///
    /// true if the block is finalized, false if the block is not finalized after all retry attempts
    pub async fn is_finalized(
        &self,
        block_hash: &str,
        max_attempts: u32,
        retry_delay_sec: u64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mut attempts = 0;

        loop {
            attempts += 1;

            // Connect to the F1r3fly node
            let mut deploy_service_client = DeployServiceClient::connect(format!(
                "http://{}:{}/",
                self.node_host, self.grpc_port
            ))
            .await?;

            // Query if the block is finalized
            let query = IsFinalizedQuery {
                hash: block_hash.to_string(),
            };

            match deploy_service_client.is_finalized(query).await {
                Ok(response) => {
                    let finalized_response = response.get_ref();
                    if let Some(message) = &finalized_response.message {
                        match message {
                            IsFinalizedResponseMessage::Error(_) => {
                                return Err("Error checking finalization status".into());
                            }
                            IsFinalizedResponseMessage::IsFinalized(is_finalized) => {
                                if *is_finalized {
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    if attempts >= max_attempts {
                        return Err("Failed to connect to node after maximum attempts".into());
                    }
                }
            }

            if attempts >= max_attempts {
                return Ok(false);
            }

            // Wait before retrying
            tokio::time::sleep(tokio::time::Duration::from_secs(retry_delay_sec)).await;
        }
    }

    /// Get the block hash for a given deploy ID by querying the HTTP API
    ///
    /// # Arguments
    ///
    /// * `deploy_id` - The deploy ID to look up
    /// * `http_port` - The HTTP port for the deploy API endpoint
    ///
    /// # Returns
    ///
    /// Some(block_hash) if the deploy is included in a block, None if not yet included, or an error
    pub async fn get_deploy_block_hash(
        &self,
        deploy_id: &str,
        http_port: u16,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let url = format!(
            "http://{}:{}/api/deploy/{}",
            self.node_host, http_port, deploy_id
        );
        let client = reqwest::Client::new();

        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let deploy_info: serde_json::Value = response.json().await?;

                    // Extract blockHash from the response
                    if let Some(block_hash) = deploy_info.get("blockHash").and_then(|v| v.as_str())
                    {
                        Ok(Some(block_hash.to_string()))
                    } else {
                        Ok(None) // Deploy exists but no blockHash yet
                    }
                } else if response.status().as_u16() == 404 {
                    Ok(None) // Deploy not found yet
                } else {
                    let status = response.status();
                    let error_body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unable to read response body".to_string());

                    // Handle the case where the deploy exists but isn't in a block yet
                    if error_body.contains("Couldn't find block containing deploy with id:") {
                        Ok(None) // Deploy exists but not in a block yet
                    } else {
                        Err(format!(
                            "HTTP error {}: {} - Response: {}",
                            status,
                            status.canonical_reason().unwrap_or("Unknown"),
                            error_body
                        )
                        .into())
                    }
                }
            }
            Err(e) => Err(format!("Network error: {}", e).into()),
        }
    }

    /// Gets comprehensive information about a deploy by ID
    ///
    /// # Arguments
    ///
    /// * `deploy_id` - The deploy ID to look up
    /// * `http_port` - HTTP port for API queries
    ///
    /// # Returns
    ///
    /// DeployInfo struct with deploy details and status
    pub async fn get_deploy_info(
        &self,
        deploy_id: &str,
        http_port: u16,
    ) -> Result<DeployInfo, Box<dyn std::error::Error>> {
        let url = format!(
            "http://{}:{}/api/deploy/{}",
            self.node_host, http_port, deploy_id
        );
        let client = reqwest::Client::new();

        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let deploy_data: serde_json::Value = response.json().await?;

                    // Parse the response into DeployInfo
                    let deploy_info = DeployInfo {
                        deploy_id: deploy_id.to_string(),
                        block_hash: deploy_data
                            .get("blockHash")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        sender: deploy_data
                            .get("sender")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        seq_num: deploy_data.get("seqNum").and_then(|v| v.as_u64()),
                        sig: deploy_data
                            .get("sig")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        sig_algorithm: deploy_data
                            .get("sigAlgorithm")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        shard_id: deploy_data
                            .get("shardId")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        version: deploy_data.get("version").and_then(|v| v.as_u64()),
                        timestamp: deploy_data.get("timestamp").and_then(|v| v.as_u64()),
                        status: DeployStatus::Included,
                    };
                    Ok(deploy_info)
                } else if response.status().as_u16() == 404 {
                    Ok(DeployInfo {
                        deploy_id: deploy_id.to_string(),
                        block_hash: None,
                        sender: None,
                        seq_num: None,
                        sig: None,
                        sig_algorithm: None,
                        shard_id: None,
                        version: None,
                        timestamp: None,
                        status: DeployStatus::NotFound,
                    })
                } else {
                    let status = response.status();
                    let error_body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unable to read response body".to_string());

                    // Handle the case where the deploy exists but isn't in a block yet
                    if error_body.contains("Couldn't find block containing deploy with id:") {
                        Ok(DeployInfo {
                            deploy_id: deploy_id.to_string(),
                            block_hash: None,
                            sender: None,
                            seq_num: None,
                            sig: None,
                            sig_algorithm: None,
                            shard_id: None,
                            version: None,
                            timestamp: None,
                            status: DeployStatus::Pending,
                        })
                    } else {
                        Ok(DeployInfo {
                            deploy_id: deploy_id.to_string(),
                            block_hash: None,
                            sender: None,
                            seq_num: None,
                            sig: None,
                            sig_algorithm: None,
                            shard_id: None,
                            version: None,
                            timestamp: None,
                            status: DeployStatus::Error(format!(
                                "HTTP error {}: {}",
                                status, error_body
                            )),
                        })
                    }
                }
            }
            Err(e) => Ok(DeployInfo {
                deploy_id: deploy_id.to_string(),
                block_hash: None,
                sender: None,
                seq_num: None,
                sig: None,
                sig_algorithm: None,
                shard_id: None,
                version: None,
                timestamp: None,
                status: DeployStatus::Error(format!("Network error: {}", e)),
            }),
        }
    }

    /// Gets blocks in the main chain
    ///
    /// # Arguments
    ///
    /// * `depth` - Number of blocks to retrieve from the main chain
    ///
    /// # Returns
    ///
    /// A vector of LightBlockInfo representing blocks in the main chain
    pub async fn show_main_chain(
        &self,
        depth: u32,
    ) -> Result<Vec<LightBlockInfo>, Box<dyn std::error::Error>> {
        use models::casper::v1::block_info_response::Message;
        use models::casper::v1::deploy_service_client::DeployServiceClient;

        // Connect to the F1r3fly node
        let mut deploy_service_client =
            DeployServiceClient::connect(format!("http://{}:{}/", self.node_host, self.grpc_port))
                .await?;

        // Create the query
        let query = BlocksQuery {
            depth: depth as i32,
        };

        // Send the query and collect streaming response
        let mut stream = deploy_service_client
            .show_main_chain(query)
            .await?
            .into_inner();

        let mut blocks = Vec::new();
        while let Some(response) = stream.message().await? {
            if let Some(message) = response.message {
                match message {
                    Message::Error(service_error) => {
                        return Err(
                            format!("gRPC Error: {}", service_error.messages.join("; ")).into()
                        );
                    }
                    Message::BlockInfo(block_info) => {
                        blocks.push(block_info);
                    }
                }
            }
        }

        Ok(blocks)
    }

    /// Gets the current block number from the blockchain
    ///
    /// # Returns
    ///
    /// The current block number if successful, otherwise an error
    pub async fn get_current_block_number(&self) -> Result<i64, Box<dyn std::error::Error>> {
        // Get the most recent block using show_main_chain with depth 1
        let blocks = self.show_main_chain(1).await?;

        if let Some(latest_block) = blocks.first() {
            Ok(latest_block.block_number)
        } else {
            // Fallback to 0 if no blocks found (genesis case)
            Ok(0)
        }
    }

    /// Builds and signs a deploy message
    ///
    /// # Arguments
    ///
    /// * `code` - Rholang source code to deploy
    /// * `phlo_limit` - Maximum amount of phlo to use for execution
    /// * `language` - Language of the deploy (typically "rholang")
    /// * `valid_after_block_number` - Block number after which the deploy is valid
    ///
    /// # Returns
    ///
    /// A signed `DeployDataProto` ready to be sent to the node
    fn build_deploy_msg(
        &self,
        code: String,
        phlo_limit: i64,
        language: String,
        valid_after_block_number: i64,
    ) -> DeployDataProto {
        // Get current timestamp in milliseconds
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Failed to get system time")
            .as_millis() as i64;

        // Create a projection with only the fields used for signature calculation
        // IMPORTANT: The language field is deliberately excluded from signature calculation
        let projection = DeployDataProto {
            term: code.clone(),
            timestamp,
            phlo_price: 1,
            phlo_limit,
            valid_after_block_number,
            shard_id: "root".into(),
            language: String::new(), // Excluded from signature calculation
            sig: prost::bytes::Bytes::new(),
            deployer: prost::bytes::Bytes::new(),
            sig_algorithm: String::new(),
        };

        // Serialize the projection for hashing
        let serialized = projection.encode_to_vec();

        // Hash with Blake2b256
        let digest = blake2b_256_hash(&serialized);

        // Sign the digest with secp256k1
        let secp = Secp256k1::new();
        let message = Secp256k1Message::from_digest(digest.into());
        let signature = secp.sign_ecdsa(&message, &self.signing_key);

        // Get signature in DER format
        let sig_bytes = signature.serialize_der().to_vec();

        // Get the public key in uncompressed format
        let public_key = self.signing_key.public_key(&secp);
        let pub_key_bytes = public_key.serialize_uncompressed().to_vec();

        // Return the complete deploy message
        DeployDataProto {
            term: code,
            timestamp,
            phlo_price: 1,
            phlo_limit,
            valid_after_block_number,
            shard_id: "root".into(),
            language,
            sig: prost::bytes::Bytes::from(sig_bytes),
            sig_algorithm: "secp256k1".into(),
            deployer: prost::bytes::Bytes::from(pub_key_bytes),
        }
    }
}

/// Extracts a simplified string representation from a Par object
fn extract_par_data(par: &Par) -> Option<String> {
    // Check for expressions
    if !par.exprs.is_empty() && par.exprs[0].expr_instance.is_some() {
        // Extract data from the first expression
        let expr = &par.exprs[0];
        if let Some(instance) = &expr.expr_instance {
            match instance {
                // Handle different types of expressions
                models::rhoapi::expr::ExprInstance::GString(s) => Some(format!("\"{}\"", s)),
                models::rhoapi::expr::ExprInstance::GInt(i) => Some(i.to_string()),
                models::rhoapi::expr::ExprInstance::GBool(b) => Some(b.to_string()),
                // Add other types as needed
                _ => Some("Complex expression".to_string()),
            }
        } else {
            None
        }
    }
    // Check for sends
    else if !par.sends.is_empty() {
        Some("Send operation".to_string())
    }
    // Check for receives
    else if !par.receives.is_empty() {
        Some("Receive operation".to_string())
    }
    // Check for news
    else if !par.news.is_empty() {
        Some("New declaration".to_string())
    }
    // Empty or unsupported Par object
    else {
        None
    }
}

/// Computes a Blake2b 256-bit hash of the provided data
fn blake2b_256_hash(data: &[u8]) -> [u8; 32] {
    let mut blake = Blake2b::<U32>::new();
    blake.update(data);
    let hash = blake.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(&hash);
    result
}
