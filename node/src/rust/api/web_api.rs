//! Web API implementation for F1r3fly node

use crate::rust::api::serde_types::block_info::BlockInfoSerde;
use crate::rust::api::serde_types::deploy_info::TransferInfoSerde;
use crate::rust::api::serde_types::light_block_info::LightBlockInfoSerde;
use crate::rust::web::block_info_enricher::extract_transfers_from_report;
use crate::rust::web::transaction::TransactionResponse;
use casper::rust::api::block_report_api::BlockReportAPI;
use crate::rust::web::version_info::get_version_info_str;
use casper::rust::api::block_api::{BlockAPI, DeployNotFoundError};
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::ProposeFunction;
use comm::rust::discovery::node_discovery::NodeDiscovery;
use comm::rust::rp::connect::ConnectionsCell;
#[cfg(feature = "schnorr_secp256k1_experimental")]
use crypto::rust::signatures::{
    frost_secp256k1::FrostSecp256k1, schnorr_secp256k1::SchnorrSecp256k1,
};
use crypto::rust::{
    public_key::PublicKey, signatures::signatures_alg::SignaturesAlg, signatures::signed::Signed,
};
use eyre::{eyre, Result};
use hex;
use models::casper::{DataWithBlockInfo, LightBlockInfo};
use models::rust::casper::protocol::casper_message::DeployData;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::warn;
use utoipa::ToSchema;

const FIND_DEPLOY_RETRY_INTERVAL_MS: u64 = 50;
const FIND_DEPLOY_MAX_ATTEMPTS: u16 = 1;

fn find_deploy_retry_interval_ms() -> u64 {
    FIND_DEPLOY_RETRY_INTERVAL_MS
}

fn find_deploy_max_attempts() -> u16 {
    FIND_DEPLOY_MAX_ATTEMPTS
}

/// Web API trait defining the interface for HTTP endpoints
#[async_trait::async_trait]
pub trait WebApi {
    /// Get API status information
    async fn status(&self) -> Result<ApiStatus>;

    /// Prepare deploy request
    async fn prepare_deploy(&self, request: Option<PrepareRequest>) -> Result<PrepareResponse>;

    /// Deploy a contract
    async fn deploy(&self, request: DeployRequest) -> Result<String>;

    /// Listen for data at a name
    async fn listen_for_data_at_name(
        &self,
        request: DataAtNameRequest,
    ) -> Result<DataAtNameResponse>;

    /// Get data at a par (parallel expression)
    async fn get_data_at_par(
        &self,
        request: DataAtNameByBlockHashRequest,
    ) -> Result<RhoDataResponse>;

    /// Get the last finalized block
    async fn last_finalized_block(&self) -> Result<BlockInfoSerde>;

    /// Get a specific block by hash
    async fn get_block(&self, hash: String) -> Result<BlockInfoSerde>;

    /// Get blocks with specified depth
    async fn get_blocks(&self, depth: i32) -> Result<Vec<LightBlockInfoSerde>>;

    /// Find a deploy by ID
    async fn find_deploy(&self, deploy_id: String) -> Result<LightBlockInfoSerde>;

    /// Find a deploy by ID, returning deploy execution details with block context
    async fn find_deploy_detail(&self, deploy_id: String) -> Result<DeployDetailResponse>;

    /// Find a deploy by ID, returning minimal response
    async fn find_deploy_minimal(&self, deploy_id: String) -> Result<DeployLookupResponse>;

    /// Perform exploratory deploy
    async fn exploratory_deploy(
        &self,
        term: String,
        block_hash: Option<String>,
        use_pre_state_hash: bool,
    ) -> Result<RhoDataResponse>;

    /// Get blocks by height range
    async fn get_blocks_by_heights(
        &self,
        start_block_number: i64,
        end_block_number: i64,
    ) -> Result<Vec<LightBlockInfoSerde>>;

    /// Check if a block is finalized
    async fn is_finalized(&self, hash: String) -> Result<bool>;

    /// Get transaction by hash
    async fn get_transaction(&self, hash: String) -> Result<TransactionResponse>;
}

/// Web API implementation
pub struct WebApiImpl {
    api_max_blocks_limit: i32,
    dev_mode: bool,
    network_id: String,
    shard_id: String,
    min_phlo_price: i64,
    native_token_name: String,
    native_token_symbol: String,
    native_token_decimals: u32,
    is_node_read_only: bool,
    engine_cell: Arc<EngineCell>,
    block_report_api: BlockReportAPI,
    transfer_unforgeable: models::rhoapi::Par,
    rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
    connections_cell: ConnectionsCell,
    node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
    trigger_propose_f: Option<Arc<ProposeFunction>>,
}

impl WebApiImpl {
    pub fn new(
        api_max_blocks_limit: i32,
        dev_mode: bool,
        network_id: String,
        shard_id: String,
        min_phlo_price: i64,
        native_token_name: String,
        native_token_symbol: String,
        native_token_decimals: u32,
        is_node_read_only: bool,
        block_report_api: BlockReportAPI,
        transfer_unforgeable: models::rhoapi::Par,
        engine_cell: Arc<EngineCell>,
        rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
        connections_cell: ConnectionsCell,
        node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
        trigger_propose_f: Option<Arc<ProposeFunction>>,
    ) -> Self {
        Self {
            api_max_blocks_limit,
            dev_mode,
            network_id,
            shard_id,
            min_phlo_price,
            native_token_name,
            native_token_symbol,
            native_token_decimals,
            is_node_read_only,
            engine_cell,
            block_report_api,
            transfer_unforgeable,
            rp_conf_cell,
            connections_cell,
            node_discovery,
            trigger_propose_f,
        }
    }
    /// Enrich a BlockInfoSerde with transfer data from BlockReportAPI.
    /// On success: each deploy gets `Some(transfers)`.
    /// On failure (validator node): each deploy gets `None` (field omitted).
    async fn enrich_transfers(
        &self,
        serde: &mut BlockInfoSerde,
        block_hash_hex: String,
    ) {
        let block_hash_bytes: prost::bytes::Bytes = match hex::decode(&block_hash_hex) {
            Ok(bytes) => bytes.into(),
            Err(_) => {
                for deploy in &mut serde.deploys {
                    deploy.transfers = None;
                }
                return;
            }
        };
        match self.block_report_api.block_report(block_hash_bytes, false).await {
            Ok(report) => {
                let transfers_by_deploy =
                    extract_transfers_from_report(&report, &self.transfer_unforgeable);
                for deploy in &mut serde.deploys {
                    deploy.transfers = Some(
                        transfers_by_deploy
                            .get(&deploy.sig)
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .map(TransferInfoSerde::from)
                            .collect(),
                    );
                }
            }
            Err(_) => {
                for deploy in &mut serde.deploys {
                    deploy.transfers = None;
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl WebApi for WebApiImpl {
    async fn status(&self) -> Result<ApiStatus> {
        const STATUS_SLOW_THRESHOLD: Duration = Duration::from_millis(500);
        let total_start = Instant::now();

        let rp_conf_start = Instant::now();
        let rp_conf = self.rp_conf_cell.read()?;
        let rp_conf_elapsed = rp_conf_start.elapsed();

        let connections_start = Instant::now();
        let address = rp_conf.local.to_address();
        let connections = self.connections_cell.read()?;
        let connections_elapsed = connections_start.elapsed();

        let discovery_start = Instant::now();
        let discovered_nodes = self.node_discovery.peers()?;
        let discovery_elapsed = discovery_start.elapsed();

        let peers = connections.len() as i32;
        let nodes = discovered_nodes.len() as i32;

        // Create a set of connected peer IDs for quick lookup
        let connected_ids: std::collections::HashSet<_> =
            connections.iter().map(|p| p.id.key.clone()).collect();

        // Convert PeerNode to PeerInfoData with connection status
        let peer_list: Vec<PeerInfoData> = discovered_nodes
            .iter()
            .map(|node| PeerInfoData {
                address: node.to_address(),
                node_id: node.id.to_string(),
                host: node.endpoint.host.clone(),
                protocol_port: node.endpoint.tcp_port as i32,
                discovery_port: node.endpoint.udp_port as i32,
                is_connected: connected_ids.contains(&node.id.key),
            })
            .collect();

        let total_elapsed = total_start.elapsed();
        if total_elapsed >= STATUS_SLOW_THRESHOLD {
            warn!(
                ?total_elapsed,
                ?rp_conf_elapsed,
                ?connections_elapsed,
                ?discovery_elapsed,
                peers,
                nodes,
                "Web API status assembly is slow"
            );
        }

        Ok(ApiStatus {
            version: VersionInfo {
                api: "1".to_string(),
                node: get_version_info_str(),
            },
            address,
            network_id: self.network_id.clone(),
            shard_id: self.shard_id.clone(),
            peers,
            nodes,
            min_phlo_price: self.min_phlo_price,
            peer_list,
            native_token_name: self.native_token_name.clone(),
            native_token_symbol: self.native_token_symbol.clone(),
            native_token_decimals: self.native_token_decimals,
        })
    }

    async fn prepare_deploy(&self, request: Option<PrepareRequest>) -> Result<PrepareResponse> {
        let seq_number = BlockAPI::get_latest_message(&self.engine_cell)
            .await
            .map(|message| message.sequence_number)
            .unwrap_or(-1);

        let names = if let Some(req) = request {
            let deployer_bytes = hex::decode(&req.deployer)
                .map_err(|e| eyre!("Deployer is not valid hex format: {}", e))?;
            let name_bytes =
                BlockAPI::preview_private_names(&deployer_bytes, req.timestamp, req.name_qty)?;
            name_bytes.into_iter().map(hex::encode).collect()
        } else {
            vec![]
        };

        Ok(PrepareResponse { names, seq_number })
    }

    async fn deploy(&self, request: DeployRequest) -> Result<String> {
        // Convert request to signed deploy
        let signed_deploy = to_signed_deploy(&request)?;

        // Deploy using BlockAPI
        BlockAPI::deploy(
            &self.engine_cell,
            signed_deploy,
            &self.trigger_propose_f,
            self.min_phlo_price,
            self.is_node_read_only,
            &self.shard_id,
        )
        .await
    }

    async fn listen_for_data_at_name(
        &self,
        request: DataAtNameRequest,
    ) -> Result<DataAtNameResponse> {
        let res = BlockAPI::get_listening_name_data_response(
            &self.engine_cell,
            request.depth,
            to_par(request.name)?,
            self.api_max_blocks_limit,
        )
        .await?;

        Ok(to_data_at_name_response(res))
    }

    async fn get_data_at_par(
        &self,
        request: DataAtNameByBlockHashRequest,
    ) -> Result<RhoDataResponse> {
        let (pars, block) = BlockAPI::get_data_at_par(
            &self.engine_cell,
            &to_par(request.name)?,
            request.block_hash,
            request.use_pre_state_hash,
        )
        .await?;

        Ok(to_rho_data_response(pars, block, 0))
    }

    async fn last_finalized_block(&self) -> Result<BlockInfoSerde> {
        let block_info = BlockAPI::last_finalized_block(&self.engine_cell).await?;
        let mut serde = BlockInfoSerde::from(block_info);
        let block_hash = serde.block_info.block_hash.clone();
        self.enrich_transfers(&mut serde, block_hash).await;
        Ok(serde)
    }

    async fn get_block(&self, hash: String) -> Result<BlockInfoSerde> {
        let block_info = BlockAPI::get_block(&self.engine_cell, &hash).await?;
        let mut serde = BlockInfoSerde::from(block_info);
        let block_hash = serde.block_info.block_hash.clone();
        self.enrich_transfers(&mut serde, block_hash).await;
        Ok(serde)
    }

    async fn get_blocks(&self, depth: i32) -> Result<Vec<LightBlockInfoSerde>> {
        let blocks =
            BlockAPI::get_blocks(&self.engine_cell, depth, self.api_max_blocks_limit).await?;

        Ok(blocks
            .into_iter()
            .map(|block| LightBlockInfoSerde::from(block))
            .collect())
    }

    async fn find_deploy(&self, deploy_id: String) -> Result<LightBlockInfoSerde> {
        let deploy_id_bytes =
            hex::decode(&deploy_id).map_err(|e| eyre!("Invalid deploy ID format: {}", e))?;

        let retry_interval_ms = find_deploy_retry_interval_ms();
        let max_attempts = find_deploy_max_attempts();

        let mut attempt: u16 = 1;
        loop {
            match BlockAPI::find_deploy(&self.engine_cell, &deploy_id_bytes).await {
                Ok(block) => return Ok(LightBlockInfoSerde::from(block)),
                Err(err) => {
                    let not_found = err.downcast_ref::<DeployNotFoundError>().is_some();

                    if !not_found || attempt >= max_attempts {
                        return Err(err);
                    }

                    tracing::debug!(
                        ?attempt,
                        ?max_attempts,
                        ?retry_interval_ms,
                        ?deploy_id,
                        "Waiting for deploy to become visible in block DAG"
                    );
                    sleep(Duration::from_millis(retry_interval_ms)).await;
                    attempt += 1;
                }
            }
        }
    }

    async fn find_deploy_detail(&self, deploy_id: String) -> Result<DeployDetailResponse> {
        let light_block = self.find_deploy(deploy_id.clone()).await?;
        let block_info = self.get_block(light_block.block_hash.clone()).await?;

        let matching_deploy = block_info
            .deploys
            .iter()
            .find(|d| d.sig == deploy_id)
            .cloned();

        match matching_deploy {
            Some(deploy) => Ok(DeployDetailResponse {
                block_hash: light_block.block_hash,
                block_number: light_block.block_number,
                timestamp: light_block.timestamp,
                deployer: deploy.deployer,
                term: deploy.term,
                cost: deploy.cost,
                errored: deploy.errored,
                system_deploy_error: deploy.system_deploy_error,
                phlo_price: deploy.phlo_price,
                phlo_limit: deploy.phlo_limit,
                sig: deploy.sig,
                sig_algorithm: deploy.sig_algorithm,
                valid_after_block_number: deploy.valid_after_block_number,
                transfers: deploy.transfers,
            }),
            None => Err(eyre!("Deploy {} found in block {} but not in deploy list", deploy_id, light_block.block_hash)),
        }
    }

    async fn find_deploy_minimal(&self, deploy_id: String) -> Result<DeployLookupResponse> {
        let light_block = self.find_deploy(deploy_id.clone()).await?;
        let block_info = self.get_block(light_block.block_hash.clone()).await?;

        let cost = block_info
            .deploys
            .iter()
            .find(|d| d.sig == deploy_id)
            .map(|d| d.cost)
            .unwrap_or(0);

        Ok(DeployLookupResponse::from_block_and_cost(light_block, cost))
    }

    async fn exploratory_deploy(
        &self,
        term: String,
        block_hash: Option<String>,
        use_pre_state_hash: bool,
    ) -> Result<RhoDataResponse> {
        let (pars, block, cost) = BlockAPI::exploratory_deploy(
            &self.engine_cell,
            term,
            block_hash,
            use_pre_state_hash,
            self.dev_mode,
        )
        .await?;

        Ok(to_rho_data_response(pars, block, cost))
    }

    async fn get_blocks_by_heights(
        &self,
        start_block_number: i64,
        end_block_number: i64,
    ) -> Result<Vec<LightBlockInfoSerde>> {
        let blocks = BlockAPI::get_blocks_by_heights(
            &self.engine_cell,
            start_block_number,
            end_block_number,
            self.api_max_blocks_limit,
        )
        .await?;

        Ok(blocks
            .into_iter()
            .map(|block| LightBlockInfoSerde::from(block))
            .collect())
    }

    async fn is_finalized(&self, hash: String) -> Result<bool> {
        BlockAPI::is_finalized(&self.engine_cell, &hash).await
    }

    async fn get_transaction(&self, _hash: String) -> Result<TransactionResponse> {
        Err(eyre!("Deprecated: use /api/deploy/{{id}}?view=full for transfers or gRPC getEventByHash for raw events"))
    }
}

// Rholang terms interesting for translation to JSON
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(no_recursion)]
pub enum RhoExpr {
    // === Collections ===
    ExprPar { data: Vec<RhoExpr> },
    ExprTuple { data: Vec<RhoExpr> },
    ExprList { data: Vec<RhoExpr> },
    ExprSet { data: Vec<RhoExpr> },
    ExprMap { data: HashMap<String, RhoExpr> },

    // === Primitives ===
    ExprBool { data: bool },
    ExprInt { data: i64 },
    ExprString { data: String },
    ExprUri { data: String },
    ExprBytes { data: String },

    // === Extended numerics ===
    ExprFloat { data: f64 },
    ExprBigInt { data: String },
    ExprBigRat { numerator: String, denominator: String },
    ExprFixedPoint { value: String, scale: u32 },

    // === Unforgeable names ===
    ExprUnforg { data: RhoUnforg },

    // === Bundle (with permissions) ===
    ExprBundle { data: Box<RhoExpr>, read: bool, write: bool },

    // === Unary operators ===
    ExprNot { data: Box<RhoExpr> },
    ExprNeg { data: Box<RhoExpr> },

    // === Binary arithmetic ===
    ExprPlus { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprMinus { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprMult { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprDiv { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprMod { left: Box<RhoExpr>, right: Box<RhoExpr> },

    // === Comparison ===
    ExprLt { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprLte { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprGt { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprGte { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprEq { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprNeq { left: Box<RhoExpr>, right: Box<RhoExpr> },

    // === Logical ===
    ExprAnd { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprOr { left: Box<RhoExpr>, right: Box<RhoExpr> },

    // === String operations ===
    ExprConcat { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprInterpolate { left: Box<RhoExpr>, right: Box<RhoExpr> },
    ExprDiff { left: Box<RhoExpr>, right: Box<RhoExpr> },

    // === Pattern matching ===
    ExprMatches { target: Box<RhoExpr>, pattern: Box<RhoExpr> },

    // === Method call ===
    ExprMethod { target: Box<RhoExpr>, name: String, args: Vec<RhoExpr> },

    // === Variable reference ===
    ExprVar { index: i32 },

    // === System ===
    ExprSysAuthToken,

    // === Catch-all for unknown/unhandled types ===
    ExprUnknown { type_name: String },
}

/// Unforgeable name types
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum RhoUnforg {
    UnforgPrivate { data: String },
    UnforgDeploy { data: String },
    UnforgDeployer { data: String },
    UnforgSysAuthToken,
}

// API request & response types
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeployRequest {
    pub data: DeployData,
    pub deployer: String,
    pub signature: String,
    #[serde(rename = "sigAlgorithm")]
    pub sig_algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ExploreDeployRequest {
    pub term: String,
    #[serde(rename = "blockHash")]
    pub block_hash: String,
    #[serde(rename = "usePreStateHash")]
    pub use_pre_state_hash: bool,
}

/// Simple explore deploy request with only the term field.
/// Used by the /explore-deploy endpoint which doesn't require block hash or pre-state hash.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SimpleExploreDeployRequest {
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DataAtNameRequest {
    /// For simplicity only one Unforgeable name is allowed
    /// instead of the whole RhoExpr (proto Par)
    pub name: RhoUnforg,
    /// Number of blocks in the past to search for data
    pub depth: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DataAtNameByBlockHashRequest {
    pub name: RhoUnforg,
    #[serde(rename = "blockHash")]
    pub block_hash: String,
    #[serde(rename = "usePreStateHash")]
    pub use_pre_state_hash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DataAtNameResponse {
    pub exprs: Vec<RhoExprWithBlock>,
    pub length: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RhoExprWithBlock {
    pub expr: RhoExpr,
    pub block: LightBlockInfoSerde,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ExploratoryDeployResponse {
    pub expr: Vec<RhoExpr>,
    pub block: LightBlockInfoSerde,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RhoDataResponse {
    pub expr: Vec<RhoExpr>,
    pub block: LightBlockInfoSerde,
    pub cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PrepareRequest {
    pub deployer: String,
    pub timestamp: i64,
    #[serde(rename = "nameQty")]
    pub name_qty: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PrepareResponse {
    pub names: Vec<String>,
    #[serde(rename = "seqNumber")]
    pub seq_number: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PeerInfoData {
    pub address: String,
    #[serde(rename = "nodeId")]
    pub node_id: String,
    pub host: String,
    #[serde(rename = "protocolPort")]
    pub protocol_port: i32,
    #[serde(rename = "discoveryPort")]
    pub discovery_port: i32,
    #[serde(rename = "isConnected")]
    pub is_connected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiStatus {
    pub version: VersionInfo,
    pub address: String,
    #[serde(rename = "networkId")]
    pub network_id: String,
    #[serde(rename = "shardId")]
    pub shard_id: String,
    pub peers: i32,
    pub nodes: i32,
    #[serde(rename = "minPhloPrice")]
    pub min_phlo_price: i64,
    #[serde(rename = "peerList")]
    pub peer_list: Vec<PeerInfoData>,
    /// Full display name of the native token. Baked into genesis state via
    /// the `TokenMetadata` Rholang contract at `rho:system:tokenMetadata`.
    #[serde(rename = "nativeTokenName")]
    pub native_token_name: String,
    /// Ticker symbol of the native token (e.g. "F1R3").
    #[serde(rename = "nativeTokenSymbol")]
    pub native_token_symbol: String,
    /// Decimal places used to display the native token (dust per token = 10^decimals).
    #[serde(rename = "nativeTokenDecimals")]
    pub native_token_decimals: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct VersionInfo {
    pub api: String,
    pub node: String,
}

/// Deploy detail response with execution info and block context
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeployDetailResponse {
    #[serde(rename = "blockHash")]
    pub block_hash: String,
    #[serde(rename = "blockNumber")]
    pub block_number: i64,
    pub timestamp: i64,
    pub deployer: String,
    pub term: String,
    pub cost: u64,
    pub errored: bool,
    #[serde(rename = "systemDeployError")]
    pub system_deploy_error: String,
    #[serde(rename = "phloPrice")]
    pub phlo_price: i64,
    #[serde(rename = "phloLimit")]
    pub phlo_limit: i64,
    pub sig: String,
    #[serde(rename = "sigAlgorithm")]
    pub sig_algorithm: String,
    #[serde(rename = "validAfterBlockNumber")]
    pub valid_after_block_number: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transfers: Option<Vec<TransferInfoSerde>>,
}

/// Minimal deploy lookup response containing essential fields plus cost
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeployLookupResponse {
    #[serde(rename = "blockHash")]
    pub block_hash: String,
    #[serde(rename = "blockNumber")]
    pub block_number: i64,
    pub timestamp: i64,
    pub sender: String,
    #[serde(rename = "seqNum")]
    pub seq_num: i64,
    pub sig: String,
    #[serde(rename = "sigAlgorithm")]
    pub sig_algorithm: String,
    #[serde(rename = "shardId")]
    pub shard_id: String,
    pub version: i64,
    pub cost: u64,
}

impl DeployLookupResponse {
    pub fn from_block_and_cost(info: LightBlockInfoSerde, cost: u64) -> Self {
        Self {
            block_hash: info.block_hash,
            block_number: info.block_number,
            timestamp: info.timestamp,
            sender: info.sender,
            seq_num: info.seq_num,
            sig: info.sig,
            sig_algorithm: info.sig_algorithm,
            shard_id: info.shard_id,
            version: info.version,
            cost,
        }
    }
}

// Error types

#[derive(Debug)]
pub enum WebApiError {
    BlockApiError(String),
    SignatureError(String),
    InvalidFormat(String),
}

impl std::fmt::Display for WebApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebApiError::BlockApiError(msg) => write!(f, "Block API error: {}", msg),
            WebApiError::SignatureError(msg) => write!(f, "Signature error: {}", msg),
            WebApiError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for WebApiError {}

// Conversion functions

/// Convert DeployRequest to Signed DeployData
fn to_signed_deploy(request: &DeployRequest) -> Result<Signed<DeployData>> {
    // Decode hex strings
    let pk_bytes = hex::decode(&request.deployer)
        .map_err(|e| eyre!("Public key is not valid base16 format: {}", e))?;

    let sig_bytes = hex::decode(&request.signature)
        .map_err(|e| eyre!("Signature is not valid base16 format: {}", e))?;

    // Create public key
    let pk = PublicKey::from_bytes(&pk_bytes);

    // Look up signature algorithm by name
    let sig_alg: Box<dyn SignaturesAlg> = match request.sig_algorithm.as_str() {
        "secp256k1" => Box::new(crypto::rust::signatures::secp256k1::Secp256k1),
        "secp256k1-eth" => Box::new(crypto::rust::signatures::secp256k1_eth::Secp256k1Eth),
        "ed25519" => Box::new(crypto::rust::signatures::ed25519::Ed25519),
        #[cfg(feature = "schnorr_secp256k1_experimental")]
        "schnorr-secp256k1" => Box::new(SchnorrSecp256k1),
        #[cfg(feature = "schnorr_secp256k1_experimental")]
        "frost-secp256k1" => Box::new(FrostSecp256k1),
        _ => {
            return Err(eyre!(
                "Signature algorithm not supported: {}",
                request.sig_algorithm
            ))
        }
    };

    // Create DeployData (use the data from request)
    let deploy_data = request.data.clone();

    // Create signed deploy
    Signed::from_signed_data(deploy_data, pk, sig_bytes.into(), sig_alg)
        .map_err(|e| eyre!("Invalid signature: {}", e))?
        .ok_or_else(|| eyre!("Failed to create signed deploy"))
}

// Conversion functions for protobuf generated types
use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::{Bundle, Expr, GDeployId, GDeployerId, GPrivate};
use models::rhoapi::{GUnforgeable, Par};

/// Convert RhoUnforg to protobuf GUnforgeable.
/// Hex decode errors produce empty bytes with a warning log.
fn unforg_to_unforg_proto(unforg: RhoUnforg) -> eyre::Result<UnfInstance> {
    fn decode_hex(data: &str) -> eyre::Result<Vec<u8>> {
        hex::decode(data).map_err(|e| eyre::eyre!("Invalid hex in unforgeable name '{}': {}", data, e))
    }
    Ok(match unforg {
        RhoUnforg::UnforgPrivate { data } => UnfInstance::GPrivateBody(GPrivate {
            id: decode_hex(&data)?.into(),
        }),
        RhoUnforg::UnforgDeploy { data } => UnfInstance::GDeployIdBody(GDeployId {
            sig: decode_hex(&data)?.into(),
        }),
        RhoUnforg::UnforgDeployer { data } => UnfInstance::GDeployerIdBody(GDeployerId {
            public_key: decode_hex(&data)?.into(),
        }),
        RhoUnforg::UnforgSysAuthToken => {
            use models::rhoapi::GSysAuthToken;
            UnfInstance::GSysAuthTokenBody(GSysAuthToken {})
        }
    })
}

/// Convert DataAtNameRequest to Par. Returns error if hex decode fails.
fn to_par(rho_unforg: RhoUnforg) -> eyre::Result<Par> {
    Ok(Par {
        unforgeables: vec![GUnforgeable {
            unf_instance: Some(unforg_to_unforg_proto(rho_unforg)?),
        }],
        ..Default::default()
    })
}

/// Convert Par to RhoExpr - equivalent to Scala's exprFromParProto function
fn expr_from_par_proto(par: Par) -> Option<RhoExpr> {
    let has_process_fields = !par.sends.is_empty()
        || !par.receives.is_empty()
        || !par.news.is_empty()
        || !par.matches.is_empty()
        || !par.connectives.is_empty();

    let exprs = par.exprs.into_iter().filter_map(expr_from_expr_proto);
    let unforg_exprs = par.unforgeables.into_iter().filter_map(unforg_from_proto);
    let bundle_exprs = par.bundles.into_iter().filter_map(expr_from_bundle_proto);

    let all_exprs: Vec<RhoExpr> = exprs.chain(unforg_exprs).chain(bundle_exprs).collect();

    if all_exprs.len() == 1 {
        all_exprs.into_iter().next()
    } else if all_exprs.is_empty() {
        if has_process_fields {
            // Par has process-level constructs (sends, receives, etc.) but no data expressions
            Some(RhoExpr::ExprUnknown { type_name: "Process".to_string() })
        } else {
            None // Truly empty Par (Nil)
        }
    } else {
        Some(RhoExpr::ExprPar { data: all_exprs })
    }
}

/// Convert Expr to RhoExpr — handles all Rholang expression types.
fn expr_from_expr_proto(expr: Expr) -> Option<RhoExpr> {
    use models::rhoapi::expr::ExprInstance;
    use num_bigint::BigInt;

    let instance = expr.expr_instance?;
    Some(match instance {
        // Primitives
        ExprInstance::GBool(v) => RhoExpr::ExprBool { data: v },
        ExprInstance::GInt(v) => RhoExpr::ExprInt { data: v },
        ExprInstance::GString(v) => RhoExpr::ExprString { data: v },
        ExprInstance::GUri(v) => RhoExpr::ExprUri { data: v },
        ExprInstance::GByteArray(bytes) => RhoExpr::ExprBytes { data: hex::encode(&bytes) },

        // Extended numerics
        ExprInstance::GDouble(bits) => RhoExpr::ExprFloat { data: f64::from_bits(bits) },
        ExprInstance::GBigInt(bytes) => {
            let n = BigInt::from_signed_bytes_be(&bytes);
            RhoExpr::ExprBigInt { data: n.to_string() }
        }
        ExprInstance::GBigRat(rat) => {
            let num = BigInt::from_signed_bytes_be(&rat.numerator);
            let den = BigInt::from_signed_bytes_be(&rat.denominator);
            RhoExpr::ExprBigRat { numerator: num.to_string(), denominator: den.to_string() }
        }
        ExprInstance::GFixedPoint(fp) => {
            let unscaled = BigInt::from_signed_bytes_be(&fp.unscaled);
            RhoExpr::ExprFixedPoint { value: unscaled.to_string(), scale: fp.scale }
        }

        // Collections
        ExprInstance::ETupleBody(tuple) => {
            RhoExpr::ExprTuple { data: tuple.ps.into_iter().filter_map(expr_from_par_proto).collect() }
        }
        ExprInstance::EListBody(list) => {
            RhoExpr::ExprList { data: list.ps.into_iter().filter_map(expr_from_par_proto).collect() }
        }
        ExprInstance::ESetBody(set) => {
            RhoExpr::ExprSet { data: set.ps.into_iter().filter_map(expr_from_par_proto).collect() }
        }
        ExprInstance::EMapBody(map) => {
            let mut data = HashMap::new();
            for kv in map.kvs {
                if let (Some(key_par), Some(value_par)) = (kv.key, kv.value) {
                    if let (Some(key_expr), Some(value_expr)) = (expr_from_par_proto(key_par), expr_from_par_proto(value_par)) {
                        let key = extract_key_from_expr(&key_expr);
                        data.insert(key, value_expr);
                    }
                }
            }
            RhoExpr::ExprMap { data }
        }
        ExprInstance::EPathmapBody(pm) => {
            RhoExpr::ExprList { data: pm.ps.into_iter().filter_map(expr_from_par_proto).collect() }
        }
        ExprInstance::EZipperBody(z) => {
            let pathmap = z.pathmap.map(|pm| {
                RhoExpr::ExprList { data: pm.ps.into_iter().filter_map(expr_from_par_proto).collect() }
            });
            RhoExpr::ExprTuple {
                data: vec![
                    pathmap.unwrap_or(RhoExpr::ExprList { data: vec![] }),
                    RhoExpr::ExprList {
                        data: z.current_path.into_iter().map(|b| RhoExpr::ExprBytes { data: hex::encode(&b) }).collect(),
                    },
                ],
            }
        }

        // Unary operators
        ExprInstance::ENotBody(op) => {
            RhoExpr::ExprNot { data: Box::new(par_to_expr(op.p)) }
        }
        ExprInstance::ENegBody(op) => {
            RhoExpr::ExprNeg { data: Box::new(par_to_expr(op.p)) }
        }

        // Binary arithmetic
        ExprInstance::EPlusBody(op) => {
            RhoExpr::ExprPlus { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EMinusBody(op) => {
            RhoExpr::ExprMinus { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EMultBody(op) => {
            RhoExpr::ExprMult { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EDivBody(op) => {
            RhoExpr::ExprDiv { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EModBody(op) => {
            RhoExpr::ExprMod { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }

        // Comparison
        ExprInstance::ELtBody(op) => {
            RhoExpr::ExprLt { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::ELteBody(op) => {
            RhoExpr::ExprLte { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EGtBody(op) => {
            RhoExpr::ExprGt { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EGteBody(op) => {
            RhoExpr::ExprGte { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EEqBody(op) => {
            RhoExpr::ExprEq { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::ENeqBody(op) => {
            RhoExpr::ExprNeq { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }

        // Logical
        ExprInstance::EAndBody(op) => {
            RhoExpr::ExprAnd { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EOrBody(op) => {
            RhoExpr::ExprOr { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }

        // String operations
        ExprInstance::EPlusPlusBody(op) => {
            RhoExpr::ExprConcat { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EPercentPercentBody(op) => {
            RhoExpr::ExprInterpolate { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }
        ExprInstance::EMinusMinusBody(op) => {
            RhoExpr::ExprDiff { left: Box::new(par_to_expr(op.p1)), right: Box::new(par_to_expr(op.p2)) }
        }

        // Pattern matching
        ExprInstance::EMatchesBody(op) => {
            RhoExpr::ExprMatches { target: Box::new(par_to_expr(op.target)), pattern: Box::new(par_to_expr(op.pattern)) }
        }

        // Method call
        ExprInstance::EMethodBody(method) => {
            RhoExpr::ExprMethod {
                target: Box::new(par_to_expr(method.target)),
                name: method.method_name,
                args: method.arguments.into_iter().filter_map(expr_from_par_proto).collect(),
            }
        }

        // Variable
        ExprInstance::EVarBody(var) => {
            let index = var.v.and_then(|v| v.var_instance).map(|vi| match vi {
                models::rhoapi::var::VarInstance::BoundVar(i) => i,
                models::rhoapi::var::VarInstance::FreeVar(i) => i,
                models::rhoapi::var::VarInstance::Wildcard(_) => -1,
            }).unwrap_or(-1);
            RhoExpr::ExprVar { index }
        }
    })
}

/// Convert an optional Par to RhoExpr, falling back to ExprUnknown for None.
fn par_to_expr(par: Option<Par>) -> RhoExpr {
    par.and_then(expr_from_par_proto)
        .unwrap_or(RhoExpr::ExprUnknown { type_name: "Nil".to_string() })
}

/// Convert GUnforgeable to RhoExpr.
fn unforg_from_proto(unforg: GUnforgeable) -> Option<RhoExpr> {
    use models::rhoapi::g_unforgeable::UnfInstance;

    Some(match unforg.unf_instance? {
        UnfInstance::GPrivateBody(private) => RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgPrivate {
                data: hex::encode(&private.id),
            },
        },
        UnfInstance::GDeployIdBody(deploy_id) => RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgDeploy {
                data: hex::encode(&deploy_id.sig),
            },
        },
        UnfInstance::GDeployerIdBody(deployer_id) => RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgDeployer {
                data: hex::encode(&deployer_id.public_key),
            },
        },
        UnfInstance::GSysAuthTokenBody(_) => RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgSysAuthToken,
        },
    })
}

/// Convert Bundle to RhoExpr, preserving read/write permissions.
fn expr_from_bundle_proto(bundle: Bundle) -> Option<RhoExpr> {
    let body_expr = bundle.body.and_then(expr_from_par_proto)
        .unwrap_or(RhoExpr::ExprUnknown { type_name: "Nil".to_string() });
    Some(RhoExpr::ExprBundle {
        data: Box::new(body_expr),
        read: bundle.read_flag,
        write: bundle.write_flag,
    })
}

/// Extract a string key from a RhoExpr for map keys.
/// Primitive types use natural string representation; complex types use JSON serialization.
fn extract_key_from_expr(expr: &RhoExpr) -> String {
    match expr {
        RhoExpr::ExprString { data } => data.clone(),
        RhoExpr::ExprInt { data } => data.to_string(),
        RhoExpr::ExprBool { data } => data.to_string(),
        RhoExpr::ExprFloat { data } => data.to_string(),
        RhoExpr::ExprBigInt { data } => data.clone(),
        RhoExpr::ExprUri { data } => data.clone(),
        RhoExpr::ExprBytes { data } => data.clone(),
        RhoExpr::ExprUnforg { data } => match data {
            RhoUnforg::UnforgPrivate { data } => data.clone(),
            RhoUnforg::UnforgDeploy { data } => data.clone(),
            RhoUnforg::UnforgDeployer { data } => data.clone(),
            RhoUnforg::UnforgSysAuthToken => "SysAuthToken".to_string(),
        },
        // Complex types: serialize to JSON string
        other => serde_json::to_string(other).unwrap_or_else(|_| format!("{:?}", other)),
    }
}

/// Convert (Vec<DataWithBlockInfo>, i32) to DataAtNameResponse
/// Equivalent to Scala's toDataAtNameResponse function
fn to_data_at_name_response(req: (Vec<DataWithBlockInfo>, i32)) -> DataAtNameResponse {
    let (dbs, length) = req;

    let exprs_with_block: Vec<RhoExprWithBlock> = dbs
        .into_iter()
        .rev() // Reverse to match Scala's foldLeft behavior (+: prepends)
        .map(|data| {
            // Convert post_block_data (Vec<Par>) to Vec<RhoExpr> using expr_from_par_proto
            let exprs: Vec<RhoExpr> = data
                .post_block_data
                .into_iter()
                .filter_map(expr_from_par_proto)
                .collect();

            // Implements semantic of Par with Unit: P | Nil ==> P
            let expr = if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                RhoExpr::ExprPar { data: exprs }
            };

            // Convert LightBlockInfo to LightBlockInfoSerde
            let block = data
                .block
                .map(|block_info| LightBlockInfoSerde::from(block_info))
                .unwrap_or_default();
            RhoExprWithBlock { expr, block }
        })
        .collect();

    DataAtNameResponse {
        exprs: exprs_with_block,
        length,
    }
}

/// Convert (Vec<Par>, LightBlockInfo) to RhoDataResponse
/// Equivalent to Scala's toRhoDataResponse function
fn to_rho_data_response(pars: Vec<Par>, light_block_info: LightBlockInfo, cost: u64) -> RhoDataResponse {
    let rho_exprs: Vec<RhoExpr> = pars.into_iter().filter_map(expr_from_par_proto).collect();
    let block = LightBlockInfoSerde::from(light_block_info);

    RhoDataResponse {
        expr: rho_exprs,
        block,
        cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::rhoapi::expr::ExprInstance;
    use models::rhoapi::g_unforgeable::UnfInstance;
    use models::rhoapi::{
        Bundle, EList, EMap, ESet, ETuple, GDeployId, GDeployerId, GPrivate, KeyValuePair,
    };

    #[test]
    fn test_deploy_lookup_response_from_light_block_info() {
        let light_block = LightBlockInfoSerde {
            block_hash: "7bf8abc123".to_string(),
            sender: "0487def456".to_string(),
            seq_num: 17453,
            sig: "3044abcdef".to_string(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "root".to_string(),
            extra_bytes: vec![],
            version: 1,
            timestamp: 1770028092477,
            header_extra_bytes: vec![],
            parents_hash_list: vec!["parent1".to_string(), "parent2".to_string()],
            block_number: 52331,
            pre_state_hash: "preState123".to_string(),
            post_state_hash: "postState456".to_string(),
            body_extra_bytes: vec![],
            bonds: vec![
                super::super::serde_types::light_block_info::BondInfoJson {
                    validator: "validator1".to_string(),
                    stake: 100,
                },
                super::super::serde_types::light_block_info::BondInfoJson {
                    validator: "validator2".to_string(),
                    stake: 200,
                },
            ],
            block_size: "4096".to_string(),
            deploy_count: 5,
            fault_tolerance: 0.5,
            justifications: vec![
                super::super::serde_types::light_block_info::JustificationInfoJson {
                    validator: "validator1".to_string(),
                    latest_block_hash: "latestBlockHash1".to_string(),
                },
            ],
            rejected_deploys: vec![],
        };

        let result = DeployLookupResponse::from_block_and_cost(light_block, 0);

        assert_eq!(result.block_hash, "7bf8abc123");
        assert_eq!(result.block_number, 52331);
        assert_eq!(result.timestamp, 1770028092477);
        assert_eq!(result.sender, "0487def456");
        assert_eq!(result.seq_num, 17453);
        assert_eq!(result.sig, "3044abcdef");
        assert_eq!(result.sig_algorithm, "secp256k1");
        assert_eq!(result.shard_id, "root");
        assert_eq!(result.version, 1);
    }

    #[test]
    fn test_deploy_lookup_response_excludes_heavy_fields() {
        let response = DeployLookupResponse {
            block_hash: "hash".to_string(),
            block_number: 1,
            timestamp: 123,
            sender: "sender".to_string(),
            seq_num: 1,
            sig: "sig".to_string(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "root".to_string(),
            version: 1,
            cost: 0,
        };

        let json = serde_json::to_value(&response).unwrap();

        // Should contain only the 9 minimal fields
        assert!(json.get("blockHash").is_some());
        assert!(json.get("blockNumber").is_some());
        assert!(json.get("timestamp").is_some());
        assert!(json.get("sender").is_some());
        assert!(json.get("seqNum").is_some());
        assert!(json.get("sig").is_some());
        assert!(json.get("sigAlgorithm").is_some());
        assert!(json.get("shardId").is_some());
        assert!(json.get("version").is_some());

        // Should NOT contain heavy fields
        assert!(json.get("bonds").is_none());
        assert!(json.get("justifications").is_none());
        assert!(json.get("parentsHashList").is_none());
        assert!(json.get("preStateHash").is_none());
        assert!(json.get("postStateHash").is_none());
        assert!(json.get("faultTolerance").is_none());
        assert!(json.get("deployCount").is_none());
        assert!(json.get("blockSize").is_none());
    }

    #[test]
    fn test_deploy_lookup_response_handles_empty_string_fields() {
        let mut light_block = LightBlockInfoSerde {
            block_hash: "7bf8abc123".to_string(),
            sender: "0487def456".to_string(),
            seq_num: 17453,
            sig: "3044abcdef".to_string(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "root".to_string(),
            extra_bytes: vec![],
            version: 1,
            timestamp: 1770028092477,
            header_extra_bytes: vec![],
            parents_hash_list: vec![],
            block_number: 52331,
            pre_state_hash: String::new(),
            post_state_hash: String::new(),
            body_extra_bytes: vec![],
            bonds: vec![],
            block_size: String::new(),
            deploy_count: 0,
            fault_tolerance: 0.0,
            justifications: vec![],
            rejected_deploys: vec![],
        };
        light_block.block_hash = String::new();
        light_block.sender = String::new();
        light_block.sig = String::new();
        light_block.sig_algorithm = String::new();
        light_block.shard_id = String::new();

        let result = DeployLookupResponse::from_block_and_cost(light_block, 0);

        assert_eq!(result.block_hash, "");
        assert_eq!(result.sender, "");
        assert_eq!(result.sig, "");
        assert_eq!(result.sig_algorithm, "");
        assert_eq!(result.shard_id, "");
    }

    #[test]
    fn test_deploy_lookup_response_handles_zero_numeric_fields() {
        let light_block = LightBlockInfoSerde {
            block_hash: "hash".to_string(),
            sender: "sender".to_string(),
            seq_num: 0,
            sig: "sig".to_string(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "root".to_string(),
            extra_bytes: vec![],
            version: 0,
            timestamp: 0,
            header_extra_bytes: vec![],
            parents_hash_list: vec![],
            block_number: 0,
            pre_state_hash: String::new(),
            post_state_hash: String::new(),
            body_extra_bytes: vec![],
            bonds: vec![],
            block_size: String::new(),
            deploy_count: 0,
            fault_tolerance: 0.0,
            justifications: vec![],
            rejected_deploys: vec![],
        };

        let result = DeployLookupResponse::from_block_and_cost(light_block, 0);

        assert_eq!(result.block_number, 0);
        assert_eq!(result.timestamp, 0);
        assert_eq!(result.seq_num, 0);
        assert_eq!(result.version, 0);
    }

    #[test]
    fn test_deploy_lookup_response_correct_regardless_of_bonds_list_size() {
        let large_bonds: Vec<super::super::serde_types::light_block_info::BondInfoJson> = (1
            ..=1000)
            .map(
                |i| super::super::serde_types::light_block_info::BondInfoJson {
                    validator: format!("validator{}", i),
                    stake: i,
                },
            )
            .collect();

        let light_block = LightBlockInfoSerde {
            block_hash: "7bf8abc123".to_string(),
            sender: "0487def456".to_string(),
            seq_num: 17453,
            sig: "3044abcdef".to_string(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "root".to_string(),
            extra_bytes: vec![],
            version: 1,
            timestamp: 1770028092477,
            header_extra_bytes: vec![],
            parents_hash_list: vec![],
            block_number: 52331,
            pre_state_hash: String::new(),
            post_state_hash: String::new(),
            body_extra_bytes: vec![],
            bonds: large_bonds,
            block_size: String::new(),
            deploy_count: 0,
            fault_tolerance: 0.0,
            justifications: vec![],
            rejected_deploys: vec![],
        };

        let result = DeployLookupResponse::from_block_and_cost(light_block, 0);

        // Response should be the same regardless of bonds size
        assert_eq!(result.block_hash, "7bf8abc123");
        assert_eq!(result.block_number, 52331);
    }

    #[test]
    fn test_deploy_request_serialization() {
        let request = DeployRequest {
            data: DeployData {
                term: "contract".to_string(),
                time_stamp: 1234567890,
                phlo_price: 1,
                phlo_limit: 1000000,
                valid_after_block_number: 0,
                shard_id: "".to_string(),
                expiration_timestamp: None,
            },
            deployer: "0123456789abcdef".to_string(),
            signature: "fedcba9876543210".to_string(),
            sig_algorithm: "secp256k1".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: DeployRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request.deployer, deserialized.deployer);
        assert_eq!(request.signature, deserialized.signature);
        assert_eq!(request.sig_algorithm, deserialized.sig_algorithm);
    }

    #[test]
    fn test_rho_expr_serialization() {
        let expr = RhoExpr::ExprBool { data: true };
        let json = serde_json::to_string(&expr).unwrap();
        let deserialized: RhoExpr = serde_json::from_str(&json).unwrap();

        match deserialized {
            RhoExpr::ExprBool { data } => assert!(data),
            _ => panic!("Expected ExprBool"),
        }
    }

    #[test]
    fn test_expr_from_par_proto_empty() {
        let par = Par::default();
        let result = expr_from_par_proto(par);
        assert!(result.is_none());
    }

    #[test]
    fn test_expr_from_par_proto_single_bool() {
        let par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GBool(true)),
            }],
            ..Default::default()
        };
        let result = expr_from_par_proto(par);
        assert!(matches!(result, Some(RhoExpr::ExprBool { data: true })));
    }

    #[test]
    fn test_expr_from_par_proto_multiple_exprs() {
        let par = Par {
            exprs: vec![
                Expr {
                    expr_instance: Some(ExprInstance::GBool(true)),
                },
                Expr {
                    expr_instance: Some(ExprInstance::GInt(42)),
                },
            ],
            ..Default::default()
        };
        let result = expr_from_par_proto(par);
        match result {
            Some(RhoExpr::ExprPar { data }) => {
                assert_eq!(data.len(), 2);
                assert!(matches!(data[0], RhoExpr::ExprBool { data: true }));
                assert!(matches!(data[1], RhoExpr::ExprInt { data: 42 }));
            }
            _ => panic!("Expected ExprPar with 2 elements"),
        }
    }

    #[test]
    fn test_expr_from_expr_proto_primitive_types() {
        // Test GBool
        let expr = Expr {
            expr_instance: Some(ExprInstance::GBool(true)),
        };
        let result = expr_from_expr_proto(expr);
        assert!(matches!(result, Some(RhoExpr::ExprBool { data: true })));

        // Test GInt
        let expr = Expr {
            expr_instance: Some(ExprInstance::GInt(42)),
        };
        let result = expr_from_expr_proto(expr);
        assert!(matches!(result, Some(RhoExpr::ExprInt { data: 42 })));

        // Test GString
        let expr = Expr {
            expr_instance: Some(ExprInstance::GString("hello".to_string())),
        };
        let result = expr_from_expr_proto(expr);
        assert!(matches!(result, Some(RhoExpr::ExprString { data }) if data == "hello"));

        // Test GUri
        let expr = Expr {
            expr_instance: Some(ExprInstance::GUri("rho:io:stdout".to_string())),
        };
        let result = expr_from_expr_proto(expr);
        assert!(matches!(result, Some(RhoExpr::ExprUri { data }) if data == "rho:io:stdout"));

        // Test GByteArray
        let expr = Expr {
            expr_instance: Some(ExprInstance::GByteArray(vec![0x01, 0x02, 0x03])),
        };
        let result = expr_from_expr_proto(expr);
        assert!(matches!(result, Some(RhoExpr::ExprBytes { data }) if data == "010203"));
    }

    #[test]
    fn test_expr_from_expr_proto_tuple() {
        let tuple = ETuple {
            ps: vec![
                Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GInt(1)),
                    }],
                    ..Default::default()
                },
                Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GString("hello".to_string())),
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let expr = Expr {
            expr_instance: Some(ExprInstance::ETupleBody(tuple)),
        };
        let result = expr_from_expr_proto(expr);
        match result {
            Some(RhoExpr::ExprTuple { data }) => {
                assert_eq!(data.len(), 2);
                assert!(matches!(data[0], RhoExpr::ExprInt { data: 1 }));
                assert!(matches!(data[1], RhoExpr::ExprString { data: ref d } if d == "hello"));
            }
            _ => panic!("Expected ExprTuple"),
        }
    }

    #[test]
    fn test_expr_from_expr_proto_list() {
        let list = EList {
            ps: vec![
                Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GInt(1)),
                    }],
                    ..Default::default()
                },
                Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GInt(2)),
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let expr = Expr {
            expr_instance: Some(ExprInstance::EListBody(list)),
        };
        let result = expr_from_expr_proto(expr);
        match result {
            Some(RhoExpr::ExprList { data }) => {
                assert_eq!(data.len(), 2);
                assert!(matches!(data[0], RhoExpr::ExprInt { data: 1 }));
                assert!(matches!(data[1], RhoExpr::ExprInt { data: 2 }));
            }
            _ => panic!("Expected ExprList"),
        }
    }

    #[test]
    fn test_expr_from_expr_proto_set() {
        let set = ESet {
            ps: vec![
                Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GString("a".to_string())),
                    }],
                    ..Default::default()
                },
                Par {
                    exprs: vec![Expr {
                        expr_instance: Some(ExprInstance::GString("b".to_string())),
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let expr = Expr {
            expr_instance: Some(ExprInstance::ESetBody(set)),
        };
        let result = expr_from_expr_proto(expr);
        match result {
            Some(RhoExpr::ExprSet { data }) => {
                assert_eq!(data.len(), 2);
                assert!(matches!(data[0], RhoExpr::ExprString { data: ref d } if d == "a"));
                assert!(matches!(data[1], RhoExpr::ExprString { data: ref d } if d == "b"));
            }
            _ => panic!("Expected ExprSet"),
        }
    }

    #[test]
    fn test_expr_from_expr_proto_map() {
        let map = EMap {
            kvs: vec![
                KeyValuePair {
                    key: Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::GString("key1".to_string())),
                        }],
                        ..Default::default()
                    }),
                    value: Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::GInt(42)),
                        }],
                        ..Default::default()
                    }),
                },
                KeyValuePair {
                    key: Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::GString("key2".to_string())),
                        }],
                        ..Default::default()
                    }),
                    value: Some(Par {
                        exprs: vec![Expr {
                            expr_instance: Some(ExprInstance::GString("value2".to_string())),
                        }],
                        ..Default::default()
                    }),
                },
            ],
            ..Default::default()
        };

        let expr = Expr {
            expr_instance: Some(ExprInstance::EMapBody(map)),
        };
        let result = expr_from_expr_proto(expr);
        match result {
            Some(RhoExpr::ExprMap { data }) => {
                assert_eq!(data.len(), 2);
                assert!(data.contains_key("key1"));
                assert!(data.contains_key("key2"));
                assert!(matches!(data["key1"], RhoExpr::ExprInt { data: 42 }));
                assert!(
                    matches!(data["key2"], RhoExpr::ExprString { data: ref d } if d == "value2")
                );
            }
            _ => panic!("Expected ExprMap"),
        }
    }

    #[test]
    fn test_unforg_from_proto_private() {
        let unforg = GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: vec![0x01, 0x02, 0x03],
            })),
        };
        let result = unforg_from_proto(unforg);
        match result {
            Some(RhoExpr::ExprUnforg { data }) => {
                assert!(matches!(data, RhoUnforg::UnforgPrivate { data: ref d } if d == "010203"));
            }
            _ => panic!("Expected ExprUnforg with UnforgPrivate"),
        }
    }

    #[test]
    fn test_unforg_from_proto_deploy() {
        let unforg = GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId {
                sig: vec![0x04, 0x05, 0x06],
            })),
        };
        let result = unforg_from_proto(unforg);
        match result {
            Some(RhoExpr::ExprUnforg { data }) => {
                assert!(matches!(data, RhoUnforg::UnforgDeploy { data: ref d } if d == "040506"));
            }
            _ => panic!("Expected ExprUnforg with UnforgDeploy"),
        }
    }

    #[test]
    fn test_unforg_from_proto_deployer() {
        let unforg = GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                public_key: vec![0x07, 0x08, 0x09],
            })),
        };
        let result = unforg_from_proto(unforg);
        match result {
            Some(RhoExpr::ExprUnforg { data }) => {
                assert!(matches!(data, RhoUnforg::UnforgDeployer { data: ref d } if d == "070809"));
            }
            _ => panic!("Expected ExprUnforg with UnforgDeployer"),
        }
    }

    #[test]
    fn test_expr_from_bundle_proto() {
        let bundle = Bundle {
            body: Some(Par {
                exprs: vec![Expr {
                    expr_instance: Some(ExprInstance::GString("bundle_content".to_string())),
                }],
                ..Default::default()
            }),
            write_flag: true,
            read_flag: false,
        };
        let result = expr_from_bundle_proto(bundle);
        assert!(matches!(
            result,
            Some(RhoExpr::ExprBundle { ref data, write: true, read: false })
            if matches!(data.as_ref(), RhoExpr::ExprString { data } if data == "bundle_content")
        ));
    }

    #[test]
    fn test_expr_from_bundle_proto_empty() {
        let bundle = Bundle {
            body: None,
            write_flag: false,
            read_flag: true,
        };
        let result = expr_from_bundle_proto(bundle);
        // Empty body bundle returns ExprBundle with ExprUnknown body
        assert!(matches!(
            result,
            Some(RhoExpr::ExprBundle { read: true, write: false, .. })
        ));
    }

    #[test]
    fn test_extract_key_from_expr() {
        // Test string key
        let expr = RhoExpr::ExprString {
            data: "hello".to_string(),
        };
        assert_eq!(extract_key_from_expr(&expr), "hello");

        // Test int key
        let expr = RhoExpr::ExprInt { data: 42 };
        assert_eq!(extract_key_from_expr(&expr), "42");

        // Test bool key
        let expr = RhoExpr::ExprBool { data: true };
        assert_eq!(extract_key_from_expr(&expr), "true");

        // Test URI key
        let expr = RhoExpr::ExprUri {
            data: "rho:io:stdout".to_string(),
        };
        assert_eq!(extract_key_from_expr(&expr), "rho:io:stdout");

        // Test bytes key
        let expr = RhoExpr::ExprBytes {
            data: "010203".to_string(),
        };
        assert_eq!(extract_key_from_expr(&expr), "010203");

        // Test unforgeable keys
        let expr = RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgPrivate {
                data: "private".to_string(),
            },
        };
        assert_eq!(extract_key_from_expr(&expr), "private");

        // Test complex key type — serialized to JSON
        let expr = RhoExpr::ExprPar { data: vec![] };
        let key = extract_key_from_expr(&expr);
        assert!(!key.is_empty(), "complex keys should serialize to non-empty string");
    }
}
