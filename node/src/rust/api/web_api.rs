//! Web API implementation for F1r3fly node
use crate::rust::api::serde_types::light_block_info::LightBlockInfoSerde;
use crate::rust::web::transaction::{CacheTransactionAPI, TransactionAPI, TransactionResponse};
use crate::rust::web::version_info::get_version_info_str;
use casper::rust::api::block_api::BlockAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::ProposeFunction;
use comm::rust::discovery::node_discovery::NodeDiscovery;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use crypto::rust::{
    public_key::PublicKey, signatures::signatures_alg::SignaturesAlg, signatures::signed::Signed,
};
use eyre::{eyre, Result};
use hex;
use models::casper::{BlockInfo, DataWithBlockInfo, LightBlockInfo};
use models::rust::casper::protocol::casper_message::DeployData;
use serde::{Deserialize, Serialize};
use shared::rust::store::key_value_typed_store::KeyValueTypedStore;
use shared::rust::ByteString;
use std::collections::HashMap;
use std::sync::Arc;

/// Web API trait defining the interface for HTTP endpoints
#[async_trait::async_trait(?Send)] // TODO: remove the ?Send once the casper::EngineCell wrapped interfaces would be reimplemented with Send support
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
    async fn last_finalized_block(&self) -> Result<BlockInfo>;

    /// Get a specific block by hash
    async fn get_block(&self, hash: String) -> Result<BlockInfo>;

    /// Get blocks with specified depth
    async fn get_blocks(&self, depth: i32) -> Result<Vec<LightBlockInfoSerde>>;

    /// Find a deploy by ID
    async fn find_deploy(&self, deploy_id: String) -> Result<LightBlockInfoSerde>;

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
pub struct WebApiImpl<TA, TS>
where
    TA: TransactionAPI + Send + Sync + 'static,
    TS: KeyValueTypedStore<String, TransactionResponse> + Send + Sync + 'static,
{
    api_max_blocks_limit: i32,
    dev_mode: bool,
    network_id: String,
    shard_id: String,
    min_phlo_price: i64,
    is_node_read_only: bool,
    engine_cell: Arc<EngineCell>,
    cache_transaction_api: CacheTransactionAPI<TA, TS>,
    rp_conf: RPConf,
    connections_cell: ConnectionsCell,
    node_discovery: Box<dyn NodeDiscovery + Send + Sync + 'static>,
    trigger_propose_f: Option<Box<ProposeFunction>>,
}

impl<TA, TS> WebApiImpl<TA, TS>
where
    TA: TransactionAPI + Send + Sync + 'static,
    TS: KeyValueTypedStore<String, TransactionResponse> + Send + Sync + 'static,
{
    pub fn new(
        api_max_blocks_limit: i32,
        dev_mode: bool,
        network_id: String,
        shard_id: String,
        min_phlo_price: i64,
        is_node_read_only: bool,
        cache_transaction_api: CacheTransactionAPI<TA, TS>,
        engine_cell: Arc<EngineCell>,
        rp_conf: RPConf,
        connections_cell: ConnectionsCell,
        node_discovery: Box<dyn NodeDiscovery + Send + Sync + 'static>,
        trigger_propose_f: Option<Box<ProposeFunction>>,
    ) -> Self {
        Self {
            api_max_blocks_limit,
            dev_mode,
            network_id,
            shard_id,
            min_phlo_price,
            is_node_read_only,
            engine_cell,
            cache_transaction_api,
            rp_conf,
            connections_cell,
            node_discovery,
            trigger_propose_f,
        }
    }
}

#[async_trait::async_trait(?Send)]
impl<TA, TS> WebApi for WebApiImpl<TA, TS>
where
    TA: TransactionAPI + Send + Sync + 'static,
    TS: KeyValueTypedStore<String, TransactionResponse> + Send + Sync + 'static,
{
    async fn status(&self) -> Result<ApiStatus> {
        let address = self.rp_conf.local.to_address();
        let peers = self.connections_cell.read()?.len() as i32;
        let nodes = self.node_discovery.peers()?.len() as i32;

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
        })
    }

    async fn prepare_deploy(&self, request: Option<PrepareRequest>) -> Result<PrepareResponse> {
        let seq_number = BlockAPI::get_latest_message(&self.engine_cell)
            .await
            .map(|message| message.sequence_number)
            .unwrap_or(-1);

        let names = if let Some(req) = request {
            BlockAPI::preview_private_names(&req.deployer, req.timestamp, req.name_qty)?
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
            to_par(request.name),
            self.api_max_blocks_limit,
        )
        .await?;

        Ok(to_data_at_name_response(res))
    }

    async fn get_data_at_par(
        &self,
        request: DataAtNameByBlockHashRequest,
    ) -> Result<RhoDataResponse> {
        let res = BlockAPI::get_data_at_par(
            &self.engine_cell,
            &to_par(request.name),
            request.block_hash,
            request.use_pre_state_hash,
        )
        .await?;

        Ok(to_rho_data_response(res))
    }

    async fn last_finalized_block(&self) -> Result<BlockInfo> {
        BlockAPI::last_finalized_block(&self.engine_cell).await
    }

    async fn get_block(&self, hash: String) -> Result<BlockInfo> {
        BlockAPI::get_block(&self.engine_cell, &hash).await
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
        let block = BlockAPI::find_deploy(&self.engine_cell, &deploy_id_bytes).await?;

        Ok(LightBlockInfoSerde::from(block))
    }

    async fn exploratory_deploy(
        &self,
        term: String,
        block_hash: Option<String>,
        use_pre_state_hash: bool,
    ) -> Result<RhoDataResponse> {
        let res = BlockAPI::exploratory_deploy(
            &self.engine_cell,
            term,
            block_hash,
            use_pre_state_hash,
            self.dev_mode,
        )
        .await?;

        Ok(to_rho_data_response(res))
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

    async fn get_transaction(&self, hash: String) -> Result<TransactionResponse> {
        self.cache_transaction_api.get_transaction(hash).await
    }
}

// Rholang terms interesting for translation to JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RhoExpr {
    /// Nested expressions (Par, Tuple, List and Set are converted to JSON list)
    #[serde(rename = "par")]
    ExprPar { data: Vec<RhoExpr> },
    #[serde(rename = "tuple")]
    ExprTuple { data: Vec<RhoExpr> },
    #[serde(rename = "list")]
    ExprList { data: Vec<RhoExpr> },
    #[serde(rename = "set")]
    ExprSet { data: Vec<RhoExpr> },
    #[serde(rename = "map")]
    ExprMap { data: HashMap<String, RhoExpr> },

    /// Terminal expressions (here is the data)
    #[serde(rename = "bool")]
    ExprBool { data: bool },
    #[serde(rename = "int")]
    ExprInt { data: i64 },
    #[serde(rename = "string")]
    ExprString { data: String },
    #[serde(rename = "uri")]
    ExprUri { data: String },
    /// Binary data is encoded as base16 string
    #[serde(rename = "bytes")]
    ExprBytes { data: String },
    #[serde(rename = "unforg")]
    ExprUnforg { data: RhoUnforg },
}

/// Unforgeable name types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RhoUnforg {
    #[serde(rename = "private")]
    UnforgPrivate { data: String },
    #[serde(rename = "deploy")]
    UnforgDeploy { data: String },
    #[serde(rename = "deployer")]
    UnforgDeployer { data: String },
}

// API request & response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployRequest {
    pub data: DeployData,
    pub deployer: String,
    pub signature: String,
    pub sig_algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreDeployRequest {
    pub term: String,
    pub block_hash: String,
    pub use_pre_state_hash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAtNameRequest {
    /// For simplicity only one Unforgeable name is allowed
    /// instead of the whole RhoExpr (proto Par)
    pub name: RhoUnforg,
    /// Number of blocks in the past to search for data
    pub depth: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAtNameByBlockHashRequest {
    pub name: RhoUnforg,
    pub block_hash: String,
    pub use_pre_state_hash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAtNameResponse {
    pub exprs: Vec<RhoExprWithBlock>,
    pub length: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhoExprWithBlock {
    pub expr: RhoExpr,
    pub block: LightBlockInfoSerde,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploratoryDeployResponse {
    pub expr: Vec<RhoExpr>,
    pub block: LightBlockInfoSerde,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhoDataResponse {
    pub expr: Vec<RhoExpr>,
    pub block: LightBlockInfoSerde,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareRequest {
    pub deployer: ByteString,
    pub timestamp: i64,
    pub name_qty: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareResponse {
    pub names: Vec<ByteString>,
    pub seq_number: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiStatus {
    pub version: VersionInfo,
    pub address: String,
    pub network_id: String,
    pub shard_id: String,
    pub peers: i32,
    pub nodes: i32,
    pub min_phlo_price: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub api: String,
    pub node: String,
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
        "ed25519" => Box::new(crypto::rust::signatures::ed25519::Ed25519),
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

/// Convert RhoUnforg to protobuf GUnforgeable
fn unforg_to_unforg_proto(unforg: RhoUnforg) -> UnfInstance {
    match unforg {
        RhoUnforg::UnforgPrivate { data } => UnfInstance::GPrivateBody(GPrivate {
            id: hex::decode(data).unwrap_or_default().into(),
        }),
        RhoUnforg::UnforgDeploy { data } => UnfInstance::GDeployIdBody(GDeployId {
            sig: hex::decode(data).unwrap_or_default().into(),
        }),
        RhoUnforg::UnforgDeployer { data } => UnfInstance::GDeployerIdBody(GDeployerId {
            public_key: hex::decode(data).unwrap_or_default().into(),
        }),
    }
}

/// Convert DataAtNameRequest to Par
fn to_par(rho_unforg: RhoUnforg) -> Par {
    Par {
        unforgeables: vec![GUnforgeable {
            unf_instance: Some(unforg_to_unforg_proto(rho_unforg)),
        }],
        ..Default::default()
    }
}

/// Convert Par to RhoExpr - equivalent to Scala's exprFromParProto function
fn expr_from_par_proto(par: Par) -> Option<RhoExpr> {
    let exprs = par.exprs.into_iter().filter_map(expr_from_expr_proto);

    let unforg_exprs = par.unforgeables.into_iter().filter_map(unforg_from_proto);

    let bundle_exprs = par.bundles.into_iter().filter_map(expr_from_bundle_proto);

    let all_exprs: Vec<RhoExpr> = exprs.chain(unforg_exprs).chain(bundle_exprs).collect();

    // Implements semantic of Par with Unit: P | Nil ==> P
    if all_exprs.len() == 1 {
        all_exprs.into_iter().next()
    } else if all_exprs.is_empty() {
        None
    } else {
        Some(RhoExpr::ExprPar { data: all_exprs })
    }
}

/// Convert Expr to RhoExpr - equivalent to Scala's exprFromExprProto function
fn expr_from_expr_proto(expr: Expr) -> Option<RhoExpr> {
    use models::rhoapi::expr::ExprInstance;

    match expr.expr_instance? {
        // Primitive types
        ExprInstance::GBool(value) => Some(RhoExpr::ExprBool { data: value }),
        ExprInstance::GInt(value) => Some(RhoExpr::ExprInt { data: value }),
        ExprInstance::GString(value) => Some(RhoExpr::ExprString { data: value }),
        ExprInstance::GUri(value) => Some(RhoExpr::ExprUri { data: value }),
        ExprInstance::GByteArray(bytes) => {
            // Binary data as base16 string
            Some(RhoExpr::ExprBytes {
                data: hex::encode(&bytes),
            })
        }
        // Tuple
        ExprInstance::ETupleBody(tuple) => {
            let data: Vec<RhoExpr> = tuple
                .ps
                .into_iter()
                .filter_map(expr_from_par_proto)
                .collect();
            Some(RhoExpr::ExprTuple { data })
        }
        // List
        ExprInstance::EListBody(list) => {
            let data: Vec<RhoExpr> = list
                .ps
                .into_iter()
                .filter_map(expr_from_par_proto)
                .collect();
            Some(RhoExpr::ExprList { data })
        }
        // Set
        ExprInstance::ESetBody(set) => {
            let data: Vec<RhoExpr> = set.ps.into_iter().filter_map(expr_from_par_proto).collect();
            Some(RhoExpr::ExprSet { data })
        }
        // Map
        ExprInstance::EMapBody(map) => {
            let mut data = HashMap::new();
            for kv in map.kvs {
                if let (Some(key_par), Some(value_par)) = (kv.key, kv.value) {
                    let key_expr = expr_from_par_proto(key_par);
                    let value_expr = expr_from_par_proto(value_par);
                    if let (Some(key_expr), Some(value_expr)) = (key_expr, value_expr) {
                        if let Some(key) = extract_key_from_expr(&key_expr) {
                            data.insert(key, value_expr);
                        }
                    }
                }
            }
            Some(RhoExpr::ExprMap { data })
        }
        _ => None, // Other expression types not handled in the original Scala
    }
}

/// Convert GUnforgeable to RhoExpr - equivalent to Scala's unforgFromProto function
fn unforg_from_proto(unforg: GUnforgeable) -> Option<RhoExpr> {
    use models::rhoapi::g_unforgeable::UnfInstance;

    match unforg.unf_instance? {
        UnfInstance::GPrivateBody(private) => Some(RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgPrivate {
                data: hex::encode(&private.id),
            },
        }),
        UnfInstance::GDeployIdBody(deploy_id) => Some(RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgDeploy {
                data: hex::encode(&deploy_id.sig),
            },
        }),
        UnfInstance::GDeployerIdBody(deployer_id) => Some(RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgDeployer {
                data: hex::encode(&deployer_id.public_key),
            },
        }),
        _ => None, // Other unforgeable types not handled in the original Scala
    }
}

/// Convert Bundle to RhoExpr - equivalent to Scala's exprFromBundleProto function
fn expr_from_bundle_proto(bundle: Bundle) -> Option<RhoExpr> {
    if let Some(body) = bundle.body {
        expr_from_par_proto(body)
    } else {
        None
    }
}

/// Extract a string key from a RhoExpr for map keys - equivalent to Scala's key extraction logic
fn extract_key_from_expr(expr: &RhoExpr) -> Option<String> {
    match expr {
        RhoExpr::ExprString { data } => Some(data.clone()),
        RhoExpr::ExprInt { data } => Some(data.to_string()),
        RhoExpr::ExprBool { data } => Some(data.to_string()),
        RhoExpr::ExprUri { data } => Some(data.clone()),
        RhoExpr::ExprUnforg { data } => match data {
            RhoUnforg::UnforgPrivate { data } => Some(data.clone()),
            RhoUnforg::UnforgDeploy { data } => Some(data.clone()),
            RhoUnforg::UnforgDeployer { data } => Some(data.clone()),
        },
        RhoExpr::ExprBytes { data } => Some(data.clone()),
        _ => None,
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
fn to_rho_data_response(data: (Vec<Par>, LightBlockInfo)) -> RhoDataResponse {
    let (pars, light_block_info) = data;

    // Convert Vec<Par> to Vec<RhoExpr> using expr_from_par_proto
    let rho_exprs: Vec<RhoExpr> = pars.into_iter().filter_map(expr_from_par_proto).collect();

    // Convert LightBlockInfo to LightBlockInfoSerde
    let block = LightBlockInfoSerde::from(light_block_info);

    RhoDataResponse {
        expr: rho_exprs,
        block,
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
    fn test_deploy_request_serialization() {
        let request = DeployRequest {
            data: DeployData {
                term: "contract".to_string(),
                time_stamp: 1234567890,
                phlo_price: 1,
                phlo_limit: 1000000,
                valid_after_block_number: 0,
                shard_id: "".to_string(),
                language: "rho".to_string(),
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
            ..Default::default()
        };
        let result = expr_from_bundle_proto(bundle);
        assert!(matches!(result, Some(RhoExpr::ExprString { data }) if data == "bundle_content"));
    }

    #[test]
    fn test_expr_from_bundle_proto_empty() {
        let bundle = Bundle {
            body: None,
            ..Default::default()
        };
        let result = expr_from_bundle_proto(bundle);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_key_from_expr() {
        // Test string key
        let expr = RhoExpr::ExprString {
            data: "hello".to_string(),
        };
        assert_eq!(extract_key_from_expr(&expr), Some("hello".to_string()));

        // Test int key
        let expr = RhoExpr::ExprInt { data: 42 };
        assert_eq!(extract_key_from_expr(&expr), Some("42".to_string()));

        // Test bool key
        let expr = RhoExpr::ExprBool { data: true };
        assert_eq!(extract_key_from_expr(&expr), Some("true".to_string()));

        // Test URI key
        let expr = RhoExpr::ExprUri {
            data: "rho:io:stdout".to_string(),
        };
        assert_eq!(
            extract_key_from_expr(&expr),
            Some("rho:io:stdout".to_string())
        );

        // Test bytes key
        let expr = RhoExpr::ExprBytes {
            data: "010203".to_string(),
        };
        assert_eq!(extract_key_from_expr(&expr), Some("010203".to_string()));

        // Test unforgeable keys
        let expr = RhoExpr::ExprUnforg {
            data: RhoUnforg::UnforgPrivate {
                data: "private".to_string(),
            },
        };
        assert_eq!(extract_key_from_expr(&expr), Some("private".to_string()));

        // Test unsupported key type
        let expr = RhoExpr::ExprPar { data: vec![] };
        assert!(extract_key_from_expr(&expr).is_none());
    }
}
