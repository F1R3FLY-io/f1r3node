// See node/src/main/scala/coop/rchain/node/runtime/APIServers.scala

use std::sync::Arc;

use casper::rust::api::block_report_api::BlockReportAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::state::instances::proposer_state::ProposerState;
use casper::rust::ProposeFunction;
use tokio::sync::RwLock;

use crate::rust::api::{
    deploy_grpc_service_v1::DeployGrpcServiceV1Impl,
    lsp_grpc_service::LspGrpcServiceImpl,
    propose_grpc_service_v1::ProposeGrpcServiceV1Impl,
    repl_grpc_service::ReplGrpcServiceImpl,
};
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use comm::rust::discovery::node_discovery::NodeDiscovery;
use comm::rust::rp::connect::ConnectionsCell;
use comm::rust::rp::rp_conf::RPConf;
use rholang::rust::interpreter::rho_runtime::RhoRuntimeImpl;

/// Container for all gRPC API service implementations
///
/// This struct holds instances of the four main API services:
/// - REPL: Read-Eval-Print Loop for Rholang execution
/// - Deploy: Contract deployment and blockchain query operations
/// - Propose: Block proposal operations
/// - LSP: Language Server Protocol for Rholang validation
pub struct APIServers {
    pub repl: ReplGrpcServiceImpl,
    pub propose: ProposeGrpcServiceV1Impl,
    pub deploy: DeployGrpcServiceV1Impl,
    pub lsp: LspGrpcServiceImpl,
}

impl APIServers {
    /// Build all API services with their dependencies
    ///
    /// # Parameters
    ///
    /// ## REPL Service Dependencies
    /// - `runtime`: RhoRuntime for executing Rholang code
    ///
    /// ## Propose Service Dependencies
    /// - `trigger_propose_f_opt`: Optional function to trigger block proposals
    /// - `proposer_state_ref_opt`: Optional reference to proposer state
    ///
    /// ## Deploy Service Dependencies
    /// - `api_max_blocks_limit`: Maximum number of blocks to return in queries
    /// - `dev_mode`: Enable development mode features
    /// - `propose_f_opt`: Optional propose function for auto-propose
    /// - `block_report_api`: API for block reporting
    /// - `network_id`: Network identifier
    /// - `shard_id`: Shard identifier
    /// - `min_phlo_price`: Minimum phlo price for deploys
    /// - `is_node_read_only`: Whether node is in read-only mode
    ///
    /// ## Shared Dependencies
    /// - `engine_cell`: Engine cell for Casper operations
    /// - `key_value_block_store`: Block storage
    /// - `rp_conf`: RChain Protocol configuration
    /// - `connections_cell`: P2P connections state
    /// - `node_discovery`: Node discovery service
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        // REPL dependencies
        runtime: RhoRuntimeImpl,
        // Propose dependencies
        trigger_propose_f_opt: Option<Arc<ProposeFunction>>,
        proposer_state_ref_opt: Option<Arc<RwLock<ProposerState>>>,
        // Deploy dependencies
        api_max_blocks_limit: i32,
        dev_mode: bool,
        propose_f_opt: Option<Arc<ProposeFunction>>,
        block_report_api: BlockReportAPI,
        network_id: String,
        shard_id: String,
        min_phlo_price: i64,
        is_node_read_only: bool,
        // Shared dependencies
        engine_cell: EngineCell,
        key_value_block_store: KeyValueBlockStore,
        rp_conf: RPConf,
        connections_cell: ConnectionsCell,
        node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
    ) -> Self {
        // Create REPL service
        let repl = ReplGrpcServiceImpl::new(runtime);

        // Create Propose service
        let propose = ProposeGrpcServiceV1Impl::new(
            trigger_propose_f_opt,
            proposer_state_ref_opt,
            Arc::new(engine_cell.clone()),
        );

        // Create Deploy service
        let deploy = DeployGrpcServiceV1Impl::new(
            api_max_blocks_limit,
            propose_f_opt,
            dev_mode,
            network_id,
            shard_id,
            min_phlo_price,
            is_node_read_only,
            engine_cell,
            block_report_api,
            key_value_block_store,
            rp_conf,
            connections_cell,
            node_discovery,
        );

        // Create LSP service (stateless)
        let lsp = LspGrpcServiceImpl::new();

        Self {
            repl,
            propose,
            deploy,
            lsp,
        }
    }
}
