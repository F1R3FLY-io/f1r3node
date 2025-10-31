//! OpenAPI documentation generation for F1re3fly Web API

use utoipa::OpenApi;

use crate::rust::api::serde_types::block_info::BlockInfoSerde;
use crate::rust::api::serde_types::light_block_info::LightBlockInfoSerde;
use crate::rust::api::web_api::{
    DataAtNameRequest, DataAtNameResponse, DeployRequest, ExploreDeployRequest,
};
use crate::rust::web::{admin_web_api_routes, shared_handlers, status_info};

/// Public API OpenAPI documentation
///
/// This schema includes the following endpoints:
/// - GET /status - Get API status
/// - POST /deploy - Deploy a contract
/// - POST /explore-deploy - Perform exploratory deploy
/// - POST /explore-deploy-by-block-hash - Exploratory deploy at specific block
/// - POST /data-at-name - Listen for data at name
/// - GET /blocks - Get recent blocks
/// - GET /block/{hash} - Get block by hash
#[derive(OpenApi)]
#[openapi(
    paths(
        status_info::status_info_handler,
        shared_handlers::deploy_handler,
        shared_handlers::explore_deploy_handler,
        shared_handlers::explore_deploy_by_block_hash_handler,
        shared_handlers::data_at_name_handler,
        shared_handlers::get_blocks_handler,
        shared_handlers::get_block_handler,
    ),
    components(schemas(
        DeployRequest,
        ExploreDeployRequest,
        DataAtNameRequest,
        DataAtNameResponse,
        LightBlockInfoSerde,
        BlockInfoSerde,
    )),
    info(
        title = "F1r3fly Node API",
        version = "1.0",
        description = "Public API for F1r3fly Node - a Casper blockchain node"
    )
)]
pub struct PublicApi;

/// Admin API OpenAPI documentation
///
/// This schema includes all public endpoints plus:
/// - POST /propose - Propose a new block (admin only)
#[derive(OpenApi)]
#[openapi(
    paths(
        status_info::status_info_handler,
        shared_handlers::deploy_handler,
        shared_handlers::explore_deploy_handler,
        shared_handlers::explore_deploy_by_block_hash_handler,
        shared_handlers::data_at_name_handler,
        shared_handlers::get_blocks_handler,
        shared_handlers::get_block_handler,
        admin_web_api_routes::propose_handler,
    ),
    components(schemas(
        DeployRequest,
        ExploreDeployRequest,
        DataAtNameRequest,
        DataAtNameResponse,
        LightBlockInfoSerde,
        BlockInfoSerde,
    )),
    info(
        title = "F1r3fly Node API (admin)",
        version = "1.0",
        description = "Admin API for F1r3fly Node - includes all public endpoints plus administrative functions"
    )
)]
pub struct AdminApi;
