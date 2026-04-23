//! OpenAPI documentation generation for F1re3fly Web API

use utoipa::OpenApi;

use crate::rust::web::{admin_web_api_routes, shared_handlers, status_info};

/// Public API OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        status_info::status_info_handler,
        shared_handlers::deploy_handler,
        shared_handlers::explore_deploy_handler,
        shared_handlers::explore_deploy_by_block_hash_handler,
        shared_handlers::get_blocks_handler,
        shared_handlers::get_block_handler,
    ),
    info(
        title = "F1r3fly Node API",
        version = "1.0",
        description = "Public API for F1r3fly Node - a Casper blockchain node"
    )
)]
pub struct PublicApi;

/// Admin API OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        status_info::status_info_handler,
        shared_handlers::deploy_handler,
        shared_handlers::explore_deploy_handler,
        shared_handlers::explore_deploy_by_block_hash_handler,
        shared_handlers::get_blocks_handler,
        shared_handlers::get_block_handler,
        admin_web_api_routes::propose_handler,
    ),
    info(
        title = "F1r3fly Node API (admin)",
        version = "1.0",
        description = "Admin API for F1r3fly Node - includes all public endpoints plus administrative functions"
    )
)]
pub struct AdminApi;
