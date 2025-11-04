use axum::{
    routing::{get, post},
    Router,
};

use crate::rust::web::{
    admin_web_api_routes::AdminWebApiRoutes, shared_handlers, shared_handlers::AppState,
};

pub struct WebApiRoutesV1;

impl WebApiRoutesV1 {
    pub fn create_router() -> Router<AppState> {
        Router::new()
            .route("/status", get(shared_handlers::status_handler))
            .route("/deploy", post(shared_handlers::deploy_handler))
            .route(
                "/explore-deploy",
                post(shared_handlers::explore_deploy_handler),
            )
            .route("/blocks", get(shared_handlers::get_blocks_handler))
            .route("/block", get(shared_handlers::get_block_handler))
    }

    pub fn create_admin_router() -> Router<AppState> {
        Self::create_router().merge(AdminWebApiRoutes::create_router())
    }
}
