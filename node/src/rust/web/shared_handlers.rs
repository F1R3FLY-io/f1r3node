use std::sync::Arc;

use crate::rust::api::{
    admin_web_api::AdminWebApi,
    web_api::{DataAtNameRequest, DeployRequest, ExploreDeployRequest, WebApi},
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use casper::rust::api::block_report_api::BlockReportAPI;
use comm::rust::{
    discovery::node_discovery::NodeDiscovery,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
};
use shared::rust::shared::f1r3fly_events::EventStream;

#[derive(Clone)]
pub struct AppState {
    pub admin_web_api: Arc<dyn AdminWebApi + Send + Sync + 'static>,
    pub web_api: Arc<dyn WebApi + Send + Sync + 'static>,
    pub block_report_api: Arc<BlockReportAPI>,
    pub rp_conf: Arc<RPConf>,
    pub connections_cell: Arc<ConnectionsCell>,
    pub node_discovery: Arc<dyn NodeDiscovery + Send + Sync + 'static>,
    pub event_stream: Arc<EventStream>,
}

impl AppState {
    pub fn new(
        admin_web_api: Arc<dyn AdminWebApi + Send + Sync + 'static>,
        web_api: Arc<dyn WebApi + Send + Sync + 'static>,
        block_report_api: Arc<BlockReportAPI>,
        rp_conf: Arc<RPConf>,
        connections_cell: Arc<ConnectionsCell>,
        node_discovery: Arc<dyn NodeDiscovery + Send + Sync + 'static>,
        event_consumer: Arc<EventStream>,
    ) -> Self {
        Self {
            admin_web_api,
            web_api,
            block_report_api,
            rp_conf,
            connections_cell,
            node_discovery,
            event_stream: event_consumer,
        }
    }
}

pub struct AppError(pub eyre::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<eyre::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[utoipa::path(
    get,
    path = "/status",
    responses(
        (status = 200, description = "API status information"),
        (status = 400, description = "Bad request or internal error"),
    ),
    tag = "Status"
)]
pub async fn status_handler(State(app_state): State<AppState>) -> Response {
    match app_state.web_api.status().await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/deploy",
    request_body = DeployRequest,
    responses(
        (status = 200, description = "Deploy successful"),
        (status = 400, description = "Invalid deploy request or signature error"),
    ),
    tag = "Deployment"
)]
pub async fn deploy_handler(
    State(app_state): State<AppState>,
    Json(request): Json<DeployRequest>,
) -> Response {
    match app_state.web_api.deploy(request).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/explore-deploy",
    request_body(content = String, content_type = "application/json"),
    responses(
        (status = 200, description = "Exploratory deploy successful"),
        (status = 400, description = "Invalid term or execution error"),
    ),
    tag = "Deployment"
)]
pub async fn explore_deploy_handler(
    State(app_state): State<AppState>,
    Json(term): Json<String>,
) -> Response {
    match app_state
        .web_api
        .exploratory_deploy(term, None, false)
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/explore-deploy-by-block-hash",
    request_body = ExploreDeployRequest,
    responses(
        (status = 200, description = "Exploratory deploy successful"),
        (status = 400, description = "Invalid term, block hash, or execution error"),
    ),
    tag = "Deployment"
)]
pub async fn explore_deploy_by_block_hash_handler(
    State(app_state): State<AppState>,
    Json(request): Json<ExploreDeployRequest>,
) -> Response {
    match app_state
        .web_api
        .exploratory_deploy(request.term, Some(request.block_hash), false)
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/data-at-name",
    request_body = DataAtNameRequest,
    responses(
        (status = 200, description = "Data retrieval successful"),
        (status = 400, description = "Invalid name or depth parameter"),
    ),
    tag = "Data"
)]
pub async fn data_at_name_handler(
    State(app_state): State<AppState>,
    Json(request): Json<DataAtNameRequest>,
) -> Response {
    match app_state.web_api.listen_for_data_at_name(request).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/blocks",
    responses(
        (status = 200, description = "Blocks retrieved successfully"),
        (status = 400, description = "Error retrieving blocks"),
    ),
    tag = "Blocks"
)]
pub async fn get_blocks_handler(State(app_state): State<AppState>) -> Response {
    match app_state.web_api.get_blocks(10).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/block/{hash}",
    params(
        ("hash" = String, Path, description = "Block hash in hex format")
    ),
    responses(
        (status = 200, description = "Block information retrieved successfully"),
        (status = 400, description = "Block not found or invalid hash"),
    ),
    tag = "Blocks"
)]
pub async fn get_block_handler(
    State(app_state): State<AppState>,
    Path(hash): Path<String>,
) -> Response {
    match app_state.web_api.get_block(hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}
