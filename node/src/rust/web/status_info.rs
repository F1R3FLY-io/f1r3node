use axum::{extract::State, response::Json, routing::get, Router};
use serde::Serialize;
use utoipa::ToSchema;

use crate::rust::web::{
    shared_handlers::{AppError, AppState},
    version_info::get_version_info_str,
};
pub struct StatusInfo;

#[derive(Debug, Serialize, ToSchema)]
pub struct Status {
    pub address: String,
    pub version: String,
    pub peers: i32,
    pub nodes: i32,
}

impl StatusInfo {
    pub fn create_router() -> Router<AppState> {
        Router::new().route("/", get(status_info_handler))
    }
}

#[utoipa::path(
        get,
        path = "/status",
        responses(
            (status = 200, description = "Node status information", body = Status),
        ),
        tag = "System"
    )]
pub async fn status_info_handler(
    State(app_state): State<AppState>,
) -> Result<Json<Status>, AppError> {
    let rp_conf = app_state.rp_conf_cell.read()?;
    let address = rp_conf.local.to_address();
    let peers = app_state.connections_cell.read()?.len() as i32;
    let nodes = app_state.node_discovery.peers()?.len() as i32;

    Ok(Json(Status {
        address,
        version: get_version_info_str(),
        peers,
        nodes,
    }))
}
