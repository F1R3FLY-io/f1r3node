use axum::{extract::State, response::Json, routing::get, Router};
use serde::Serialize;
use std::collections::HashSet;
use utoipa::ToSchema;

use crate::rust::web::{
    shared_handlers::{AppError, AppState},
    version_info::get_version_info_str,
};
pub struct StatusInfo;

#[derive(Debug, Serialize, ToSchema)]
pub struct PeerInfo {
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

#[derive(Debug, Serialize, ToSchema)]
pub struct Status {
    pub address: String,
    pub version: String,
    pub peers: i32,
    pub nodes: i32,
    #[serde(rename = "peerList")]
    pub peer_list: Vec<PeerInfo>,
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
    let connections = app_state.connections_cell.read()?;
    let discovered_nodes = app_state.node_discovery.peers()?;

    let peers = connections.len() as i32;
    let nodes = discovered_nodes.len() as i32;

    // Create a set of connected peer IDs for quick lookup
    let connected_ids: HashSet<_> = connections.iter().map(|p| p.id.key.clone()).collect();

    // Convert PeerNode to PeerInfo with connection status
    let peer_list: Vec<PeerInfo> = discovered_nodes
        .iter()
        .map(|node| PeerInfo {
            address: node.to_address(),
            node_id: node.id.to_string(),
            host: node.endpoint.host.clone(),
            protocol_port: node.endpoint.tcp_port as i32,
            discovery_port: node.endpoint.udp_port as i32,
            is_connected: connected_ids.contains(&node.id.key),
        })
        .collect();

    Ok(Json(Status {
        address,
        version: get_version_info_str(),
        peers,
        nodes,
        peer_list,
    }))
}
