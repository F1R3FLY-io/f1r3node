use axum::{
    response::{IntoResponse, Response},
    routing::get,
    Router,
};

use std::env;

pub fn get_version_info() -> (&'static str, &'static str) {
    let version = env!("CARGO_PKG_VERSION");
    let git_hash = env!("GIT_HASH_SHORT");
    (version, git_hash)
}

pub fn get_version_info_str() -> String {
    let (version, git_hash) = get_version_info();
    format!("F1r3fly Node {} ({})", version, git_hash)
}

pub struct VersionInfo;

impl VersionInfo {
    pub fn create_router() -> Router {
        Router::new().route("/", get(version_info_handler))
    }
}

#[utoipa::path(
        get,
        path = "/version",
        responses(
            (status = 200, description = "Node version information", body = String),
        ),
        tag = "System"
    )]
pub async fn version_info_handler() -> Response {
    get_version_info_str().into_response()
}
