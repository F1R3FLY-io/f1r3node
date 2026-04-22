use axum::{
    extract::{Path, Query, State},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::Deserialize;

use crate::rust::{
    api::{
        serde_types::block_info::BlockInfoSerde,
        web_api::{
            DataAtNameByBlockHashRequest, DeployResponse, PrepareRequest, PrepareResponse,
            RhoDataResponse,
        },
    },
    web::{
        shared_handlers::{self, AppError, AppState},
        transaction::TransactionResponse,
    },
};

#[derive(Debug, Deserialize)]
pub struct ViewQuery {
    pub view: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BlockHashQuery {
    pub block_hash: Option<String>,
}

pub struct WebApiRoutes;

impl WebApiRoutes {
    pub fn create_router() -> Router<AppState> {
        Router::new()
            .route("/status", get(shared_handlers::status_handler))
            .route("/prepare-deploy", get(prepare_deploy_get_handler))
            .route("/prepare-deploy", post(prepare_deploy_post_handler))
            .route("/deploy", post(shared_handlers::deploy_handler))
            .route(
                "/explore-deploy",
                post(shared_handlers::explore_deploy_handler),
            )
            .route(
                "/explore-deploy-by-block-hash",
                post(shared_handlers::explore_deploy_by_block_hash_handler),
            )
            .route("/data-at-name", post(shared_handlers::data_at_name_handler))
            .route(
                "/data-at-name-by-block-hash",
                post(data_at_name_by_block_hash_handler),
            )
            .route("/last-finalized-block", get(last_finalized_block_handler))
            .route("/block/{hash}", get(shared_handlers::get_block_handler))
            .route("/blocks", get(shared_handlers::get_blocks_handler))
            .route("/blocks/{start}/{end}", get(get_blocks_by_heights_handler))
            .route("/blocks/{depth}", get(get_blocks_by_depth_handler))
            .route("/deploy/{deploy_id}", get(find_deploy_handler))
            .route("/is-finalized/{hash}", get(is_finalized_handler))
            .route("/balance/{address}", get(balance_handler))
            .route("/registry/{uri}", get(registry_handler))
            .route("/validators", get(validators_handler))
            .route("/epoch", get(epoch_handler))
            .route("/transactions/{hash}", get(get_transaction_handler))
    }
}

#[utoipa::path(
    get,
    path = "/api/prepare-deploy",
    responses(
        (status = 200, description = "Prepare deploy response", body = PrepareResponse),
        (status = 400, description = "Bad request or internal error")
    ),
    tag = "WebAPI"
)]
pub async fn prepare_deploy_get_handler(State(app_state): State<AppState>) -> Response {
    match app_state.web_api.prepare_deploy(None).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/api/prepare-deploy",
    request_body = PrepareRequest,
    responses(
        (status = 200, description = "Prepare deploy response", body = PrepareResponse),
        (status = 400, description = "Bad request or internal error")
    ),
    tag = "WebAPI"
)]
pub async fn prepare_deploy_post_handler(
    State(app_state): State<AppState>,
    Json(request): Json<PrepareRequest>,
) -> Response {
    match app_state.web_api.prepare_deploy(Some(request)).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/api/data-at-name-by-block-hash",
    request_body = DataAtNameByBlockHashRequest,
    responses(
        (status = 200, description = "Data at name response", body = RhoDataResponse),
        (status = 400, description = "Bad request or invalid parameters"),

    ),
    tag = "WebAPI"
)]
pub async fn data_at_name_by_block_hash_handler(
    State(app_state): State<AppState>,
    Json(request): Json<DataAtNameByBlockHashRequest>,
) -> Response {
    match app_state.web_api.get_data_at_par(request).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/last-finalized-block",
    params(
        ("view" = Option<String>, Query, description = "Response view: 'full' (default) includes deploys, 'summary' block header only"),
    ),
    responses(
        (status = 200, description = "Last finalized block", body = BlockInfoSerde),
        (status = 400, description = "Bad request or block not found")
    ),
    tag = "WebAPI"
)]
pub async fn last_finalized_block_handler(
    State(app_state): State<AppState>,
    Query(query): Query<ViewQuery>,
) -> Response {
    use crate::rust::api::web_api::ViewMode;

    let view = match query.view.as_deref() {
        Some("summary") => ViewMode::Summary,
        _ => ViewMode::Full,
    };
    match app_state.web_api.last_finalized_block(view).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/blocks/{start}/{end}",
    params(
        ("start" = i64, Path, description = "Start block height"),
        ("end" = i64, Path, description = "End block height"),
        ("view" = Option<String>, Query, description = "Response view: 'summary' (default) block headers only, 'full' includes deploys"),
    ),
    responses(
        (status = 200, description = "Blocks by height range", body = Vec<BlockInfoSerde>),
        (status = 400, description = "Bad request or invalid height range")
    ),
    tag = "WebAPI"
)]
pub async fn get_blocks_by_heights_handler(
    State(app_state): State<AppState>,
    Path((start, end)): Path<(i64, i64)>,
    Query(query): Query<ViewQuery>,
) -> Response {
    use crate::rust::api::web_api::ViewMode;

    let view = match query.view.as_deref() {
        Some("full") => ViewMode::Full,
        _ => ViewMode::Summary,
    };
    match app_state.web_api.get_blocks_by_heights(start, end, view).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/blocks/{depth}",
    params(
        ("depth" = i32, Path, description = "Block depth"),
        ("view" = Option<String>, Query, description = "Response view: 'summary' (default) block headers only, 'full' includes deploys"),
    ),
    responses(
        (status = 200, description = "Blocks by depth", body = Vec<BlockInfoSerde>),
        (status = 400, description = "Bad request or invalid depth")
    ),
    tag = "WebAPI"
)]
pub async fn get_blocks_by_depth_handler(
    State(app_state): State<AppState>,
    Path(depth): Path<i32>,
    Query(query): Query<ViewQuery>,
) -> Response {
    use crate::rust::api::web_api::ViewMode;

    let view = match query.view.as_deref() {
        Some("full") => ViewMode::Full,
        _ => ViewMode::Summary,
    };
    match app_state.web_api.get_blocks(depth, view).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/deploy/{deploy_id}",
    params(
        ("deploy_id" = String, Path, description = "Deploy ID"),
        ("view" = Option<String>, Query, description = "Response view: 'full' (default) returns all fields, 'summary' returns core fields only"),
    ),
    responses(
        (status = 200, description = "Deploy information", body = DeployResponse),
        (status = 404, description = "Deploy not found"),
        (status = 400, description = "Bad request")
    ),
    tag = "WebAPI"
)]
pub async fn find_deploy_handler(
    State(app_state): State<AppState>,
    Path(deploy_id): Path<String>,
    Query(query): Query<ViewQuery>,
) -> Response {
    use crate::rust::api::web_api::ViewMode;

    let view = match query.view.as_deref() {
        Some("summary") => ViewMode::Summary,
        _ => ViewMode::Full,
    };

    match app_state.web_api.find_deploy(deploy_id, view).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => {
            if e.downcast_ref::<casper::rust::api::block_api::DeployNotFoundError>().is_some() {
                (axum::http::StatusCode::NOT_FOUND, format!("{}", e)).into_response()
            } else {
                AppError(e).into_response()
            }
        }
    }
}

#[utoipa::path(
    get,
    path = "/api/is-finalized/{hash}",
    params(
        ("hash" = String, Path, description = "Block hash"),
    ),
    responses(
        (status = 200, description = "Finalization status", body = bool),
        (status = 400, description = "Bad request or invalid hash")
    ),
    tag = "WebAPI"
)]
pub async fn is_finalized_handler(
    State(app_state): State<AppState>,
    Path(hash): Path<String>,
) -> Response {
    match app_state.web_api.is_finalized(hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

use crate::rust::api::web_api::{BalanceResponse, RegistryResponse, ValidatorsResponse, EpochResponse};

#[utoipa::path(
    get,
    path = "/api/balance/{address}",
    params(
        ("address" = String, Path, description = "Wallet address (hex public key)"),
        ("block_hash" = Option<String>, Query, description = "Block hash to query against (defaults to LFB)"),
    ),
    responses(
        (status = 200, description = "Balance for address", body = BalanceResponse),
        (status = 400, description = "Bad request or address not found")
    ),
    tag = "Query"
)]
pub async fn balance_handler(
    State(app_state): State<AppState>,
    Path(address): Path<String>,
    Query(query): Query<BlockHashQuery>,
) -> Response {
    match app_state.web_api.get_balance(address, query.block_hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/registry/{uri}",
    params(
        ("uri" = String, Path, description = "Registry URI (e.g. rho:id:...)"),
        ("block_hash" = Option<String>, Query, description = "Block hash to query against (defaults to LFB)"),
    ),
    responses(
        (status = 200, description = "Registry entry", body = RegistryResponse),
        (status = 400, description = "Bad request or URI not found")
    ),
    tag = "Query"
)]
pub async fn registry_handler(
    State(app_state): State<AppState>,
    Path(uri): Path<String>,
    Query(query): Query<BlockHashQuery>,
) -> Response {
    match app_state.web_api.get_registry(uri, query.block_hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/validators",
    params(
        ("block_hash" = Option<String>, Query, description = "Block hash to query against (defaults to LFB)"),
    ),
    responses(
        (status = 200, description = "Active validator set", body = ValidatorsResponse),
        (status = 400, description = "Bad request")
    ),
    tag = "Query"
)]
pub async fn validators_handler(
    State(app_state): State<AppState>,
    Query(query): Query<BlockHashQuery>,
) -> Response {
    match app_state.web_api.get_validators(query.block_hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/epoch",
    params(
        ("block_hash" = Option<String>, Query, description = "Block hash to query against (defaults to LFB)"),
    ),
    responses(
        (status = 200, description = "Current epoch info", body = EpochResponse),
        (status = 400, description = "Bad request")
    ),
    tag = "Query"
)]
pub async fn epoch_handler(
    State(app_state): State<AppState>,
    Query(query): Query<BlockHashQuery>,
) -> Response {
    match app_state.web_api.get_epoch(query.block_hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/transactions/{hash}",
    params(
        ("hash" = String, Path, description = "Transaction hash"),
    ),
    responses(
        (status = 200, description = "Transaction information", body = TransactionResponse),
        (status = 400, description = "Bad request or transaction not found")
    ),
    tag = "WebAPI"
)]
pub async fn get_transaction_handler(
    State(app_state): State<AppState>,
    Path(hash): Path<String>,
) -> Response {
    match app_state.web_api.get_transaction(hash).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::StatusCode;
    use std::sync::Arc;
    use tower::ServiceExt;

    use crate::rust::api::web_api::{
        ApiStatus, DataAtNameByBlockHashRequest, DataAtNameRequest, DataAtNameResponse,
        DeployRequest, DeployResponse, ViewMode, RhoDataResponse, WebApi,
    };
    use crate::rust::web::transaction::TransactionResponse;

    /// Stub WebApi that returns sample DeployResponse for testing.
    struct StubWebApi;

    fn sample_deploy_response(view: ViewMode) -> DeployResponse {
        let is_full = view == ViewMode::Full;
        DeployResponse {
            deploy_id: "abc123def".to_string(),
            block_hash: "7bf8abc123".to_string(),
            block_number: 52331,
            timestamp: 1770028092477,
            cost: 100,
            errored: false,
            is_finalized: true,
            deployer: if is_full { Some("0487def456".to_string()) } else { None },
            term: if is_full { Some("new ret in { ret!(42) }".to_string()) } else { None },
            system_deploy_error: if is_full { Some(String::new()) } else { None },
            phlo_price: if is_full { Some(10) } else { None },
            phlo_limit: if is_full { Some(100000) } else { None },
            sig_algorithm: if is_full { Some("secp256k1".to_string()) } else { None },
            valid_after_block_number: if is_full { Some(0) } else { None },
            transfers: if is_full { Some(vec![]) } else { None },
        }
    }

    #[async_trait::async_trait]
    impl WebApi for StubWebApi {
        async fn status(&self) -> eyre::Result<ApiStatus> {
            unimplemented!()
        }
        async fn prepare_deploy(
            &self,
            _: Option<crate::rust::api::web_api::PrepareRequest>,
        ) -> eyre::Result<crate::rust::api::web_api::PrepareResponse> {
            unimplemented!()
        }
        async fn deploy(&self, _: DeployRequest) -> eyre::Result<String> {
            unimplemented!()
        }
        async fn listen_for_data_at_name(
            &self,
            _: DataAtNameRequest,
        ) -> eyre::Result<DataAtNameResponse> {
            unimplemented!()
        }
        async fn get_data_at_par(
            &self,
            _: DataAtNameByBlockHashRequest,
        ) -> eyre::Result<RhoDataResponse> {
            unimplemented!()
        }
        async fn last_finalized_block(
            &self,
            _: ViewMode,
        ) -> eyre::Result<crate::rust::api::serde_types::block_info::BlockInfoSerde> {
            unimplemented!()
        }
        async fn get_block(
            &self,
            _: String,
            _: ViewMode,
        ) -> eyre::Result<crate::rust::api::serde_types::block_info::BlockInfoSerde> {
            unimplemented!()
        }
        async fn get_blocks(
            &self,
            _: i32,
            _: ViewMode,
        ) -> eyre::Result<Vec<crate::rust::api::serde_types::block_info::BlockInfoSerde>> {
            unimplemented!()
        }
        async fn find_deploy(&self, _: String, view: ViewMode) -> eyre::Result<DeployResponse> {
            Ok(sample_deploy_response(view))
        }
        async fn exploratory_deploy(
            &self,
            _: String,
            _: Option<String>,
            _: bool,
        ) -> eyre::Result<RhoDataResponse> {
            unimplemented!()
        }
        async fn get_blocks_by_heights(
            &self,
            _: i64,
            _: i64,
            _: ViewMode,
        ) -> eyre::Result<Vec<crate::rust::api::serde_types::block_info::BlockInfoSerde>> {
            unimplemented!()
        }
        async fn is_finalized(&self, _: String) -> eyre::Result<bool> {
            unimplemented!()
        }
        async fn get_balance(&self, _: String, _: Option<String>) -> eyre::Result<crate::rust::api::web_api::BalanceResponse> {
            unimplemented!()
        }
        async fn get_registry(&self, _: String, _: Option<String>) -> eyre::Result<crate::rust::api::web_api::RegistryResponse> {
            unimplemented!()
        }
        async fn get_validators(&self, _: Option<String>) -> eyre::Result<crate::rust::api::web_api::ValidatorsResponse> {
            unimplemented!()
        }
        async fn get_epoch(&self, _: Option<String>) -> eyre::Result<crate::rust::api::web_api::EpochResponse> {
            unimplemented!()
        }
        async fn get_transaction(&self, _: String) -> eyre::Result<TransactionResponse> {
            unimplemented!()
        }
    }

    async fn test_find_deploy_handler(
        State(web_api): State<Arc<dyn WebApi + Send + Sync>>,
        Path(deploy_id): Path<String>,
        Query(query): Query<ViewQuery>,
    ) -> Response {
        let view = match query.view.as_deref() {
            Some("summary") => ViewMode::Summary,
            _ => ViewMode::Full,
        };
        match web_api.find_deploy(deploy_id, view).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => AppError(e).into_response(),
        }
    }

    fn test_router() -> Router {
        let web_api: Arc<dyn WebApi + Send + Sync> = Arc::new(StubWebApi);
        Router::new()
            .route("/deploy/{deploy_id}", get(test_find_deploy_handler))
            .with_state(web_api)
    }

    async fn body_to_string(body: Body) -> String {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_find_deploy_returns_full_response_by_default() {
        let app = test_router();

        let request: axum::http::Request<Body> = axum::http::Request::builder()
            .uri("/deploy/abc123def")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = body_to_string(response.into_body()).await;
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();

        // Core fields always present
        assert_eq!(json["deployId"], "abc123def");
        assert_eq!(json["blockHash"], "7bf8abc123");
        assert_eq!(json["blockNumber"], 52331);
        assert_eq!(json["timestamp"], 1770028092477i64);
        assert_eq!(json["cost"], 100);
        assert_eq!(json["errored"], false);
        assert_eq!(json["isFinalized"], true);

        // Full view includes deploy execution details
        assert_eq!(json["deployer"], "0487def456");
        assert!(json.get("term").is_some());
        assert!(json.get("phloPrice").is_some());
        assert!(json.get("phloLimit").is_some());
        assert!(json.get("sigAlgorithm").is_some());
        assert!(json.get("transfers").is_some());
    }

    #[tokio::test]
    async fn test_find_deploy_returns_summary_response() {
        let app = test_router();

        let request: axum::http::Request<Body> = axum::http::Request::builder()
            .uri("/deploy/abc123def?view=summary")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = body_to_string(response.into_body()).await;
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();

        // Core fields present
        assert_eq!(json["deployId"], "abc123def");
        assert_eq!(json["blockHash"], "7bf8abc123");
        assert_eq!(json["blockNumber"], 52331);
        assert_eq!(json["cost"], 100);
        assert_eq!(json["isFinalized"], true);

        // Full-only fields omitted
        assert!(json.get("deployer").is_none());
        assert!(json.get("term").is_none());
        assert!(json.get("phloPrice").is_none());
        assert!(json.get("phloLimit").is_none());
        assert!(json.get("sigAlgorithm").is_none());
        assert!(json.get("transfers").is_none());
    }

    #[tokio::test]
    async fn test_find_deploy_unknown_view_defaults_to_full() {
        let app = test_router();

        let request: axum::http::Request<Body> = axum::http::Request::builder()
            .uri("/deploy/abc123def?view=unknown")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = body_to_string(response.into_body()).await;
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();

        // Unknown view falls back to full
        assert!(json.get("deployer").is_some());
        assert!(json.get("term").is_some());
    }
}
