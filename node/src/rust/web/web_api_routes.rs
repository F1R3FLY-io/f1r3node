use axum::{
    extract::{Path, State},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};

use crate::rust::{
    api::{
        serde_types::{block_info::BlockInfoSerde, light_block_info::LightBlockInfoSerde},
        web_api::{DataAtNameByBlockHashRequest, PrepareRequest, PrepareResponse, RhoDataResponse},
    },
    web::{
        shared_handlers::{self, AppError, AppState},
        transaction::TransactionResponse,
    },
};

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
    responses(
        (status = 200, description = "Last finalized block", body = BlockInfoSerde),
        (status = 400, description = "Bad request or block not found")
    ),
    tag = "WebAPI"
)]
pub async fn last_finalized_block_handler(State(app_state): State<AppState>) -> Response {
    match app_state.web_api.last_finalized_block().await {
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
    ),
    responses(
        (status = 200, description = "Blocks by height range", body = Vec<LightBlockInfoSerde>),
        (status = 400, description = "Bad request or invalid height range")
    ),
    tag = "WebAPI"
)]
pub async fn get_blocks_by_heights_handler(
    State(app_state): State<AppState>,
    Path((start, end)): Path<(i64, i64)>,
) -> Response {
    match app_state.web_api.get_blocks_by_heights(start, end).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/blocks/{depth}",
    params(
        ("depth" = i32, Path, description = "Block depth"),
    ),
    responses(
        (status = 200, description = "Blocks by depth", body = Vec<LightBlockInfoSerde>),
        (status = 400, description = "Bad request or invalid depth")
    ),
    tag = "WebAPI"
)]
pub async fn get_blocks_by_depth_handler(
    State(app_state): State<AppState>,
    Path(depth): Path<i32>,
) -> Response {
    match app_state.web_api.get_blocks(depth).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/api/deploy/{deploy_id}",
    params(
        ("deploy_id" = String, Path, description = "Deploy ID"),
    ),
    responses(
        (status = 200, description = "Deploy information", body = LightBlockInfoSerde),
        (status = 400, description = "Bad request or deploy not found")
    ),
    tag = "WebAPI"
)]
pub async fn find_deploy_handler(
    State(app_state): State<AppState>,
    Path(deploy_id): Path<String>,
) -> Response {
    match app_state.web_api.find_deploy(deploy_id).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => AppError(e).into_response(),
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
