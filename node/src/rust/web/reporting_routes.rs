use axum::{
    extract::{Query, State},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::rust::{
    api::serde_types::block_event_info::BlockEventInfoSerde, web::shared_handlers::AppState,
};

pub struct ReportingRoutes;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum ReportResponse {
    #[serde(rename = "block-traces-report")]
    BlockTracesReport { report: BlockEventInfoSerde },
    #[serde(rename = "block-report-error")]
    BlockReportError { error_message: String },
}

#[derive(Debug, Deserialize)]
pub struct TraceQuery {
    #[serde(rename = "blockHash")]
    pub block_hash: String,
    #[serde(rename = "forceReplay")]
    pub force_replay: Option<bool>,
}

impl ReportingRoutes {
    pub fn create_router() -> Router<AppState> {
        Router::new().route("/trace", get(trace_handler))
    }
}

#[utoipa::path(
        get,
        path = "/reporting/trace",
        params(
            ("blockHash" = String, Query, description = "Block hash"),
            ("forceReplay" = Option<bool>, Query, description = "Force replay"),
        ),
        responses(
            (status = 200, description = "Block trace report", body = ReportResponse),
            (status = 400, description = "Invalid parameters"),
        ),
        tag = "Reporting"
    )]
async fn trace_handler(
    State(app_state): State<AppState>,
    Query(params): Query<TraceQuery>,
) -> Response {
    // Validate block hash parameter - equivalent to Scala's BlockHashParam validation
    if params.block_hash.is_empty() {
        let error_response = ReportResponse::BlockReportError {
            error_message: "block_hash parameter is required".to_string(),
        };
        return Json(error_response).into_response();
    }

    let force_replay = params.force_replay.unwrap_or(false);

    let result = app_state
        .block_report_api
        .block_report(
            Blake2b256Hash::from_hex(&params.block_hash).to_bytes_prost(),
            force_replay,
        )
        .await;

    let response = match result {
        Ok(block_event_info) => ReportResponse::BlockTracesReport {
            report: block_event_info.into(),
        },
        Err(error) => ReportResponse::BlockReportError {
            error_message: error.to_string(),
        },
    };

    Json(response).into_response()
}
