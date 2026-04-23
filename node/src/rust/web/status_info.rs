use axum::{extract::State, response::Json, routing::get, Router};

use crate::rust::api::web_api::ApiStatus;
use crate::rust::web::shared_handlers::{AppError, AppState};

pub struct StatusInfo;

impl StatusInfo {
    pub fn create_router() -> Router<AppState> {
        Router::new().route("/", get(status_info_handler))
    }
}

#[utoipa::path(
        get,
        path = "/status",
        responses(
            (status = 200, description = "Node status information", body = ApiStatus),
        ),
        tag = "System"
    )]
pub async fn status_info_handler(
    State(app_state): State<AppState>,
) -> Result<Json<ApiStatus>, AppError> {
    let status = app_state.web_api.status().await?;
    Ok(Json(status))
}
