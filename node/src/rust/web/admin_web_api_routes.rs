use crate::rust::web::shared_handlers::{AppError, AppState};
use axum::{extract::State, routing::post, Router};

pub struct AdminWebApiRoutes;

impl AdminWebApiRoutes {
    /// Creates the admin Web API router
    pub fn create_router() -> Router<AppState> {
        Router::new().route("/propose", post(propose_handler))
    }
}

#[utoipa::path(
        post,
        path = "/api/propose",
        responses(
            (status = 200, description = "Block proposed successfully", body = String),
            (status = 403, description = "Read-only node or not authorized"),
        ),
        tag = "AdminAPI"
    )]
pub async fn propose_handler(State(app_state): State<AppState>) -> Result<String, AppError> {
    let result = app_state.admin_web_api.propose().await?;
    Ok(result)
}
