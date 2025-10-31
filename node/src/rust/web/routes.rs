use axum::{response::Response, routing::get, Router};
use tower_http::cors::{Any, CorsLayer};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::rust::web::{
    admin_web_api_routes::AdminWebApiRoutes,
    events_info,
    reporting_routes::ReportingRoutes,
    shared_handlers::AppState,
    status_info, version_info,
    web_api_docs::{AdminApi, PublicApi},
    web_api_routes::WebApiRoutes,
    web_api_routes_v1::WebApiRoutesV1,
};

pub struct Routes;

impl Routes {
    pub fn create_main_routes(reporting_enabled: bool, app_state: AppState) -> Router<AppState> {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .allow_credentials(false);

        let mut router = Router::new()
            // System routes
            .route("/metrics", get(metrics_handler))
            .route("/version", get(version_info::version_info_handler))
            .route("/status", get(status_info::status_info_handler))
            .route("/ws/events", get(events_info::events_info_handler));

        // Web API routes
        let web_api_routes = WebApiRoutes::create_router();
        let reporting_routes = if reporting_enabled {
            ReportingRoutes::create_router()
        } else {
            Router::<AppState>::new()
        };

        router = router
            .nest("/api", web_api_routes.merge(reporting_routes))
            .nest("/api/v1", WebApiRoutesV1::create_router())
            .merge(
                SwaggerUi::new("/swagger-ui/{_:.*}")
                    .url("/api-doc/openapi.json", PublicApi::openapi()),
            );

        // Legacy reporting routes (if enabled)
        if reporting_enabled {
            router = router.nest("/reporting", ReportingRoutes::create_router());
        }

        router.layer(cors).with_state(app_state)
    }

    pub fn create_admin_routes(app_state: AppState) -> Router<AppState> {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .allow_credentials(false);

        let admin_routes = AdminWebApiRoutes::create_router();
        let reporting_routes = ReportingRoutes::create_router();

        Router::new()
            .nest("/api", admin_routes.merge(reporting_routes))
            .nest("/api/v1", WebApiRoutesV1::create_admin_router())
            .merge(
                SwaggerUi::new("/swagger-ui/{_:.*}")
                    .url("/api-doc/openapi.json", AdminApi::openapi()),
            )
            .layer(cors)
            .with_state(app_state)
    }
}

#[utoipa::path(
        get,
        path = "/metrics",
        responses(
            (status = 200, description = "Prometheus metrics"),
        ),
        tag = "System"
    )]
async fn metrics_handler() -> Response {
    // TODO: Add metrics
    todo!()
}
