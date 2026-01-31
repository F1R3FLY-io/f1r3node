// See rholang/src/main/scala/coop/rchain/rholang/externalservices/GrpcClient.scala
// Ported from Scala PR #140
//
// Uses enum-based dispatch instead of trait objects for async compatibility.

use models::rust::rholang::grpc_client::{GrpcClient, GrpcClientError};

/// GrpcClientService using enum dispatch for async compatibility
#[derive(Clone)]
pub enum GrpcClientService {
    /// Real implementation that makes gRPC calls
    Real,
    /// NoOp implementation for observer nodes
    NoOp,
}

impl GrpcClientService {
    pub fn new_real() -> Self {
        tracing::debug!("RealGrpcClientService created");
        Self::Real
    }

    pub fn new_noop() -> Self {
        tracing::debug!("NoOpGrpcClientService created - gRPC calls are disabled");
        Self::NoOp
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Real)
    }

    pub async fn tell(
        &self,
        client_host: &str,
        client_port: u64,
        notification_payload: &str,
    ) -> Result<(), GrpcClientError> {
        match self {
            Self::Real => {
                GrpcClient::init_client_and_tell(client_host, client_port, notification_payload)
                    .await
            }
            Self::NoOp => {
                tracing::debug!(
                    "GrpcClientService is disabled - tell request ignored: host={}, port={}, payload={}",
                    client_host,
                    client_port,
                    notification_payload
                );
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_service_is_disabled() {
        let service = GrpcClientService::new_noop();
        assert!(!service.is_enabled());
    }

    #[test]
    fn test_real_service_is_enabled() {
        let service = GrpcClientService::new_real();
        assert!(service.is_enabled());
    }

    #[tokio::test]
    async fn test_noop_service_returns_ok() {
        let service = GrpcClientService::new_noop();
        let result = service.tell("http://localhost", 8080, "test").await;
        assert!(result.is_ok());
    }
}
