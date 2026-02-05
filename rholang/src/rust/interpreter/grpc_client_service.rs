// See rholang/src/main/scala/coop/rchain/rholang/externalservices/GrpcClient.scala
// Ported from Scala PR #140
//
// Uses enum-based dispatch instead of trait objects for async compatibility.

use models::rust::rholang::grpc_client::{GrpcClient, GrpcClientError};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

/// Mock configuration for GrpcClient service
#[derive(Clone)]
pub struct GrpcClientMockConfig {
    /// Expected host for the mock to succeed
    pub expected_host: String,
    /// Expected port for the mock to succeed
    pub expected_port: u64,
    /// Track if the service was called
    was_called: Arc<AtomicBool>,
}

impl GrpcClientMockConfig {
    /// Create mock that expects a specific host and port
    pub fn create(expected_host: &str, expected_port: u64) -> Self {
        Self {
            expected_host: expected_host.to_string(),
            expected_port,
            was_called: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if the service was called
    pub fn was_called(&self) -> bool {
        self.was_called.load(Ordering::SeqCst)
    }
}

/// GrpcClientService using enum dispatch for async compatibility
#[derive(Clone)]
pub enum GrpcClientService {
    /// Real implementation that makes gRPC calls
    Real,
    /// NoOp implementation for observer nodes
    NoOp,
    /// Mock implementation for testing
    Mock(GrpcClientMockConfig),
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

    pub fn new_mock(config: GrpcClientMockConfig) -> Self {
        tracing::debug!("MockGrpcClientService created");
        Self::Mock(config)
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Real | Self::Mock(_))
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
            Self::Mock(config) => {
                config.was_called.store(true, Ordering::SeqCst);
                if client_host == config.expected_host && client_port == config.expected_port {
                    Ok(())
                } else {
                    Err(GrpcClientError::ConnectionError(format!(
                        "Mock connection error: expected {}:{} but got {}:{}",
                        config.expected_host, config.expected_port, client_host, client_port
                    )))
                }
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
