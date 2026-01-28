// See rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala
// Ported from Scala PR #140
//
// Uses enum-based dispatch instead of trait objects for async compatibility.

use super::grpc_client_service::GrpcClientService;
use super::openai_service::{create_noop_openai_service, create_openai_service, OpenAIConfig, SharedOpenAIService};

/// ExternalServices configuration and instances
/// Uses enum to distinguish between node types
#[derive(Clone)]
pub struct ExternalServices {
    pub openai: SharedOpenAIService,
    pub grpc_client: GrpcClientService,
    pub openai_enabled: bool,
    pub is_validator: bool,
}

impl ExternalServices {
    /// Create external services for a validator node
    pub fn for_validator(config: &OpenAIConfig) -> Self {
        Self {
            openai: create_openai_service(config),
            grpc_client: GrpcClientService::new_real(),
            openai_enabled: config.enabled,
            is_validator: true,
        }
    }

    /// Create external services for an observer node
    /// Observers have OpenAI and GrpcTell disabled for security
    pub fn for_observer() -> Self {
        Self {
            openai: create_noop_openai_service(),
            grpc_client: GrpcClientService::new_noop(),
            openai_enabled: false,
            is_validator: false,
        }
    }

    /// Create NoOp external services (all services disabled)
    /// Useful for testing
    pub fn noop() -> Self {
        Self {
            openai: create_noop_openai_service(),
            grpc_client: GrpcClientService::new_noop(),
            openai_enabled: false,
            is_validator: false,
        }
    }

    /// Factory function to create external services based on node type
    /// Matches Scala object ExternalServices.forNodeType
    pub fn for_node_type(is_validator: bool, config: &OpenAIConfig) -> Self {
        if is_validator {
            Self::for_validator(config)
        } else {
            Self::for_observer()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_services() {
        let services = ExternalServices::noop();
        assert!(!services.openai_enabled);
        assert!(!services.is_validator);
    }

    #[test]
    fn test_observer_services() {
        let services = ExternalServices::for_observer();
        assert!(!services.openai_enabled);
        assert!(!services.is_validator);
    }

    #[test]
    fn test_for_node_type_observer() {
        let config = OpenAIConfig::disabled();
        let services = ExternalServices::for_node_type(false, &config);
        assert!(!services.is_validator);
        assert!(!services.openai_enabled);
    }

    #[test]
    fn test_for_node_type_validator_disabled() {
        let config = OpenAIConfig::disabled();
        let services = ExternalServices::for_node_type(true, &config);
        assert!(services.is_validator);
        assert!(!services.openai_enabled);
    }
}
