// See rholang/src/main/scala/coop/rchain/rholang/externalservices/ExternalServices.scala
// Ported from Scala PR #140
//
// Uses enum-based dispatch instead of trait objects for async compatibility.

use super::grpc_client_service::GrpcClientService;
use super::ollama_service::{create_noop_ollama_service, create_ollama_service, OllamaConfig, SharedOllamaService};
use super::openai_service::{create_noop_openai_service, create_openai_service, OpenAIConfig, SharedOpenAIService};

/// ExternalServices configuration and instances
/// Uses enum to distinguish between node types
#[derive(Clone)]
pub struct ExternalServices {
    pub openai: SharedOpenAIService,
    pub ollama: SharedOllamaService,
    pub grpc_client: GrpcClientService,
    pub openai_enabled: bool,
    pub ollama_enabled: bool,
    pub is_validator: bool,
}

impl ExternalServices {
    /// Create external services for a validator node
    pub fn for_validator(openai_config: &OpenAIConfig, ollama_config: &OllamaConfig) -> Self {
        Self {
            openai: create_openai_service(openai_config),
            ollama: create_ollama_service(ollama_config),
            grpc_client: GrpcClientService::new_real(),
            openai_enabled: openai_config.enabled,
            ollama_enabled: ollama_config.enabled,
            is_validator: true,
        }
    }

    /// Create external services for an observer node
    /// Observers have OpenAI, Ollama and GrpcTell disabled for security
    pub fn for_observer() -> Self {
        Self {
            openai: create_noop_openai_service(),
            ollama: create_noop_ollama_service(),
            grpc_client: GrpcClientService::new_noop(),
            openai_enabled: false,
            ollama_enabled: false,
            is_validator: false,
        }
    }


    /// Create NoOp external services (all services disabled)
    /// Useful for testing
    pub fn noop() -> Self {
        Self {
            openai: create_noop_openai_service(),
            ollama: create_noop_ollama_service(),
            grpc_client: GrpcClientService::new_noop(),
            openai_enabled: false,
            ollama_enabled: false,
            is_validator: false,
        }
    }

    /// Factory function to create external services based on node type
    /// Matches Scala object ExternalServices.forNodeType
    /// Factory function to create external services based on node type
    /// Matches Scala object ExternalServices.forNodeType
    pub fn for_node_type(is_validator: bool, openai_config: &OpenAIConfig, ollama_config: &OllamaConfig) -> Self {
        if is_validator {
            Self::for_validator(openai_config, ollama_config)
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
        assert!(!services.ollama_enabled);
        assert!(!services.is_validator);
    }

    #[test]
    fn test_observer_services() {
        let services = ExternalServices::for_observer();
        assert!(!services.openai_enabled);
        assert!(!services.ollama_enabled);
        assert!(!services.is_validator);
    }

    #[test]
    fn test_for_node_type_observer() {
        let openai_config = OpenAIConfig::disabled();
        let ollama_config = OllamaConfig::disabled();
        let services = ExternalServices::for_node_type(false, &openai_config, &ollama_config);
        assert!(!services.is_validator);
        assert!(!services.openai_enabled);
        assert!(!services.ollama_enabled);
    }

    #[test]
    fn test_for_node_type_validator_disabled() {
        let openai_config = OpenAIConfig::disabled();
        let ollama_config = OllamaConfig::disabled();
        let services = ExternalServices::for_node_type(true, &openai_config, &ollama_config);
        assert!(services.is_validator);
        assert!(!services.openai_enabled);
        assert!(!services.ollama_enabled);
    }
}
