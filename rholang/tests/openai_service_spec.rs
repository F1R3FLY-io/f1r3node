// See rholang/src/test/scala/coop/rchain/rholang/interpreter/OpenAIServiceSpec.scala
// Ported from Scala PR #123

//! Tests for OpenAI service configuration and behavior.
//! Matches Scala's OpenAIServiceSpec test suite.

use rholang::rust::interpreter::openai_service::{OpenAIConfig, OpenAIService};

/// Helper to parse boolean values (mirrors Scala's parseEnvValue logic)
fn parse_bool_value(value: &str) -> Option<bool> {
    match value.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Helper to select service based on config (mirrors Scala's selectService logic)
fn select_service(
    config_enabled: Option<bool>,
    env_enabled: Option<bool>,
    has_api_key: bool,
) -> &'static str {
    let is_enabled = env_enabled.or(config_enabled).unwrap_or(false);
    if is_enabled {
        if has_api_key {
            "OpenAIServiceImpl" // Would create real service
        } else {
            "IllegalStateException" // Would throw exception
        }
    } else {
        "DisabledOpenAIService" // Would create disabled service
    }
}

/// Helper to select service with validation (mirrors Scala's selectServiceWithValidation logic)
fn select_service_with_validation(
    enabled: bool,
    has_api_key: bool,
    validation_enabled: bool,
    validation_succeeds: bool,
) -> &'static str {
    if enabled {
        if has_api_key {
            if validation_enabled {
                if validation_succeeds {
                    "OpenAIServiceImpl" // Validation passed
                } else {
                    "IllegalStateException" // Validation failed
                }
            } else {
                "OpenAIServiceImpl" // Validation skipped
            }
        } else {
            "IllegalStateException" // No API key
        }
    } else {
        "DisabledOpenAIService" // Service disabled
    }
}

/// Helper to resolve API key priority (mirrors Scala's resolveApiKey logic)
fn resolve_api_key(config_key: Option<&str>, env_key: Option<&str>) -> Option<String> {
    let api_key_from_config = config_key.filter(|k| !k.is_empty());
    api_key_from_config
        .or(env_key.filter(|k| !k.is_empty()))
        .map(|s| s.to_string())
}

// =============================================================================
// Tests for DisabledOpenAIService (NoOp) behavior
// =============================================================================

mod disabled_service_tests {
    use super::*;

    #[tokio::test]
    async fn noop_service_returns_ok_for_tts_create_audio_speech() {
        let service = OpenAIService::new_noop();
        let result = service.create_audio_speech("test prompt", "test.mp3").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn noop_service_returns_empty_for_dalle3_create_image() {
        let service = OpenAIService::new_noop();
        let result = service.dalle3_create_image("test prompt").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[tokio::test]
    async fn noop_service_returns_empty_for_gpt4_text_completion() {
        let service = OpenAIService::new_noop();
        let result = service.gpt4_chat_completion("test prompt").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn noop_service_is_not_enabled() {
        let service = OpenAIService::new_noop();
        assert!(!service.is_enabled());
    }
}

// =============================================================================
// Tests for environment variable parsing
// =============================================================================

mod env_parsing_tests {
    use super::*;

    #[test]
    fn parse_valid_true_values() {
        assert_eq!(parse_bool_value("true"), Some(true));
        assert_eq!(parse_bool_value("TRUE"), Some(true));
        assert_eq!(parse_bool_value("True"), Some(true));
        assert_eq!(parse_bool_value("1"), Some(true));
        assert_eq!(parse_bool_value("yes"), Some(true));
        assert_eq!(parse_bool_value("YES"), Some(true));
        assert_eq!(parse_bool_value("on"), Some(true));
        assert_eq!(parse_bool_value("ON"), Some(true));
    }

    #[test]
    fn parse_valid_false_values() {
        assert_eq!(parse_bool_value("false"), Some(false));
        assert_eq!(parse_bool_value("FALSE"), Some(false));
        assert_eq!(parse_bool_value("False"), Some(false));
        assert_eq!(parse_bool_value("0"), Some(false));
        assert_eq!(parse_bool_value("no"), Some(false));
        assert_eq!(parse_bool_value("NO"), Some(false));
        assert_eq!(parse_bool_value("off"), Some(false));
        assert_eq!(parse_bool_value("OFF"), Some(false));
    }

    #[test]
    fn parse_invalid_values_returns_none() {
        assert_eq!(parse_bool_value("maybe"), None);
        assert_eq!(parse_bool_value("2"), None);
        assert_eq!(parse_bool_value(""), None);
        assert_eq!(parse_bool_value("invalid"), None);
    }
}

// =============================================================================
// Tests for service instantiation behavior
// =============================================================================

mod service_selection_tests {
    use super::*;

    #[test]
    fn env_var_takes_precedence_over_config() {
        // Env says false, config says true -> disabled
        assert_eq!(
            select_service(Some(true), Some(false), true),
            "DisabledOpenAIService"
        );
        // Env says true, config says false -> enabled
        assert_eq!(
            select_service(Some(false), Some(true), true),
            "OpenAIServiceImpl"
        );
    }

    #[test]
    fn env_var_fallback_when_config_not_set() {
        assert_eq!(
            select_service(None, Some(true), true),
            "OpenAIServiceImpl"
        );
        assert_eq!(
            select_service(None, Some(false), true),
            "DisabledOpenAIService"
        );
    }

    #[test]
    fn default_fallback_when_neither_set() {
        assert_eq!(select_service(None, None, true), "DisabledOpenAIService");
        assert_eq!(select_service(None, None, false), "DisabledOpenAIService");
    }

    #[test]
    fn api_key_validation_still_applies() {
        assert_eq!(
            select_service(Some(true), None, false),
            "IllegalStateException"
        );
        assert_eq!(
            select_service(None, Some(true), false),
            "IllegalStateException"
        );
    }

    #[test]
    fn legacy_service_selection_logic() {
        // Test all combinations (enabled, has_api_key)
        assert_eq!(
            select_service(Some(false), None, false),
            "DisabledOpenAIService"
        );
        assert_eq!(
            select_service(Some(false), None, true),
            "DisabledOpenAIService"
        );
        assert_eq!(
            select_service(Some(true), None, false),
            "IllegalStateException"
        );
        assert_eq!(
            select_service(Some(true), None, true),
            "OpenAIServiceImpl"
        );
    }
}

// =============================================================================
// Tests for API key resolution priority
// =============================================================================

mod api_key_resolution_tests {
    use super::*;

    #[test]
    fn config_key_takes_priority() {
        assert_eq!(
            resolve_api_key(Some("config-key"), Some("env-key")),
            Some("config-key".to_string())
        );
    }

    #[test]
    fn falls_back_to_env_key() {
        assert_eq!(
            resolve_api_key(None, Some("env-key")),
            Some("env-key".to_string())
        );
        assert_eq!(
            resolve_api_key(Some(""), Some("env-key")),
            Some("env-key".to_string())
        );
    }

    #[test]
    fn returns_none_if_neither_available() {
        assert_eq!(resolve_api_key(None, None), None);
        assert_eq!(resolve_api_key(Some(""), Some("")), None);
    }
}

// =============================================================================
// Tests for API key validation logic
// =============================================================================

mod validation_logic_tests {
    use super::*;

    #[test]
    fn validation_enabled_and_succeeds() {
        assert_eq!(
            select_service_with_validation(true, true, true, true),
            "OpenAIServiceImpl"
        );
    }

    #[test]
    fn validation_enabled_but_fails() {
        assert_eq!(
            select_service_with_validation(true, true, true, false),
            "IllegalStateException"
        );
    }

    #[test]
    fn validation_disabled_skipped() {
        // validation_succeeds doesn't matter when validation is disabled
        assert_eq!(
            select_service_with_validation(true, true, false, false),
            "OpenAIServiceImpl"
        );
    }

    #[test]
    fn service_disabled_makes_validation_irrelevant() {
        assert_eq!(
            select_service_with_validation(false, true, true, true),
            "DisabledOpenAIService"
        );
    }

    #[test]
    fn no_api_key_makes_validation_irrelevant() {
        assert_eq!(
            select_service_with_validation(true, false, true, true),
            "IllegalStateException"
        );
    }
}

// =============================================================================
// Tests for OpenAIConfig
// =============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn disabled_config_has_correct_defaults() {
        let config = OpenAIConfig::disabled();
        assert!(!config.enabled);
        assert!(config.api_key.is_none());
        assert!(config.validate_api_key); // Default is true
        assert_eq!(config.validation_timeout_sec, 15); // Default is 15
    }

    #[test]
    fn from_config_creates_noop_when_disabled() {
        let config = OpenAIConfig::disabled();
        let service = OpenAIService::from_config(&config);
        assert!(!service.is_enabled());
    }

    #[test]
    fn from_config_values_uses_config_defaults_when_no_env() {
        // Clear relevant env vars to test config-only path
        std::env::remove_var("OPENAI_ENABLED");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_SCALA_CLIENT_API_KEY");
        std::env::remove_var("OPENAI_VALIDATE_API_KEY");
        std::env::remove_var("OPENAI_VALIDATION_TIMEOUT_SEC");

        // Disabled config should not require API key
        let config = OpenAIConfig::from_config_values(false, String::new(), false, 30);
        assert!(!config.enabled);
        assert!(config.api_key.is_none());
        assert!(!config.validate_api_key);
        assert_eq!(config.validation_timeout_sec, 30);
    }

    #[test]
    fn from_env_falls_back_to_from_config_values_defaults() {
        // Clear relevant env vars
        std::env::remove_var("OPENAI_ENABLED");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_SCALA_CLIENT_API_KEY");
        std::env::remove_var("OPENAI_VALIDATE_API_KEY");
        std::env::remove_var("OPENAI_VALIDATION_TIMEOUT_SEC");

        let config = OpenAIConfig::from_env();
        assert!(!config.enabled); // Default is false
        assert!(config.api_key.is_none());
        assert!(config.validate_api_key); // Default is true
        assert_eq!(config.validation_timeout_sec, 15); // Default is 15
    }
}

