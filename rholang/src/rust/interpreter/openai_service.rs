// See rholang/src/main/scala/coop/rchain/rholang/externalservices/OpenAIService.scala
// Ported from Scala PR #123 and #140
//
// This module provides OpenAI integration with configuration support.
// Uses enum-based dispatch instead of trait objects for async compatibility.

use dotenv::dotenv;
use openai_api_rs::v1::{
    api::OpenAIClient,
    audio::{self, AudioSpeechRequest, TTS_1},
    chat_completion::{self, ChatCompletionRequest},
    common::GPT4_O_MINI,
    image::ImageGenerationRequest,
};
use std::env;
use std::sync::Arc;
use std::time::Duration;

use super::errors::InterpreterError;

/// Configuration for OpenAI service
/// Matches Scala HOCON config structure from PR #123:
/// ```hocon
/// openai {
///   enabled = false
///   api-key = ""
///   validate-api-key = true
///   validation-timeout-sec = 15
/// }
/// ```
#[derive(Clone, Debug)]
pub struct OpenAIConfig {
    /// Whether OpenAI service is enabled
    /// Priority: 1. OPENAI_ENABLED env, 2. config file, 3. default (false)
    pub enabled: bool,
    /// API key for OpenAI (only required if enabled)
    /// Priority: 1. config file, 2. OPENAI_API_KEY env, 3. OPENAI_SCALA_CLIENT_API_KEY env
    pub api_key: Option<String>,
    /// Whether to validate API key at startup by calling a lightweight endpoint
    /// Default: true
    pub validate_api_key: bool,
    /// Timeout for API key validation call in seconds
    /// Default: 15
    pub validation_timeout_sec: u64,
}

impl OpenAIConfig {
    /// Load configuration from environment variables only
    /// Matches Scala resolution logic from OpenAIServiceImpl
    /// Use this when no HOCON config is available
    pub fn from_env() -> Self {
        // Use empty config values as base, let env vars take priority
        Self::from_config_values(false, String::new(), true, 15)
    }

    /// Load configuration merging HOCON config values with environment variables
    /// Priority order per Issue #127:
    /// - enabled: 1. OPENAI_ENABLED env, 2. config value, 3. default (false)
    /// - api_key: 1. OPENAI_API_KEY env, 2. OPENAI_SCALA_CLIENT_API_KEY env, 3. config value
    /// - validate_api_key: 1. OPENAI_VALIDATE_API_KEY env, 2. config value, 3. default (true)
    /// - validation_timeout_sec: 1. OPENAI_VALIDATION_TIMEOUT_SEC env, 2. config value, 3. default (15)
    pub fn from_config_values(
        config_enabled: bool,
        config_api_key: String,
        config_validate_api_key: bool,
        config_validation_timeout_sec: u64,
    ) -> Self {
        dotenv().ok();

        // enabled: env var takes priority over config
        let enabled = parse_bool_env("OPENAI_ENABLED").unwrap_or(config_enabled);

        // api_key: env vars take priority, then config
        // Matches Scala's (apiKeyFromEnv orElse apiKeyFromConfig) behavior
        let api_key = if enabled {
            let key = env::var("OPENAI_API_KEY")
                .or_else(|_| env::var("OPENAI_SCALA_CLIENT_API_KEY"))
                .ok()
                .filter(|k| !k.is_empty())
                .or_else(|| {
                    if !config_api_key.is_empty() {
                        Some(config_api_key.clone())
                    } else {
                        None
                    }
                });

            if key.is_none() {
                tracing::error!(
                    "OpenAI is enabled but no API key provided. \
                     Set OPENAI_API_KEY or OPENAI_SCALA_CLIENT_API_KEY environment variable, \
                     or configure 'openai.api-key' in the config file."
                );
                panic!(
                    "OpenAI API key is not configured. Provide it via env var OPENAI_API_KEY, \
                     OPENAI_SCALA_CLIENT_API_KEY, or openai.api-key config when openai is enabled."
                );
            }
            key
        } else {
            None
        };

        // validate_api_key: env var takes priority over config
        let validate_api_key =
            parse_bool_env("OPENAI_VALIDATE_API_KEY").unwrap_or(config_validate_api_key);

        // validation_timeout_sec: env var takes priority over config
        let validation_timeout_sec = env::var("OPENAI_VALIDATION_TIMEOUT_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(config_validation_timeout_sec);

        Self {
            enabled,
            api_key,
            validate_api_key,
            validation_timeout_sec,
        }
    }

    /// Create a disabled configuration
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            api_key: None,
            validate_api_key: true,
            validation_timeout_sec: 15,
        }
    }

    /// Create a test configuration with specified settings
    #[cfg(test)]
    pub fn for_test(enabled: bool, api_key: Option<String>) -> Self {
        Self {
            enabled,
            api_key,
            validate_api_key: false, // Skip validation in tests
            validation_timeout_sec: 5,
        }
    }
}


/// Parse boolean environment variable with Scala-compatible values
/// Accepts: true/false, 1/0, yes/no, on/off (case insensitive)
pub fn parse_bool_env(name: &str) -> Option<bool> {
    env::var(name).ok().and_then(|v| {
        match v.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None, // Invalid values return None
        }
    })
}

/// Validate API key by calling a lightweight endpoint (models list).
/// Matches Scala's OpenAIServiceImpl.validateApiKeyOrFail behavior.
/// 
/// This function panics if validation fails, failing fast at startup
/// rather than at first use.
fn validate_api_key_or_fail_sync(api_key: &str, timeout_sec: u64) {
    use std::thread;
    
    let api_key = api_key.to_string();
    let timeout = Duration::from_secs(timeout_sec);
    
    // Run validation in blocking context since we're called during initialization
    let result = thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime for validation");
        
        rt.block_on(async {
            tokio::time::timeout(timeout, validate_api_key_async(&api_key)).await
        })
    }).join();
    
    match result {
        Ok(Ok(Ok(model_count))) => {
            tracing::info!("OpenAI API key validated ({} models available)", model_count);
        }
        Ok(Ok(Err(e))) => {
            tracing::error!("OpenAI API key validation failed: {}", e);
            panic!(
                "OpenAI API key validation failed. Check 'openai.api-key' or \
                 'OPENAI_API_KEY' or 'OPENAI_SCALA_CLIENT_API_KEY'. Error: {}",
                e
            );
        }
        Ok(Err(_)) => {
            tracing::error!("OpenAI API key validation timed out after {} seconds", timeout_sec);
            panic!(
                "OpenAI API key validation timed out after {} seconds. \
                 Check network connectivity or increase validation-timeout-sec.",
                timeout_sec
            );
        }
        Err(_) => {
            panic!("OpenAI API key validation thread panicked");
        }
    }
}

/// Async helper to validate API key by listing models
async fn validate_api_key_async(api_key: &str) -> Result<usize, String> {
    let mut client = OpenAIClient::builder()
        .with_api_key(api_key.to_string())
        .build()
        .map_err(|e| format!("Failed to build OpenAI client: {}", e))?;
    
    // Use list_model to validate the API key (lightweight endpoint)
    // This matches Scala's service.listModels call
    match client.list_models().await {
        Ok(response) => Ok(response.data.len()),
        Err(e) => Err(format!("API call failed: {}", e)),
    }
}

/// Mock configuration for OpenAI service
#[derive(Clone)]
pub struct OpenAIMockConfig {
    pub gpt4_response: Option<Result<String, String>>,
    pub dalle3_response: Option<Result<String, String>>,
    pub tts_response: Option<Result<Vec<u8>, String>>,
}

impl OpenAIMockConfig {
    /// Create mock that returns a single GPT4 completion
    pub fn single_completion(text: &str) -> Self {
        Self {
            gpt4_response: Some(Ok(text.to_string())),
            dalle3_response: None,
            tts_response: None,
        }
    }

    /// Create mock that returns a single DALL-E 3 image URL
    pub fn single_dalle3(url: &str) -> Self {
        Self {
            gpt4_response: None,
            dalle3_response: Some(Ok(url.to_string())),
            tts_response: None,
        }
    }

    /// Create mock that returns TTS audio bytes
    pub fn single_tts_audio(audio: Vec<u8>) -> Self {
        Self {
            gpt4_response: None,
            dalle3_response: None,
            tts_response: Some(Ok(audio)),
        }
    }

    /// Create mock that returns an error on first call
    pub fn error_on_first_call() -> Self {
        Self {
            gpt4_response: Some(Err("Mock error on first call".to_string())),
            dalle3_response: Some(Err("Mock error on first call".to_string())),
            tts_response: Some(Err("Mock error on first call".to_string())),
        }
    }
}

/// OpenAI service implementation using enum dispatch for async compatibility
/// This avoids the dyn-compatibility issues with async trait methods
#[derive(Clone)]
pub enum OpenAIService {
    /// Real implementation that calls OpenAI API
    Real(Arc<tokio::sync::Mutex<OpenAIClient>>),
    /// NoOp implementation that returns empty results
    NoOp,
    /// Mock implementation for testing
    Mock(OpenAIMockConfig),
}

impl OpenAIService {
    /// Create a new real OpenAI service with the given API key
    /// Note: Call validate_api_key_or_fail separately if validation is needed
    pub fn new_real(api_key: &str) -> Self {
        let client = OpenAIClient::builder()
            .with_api_key(api_key.to_string())
            .build()
            .expect("Failed to build OpenAI client");

        tracing::info!("OpenAI service initialized successfully");
        Self::Real(Arc::new(tokio::sync::Mutex::new(client)))
    }

    /// Create a NoOp service
    pub fn new_noop() -> Self {
        tracing::debug!("NoOpOpenAIService created - OpenAI functionality is disabled");
        Self::NoOp
    }

    /// Create from configuration with optional validation
    /// Matches Scala OpenAIServiceImpl.instance behavior
    pub fn from_config(config: &OpenAIConfig) -> Self {
        if config.enabled {
            if let Some(ref api_key) = config.api_key {
                let service = Self::new_real(api_key);
                
                // Validate API key if configured (matches Scala validateApiKeyOrFail)
                if config.validate_api_key {
                    validate_api_key_or_fail_sync(api_key, config.validation_timeout_sec);
                }
                
                service
            } else {
                Self::new_noop()
            }
        } else {
            Self::new_noop()
        }
    }

    /// Check if this service is enabled
    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Real(_))
    }

    /// Create audio speech from text (text-to-speech)
    pub async fn create_audio_speech(
        &self,
        input: &str,
        output_path: &str,
    ) -> Result<(), InterpreterError> {
        match self {
            Self::Real(client) => {
                let request = AudioSpeechRequest::new(
                    TTS_1.to_string(),
                    input.to_string(),
                    audio::VOICE_SHIMMER.to_string(),
                    output_path.to_string(),
                );

                let mut client = client.lock().await;
                let response = client.audio_speech(request).await?;
                if !response.result {
                    return Err(InterpreterError::OpenAIError(format!(
                        "Failed to create audio speech: {:?}",
                        response.headers
                    )));
                }
                Ok(())
            }
            Self::NoOp => {
                tracing::debug!("OpenAI service is disabled - ttsCreateAudioSpeech request ignored");
                Ok(())
            }
            Self::Mock(config) => {
                match &config.tts_response {
                    Some(Ok(_)) => Ok(()),
                    Some(Err(e)) => Err(InterpreterError::OpenAIError(e.clone())),
                    None => Ok(()),
                }
            }
        }
    }

    /// Create image using DALL-E 3
    pub async fn dalle3_create_image(&self, prompt: &str) -> Result<String, InterpreterError> {
        match self {
            Self::Real(client) => {
                let request = ImageGenerationRequest {
                    prompt: prompt.to_string(),
                    model: Some("dall-e-3".to_string()),
                    n: Some(1),
                    size: Some("1024x1024".to_string()),
                    response_format: Some("url".to_string()),
                    user: None,
                };

                let mut client = client.lock().await;
                let response = client.image_generation(request).await?;
                let image_url = response.data[0].url.clone();
                Ok(image_url)
            }
            Self::NoOp => {
                tracing::debug!("OpenAI service is disabled - dalle3CreateImage request ignored");
                Ok(String::new())
            }
            Self::Mock(config) => {
                match &config.dalle3_response {
                    Some(Ok(url)) => Ok(url.clone()),
                    Some(Err(e)) => Err(InterpreterError::OpenAIError(e.clone())),
                    None => Ok(String::new()),
                }
            }
        }
    }

    /// Get text completion using GPT-4
    pub async fn gpt4_chat_completion(&self, prompt: &str) -> Result<String, InterpreterError> {
        match self {
            Self::Real(client) => {
                let request = ChatCompletionRequest::new(
                    GPT4_O_MINI.to_string(),
                    vec![chat_completion::ChatCompletionMessage {
                        role: chat_completion::MessageRole::user,
                        content: chat_completion::Content::Text(prompt.to_string()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    }],
                );

                let mut client = client.lock().await;
                let response = client.chat_completion(request).await?;
                let answer = response.choices[0]
                    .message
                    .content
                    .clone()
                    .unwrap_or_else(|| "No answer".to_string());

                Ok(answer)
            }
            Self::NoOp => {
                tracing::debug!("OpenAI service is disabled - gpt4TextCompletion request ignored");
                Ok(String::new())
            }
            Self::Mock(config) => {
                match &config.gpt4_response {
                    Some(Ok(text)) => Ok(text.clone()),
                    Some(Err(e)) => Err(InterpreterError::OpenAIError(e.clone())),
                    None => Ok(String::new()),
                }
            }
        }
    }
}

/// Type alias for thread-safe OpenAI service
pub type SharedOpenAIService = Arc<tokio::sync::Mutex<OpenAIService>>;

/// Create a shared OpenAI service based on configuration
pub fn create_openai_service(config: &OpenAIConfig) -> SharedOpenAIService {
    Arc::new(tokio::sync::Mutex::new(OpenAIService::from_config(config)))
}

/// Create a NoOp OpenAI service
pub fn create_noop_openai_service() -> SharedOpenAIService {
    Arc::new(tokio::sync::Mutex::new(OpenAIService::new_noop()))
}

/// Create a Mock OpenAI service for testing
pub fn create_mock_openai_service(config: OpenAIMockConfig) -> SharedOpenAIService {
    Arc::new(tokio::sync::Mutex::new(OpenAIService::Mock(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_config() {
        let config = OpenAIConfig::disabled();
        assert!(!config.enabled);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_noop_service() {
        let service = OpenAIService::new_noop();
        assert!(!service.is_enabled());
    }

    #[tokio::test]
    async fn test_noop_service_returns_empty() {
        let service = OpenAIService::new_noop();

        assert!(service.create_audio_speech("test", "test.mp3").await.is_ok());
        assert_eq!(service.dalle3_create_image("test").await.unwrap(), "");
        assert_eq!(service.gpt4_chat_completion("test").await.unwrap(), "");
    }
}
