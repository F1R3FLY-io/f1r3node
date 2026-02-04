use std::env;
use std::sync::Arc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use super::errors::InterpreterError;

#[derive(Clone, Debug)]
pub struct OllamaConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model: String,
    pub timeout_sec: u64,
    pub validate_connection: bool,
}

impl OllamaConfig {
    pub fn from_env() -> Self {
        Self::from_config_values(
            false,
            "http://localhost:11434".to_string(),
            "llama4:latest".to_string(),
            30,    // default timeout
            true,  // default validate_connection
        )
    }

    pub fn from_config_values(
        config_enabled: bool,
        config_base_url: String,
        config_model: String,
        config_timeout_sec: u64,
        config_validate_connection: bool,
    ) -> Self {
        let enabled = parse_bool_env("OLLAMA_ENABLED").unwrap_or(config_enabled);
        let base_url = env::var("OLLAMA_BASE_URL").unwrap_or(config_base_url);
        let model = env::var("OLLAMA_MODEL").unwrap_or(config_model);
        let timeout_sec = env::var("OLLAMA_TIMEOUT_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(config_timeout_sec);
        let validate_connection = parse_bool_env("OLLAMA_VALIDATE_CONNECTION")
            .unwrap_or(config_validate_connection);

        Self {
            enabled,
            base_url,
            model,
            timeout_sec,
            validate_connection,
        }
    }
    
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:11434".to_string(),
            model: "llama4:latest".to_string(),
            timeout_sec: 30,
            validate_connection: false,
        }
    }
}

pub fn parse_bool_env(name: &str) -> Option<bool> {
    env::var(name).ok().and_then(|v| {
        match v.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Serialize, Debug)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct GenerateResponse {
    response: String,
}

#[derive(Deserialize, Debug)]
struct ModelInfo {
    name: String,
}

#[derive(Deserialize, Debug)]
struct ListModelsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Clone)]
pub enum OllamaService {
    Real {
        client: Client,
        base_url: String,
        model: String,
        timeout_sec: u64,
    },
    Mock {
        chat_response: String,
        generate_response: String,
        models_response: Vec<String>,
    },
    Disabled,
}

impl OllamaService {
    pub fn new_real(base_url: &str, model: &str, timeout_sec: u64) -> Self {
        Self::Real {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_sec))
                .build()
                .unwrap_or_else(|_| Client::new()),
            base_url: base_url.to_string(),
            model: model.to_string(),
            timeout_sec,
        }
    }

    pub fn new_mock(chat_response: String, generate_response: String, models_response: Vec<String>) -> Self {
        Self::Mock {
            chat_response,
            generate_response,
            models_response,
        }
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }

    pub fn from_config(config: &OllamaConfig) -> Self {
        if config.enabled {
            Self::new_real(&config.base_url, &config.model, config.timeout_sec)
        } else {
            Self::new_disabled()
        }
    }

    pub async fn chat(&self, model_override: Option<&str>, messages: Vec<ChatMessage>) -> Result<String, InterpreterError> {
        match self {
            Self::Real { client, base_url, model, .. } => {
                // Empty model string falls back to default (parity with Scala)
                let actual_model = match model_override {
                    Some(m) if !m.is_empty() => m.to_string(),
                    _ => model.clone(),
                };

                // Log prompt preview (truncated to 250 chars like Scala)
                let prompt_preview = if let Some(last_msg) = messages.last() {
                    if last_msg.content.len() <= 250 {
                        last_msg.content.clone()
                    } else {
                        format!("{}...", &last_msg.content[..250])
                    }
                } else {
                    String::new()
                };
                tracing::info!("Ollama chat - model: '{}', prompt: '{}'", actual_model, prompt_preview);

                let req = ChatRequest {
                    model: actual_model.clone(),
                    messages,
                    stream: false,
                };

                let res = client.post(format!("{}/api/chat", base_url))
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Ollama request failed: {}", e)))?;

                if !res.status().is_success() {
                    let status = res.status();
                    let error_body = res.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                    return Err(InterpreterError::OllamaError(format!("Ollama error: {} - {}", status, error_body)));
                }

                let body: ChatResponse = res.json()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Failed to parse response: {}", e)))?;

                tracing::info!("Ollama chat completion succeeded for model: {}", actual_model);
                Ok(body.message.content)
            }
            Self::Mock { chat_response, .. } => Ok(chat_response.clone()),
            Self::Disabled => Err(InterpreterError::OllamaError(
                "Ollama service is disabled via configuration".to_string()
            )),
        }
    }

    pub async fn generate(&self, model_override: Option<&str>, prompt: &str) -> Result<String, InterpreterError> {
        match self {
            Self::Real { client, base_url, model, .. } => {
                // Empty model string falls back to default (parity with Scala)
                let actual_model = match model_override {
                    Some(m) if !m.is_empty() => m.to_string(),
                    _ => model.clone(),
                };

                // Log prompt preview (truncated to 250 chars like Scala)
                let prompt_preview = if prompt.len() <= 250 {
                    prompt.to_string()
                } else {
                    format!("{}...", &prompt[..250])
                };
                tracing::info!("Ollama generate - model: '{}', prompt: '{}'", actual_model, prompt_preview);

                let req = GenerateRequest {
                    model: actual_model.clone(),
                    prompt: prompt.to_string(),
                    stream: false,
                };

                let res = client.post(format!("{}/api/generate", base_url))
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Ollama request failed: {}", e)))?;

                if !res.status().is_success() {
                    let status = res.status();
                    let error_body = res.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                    return Err(InterpreterError::OllamaError(format!("Ollama error: {} - {}", status, error_body)));
                }

                let body: GenerateResponse = res.json()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Failed to parse response: {}", e)))?;

                tracing::info!("Ollama generate succeeded for model: {}", actual_model);
                Ok(body.response)
            }
            Self::Mock { generate_response, .. } => Ok(generate_response.clone()),
            Self::Disabled => Err(InterpreterError::OllamaError(
                "Ollama service is disabled via configuration".to_string()
            )),
        }
    }

    pub async fn list_models(&self) -> Result<Vec<String>, InterpreterError> {
        match self {
            Self::Real { client, base_url, .. } => {
                tracing::info!("Ollama list_models - fetching available models");

                let res = client.get(format!("{}/api/tags", base_url))
                    .send()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Ollama request failed: {}", e)))?;
                
                if !res.status().is_success() {
                    let status = res.status();
                    let error_body = res.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                    return Err(InterpreterError::OllamaError(format!("Ollama error: {} - {}", status, error_body)));
                }

                let body: ListModelsResponse = res.json()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Failed to parse response: {}", e)))?;
                
                let models: Vec<String> = body.models.into_iter().map(|m| m.name).collect();
                tracing::info!("Ollama list_models succeeded, found {} models", models.len());
                Ok(models)
            }
            Self::Mock { models_response, .. } => Ok(models_response.clone()),
            Self::Disabled => Err(InterpreterError::OllamaError(
                "Ollama service is disabled via configuration".to_string()
            )),
        }
    }
}

pub type SharedOllamaService = Arc<tokio::sync::Mutex<OllamaService>>;

pub fn create_ollama_service(config: &OllamaConfig) -> SharedOllamaService {
    Arc::new(tokio::sync::Mutex::new(OllamaService::from_config(config)))
}

pub fn create_disabled_ollama_service() -> SharedOllamaService {
    Arc::new(tokio::sync::Mutex::new(OllamaService::new_disabled()))
}

/// Create Ollama service with connection validation (matches Scala's validateConnectionOrFail)
/// This should be used during node startup to ensure Ollama is reachable.
/// Returns an error if validate_connection is true and Ollama is unreachable.
pub async fn create_ollama_service_validated(config: &OllamaConfig) -> Result<SharedOllamaService, InterpreterError> {
    if !config.enabled {
        tracing::info!("Ollama service is disabled");
        return Ok(create_disabled_ollama_service());
    }
    
    let service = OllamaService::from_config(config);
    
    if config.validate_connection {
        tracing::info!("Validating Ollama connection to {}", config.base_url);
        // Test connection by listing models (same as Scala's validateConnectionOrFail)
        match service.list_models().await {
            Ok(models) => {
                tracing::info!("Ollama service connection validated successfully at {} ({} models available)", 
                    config.base_url, models.len());
            }
            Err(e) => {
                return Err(InterpreterError::OllamaError(format!(
                    "Ollama service connection validation failed. Check that Ollama is running on {}: {}",
                    config.base_url, e
                )));
            }
        }
    } else {
        tracing::info!("Ollama connection validation is disabled by config 'validate_connection=false'");
    }
    
    Ok(Arc::new(tokio::sync::Mutex::new(service)))
}
