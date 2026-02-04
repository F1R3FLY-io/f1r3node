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
}

impl OllamaConfig {
    pub fn from_env() -> Self {
        Self::from_config_values(false, "http://localhost:11434".to_string(), "llama3".to_string())
    }

    pub fn from_config_values(
        config_enabled: bool,
        config_base_url: String,
        config_model: String,
    ) -> Self {
        let enabled = parse_bool_env("OLLAMA_ENABLED").unwrap_or(config_enabled);
        let base_url = env::var("OLLAMA_BASE_URL").unwrap_or(config_base_url);
        let model = env::var("OLLAMA_MODEL").unwrap_or(config_model);

        Self {
            enabled,
            base_url,
            model,
        }
    }
    
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
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
    },
    Mock {
        chat_response: String,
        generate_response: String,
        models_response: Vec<String>,
    },
    NoOp,
}

impl OllamaService {
    pub fn new_real(base_url: &str, model: &str) -> Self {
        Self::Real {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }

    pub fn new_mock(chat_response: String, generate_response: String, models_response: Vec<String>) -> Self {
        Self::Mock {
            chat_response,
            generate_response,
            models_response,
        }
    }

    pub fn new_noop() -> Self {
        Self::NoOp
    }

    pub fn from_config(config: &OllamaConfig) -> Self {
        if config.enabled {
            Self::new_real(&config.base_url, &config.model)
        } else {
            Self::new_noop()
        }
    }

    pub async fn chat(&self, model_override: Option<&str>, messages: Vec<ChatMessage>) -> Result<String, InterpreterError> {
        match self {
            Self::Real { client, base_url, model } => {
                let req = ChatRequest {
                    model: model_override.map(|s| s.to_string()).unwrap_or(model.clone()),
                    messages,
                    stream: false,
                };

                let res = client.post(format!("{}/api/chat", base_url))
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Ollama request failed: {}", e)))?;

                if !res.status().is_success() {
                    return Err(InterpreterError::OllamaError(format!("Ollama error: {}", res.status())));
                }

                let body: ChatResponse = res.json()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Failed to parse response: {}", e)))?;

                Ok(body.message.content)
            }
            Self::Mock { chat_response, .. } => Ok(chat_response.clone()),
            Self::NoOp => Ok(String::new()),
        }
    }

    pub async fn generate(&self, model_override: Option<&str>, prompt: &str) -> Result<String, InterpreterError> {
        match self {
            Self::Real { client, base_url, model } => {
                let req = GenerateRequest {
                    model: model_override.map(|s| s.to_string()).unwrap_or(model.clone()),
                    prompt: prompt.to_string(),
                    stream: false,
                };

                let res = client.post(format!("{}/api/generate", base_url))
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Ollama request failed: {}", e)))?;

                if !res.status().is_success() {
                    return Err(InterpreterError::OllamaError(format!("Ollama error: {}", res.status())));
                }

                let body: GenerateResponse = res.json()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Failed to parse response: {}", e)))?;

                Ok(body.response)
            }
            Self::Mock { generate_response, .. } => Ok(generate_response.clone()),
            Self::NoOp => Ok(String::new()),
        }
    }

    pub async fn list_models(&self) -> Result<Vec<String>, InterpreterError> {
        match self {
            Self::Real { client, base_url, .. } => {
                let res = client.get(format!("{}/api/tags", base_url))
                    .send()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Ollama request failed: {}", e)))?;
                
                if !res.status().is_success() {
                    return Err(InterpreterError::OllamaError(format!("Ollama error: {}", res.status())));
                }

                let body: ListModelsResponse = res.json()
                    .await
                    .map_err(|e| InterpreterError::OllamaError(format!("Failed to parse response: {}", e)))?;
                
                Ok(body.models.into_iter().map(|m| m.name).collect())
            }
            Self::Mock { models_response, .. } => Ok(models_response.clone()),
            Self::NoOp => Ok(vec![]),
        }
    }
}

pub type SharedOllamaService = Arc<tokio::sync::Mutex<OllamaService>>;

pub fn create_ollama_service(config: &OllamaConfig) -> SharedOllamaService {
    Arc::new(tokio::sync::Mutex::new(OllamaService::from_config(config)))
}

pub fn create_noop_ollama_service() -> SharedOllamaService {
    Arc::new(tokio::sync::Mutex::new(OllamaService::new_noop()))
}
