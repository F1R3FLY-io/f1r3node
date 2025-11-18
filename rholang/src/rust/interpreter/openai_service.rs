// See rholang/src/main/scala/coop/rchain/rholang/interpreter/OpenAIService.scala

use dotenv::dotenv;
use openai_api_rs::v1::{
    api::OpenAIClient,
    audio::{self, AudioSpeechRequest, TTS_1},
    chat_completion::{self, ChatCompletionRequest},
    common::GPT4_O_MINI,
    image::ImageGenerationRequest,
};
use std::env;

use super::errors::InterpreterError;

pub struct OpenAIService {
    client: OpenAIClient,
}

impl OpenAIService {
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
            tracing::warn!(
                "Failed to load OPENAI_API_KEY environment variable, using default key '123'"
            );
            "123".to_string()
        });

        let client = OpenAIClient::builder()
            .with_api_key(api_key)
            .build()
            .expect("Failed to build OpenAI client");

        Self { client }
    }

    pub async fn create_audio_speech(
        &mut self,
        input: &str,
        output_path: &str,
    ) -> Result<(), InterpreterError> {
        let request = AudioSpeechRequest::new(
            TTS_1.to_string(),
            input.to_string(),
            audio::VOICE_SHIMMER.to_string(),
            output_path.to_string(),
        );

        let response = self.client.audio_speech(request).await?;
        if !response.result {
            return Err(InterpreterError::OpenAIError(format!(
                "Failed to create audio speech: {:?}",
                response.headers
            )));
        }

        Ok(())
    }

    pub async fn dalle3_create_image(&mut self, prompt: &str) -> Result<String, InterpreterError> {
        let request = ImageGenerationRequest {
            prompt: prompt.to_string(),
            model: Some("dall-e-3".to_string()),
            n: Some(1),
            size: Some("1024x1024".to_string()),
            response_format: Some("url".to_string()),
            user: None,
        };

        let response = self.client.image_generation(request).await?;
        let image_url = response.data[0].url.clone();
        Ok(image_url)
    }

    pub async fn gpt4_chat_completion(&mut self, prompt: &str) -> Result<String, InterpreterError> {
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

        let response = self.client.chat_completion(request).await?;
        let answer = response.choices[0]
            .message
            .content
            .clone()
            .unwrap_or_else(|| "No answer".to_string());

        Ok(answer)
    }
}
