use rholang::rust::interpreter::openai_service::{OpenAIConfig, OpenAIService};

#[tokio::main]
async fn main() {
    // Load configuration from environment variables
    // Set OPENAI_ENABLED=true and OPENAI_API_KEY=your-api-key to enable
    let config = OpenAIConfig::from_env();
    let openai_service = OpenAIService::from_config(&config);
    
    let result = openai_service
        .gpt4_chat_completion("What is Bitcoin?")
        .await;

    println!("{:?}", result);
}
