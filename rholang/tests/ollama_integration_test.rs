
use std::sync::Arc;
use tokio::sync::Mutex;
use tempfile::Builder;
use models::rhoapi::Par;
use rholang::rust::interpreter::external_services::ExternalServices;
use rholang::rust::interpreter::ollama_service::OllamaService;
use rholang::rust::interpreter::grpc_client_service::GrpcClientService;
use rholang::rust::interpreter::rho_runtime::{self, RhoRuntimeImpl};
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::errors::InterpreterError;
use rholang::rust::interpreter::storage::storage_printer;
use rspace_plus_plus::rspace::shared::rspace_store_manager::mk_rspace_store_manager;
use rspace_plus_plus::rspace::shared::lmdb_dir_store_manager::MB;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use rholang::rust::interpreter::rho_runtime::RhoRuntime;

async fn with_runtime_and_mock_ollama<F, Fut>(
    mock_service: OllamaService,
    f: F,
) where
    F: FnOnce(RhoRuntimeImpl) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let temp_dir = Builder::new()
        .prefix("ollama-test-")
        .tempdir()
        .expect("Failed to create temp dir");

    let mut store_manager = mk_rspace_store_manager(temp_dir.path().to_path_buf(), 100 * MB);
    let rspace_store = store_manager.r_space_stores().await.unwrap();

    let openai_service = rholang::rust::interpreter::openai_service::create_noop_openai_service();
    let grpc_client = GrpcClientService::new_noop();
    let ollama_service = Arc::new(Mutex::new(mock_service));

    let external_services = ExternalServices {
        openai: openai_service,
        ollama: ollama_service,
        grpc_client,
        openai_enabled: false,
        ollama_enabled: true,
        is_validator: true,
    };

    let runtime = rho_runtime::create_runtime_from_kv_store(
        rspace_store,
        Par::default(),
        false,
        &mut Vec::new(),
        Arc::new(Box::new(Matcher)),
        external_services,
    )
    .await;

    f(runtime).await;
}

fn storage_contents(runtime: &RhoRuntimeImpl) -> String {
    storage_printer::pretty_print(runtime)
}

async fn execute(
    runtime: &mut RhoRuntimeImpl,
    term: &str,
) -> Result<(), InterpreterError> {
    let result = runtime.evaluate_with_term(term).await?;
    if !result.errors.is_empty() {
        return Err(result.errors.into_iter().next().unwrap());
    }
    Ok(())
}



#[tokio::test]
async fn ollama_chat_should_return_mock_response() {
    let mock_response = "Echo: What is 2+2?";
    let mock_service = OllamaService::new_mock(
        mock_response.to_string(),
        "".to_string(),
        vec![]
    );

    with_runtime_and_mock_ollama(mock_service, |mut runtime| async move {
        // Use a public name @"return" to persist the result in storage
        let term = r#"
            new chat(`rho:ollama:chat`) in {
                chat!("llama3.2", "What is 2+2?", @"return")
            }
        "#;

        execute(&mut runtime, term).await.expect("Execution failed");
        
        let storage = storage_contents(&runtime);
        println!("Storage: {}", storage);
        
        assert!(storage.contains(mock_response), "Storage does not contain expected mock response");
    }).await;
}

#[tokio::test]
async fn ollama_generate_should_return_mock_response() {
    let mock_response = "Generated Poem";
    let mock_service = OllamaService::new_mock(
        "".to_string(),
        mock_response.to_string(),
        vec![]
    );

    with_runtime_and_mock_ollama(mock_service, |mut runtime| async move {
        let term = r#"
            new generate(`rho:ollama:generate`) in {
                generate!("llama3.2", "Write a poem", @"return")
            }
        "#;

        execute(&mut runtime, term).await.expect("Execution failed");
        
        let storage = storage_contents(&runtime);
        assert!(storage.contains(mock_response));
    }).await;
}

#[tokio::test]
async fn ollama_models_should_return_list() {
    let mock_models = vec!["model1".to_string(), "model2".to_string()];
    let mock_service = OllamaService::new_mock(
        "".to_string(),
        "".to_string(),
        mock_models.clone()
    );

    with_runtime_and_mock_ollama(mock_service, |mut runtime| async move {
        let term = r#"
            new models(`rho:ollama:models`) in {
                models!(@"return")
            }
        "#;

        execute(&mut runtime, term).await.expect("Execution failed");
        
        let storage = storage_contents(&runtime);
        println!("Storage Models: {}", storage);
        assert!(storage.contains("model1"));
        assert!(storage.contains("model2"));
    }).await;
}
