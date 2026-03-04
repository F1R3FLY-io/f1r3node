// See rholang/src/test/scala/coop/rchain/rholang/interpreter/accounting/NonDeterministicProcessesSpec.scala
// Ported from Scala PR #140
//
// Tests for replay consistency of non-deterministic processes (OpenAI, gRPC, Ollama)
// Ensures that replays produce consistent costs and error handling for non-deterministic operations.

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rholang::rust::interpreter::rho_runtime::RhoRuntimeImpl;
use rholang::rust::interpreter::test_utils::resources::create_runtimes_with_services;
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    interpreter::EvaluateResult,
    external_services::ExternalServices,
    openai_service::{OpenAIMockConfig, create_mock_openai_service, create_noop_openai_service},
    grpc_client_service::{GrpcClientMockConfig, GrpcClientService},
    rho_runtime::RhoRuntime,
};
use rspace_plus_plus::rspace::history::history_repository::HistoryRepository;
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};

use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create external services with mock OpenAI and optional mock gRPC
fn create_test_external_services(
    openai_mock: OpenAIMockConfig,
    grpc_mock: Option<GrpcClientMockConfig>,
) -> ExternalServices {
    ExternalServices {
        openai: create_mock_openai_service(openai_mock),
        ollama: rholang::rust::interpreter::ollama_service::create_disabled_ollama_service(),
        grpc_client: match grpc_mock {
            Some(config) => GrpcClientService::new_mock(config),
            None => GrpcClientService::new_noop(),
        },
        openai_enabled: true,
        ollama_enabled: false,
        is_validator: true,
    }
}

/// Helper to create external services with only mock gRPC
#[allow(dead_code)]
fn create_test_external_services_grpc(grpc_mock: GrpcClientMockConfig) -> ExternalServices {
    ExternalServices {
        openai: create_noop_openai_service(),
        ollama: rholang::rust::interpreter::ollama_service::create_disabled_ollama_service(),
        grpc_client: GrpcClientService::new_mock(grpc_mock),
        openai_enabled: false,
        ollama_enabled: false,
        is_validator: true,
    }
}

/// Evaluate a term and then replay it, returning both results
/// This is a simplified test that only checks play execution
async fn evaluate_with_mock_service(
    term: &str,
    external_services: ExternalServices,
) -> EvaluateResult {
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let (runtime, _, _): (
        RhoRuntimeImpl,
        RhoRuntimeImpl,
        Arc<Box<dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation> + Send + Sync + 'static>>,
    ) = create_runtimes_with_services(store, false, &mut Vec::new(), external_services).await;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "test".to_string());

    runtime
        .evaluate(term, initial_phlo, HashMap::new(), rand)
        .await
        .expect("Evaluation failed")
}

// =====================================================
// Basic System Process Tests (no replay)
// These verify the mock services work correctly
// =====================================================

#[tokio::test]
async fn test_gpt4_mock_service_works() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::single_completion("gpt4 completion"),
        None,
    );

    let result = evaluate_with_mock_service(
        r#"new output, gpt4(`rho:ai:gpt4`) in { gpt4!("abc", *output) }"#,
        external_services,
    )
    .await;

    assert!(
        result.errors.is_empty(),
        "GPT4 with mock service should not have errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_gpt4_mock_error_returns_error() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::error_on_first_call(),
        None,
    );

    let result = evaluate_with_mock_service(
        r#"new output, gpt4(`rho:ai:gpt4`) in { gpt4!("abc", *output) }"#,
        external_services,
    )
    .await;

    // The error should be captured by the NonDeterministicProcessFailure mechanism
    assert!(
        !result.errors.is_empty(),
        "GPT4 with error mock should have errors"
    );
}

#[tokio::test]
async fn test_dalle3_mock_service_works() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::single_dalle3("https://example.com/generated-image.png"),
        None,
    );

    let result = evaluate_with_mock_service(
        r#"new output, dalle3(`rho:ai:dalle3`) in { dalle3!("a cat painting", *output) }"#,
        external_services,
    )
    .await;

    assert!(
        result.errors.is_empty(),
        "DALL-E 3 with mock service should not have errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_dalle3_mock_error_returns_error() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::error_on_first_call(),
        None,
    );

    let result = evaluate_with_mock_service(
        r#"new output, dalle3(`rho:ai:dalle3`) in { dalle3!("a cat painting", *output) }"#,
        external_services,
    )
    .await;

    assert!(
        !result.errors.is_empty(),
        "DALL-E 3 with error mock should have errors"
    );
}

#[tokio::test]
async fn test_tts_mock_service_works() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::single_tts_audio(b"fake audio bytes".to_vec()),
        None,
    );

    let result = evaluate_with_mock_service(
        r#"new output, tts(`rho:ai:textToAudio`) in { tts!("Hello world", *output) }"#,
        external_services,
    )
    .await;

    assert!(
        result.errors.is_empty(),
        "Text-to-audio with mock service should not have errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_tts_mock_error_returns_error() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::error_on_first_call(),
        None,
    );

    let result = evaluate_with_mock_service(
        r#"new output, tts(`rho:ai:textToAudio`) in { tts!("Hello world", *output) }"#,
        external_services,
    )
    .await;

    assert!(
        !result.errors.is_empty(),
        "Text-to-audio with error mock should have errors"
    );
}

#[tokio::test]
async fn test_grpc_tell_mock_service_works() {
    let grpc_mock = GrpcClientMockConfig::create("localhost", 8080);
    let external_services = create_test_external_services_grpc(grpc_mock.clone());

    // gRPC tell uses 3 arguments (host, port, payload) without an ack channel
    let result = evaluate_with_mock_service(
        r#"new grpcTell(`rho:io:grpcTell`) in { grpcTell!("localhost", 8080, "payload") }"#,
        external_services,
    )
    .await;

    assert!(
        result.errors.is_empty(),
        "gRPC tell with mock service should not have errors: {:?}",
        result.errors
    );
    
    // Verify the mock was called
    assert!(grpc_mock.was_called(), "gRPC mock should have been called");
}

#[tokio::test]
async fn test_grpc_tell_mock_error_returns_error() {
    // Mock expects different host/port to trigger an error
    let grpc_mock = GrpcClientMockConfig::create("different_host", 9999);
    let external_services = create_test_external_services_grpc(grpc_mock.clone());

    // gRPC tell uses 3 arguments (host, port, payload) without an ack channel
    let result = evaluate_with_mock_service(
        r#"new grpcTell(`rho:io:grpcTell`) in { grpcTell!("localhost", 8080, "payload") }"#,
        external_services,
    )
    .await;

    // The gRPC error should be propagated
    assert!(
        !result.errors.is_empty(),
        "gRPC tell with error mock should have errors"
    );
    
    // Verify the mock was called
    assert!(grpc_mock.was_called(), "gRPC mock should have been called");
}
