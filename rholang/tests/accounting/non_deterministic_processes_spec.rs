// Tests for replay consistency of non-deterministic processes (OpenAI, gRPC, Ollama).
// Ensures that replays produce consistent costs and error handling for non-deterministic operations.

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rholang::rust::interpreter::rho_runtime::RhoRuntimeImpl;
use rholang::rust::interpreter::test_utils::resources::create_runtimes_with_services;
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    external_services::ExternalServices,
    grpc_client_service::{GrpcClientMockConfig, GrpcClientService},
    interpreter::EvaluateResult,
    openai_service::{create_mock_openai_service, create_noop_openai_service, OpenAIMockConfig},
    rho_runtime::RhoRuntime,
};
#[cfg(feature="chromadb")]
use rholang::rust::interpreter::chromadb_service::create_noop_chromadb_service;
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
        #[cfg(feature="chromadb")]
        chroma: create_noop_chromadb_service(),
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
        #[cfg(feature="chromadb")]
        chroma: create_noop_chromadb_service(),
    }
}

/// Evaluate a term (play only), returning the result
async fn evaluate_with_mock_service(
    term: &str,
    external_services: ExternalServices,
) -> EvaluateResult {
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let (runtime, _, _): (
        RhoRuntimeImpl,
        RhoRuntimeImpl,
        Arc<
            Box<
                dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>
                    + Send
                    + Sync
                    + 'static,
            >,
        >,
    ) = create_runtimes_with_services(store, false, &mut Vec::new(), external_services).await;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "test".to_string());

    runtime
        .evaluate(term, initial_phlo, HashMap::new(), rand)
        .await
        .expect("Evaluation failed")
}

/// Evaluate a term then replay it, returning (play_result, replay_result).
/// Evaluate-and-replay pattern for non-deterministic process testing:
///   1. Play: evaluate on normal runtime
///   2. Checkpoint: capture root hash + event log
///   3. Rig: reset replay runtime to root, load event log
///   4. Replay: evaluate same term on replay runtime
///   5. Verify: check_replay_data ensures all events were consumed
async fn evaluate_and_replay(
    term: &str,
    initial_phlo: Cost,
    external_services: ExternalServices,
) -> (EvaluateResult, EvaluateResult) {
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let (mut runtime, mut replay_runtime, _): (
        RhoRuntimeImpl,
        RhoRuntimeImpl,
        Arc<
            Box<
                dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>
                    + Send
                    + Sync
                    + 'static,
            >,
        >,
    ) = create_runtimes_with_services(store, false, &mut Vec::new(), external_services).await;

    let rand = Blake2b512Random::create_from_bytes(&[]);

    // Play phase
    let play_result = runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand.clone())
        .await
        .expect("Play evaluation failed");

    // Checkpoint: captures root hash and event log
    let checkpoint = runtime.create_checkpoint();

    // Rig replay runtime with the event log from play
    replay_runtime
        .reset(&checkpoint.root)
        .expect("Replay reset failed");
    replay_runtime
        .rig(checkpoint.log)
        .expect("Replay rig failed");

    // Replay phase: same term, same phlo, same rand
    let replay_result = replay_runtime
        .evaluate(term, initial_phlo, HashMap::new(), rand)
        .await
        .expect("Replay evaluation failed");

    // Verify all replay events were consumed
    replay_runtime
        .check_replay_data()
        .expect("Replay data check failed: unconsumed events remain");

    (play_result, replay_result)
}

/// Assert that play and replay produce consistent results
fn assert_replay_consistency(
    play_result: &EvaluateResult,
    replay_result: &EvaluateResult,
    test_name: &str,
    expect_error: bool,
) {
    if expect_error {
        assert!(
            !play_result.errors.is_empty(),
            "{test_name}: play result should have errors"
        );
        assert!(
            !replay_result.errors.is_empty(),
            "{test_name}: replay result should have errors"
        );
    } else {
        assert!(
            play_result.errors.is_empty(),
            "{test_name}: play result should not have errors: {:?}",
            play_result.errors
        );
        assert!(
            replay_result.errors.is_empty(),
            "{test_name}: replay result should not have errors: {:?}",
            replay_result.errors
        );
    }

    assert_eq!(
        play_result.cost.value, replay_result.cost.value,
        "{test_name}: replay cost ({}) should match play cost ({})",
        replay_result.cost.value, play_result.cost.value
    );
}

// =====================================================
// Basic System Process Tests (no replay)
// These verify the mock services work correctly
// =====================================================

#[tokio::test]
async fn test_gpt4_mock_service_works() {
    let external_services =
        create_test_external_services(OpenAIMockConfig::single_completion("gpt4 completion"), None);

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
    let external_services =
        create_test_external_services(OpenAIMockConfig::error_on_first_call(), None);

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
    let external_services =
        create_test_external_services(OpenAIMockConfig::error_on_first_call(), None);

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
    let external_services =
        create_test_external_services(OpenAIMockConfig::error_on_first_call(), None);

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

// =====================================================
// Replay Consistency Tests
// These verify play and replay produce identical costs
// (NonDeterministicProcessesSpec)
// =====================================================

#[tokio::test]
async fn replay_gpt4_produces_consistent_costs() {
    let external_services =
        create_test_external_services(OpenAIMockConfig::single_completion("gpt4 completion"), None);

    let (play, replay) = evaluate_and_replay(
        r#"new output, gpt4(`rho:ai:gpt4`) in { gpt4!("abc", *output) }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "GPT4 replay", false);
}

#[tokio::test]
async fn replay_gpt4_out_of_phlogistons_consistent_cost() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::single_completion(&"a".repeat(1_000_000)),
        None,
    );

    let (play, replay) = evaluate_and_replay(
        r#"new output, gpt4(`rho:ai:gpt4`) in { gpt4!("abc", *output) }"#,
        Cost::create(1000, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "GPT4 OutOfPhlogistons replay", true);
}

#[tokio::test]
async fn replay_gpt4_service_error_consistent() {
    let external_services =
        create_test_external_services(OpenAIMockConfig::error_on_first_call(), None);

    let (play, replay) = evaluate_and_replay(
        r#"new output, gpt4(`rho:ai:gpt4`) in { gpt4!("abc", *output) }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "GPT4 service error replay", true);
}

#[tokio::test]
async fn replay_dalle3_produces_consistent_costs() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::single_dalle3("https://example.com/generated-image.png"),
        None,
    );

    let (play, replay) = evaluate_and_replay(
        r#"new output, dalle3(`rho:ai:dalle3`) in { dalle3!("a cat painting", *output) }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "DALL-E 3 replay", false);
}

#[tokio::test]
async fn replay_dalle3_service_error_consistent() {
    let external_services =
        create_test_external_services(OpenAIMockConfig::error_on_first_call(), None);

    let (play, replay) = evaluate_and_replay(
        r#"new output, dalle3(`rho:ai:dalle3`) in { dalle3!("a cat painting", *output) }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "DALL-E 3 service error replay", true);
}

#[tokio::test]
async fn replay_tts_produces_consistent_costs() {
    let external_services = create_test_external_services(
        OpenAIMockConfig::single_tts_audio(b"fake audio bytes".to_vec()),
        None,
    );

    let (play, replay) = evaluate_and_replay(
        r#"new output, tts(`rho:ai:textToAudio`) in { tts!("Hello world", *output) }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "Text-to-audio replay", false);
}

#[tokio::test]
async fn replay_tts_service_error_consistent() {
    let external_services =
        create_test_external_services(OpenAIMockConfig::error_on_first_call(), None);

    let (play, replay) = evaluate_and_replay(
        r#"new output, tts(`rho:ai:textToAudio`) in { tts!("Hello world", *output) }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "Text-to-audio service error replay", true);
}

#[tokio::test]
async fn replay_grpc_tell_produces_consistent_costs() {
    let grpc_mock = GrpcClientMockConfig::create("localhost", 8080);
    let external_services = create_test_external_services_grpc(grpc_mock);

    let (play, replay) = evaluate_and_replay(
        r#"new grpcTell(`rho:io:grpcTell`) in { grpcTell!("localhost", 8080, "payload") }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "gRPC tell replay", false);
}

#[tokio::test]
async fn replay_grpc_tell_error_consistent() {
    let grpc_mock = GrpcClientMockConfig::create("different_host", 9999);
    let external_services = create_test_external_services_grpc(grpc_mock);

    let (play, replay) = evaluate_and_replay(
        r#"new grpcTell(`rho:io:grpcTell`) in { grpcTell!("localhost", 8080, "payload") }"#,
        Cost::create(i64::MAX, "test".to_string()),
        external_services,
    )
    .await;

    assert_replay_consistency(&play, &replay, "gRPC tell error replay", true);
}
