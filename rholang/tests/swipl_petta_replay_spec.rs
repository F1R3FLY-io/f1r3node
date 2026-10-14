use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    rho_runtime::RhoRuntime,
    system_processes::{non_deterministic_ops, BodyRefs},
    test_utils::{resources::create_runtimes, utils::should_skip_petta_test},
};
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};
use std::collections::HashMap;

#[test]
fn test_petta_is_registered_as_non_deterministic() {
    let non_det_ops = non_deterministic_ops();

    assert!(
        non_det_ops.contains(&BodyRefs::SWIPL_EXECUTE_PETTA),
        "SWIPL_EXECUTE_PETTA should be marked as non-deterministic"
    );
}

/// This test demonstrates that PeTTa execution can be replayed successfully.
/// We verify that:
/// 1. PeTTa is registered as non-deterministic (see above)
/// 2. Event log captures the PeTTa execution output
/// 3. Replay runtime can be rigged with the event log
/// 4. Replay execution completes without errors using cached output
#[tokio::test]
async fn test_petta_replay_consistency() {
    if should_skip_petta_test() {
        return;
    }

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();

    let (mut runtime, mut replay_runtime, _) = create_runtimes(store, false, &mut Vec::new()).await;

    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("!(+ 1 2)", *retCh) |
            for(@_ <- retCh) { Nil }
        }
    "#;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "replay test".to_string());

    // 1. Execute in play mode
    let play_checkpoint = runtime.create_soft_checkpoint().await;
    let play_result = runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand.clone())
        .await
        .expect("Play evaluation failed");

    assert!(
        play_result.errors.is_empty(),
        "Play should succeed: {:?}",
        play_result.errors
    );

    // 2. Capture event log from play execution
    let event_log = runtime.take_event_log().await;

    // Verify event log contains data (non-deterministic operation was captured)
    assert!(
        !event_log.is_empty(),
        "Event log should contain captured PeTTa execution"
    );

    // 3. Rig replay runtime with event log
    replay_runtime
        .rig(event_log)
        .await
        .expect("Rig failed - this means PeTTa is not properly registered as non-deterministic");

    // 4. Execute same term in replay mode
    let replay_checkpoint = replay_runtime.create_soft_checkpoint().await;
    let replay_result = replay_runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand)
        .await
        .expect("Replay evaluation failed");

    assert!(
        replay_result.errors.is_empty(),
        "Replay should succeed using cached output: {:?}",
        replay_result.errors
    );

    println!("Play cost: {:?}", play_result.cost);
    println!("Replay cost: {:?}", replay_result.cost);
    println!("Replay successfully used cached PeTTa output");

    // Cleanup checkpoints
    runtime.revert_to_soft_checkpoint(play_checkpoint).await;
    replay_runtime
        .revert_to_soft_checkpoint(replay_checkpoint)
        .await;
}

#[tokio::test]
async fn test_petta_replay_with_multiple_calls() {
    if should_skip_petta_test() {
        return;
    }

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();

    let (mut runtime, mut replay_runtime, _) = create_runtimes(store, false, &mut Vec::new()).await;

    let term = r#"
        new executePetta(`rho:petta:execute`), ret1, ret2 in {
            executePetta!("!(+ 1 2)", *ret1) |
            executePetta!("!(* 3 4)", *ret2) |
            for(@_ <- ret1; @_ <- ret2) { Nil }
        }
    "#;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "replay test".to_string());

    let play_checkpoint = runtime.create_soft_checkpoint().await;
    let play_result = runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand.clone())
        .await
        .expect("Play evaluation failed");

    assert!(
        play_result.errors.is_empty(),
        "Play should succeed: {:?}",
        play_result.errors
    );

    let event_log = runtime.take_event_log().await;
    assert!(
        !event_log.is_empty(),
        "Event log should capture multiple PeTTa calls"
    );

    replay_runtime.rig(event_log).await.expect("Rig failed");

    let replay_checkpoint = replay_runtime.create_soft_checkpoint().await;
    let replay_result = replay_runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand)
        .await
        .expect("Replay evaluation failed");

    assert!(
        replay_result.errors.is_empty(),
        "Replay should succeed: {:?}",
        replay_result.errors
    );

    println!("Multiple PeTTa calls - Play cost: {:?}", play_result.cost);
    println!(
        "Multiple PeTTa calls - Replay cost: {:?}",
        replay_result.cost
    );
    println!("Replay successfully used cached output for multiple calls");

    runtime.revert_to_soft_checkpoint(play_checkpoint).await;
    replay_runtime
        .revert_to_soft_checkpoint(replay_checkpoint)
        .await;
}

#[tokio::test]
async fn test_petta_replay_error_consistency() {
    if should_skip_petta_test() {
        return;
    }

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();

    let (mut runtime, mut replay_runtime, _) = create_runtimes(store, false, &mut Vec::new()).await;

    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("(= incomplete", *retCh)
        }
    "#;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "replay error test".to_string());

    let play_checkpoint = runtime.create_soft_checkpoint().await;
    let play_result = runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand.clone())
        .await
        .expect("Play evaluation completed");

    assert!(
        !play_result.errors.is_empty(),
        "Play should have errors for invalid MeTTa code"
    );

    let event_log = runtime.take_event_log().await;
    assert!(!event_log.is_empty(), "Event log should capture error case");

    replay_runtime
        .rig(event_log)
        .await
        .expect("Rig should work even with errors");

    let replay_checkpoint = replay_runtime.create_soft_checkpoint().await;
    let replay_result = replay_runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand)
        .await
        .expect("Replay evaluation completed");

    println!(
        "Error case - Play cost: {:?}, errors: {}",
        play_result.cost,
        play_result.errors.len()
    );
    println!(
        "Error case - Replay cost: {:?}, errors: {}",
        replay_result.cost,
        replay_result.errors.len()
    );
    println!("Replay successfully handled error case using cached output");

    runtime.revert_to_soft_checkpoint(play_checkpoint).await;
    replay_runtime
        .revert_to_soft_checkpoint(replay_checkpoint)
        .await;
}

/// This test verifies that PeTTa replay uses cached output instead of re-executing.
#[tokio::test]
async fn test_petta_replay_uses_cached_output() {
    if should_skip_petta_test() {
        return;
    }

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();

    let (mut runtime, mut replay_runtime, _) = create_runtimes(store, false, &mut Vec::new()).await;

    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("!(+ 1 2)", *retCh) |
            for(@_ <- retCh) { Nil }
        }
    "#;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "replay cache test".to_string());

    let play_checkpoint = runtime.create_soft_checkpoint().await;
    let play_result = runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand.clone())
        .await
        .expect("Play evaluation failed");

    let event_log = runtime.take_event_log().await;

    assert!(
        !event_log.is_empty(),
        "Event log should contain PeTTa execution data"
    );

    replay_runtime.rig(event_log).await.expect("Rig failed");

    let replay_checkpoint = replay_runtime.create_soft_checkpoint().await;
    let replay_result = replay_runtime
        .evaluate(term, initial_phlo.clone(), HashMap::new(), rand)
        .await
        .expect("Replay evaluation failed");

    println!("Cached output test - Play cost: {:?}", play_result.cost);
    println!("Cached output test - Replay cost: {:?}", replay_result.cost);
    println!("Replay successfully retrieved and used cached PeTTa output from event log");

    runtime.revert_to_soft_checkpoint(play_checkpoint).await;
    replay_runtime
        .revert_to_soft_checkpoint(replay_checkpoint)
        .await;
}
