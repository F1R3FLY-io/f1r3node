use crate::genesis::contracts::test_util::TestUtil;
use crate::util::rholang::resources::{generate_scope_id, mk_test_rnode_store_manager_shared};
use casper::rust::genesis::genesis::Genesis;
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom};
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::rho_runtime::create_runtime_from_kv_store;
use rspace_plus_plus::rspace::r#match::Match;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

async fn eval_rholang_code(code: &str, timeout: Duration) -> Result<(), String> {
    let scope_id = generate_scope_id();
    let mut kvs_manager = mk_test_rnode_store_manager_shared(scope_id);
    let r_store = kvs_manager
        .r_space_stores()
        .await
        .map_err(|e| format!("Failed to create RSpaceStore: {}", e))?;

    let matcher =
        Arc::new(Box::new(Matcher::default()) as Box<dyn Match<BindPattern, ListParWithRandom>>);

    let runtime = create_runtime_from_kv_store(
        r_store,
        std::sync::Arc::new(Genesis::default_mergeable_tags()),
        true,
        &mut vec![],
        matcher,
        rholang::rust::interpreter::external_services::ExternalServices::noop(),
    )
    .await;

    let rand = Blake2b512Random::create_from_length(128);

    match tokio::time::timeout(
        timeout,
        TestUtil::eval(code, &runtime, HashMap::new(), rand),
    )
    .await
    {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("Interpreter error: {:?}", e)),
        Err(_) => Err(format!("Timeout of {:?} expired", timeout)),
    }
}

/// Regression test for https://github.com/F1R3FLY-io/f1r3node/issues/305
///
/// shortslow.rho: direct recursive contract that calls itself 32768 times.
/// Without StackGrowingFuture, this causes stack overflow in debug builds.
#[tokio::test]
async fn deep_recursion_shortslow_should_not_stackoverflow() {
    let code =
        CompiledRholangSource::load_source("shortslow.rho").expect("Failed to load shortslow.rho");

    let result = eval_rholang_code(&code, Duration::from_secs(300)).await;
    assert!(
        result.is_ok(),
        "shortslow deep recursion failed: {:?}",
        result.err()
    );
}

/// Regression test for https://github.com/F1R3FLY-io/f1r3node/issues/306
///
/// longslow.rho: sends a 32768-char string to a channel, reads its length,
/// then recurses that many times. This exercises produce/consume + string ops
/// in addition to deep recursion, matching the exact integration test scenario.
#[tokio::test]
async fn deep_recursion_longslow_should_not_stackoverflow() {
    let code =
        CompiledRholangSource::load_source("longslow.rho").expect("Failed to load longslow.rho");

    let result = eval_rholang_code(&code, Duration::from_secs(300)).await;
    assert!(
        result.is_ok(),
        "longslow deep recursion failed: {:?}",
        result.err()
    );
}
