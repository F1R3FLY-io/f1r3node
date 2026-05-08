/// Performance benchmark for PoS cost accounting system deploys.
///
/// Measures the time taken by PreChargeDeploy and RefundDeploy in isolation
/// and counts RSpace produce/consume operations to quantify the overhead.
///
/// Run with: cargo test -p casper --test mod cost_accounting_perf -- --nocapture
/// Release:  cargo test -p casper --test mod --release cost_accounting_perf -- --nocapture
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use casper::rust::{
    errors::CasperError,
    rholang::{replay_runtime::ReplayRuntimeOps, runtime::RuntimeOps},
    util::{
        construct_deploy,
        rholang::{
            costacc::pre_charge_deploy::PreChargeDeploy, costacc::refund_deploy::RefundDeploy,
            runtime_manager::RuntimeManager, system_deploy::SystemDeployTrait,
            system_deploy_result::SystemDeployResult, system_deploy_util,
        },
    },
};
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
use models::{
    rhoapi::Par,
    rust::{block::state_hash::StateHash, casper::protocol::casper_message::ProcessedDeploy},
};
use rholang::rust::interpreter::{rho_runtime::RhoRuntime, system_processes::BlockData};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, merger::merging_logic::MergeType,
};

use crate::util::{genesis_builder::GenesisContext, rholang::resources::with_runtime_manager};

static METRICS_INIT: OnceLock<Snapshotter> = OnceLock::new();

fn get_snapshotter() -> &'static Snapshotter {
    METRICS_INIT.get_or_init(|| {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _ = metrics::set_global_recorder(recorder);
        snapshotter
    })
}

fn get_histogram_stats(snapshotter: &Snapshotter, metric_name: &str) -> (usize, f64) {
    let snapshot = snapshotter.snapshot();
    let metrics = snapshot.into_hashmap();
    let mut count = 0usize;
    let mut sum = 0.0f64;
    for (key, (_, _, value)) in metrics.iter() {
        if key.key().name() == metric_name {
            if let metrics_util::debugging::DebugValue::Histogram(samples) = value {
                for sample in samples {
                    count += 1;
                    sum += sample.into_inner();
                }
            }
        }
    }
    (count, sum)
}

fn get_counter_value(snapshotter: &Snapshotter, metric_name: &str) -> u64 {
    let snapshot = snapshotter.snapshot();
    let metrics = snapshot.into_hashmap();
    let mut total = 0u64;
    for (key, (_, _, value)) in metrics.iter() {
        let key_name = key.key().name();
        if key_name == metric_name {
            if let metrics_util::debugging::DebugValue::Counter(c) = value {
                total += c;
            }
        }
    }
    total
}

struct RSpaceCallCounts {
    produce: u64,
    consume: u64,
    install: u64,
    comm_produce: u64,
    comm_consume: u64,
}

fn snapshot_rspace_counts(snapshotter: &Snapshotter) -> RSpaceCallCounts {
    RSpaceCallCounts {
        produce: get_counter_value(snapshotter, "rspace.produce.calls"),
        consume: get_counter_value(snapshotter, "rspace.consume.calls"),
        install: get_counter_value(snapshotter, "rspace.install.calls"),
        comm_produce: get_counter_value(snapshotter, "comm.produce"),
        comm_consume: get_counter_value(snapshotter, "comm.consume"),
    }
}

fn diff_counts(before: &RSpaceCallCounts, after: &RSpaceCallCounts) -> RSpaceCallCounts {
    RSpaceCallCounts {
        produce: after.produce.saturating_sub(before.produce),
        consume: after.consume.saturating_sub(before.consume),
        install: after.install.saturating_sub(before.install),
        comm_produce: after.comm_produce.saturating_sub(before.comm_produce),
        comm_consume: after.comm_consume.saturating_sub(before.comm_consume),
    }
}

async fn play_system_deploy_timed<S: SystemDeployTrait>(
    runtime_manager: &mut RuntimeManager,
    genesis_context: &GenesisContext,
    start_state: &StateHash,
    system_deploy: &mut S,
) -> Result<(StateHash, std::time::Duration), CasperError> {
    let runtime = runtime_manager.spawn_runtime().await;
    runtime
        .set_block_data(BlockData {
            time_stamp: 0,
            block_number: 0,
            sender: genesis_context.validator_pks()[0].clone(),
            seq_num: 0,
        })
        .await;

    let mut runtime_ops = RuntimeOps::new(runtime);
    let start = Instant::now();
    let result = runtime_ops
        .play_system_deploy(start_state, system_deploy)
        .await?;
    let elapsed = start.elapsed();

    match result {
        SystemDeployResult::PlaySucceeded { state_hash, .. } => Ok((state_hash, elapsed)),
        SystemDeployResult::PlayFailed {
            processed_system_deploy,
            ..
        } => Err(CasperError::RuntimeError(format!(
            "System deploy failed: {:?}",
            processed_system_deploy
        ))),
    }
}

// Run manually: cargo test -p casper --release --test mod cost_accounting_perf -- --nocapture --include-ignored
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn measure_precharge_and_refund_cost() {
    const ITERATIONS: usize = 5;

    let snapshotter = get_snapshotter();

    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let user_pk = casper::rust::util::construct_deploy::DEFAULT_PUB.clone();
            let genesis_state = genesis_block.body.state.post_state_hash.clone();

            let mut precharge_times = Vec::new();
            let mut refund_times = Vec::new();
            let mut precharge_ops = Vec::new();
            let mut refund_ops = Vec::new();
            let mut current_state = genesis_state.clone();

            for i in 0..ITERATIONS {
                // PreChargeDeploy
                let before = snapshot_rspace_counts(snapshotter);
                let mut precharge = PreChargeDeploy {
                    charge_amount: 100_000,
                    pk: user_pk.clone(),
                    rand: Blake2b512Random::create_from_bytes(&vec![i as u8, 0]),
                };
                let (state_after_charge, precharge_time) = play_system_deploy_timed(
                    &mut runtime_manager,
                    &genesis_context,
                    &current_state,
                    &mut precharge,
                )
                .await
                .expect("PreChargeDeploy failed");
                let after = snapshot_rspace_counts(snapshotter);
                let delta = diff_counts(&before, &after);
                precharge_times.push(precharge_time);
                precharge_ops.push(delta);

                // RefundDeploy
                let before = snapshot_rspace_counts(snapshotter);
                let mut refund = RefundDeploy {
                    refund_amount: 50_000,
                    rand: Blake2b512Random::create_from_bytes(&vec![i as u8, 1]),
                };
                let (state_after_refund, refund_time) = play_system_deploy_timed(
                    &mut runtime_manager,
                    &genesis_context,
                    &state_after_charge,
                    &mut refund,
                )
                .await
                .expect("RefundDeploy failed");
                let after = snapshot_rspace_counts(snapshotter);
                let delta = diff_counts(&before, &after);
                refund_times.push(refund_time);
                refund_ops.push(delta);

                current_state = state_after_refund;
            }

            // Report timing results
            let precharge_avg_ms: f64 = precharge_times
                .iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .sum::<f64>()
                / ITERATIONS as f64;
            let refund_avg_ms: f64 = refund_times
                .iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .sum::<f64>()
                / ITERATIONS as f64;

            println!("\n================================================================");
            println!("Cost Accounting Performance ({} iterations, play path)", ITERATIONS);
            println!("================================================================");
            println!(
                "PreChargeDeploy: avg={:.1}ms, min={:.1}ms, max={:.1}ms",
                precharge_avg_ms,
                precharge_times.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::MAX, f64::min),
                precharge_times.iter().map(|d| d.as_secs_f64() * 1000.0).fold(0.0f64, f64::max),
            );
            println!(
                "RefundDeploy:    avg={:.1}ms, min={:.1}ms, max={:.1}ms",
                refund_avg_ms,
                refund_times.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::MAX, f64::min),
                refund_times.iter().map(|d| d.as_secs_f64() * 1000.0).fold(0.0f64, f64::max),
            );
            let total_per_deploy_ms = precharge_avg_ms + refund_avg_ms;
            println!(
                "Total per deploy: avg={:.1}ms",
                total_per_deploy_ms
            );

            assert!(
                total_per_deploy_ms < 300.0,
                "Performance regression: {:.1}ms per deploy exceeds 300ms threshold",
                total_per_deploy_ms
            );

            // Report RSpace operation counts
            println!("\n--- RSpace Operations Per PreChargeDeploy ---");
            for (i, ops) in precharge_ops.iter().enumerate() {
                println!(
                    "  iter {}: produce={}, consume={}, install={}, comm_produce={}, comm_consume={} (total={})",
                    i, ops.produce, ops.consume, ops.install, ops.comm_produce, ops.comm_consume,
                    ops.produce + ops.consume + ops.install,
                );
            }

            println!("\n--- RSpace Operations Per RefundDeploy ---");
            for (i, ops) in refund_ops.iter().enumerate() {
                println!(
                    "  iter {}: produce={}, consume={}, install={}, comm_produce={}, comm_consume={} (total={})",
                    i, ops.produce, ops.consume, ops.install, ops.comm_produce, ops.comm_consume,
                    ops.produce + ops.consume + ops.install,
                );
            }

            // Averages
            let avg = |ops: &[RSpaceCallCounts], f: fn(&RSpaceCallCounts) -> u64| -> f64 {
                ops.iter().map(|o| f(o) as f64).sum::<f64>() / ops.len() as f64
            };
            let avg_pre_produce = avg(&precharge_ops, |o| o.produce);
            let avg_pre_consume = avg(&precharge_ops, |o| o.consume);
            let avg_pre_install = avg(&precharge_ops, |o| o.install);
            let avg_ref_produce = avg(&refund_ops, |o| o.produce);
            let avg_ref_consume = avg(&refund_ops, |o| o.consume);
            let avg_ref_install = avg(&refund_ops, |o| o.install);

            let total_pre_ops = avg_pre_produce + avg_pre_consume + avg_pre_install;
            let total_ref_ops = avg_ref_produce + avg_ref_consume + avg_ref_install;
            let total_ops = total_pre_ops + total_ref_ops;
            let total_ms = precharge_avg_ms + refund_avg_ms;

            println!("\n--- Summary ---");
            println!("PreCharge: avg {:.0} produce + {:.0} consume + {:.0} install = {:.0} ops in {:.1}ms ({:.2}ms/op)",
                avg_pre_produce, avg_pre_consume, avg_pre_install,
                total_pre_ops, precharge_avg_ms,
                precharge_avg_ms / total_pre_ops.max(1.0),
            );
            println!("Refund:    avg {:.0} produce + {:.0} consume + {:.0} install = {:.0} ops in {:.1}ms ({:.2}ms/op)",
                avg_ref_produce, avg_ref_consume, avg_ref_install,
                total_ref_ops, refund_avg_ms,
                refund_avg_ms / total_ref_ops.max(1.0),
            );
            println!("Total:     {:.0} ops in {:.1}ms ({:.2}ms/op)",
                total_ops, total_ms, total_ms / total_ops.max(1.0),
            );
            println!("================================================================\n");

            // Sub-operation breakdown
            let produce_calls = get_counter_value(snapshotter, "rspace.produce.calls");
            let consume_calls = get_counter_value(snapshotter, "rspace.consume.calls");

            if produce_calls > 0 {
                let get_joins = get_counter_value(snapshotter, "rspace.produce.get_joins_ns");
                let log = get_counter_value(snapshotter, "rspace.produce.log_ns");
                let extract = get_counter_value(snapshotter, "rspace.produce.extract_candidate_ns");
                let match_found = get_counter_value(snapshotter, "rspace.produce.process_match_ns");
                let store = get_counter_value(snapshotter, "rspace.produce.store_data_ns");
                let total_ns = get_joins + log + extract + match_found + store;

                println!("--- Produce Sub-Op Breakdown ({} calls, total {:.1}ms) ---",
                    produce_calls, total_ns as f64 / 1_000_000.0);
                println!("  get_joins:         {:.3}ms avg ({:.1}ms total, {:.0}% of produce)",
                    get_joins as f64 / 1_000_000.0 / produce_calls as f64,
                    get_joins as f64 / 1_000_000.0,
                    get_joins as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  log_produce:       {:.3}ms avg ({:.1}ms total, {:.0}% of produce)",
                    log as f64 / 1_000_000.0 / produce_calls as f64,
                    log as f64 / 1_000_000.0,
                    log as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  extract_candidate: {:.3}ms avg ({:.1}ms total, {:.0}% of produce)",
                    extract as f64 / 1_000_000.0 / produce_calls as f64,
                    extract as f64 / 1_000_000.0,
                    extract as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  process_match:     {:.3}ms avg ({:.1}ms total, {:.0}% of produce)",
                    match_found as f64 / 1_000_000.0 / produce_calls as f64,
                    match_found as f64 / 1_000_000.0,
                    match_found as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  store_data:        {:.3}ms avg ({:.1}ms total, {:.0}% of produce)",
                    store as f64 / 1_000_000.0 / produce_calls as f64,
                    store as f64 / 1_000_000.0,
                    store as f64 / total_ns.max(1) as f64 * 100.0);
            }

            // Matcher sub-breakdown (inside extract_produce_candidate)
            let fetch_cont = get_counter_value(snapshotter, "rspace.matcher.fetch_continuations_ns");
            let fetch_data_m = get_counter_value(snapshotter, "rspace.matcher.fetch_data_ns");
            let extract_match = get_counter_value(snapshotter, "rspace.matcher.extract_first_match_ns");
            let matcher_total = fetch_cont + fetch_data_m + extract_match;
            if matcher_total > 0 {
                println!("\n--- Matcher Sub-Breakdown (inside extract_candidate, total {:.1}ms) ---",
                    matcher_total as f64 / 1_000_000.0);
                println!("  fetch_continuations: {:.1}ms ({:.0}%)",
                    fetch_cont as f64 / 1_000_000.0,
                    fetch_cont as f64 / matcher_total.max(1) as f64 * 100.0);
                println!("  fetch_data:          {:.1}ms ({:.0}%)",
                    fetch_data_m as f64 / 1_000_000.0,
                    fetch_data_m as f64 / matcher_total.max(1) as f64 * 100.0);
                println!("  extract_first_match: {:.1}ms ({:.0}%)",
                    extract_match as f64 / 1_000_000.0,
                    extract_match as f64 / matcher_total.max(1) as f64 * 100.0);
            }

            // Additional matcher stats
            let matcher_get_calls = get_counter_value(snapshotter, "rspace.matcher.get_calls");
            let conts_returned = get_counter_value(snapshotter, "rspace.matcher.continuations_returned");
            if matcher_get_calls > 0 || conts_returned > 0 {
                println!("  matcher.get calls:        {} (avg {:.1} per produce)",
                    matcher_get_calls, matcher_get_calls as f64 / produce_calls.max(1) as f64);
                println!("  continuations returned:   {} (avg {:.1} per fetch)",
                    conts_returned, conts_returned as f64 / produce_calls.max(1) as f64);
                let clone_ns = get_counter_value(snapshotter, "rspace.matcher.clone_ns");
                let fold_match_ns = get_counter_value(snapshotter, "rspace.matcher.fold_match_ns");
                if matcher_get_calls > 0 {
                    println!("  avg time per matcher.get: {:.3}ms",
                        extract_match as f64 / 1_000_000.0 / matcher_get_calls as f64);
                    println!("    clone (pattern+data): {:.3}ms avg ({:.1}ms total, {:.0}% of match time)",
                        clone_ns as f64 / 1_000_000.0 / matcher_get_calls as f64,
                        clone_ns as f64 / 1_000_000.0,
                        clone_ns as f64 / (clone_ns + fold_match_ns).max(1) as f64 * 100.0);
                    println!("    fold_match (algo):    {:.3}ms avg ({:.1}ms total, {:.0}% of match time)",
                        fold_match_ns as f64 / 1_000_000.0 / matcher_get_calls as f64,
                        fold_match_ns as f64 / 1_000_000.0,
                        fold_match_ns as f64 / (clone_ns + fold_match_ns).max(1) as f64 * 100.0);
                }
            }

            if consume_calls > 0 {
                let log = get_counter_value(snapshotter, "rspace.consume.log_ns");
                let fetch = get_counter_value(snapshotter, "rspace.consume.fetch_data_ns");
                let matching = get_counter_value(snapshotter, "rspace.consume.match_ns");
                let match_found = get_counter_value(snapshotter, "rspace.consume.process_match_ns");
                let store_cont = get_counter_value(snapshotter, "rspace.consume.store_continuation_ns");
                let total_ns = log + fetch + matching + match_found + store_cont;

                println!("\n--- Consume Sub-Op Breakdown ({} calls, total {:.1}ms) ---",
                    consume_calls, total_ns as f64 / 1_000_000.0);
                println!("  log_consume:       {:.3}ms avg ({:.1}ms total, {:.0}% of consume)",
                    log as f64 / 1_000_000.0 / consume_calls as f64,
                    log as f64 / 1_000_000.0,
                    log as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  fetch_data:        {:.3}ms avg ({:.1}ms total, {:.0}% of consume)",
                    fetch as f64 / 1_000_000.0 / consume_calls as f64,
                    fetch as f64 / 1_000_000.0,
                    fetch as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  match_patterns:    {:.3}ms avg ({:.1}ms total, {:.0}% of consume)",
                    matching as f64 / 1_000_000.0 / consume_calls as f64,
                    matching as f64 / 1_000_000.0,
                    matching as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  process_match:     {:.3}ms avg ({:.1}ms total, {:.0}% of consume)",
                    match_found as f64 / 1_000_000.0 / consume_calls as f64,
                    match_found as f64 / 1_000_000.0,
                    match_found as f64 / total_ns.max(1) as f64 * 100.0);
                println!("  store_continuation:{:.3}ms avg ({:.1}ms total, {:.0}% of consume)",
                    store_cont as f64 / 1_000_000.0 / consume_calls as f64,
                    store_cont as f64 / 1_000_000.0,
                    store_cont as f64 / total_ns.max(1) as f64 * 100.0);
            }

            // Lock acquisition and spawn overhead
            let (_produce_lock_count, _produce_lock_total) = get_histogram_stats(snapshotter, "rspace.produce.lock_acquire_seconds");
            let (_consume_lock_count, _consume_lock_total) = get_histogram_stats(snapshotter, "rspace.consume.lock_acquire_seconds");
            let (_spawn_count, _spawn_total) = get_histogram_stats(snapshotter, "reducer.eval_par.spawn_seconds");
            let (_join_count, _join_total) = get_histogram_stats(snapshotter, "reducer.eval_par.join_seconds");
            let (_eval_par_count, _) = get_histogram_stats(snapshotter, "reducer.eval_par.term_count");
            let _eval_par_calls = get_counter_value(snapshotter, "reducer.eval_par.calls");

            // Debug: list all histogram metrics
            {
                let snapshot = snapshotter.snapshot();
                let metrics = snapshot.into_hashmap();
                let mut hist_names: Vec<String> = Vec::new();
                for (key, (_, _, value)) in metrics.iter() {
                    if let metrics_util::debugging::DebugValue::Histogram(samples) = value {
                        if !samples.is_empty() {
                            hist_names.push(format!("  {} ({} samples)", key.key().name(), samples.len()));
                        }
                    }
                }
                hist_names.sort();
                println!("\n--- All Histogram Metrics ({} with data) ---", hist_names.len());
                for name in &hist_names {
                    println!("{}", name);
                }
            }

            let produce_lock_ns = get_counter_value(snapshotter, "rspace.produce.lock_acquire_ns");
            let consume_lock_ns = get_counter_value(snapshotter, "rspace.consume.lock_acquire_ns");
            let spawn_ns = get_counter_value(snapshotter, "reducer.eval_par.spawn_ns");
            let join_ns = get_counter_value(snapshotter, "reducer.eval_par.join_ns");
            let eval_par_calls = get_counter_value(snapshotter, "reducer.eval_par.calls");
            let eval_par_terms = get_counter_value(snapshotter, "reducer.eval_par.term_count");
            let total_lock_ms = (produce_lock_ns + consume_lock_ns) as f64 / 1_000_000.0;

            println!("\n--- Lock & Spawn Overhead ---");
            println!("  Produce lock acquire: {:.1}ms total ({:.3}ms avg over {} calls)",
                produce_lock_ns as f64 / 1_000_000.0,
                if produce_calls > 0 { produce_lock_ns as f64 / 1_000_000.0 / produce_calls as f64 } else { 0.0 },
                produce_calls);
            println!("  Consume lock acquire: {:.1}ms total ({:.3}ms avg over {} calls)",
                consume_lock_ns as f64 / 1_000_000.0,
                if consume_calls > 0 { consume_lock_ns as f64 / 1_000_000.0 / consume_calls as f64 } else { 0.0 },
                consume_calls);
            println!("  Total lock overhead:  {:.1}ms ({:.1}% of {:.1}ms total)",
                total_lock_ms, total_lock_ms / total_ms * 100.0, total_ms);
            println!("  eval(Par) calls: {}, terms: {}, spawn: {:.1}ms, join: {:.1}ms",
                eval_par_calls, eval_par_terms,
                spawn_ns as f64 / 1_000_000.0, join_ns as f64 / 1_000_000.0);
        },
    )
    .await
    .unwrap()
}

struct ReplayPhaseTimings {
    rig: std::time::Duration,
    precharge: std::time::Duration,
    evaluate: std::time::Duration,
    refund: std::time::Duration,
    check_replay: std::time::Duration,
    total: std::time::Duration,
}

async fn time_replay_per_deploy(
    runtime_manager: &mut RuntimeManager,
    genesis_context: &GenesisContext,
    start_state: &StateHash,
    processed_deploy: &ProcessedDeploy,
) -> Result<ReplayPhaseTimings, CasperError> {
    let replay_runtime = runtime_manager.spawn_replay_runtime().await;
    replay_runtime
        .set_block_data(BlockData {
            time_stamp: processed_deploy.deploy.data.time_stamp,
            block_number: 0,
            sender: genesis_context.validator_pks()[0].clone(),
            seq_num: 0,
        })
        .await;

    let mut replay_ops = ReplayRuntimeOps::new_from_runtime(replay_runtime);
    replay_ops
        .runtime_ops
        .runtime
        .reset(&Blake2b256Hash::from_bytes_prost(start_state))
        .await?;

    time_replay_one_deploy(&mut replay_ops, processed_deploy).await
}

async fn time_replay_one_deploy(
    replay_ops: &mut ReplayRuntimeOps,
    processed_deploy: &ProcessedDeploy,
) -> Result<ReplayPhaseTimings, CasperError> {
    let total_start = Instant::now();

    let rig_start = Instant::now();
    replay_ops.rig(processed_deploy).await?;
    let rig = rig_start.elapsed();

    let mut mergeable_channels: HashMap<Par, MergeType> = HashMap::new();

    // Precharge — matches process_deploy_with_cost_accounting precharge window
    // (BLOCK_REPLAY_DEPLOY_PRECHARGE_TIME_METRIC).
    let precharge_start = Instant::now();
    let mut precharge = PreChargeDeploy {
        charge_amount: processed_deploy.deploy.data.total_phlo_charge(),
        pk: processed_deploy.deploy.pk.clone(),
        rand: system_deploy_util::generate_pre_charge_deploy_random_seed(&processed_deploy.deploy),
    };
    let (_, mut precharge_eval) = replay_ops
        .replay_system_deploy_internal(&mut precharge, &processed_deploy.system_deploy_error)
        .await?;
    replay_ops.discard_event_log("precharge", false).await;
    if precharge_eval.errors.is_empty() {
        mergeable_channels.extend(precharge_eval.mergeable.drain());
    }
    let precharge = precharge_start.elapsed();

    // User-deploy evaluate — matches BLOCK_REPLAY_DEPLOY_EVALUATE_TIME_METRIC.
    let evaluate_start = Instant::now();
    let (_, eval_successful) = replay_ops
        .run_user_deploy(processed_deploy, &mut mergeable_channels)
        .await?;
    let evaluate = evaluate_start.elapsed();

    // Refund — matches BLOCK_REPLAY_DEPLOY_REFUND_TIME_METRIC.
    let refund_start = Instant::now();
    let mut refund = RefundDeploy {
        refund_amount: processed_deploy.refund_amount(),
        rand: system_deploy_util::generate_refund_deploy_random_seed(&processed_deploy.deploy),
    };
    let (_, mut refund_eval) = replay_ops
        .replay_system_deploy_internal(&mut refund, &None)
        .await?;
    replay_ops.discard_event_log("refund", false).await;
    if refund_eval.errors.is_empty() {
        mergeable_channels.extend(refund_eval.mergeable.drain());
    }
    let refund = refund_start.elapsed();

    let check_start = Instant::now();
    replay_ops
        .check_replay_data_with_fix(eval_successful)
        .await?;
    let check_replay = check_start.elapsed();

    let total = total_start.elapsed();

    Ok(ReplayPhaseTimings {
        rig,
        precharge,
        evaluate,
        refund,
        check_replay,
        total,
    })
}

// Run manually: cargo test -p casper --release --test mod cost_accounting_perf -- --nocapture --include-ignored
//
// Mirrors the per-deploy validator wallclock measured in production via the
// `block.replay.deploy.{rig,precharge,evaluate,refund,check_replay_data}.time`
// histograms scraped by integration test_load. Records the same windows by
// driving the public ReplayRuntimeOps surface directly.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn measure_replay_per_deploy_cost() {
    const ITERATIONS: usize = 5;

    let snapshotter = get_snapshotter();

    with_runtime_manager(
        |mut runtime_manager, genesis_context, genesis_block| async move {
            let user_sec = construct_deploy::DEFAULT_SEC.clone();
            let genesis_state = genesis_block.body.state.post_state_hash.clone();

            // Mirrors the simplest user deploy shape that still triggers the
            // full precharge → user-eval → refund pipeline (same shape as
            // comput_state_should_charge_for_deploys).
            let source = "Nil".to_string();

            let mut current_state = genesis_state;
            let mut timings: Vec<ReplayPhaseTimings> = Vec::new();

            for i in 0..ITERATIONS {
                let deploy = construct_deploy::source_deploy_now_full(
                    source.clone(),
                    Some(100_000),
                    None,
                    Some(user_sec.clone()),
                    Some(i as i64),
                    None,
                )
                .unwrap();

                let block_data = BlockData {
                    time_stamp: deploy.data.time_stamp,
                    block_number: 0,
                    sender: genesis_context.validator_pks()[0].clone(),
                    seq_num: 0,
                };
                let (post_state, mut processed_deploys, _) = runtime_manager
                    .compute_state(
                        &current_state,
                        vec![deploy],
                        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
                        block_data,
                        None,
                    )
                    .await
                    .expect("compute_state failed");

                let processed_deploy = processed_deploys.pop().expect("no processed deploy");
                assert!(
                    !processed_deploy.is_failed,
                    "deploy must not fail (iter {})",
                    i
                );

                let phase = time_replay_per_deploy(
                    &mut runtime_manager,
                    &genesis_context,
                    &current_state,
                    &processed_deploy,
                )
                .await
                .expect("replay timing failed");

                timings.push(phase);
                current_state = post_state;
            }

            let avg_ms = |get: fn(&ReplayPhaseTimings) -> std::time::Duration| -> f64 {
                timings.iter().map(|t| get(t).as_secs_f64() * 1000.0).sum::<f64>()
                    / ITERATIONS as f64
            };

            let rig_ms = avg_ms(|t| t.rig);
            let precharge_ms = avg_ms(|t| t.precharge);
            let evaluate_ms = avg_ms(|t| t.evaluate);
            let refund_ms = avg_ms(|t| t.refund);
            let check_ms = avg_ms(|t| t.check_replay);
            let total_ms = avg_ms(|t| t.total);

            println!("\n================================================================");
            println!(
                "Replay Per-Deploy Phase Breakdown ({} iterations, replay path)",
                ITERATIONS
            );
            println!("================================================================");
            println!("rig:           avg={:.1}ms", rig_ms);
            println!(
                "precharge:     avg={:.1}ms  (production high-phase target ~229ms)",
                precharge_ms
            );
            println!(
                "evaluate:      avg={:.1}ms  (production high-phase target  ~32ms)",
                evaluate_ms
            );
            println!(
                "refund:        avg={:.1}ms  (production high-phase target ~189ms)",
                refund_ms
            );
            println!("check_replay:  avg={:.1}ms", check_ms);
            println!(
                "total:         avg={:.1}ms  (production high-phase target ~450ms)",
                total_ms
            );

            // Per-iteration detail
            println!("\n--- Per-Iteration ---");
            for (i, t) in timings.iter().enumerate() {
                println!(
                    "  iter {}: rig={:.1}ms precharge={:.1}ms eval={:.1}ms refund={:.1}ms check={:.1}ms total={:.1}ms",
                    i,
                    t.rig.as_secs_f64() * 1000.0,
                    t.precharge.as_secs_f64() * 1000.0,
                    t.evaluate.as_secs_f64() * 1000.0,
                    t.refund.as_secs_f64() * 1000.0,
                    t.check_replay.as_secs_f64() * 1000.0,
                    t.total.as_secs_f64() * 1000.0,
                );
            }
            println!("================================================================\n");

            // Reducer per-op-type breakdown — splits reduce_term by AST node kind.
            let send_calls = get_counter_value(snapshotter, "reducer.eval_send.calls");
            let send_ns = get_counter_value(snapshotter, "reducer.eval_send.time_ns");
            let recv_calls = get_counter_value(snapshotter, "reducer.eval_receive.calls");
            let recv_ns = get_counter_value(snapshotter, "reducer.eval_receive.time_ns");
            let new_calls = get_counter_value(snapshotter, "reducer.eval_new.calls");
            let new_ns = get_counter_value(snapshotter, "reducer.eval_new.time_ns");
            let match_calls = get_counter_value(snapshotter, "reducer.eval_match.calls");
            let match_ns = get_counter_value(snapshotter, "reducer.eval_match.time_ns");
            let reducer_ns = send_ns + recv_ns + new_ns + match_ns;
            let reducer_calls = send_calls + recv_calls + new_calls + match_calls;

            println!("--- Reducer Per-Op-Type Breakdown (cumulative across {} iterations × full pipeline) ---", ITERATIONS);
            println!("  total dispatched calls: {} ({} send + {} receive + {} new + {} match)",
                reducer_calls, send_calls, recv_calls, new_calls, match_calls);
            let pct = |ns: u64| -> f64 {
                if reducer_ns == 0 { 0.0 } else { ns as f64 / reducer_ns as f64 * 100.0 }
            };
            let avg_ms = |ns: u64, calls: u64| -> f64 {
                if calls == 0 { 0.0 } else { ns as f64 / 1_000_000.0 / calls as f64 }
            };
            println!("  eval_send:    {:>9.1}ms total ({:>5.0}%) over {:>6} calls (avg {:.3}ms/call)",
                send_ns as f64 / 1_000_000.0, pct(send_ns), send_calls, avg_ms(send_ns, send_calls));
            println!("  eval_receive: {:>9.1}ms total ({:>5.0}%) over {:>6} calls (avg {:.3}ms/call)",
                recv_ns as f64 / 1_000_000.0, pct(recv_ns), recv_calls, avg_ms(recv_ns, recv_calls));
            println!("  eval_new:     {:>9.1}ms total ({:>5.0}%) over {:>6} calls (avg {:.3}ms/call)",
                new_ns as f64 / 1_000_000.0, pct(new_ns), new_calls, avg_ms(new_ns, new_calls));
            println!("  eval_match:   {:>9.1}ms total ({:>5.0}%) over {:>6} calls (avg {:.3}ms/call)",
                match_ns as f64 / 1_000_000.0, pct(match_ns), match_calls, avg_ms(match_ns, match_calls));
            println!("  TOTAL:        {:>9.1}ms across {} calls", reducer_ns as f64 / 1_000_000.0, reducer_calls);

            // Wrapper-overhead breakdown — closes the unaccounted gap inside
            // `evaluate_system_source` and `eval_system_deploy`.
            let evs_wrap_calls = get_counter_value(snapshotter, "block.replay.sysdeploy.eval.evaluate-source.wrapper.calls");
            let evs_wrap_ns = get_counter_value(snapshotter, "block.replay.sysdeploy.eval.evaluate-source.wrapper.time_ns");
            let esd_wrap_calls = get_counter_value(snapshotter, "block.replay.sysdeploy.eval.wrapper.calls");
            let esd_wrap_ns = get_counter_value(snapshotter, "block.replay.sysdeploy.eval.wrapper.time_ns");

            println!("\n--- Sysdeploy Wrapper Overhead (env build + bookkeeping outside phase histograms) ---");
            println!("  evaluate_source wrapper:    {:>7.1}ms over {:>4} calls (avg {:.3}ms/call)",
                evs_wrap_ns as f64 / 1_000_000.0, evs_wrap_calls, avg_ms(evs_wrap_ns, evs_wrap_calls));
            println!("  eval_system_deploy wrapper: {:>7.1}ms over {:>4} calls (avg {:.3}ms/call)",
                esd_wrap_ns as f64 / 1_000_000.0, esd_wrap_calls, avg_ms(esd_wrap_ns, esd_wrap_calls));
            println!("================================================================\n");
        },
    )
    .await
    .unwrap()
}

// Run manually: cargo test -p casper --release --test mod cost_accounting_perf -- --nocapture --include-ignored
//
// Multi-deploy block-replay benchmark — reproduces production high-phase load
// shape: NUM_BLOCKS sequential blocks each containing DEPLOYS_PER_BLOCK user
// deploys (`@N!(N)` like test_load.py), all replayed through a single
// ReplayRuntimeOps so per-channel rspace state accumulates between deploys
// the same way it does on a real validator. Per-deploy phase timings are
// averaged across all NUM_BLOCKS × DEPLOYS_PER_BLOCK runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn measure_block_replay_cost() {
    const NUM_BLOCKS: usize = 14;
    const DEPLOYS_PER_BLOCK: usize = 7;

    let snapshotter = get_snapshotter();

    with_runtime_manager(
        |runtime_manager, genesis_context, genesis_block| async move {
            let user_sec = construct_deploy::DEFAULT_SEC.clone();
            let validator = genesis_context.validator_pks()[0].clone();
            let mut current_state = genesis_block.body.state.post_state_hash.clone();

            let mut all_timings: Vec<ReplayPhaseTimings> = Vec::new();
            let mut per_block_avg_ms: Vec<f64> = Vec::new();

            for block_idx in 0..NUM_BLOCKS {
                // Build DEPLOYS_PER_BLOCK deploys with unique indices/vabn so
                // signatures differ and the deploy log doesn't collapse.
                let mut deploys = Vec::with_capacity(DEPLOYS_PER_BLOCK);
                for d_idx in 0..DEPLOYS_PER_BLOCK {
                    let global_idx = block_idx * DEPLOYS_PER_BLOCK + d_idx;
                    let source = format!("@{}!({})", global_idx, global_idx);
                    let deploy = construct_deploy::source_deploy_now_full(
                        source,
                        Some(100_000),
                        None,
                        Some(user_sec.clone()),
                        Some(global_idx as i64),
                        None,
                    )
                    .unwrap();
                    deploys.push(deploy);
                }

                let block_data = BlockData {
                    time_stamp: deploys[0].data.time_stamp,
                    block_number: block_idx as i64,
                    sender: validator.clone(),
                    seq_num: block_idx as i32,
                };

                let (post_state, processed_deploys, _) = runtime_manager
                    .compute_state(
                        &current_state,
                        deploys,
                        Vec::<casper::rust::util::rholang::system_deploy_enum::SystemDeployEnum>::new(),
                        block_data.clone(),
                        None,
                    )
                    .await
                    .expect("compute_state failed");

                for (i, pd) in processed_deploys.iter().enumerate() {
                    assert!(
                        !pd.is_failed,
                        "deploy {} failed in block {}",
                        i, block_idx
                    );
                }

                // One replay runtime per block, mirroring production.
                let replay_runtime = runtime_manager.spawn_replay_runtime().await;
                replay_runtime.set_block_data(block_data).await;
                let mut replay_ops = ReplayRuntimeOps::new_from_runtime(replay_runtime);
                replay_ops
                    .runtime_ops
                    .runtime
                    .reset(&Blake2b256Hash::from_bytes_prost(&current_state))
                    .await
                    .expect("reset failed");

                let mut block_total_ms = 0.0f64;
                for pd in processed_deploys.iter() {
                    let phase = time_replay_one_deploy(&mut replay_ops, pd)
                        .await
                        .expect("replay timing failed");
                    block_total_ms += phase.total.as_secs_f64() * 1000.0;
                    all_timings.push(phase);
                }
                per_block_avg_ms.push(block_total_ms / DEPLOYS_PER_BLOCK as f64);
                current_state = post_state;
            }

            let total_count = all_timings.len() as f64;
            let avg_ms = |get: fn(&ReplayPhaseTimings) -> std::time::Duration| -> f64 {
                all_timings
                    .iter()
                    .map(|t| get(t).as_secs_f64() * 1000.0)
                    .sum::<f64>()
                    / total_count
            };

            let rig_ms = avg_ms(|t| t.rig);
            let precharge_ms = avg_ms(|t| t.precharge);
            let evaluate_ms = avg_ms(|t| t.evaluate);
            let refund_ms = avg_ms(|t| t.refund);
            let check_ms = avg_ms(|t| t.check_replay);
            let total_ms = avg_ms(|t| t.total);

            println!("\n================================================================");
            println!(
                "Block Replay Per-Deploy Phase Breakdown ({} blocks × {} deploys = {} replays)",
                NUM_BLOCKS,
                DEPLOYS_PER_BLOCK,
                all_timings.len()
            );
            println!("================================================================");
            println!("rig:           avg={:.1}ms", rig_ms);
            println!(
                "precharge:     avg={:.1}ms  (production high-phase target ~229ms)",
                precharge_ms
            );
            println!(
                "evaluate:      avg={:.1}ms  (production high-phase target  ~32ms)",
                evaluate_ms
            );
            println!(
                "refund:        avg={:.1}ms  (production high-phase target ~189ms)",
                refund_ms
            );
            println!("check_replay:  avg={:.1}ms", check_ms);
            println!(
                "total:         avg={:.1}ms  (production high-phase target ~450ms)",
                total_ms
            );

            // State accumulation effect — does per-deploy time grow as
            // registry / vault / mergeable history accumulates?
            println!("\n--- Per-Block Avg Total ms (state accumulation effect) ---");
            for (i, ms) in per_block_avg_ms.iter().enumerate() {
                println!("  block {:>2}: avg total {:.1}ms", i, ms);
            }
            println!("================================================================\n");

            // Reducer per-op-type breakdown across the full run.
            let send_calls = get_counter_value(snapshotter, "reducer.eval_send.calls");
            let send_ns = get_counter_value(snapshotter, "reducer.eval_send.time_ns");
            let recv_calls = get_counter_value(snapshotter, "reducer.eval_receive.calls");
            let recv_ns = get_counter_value(snapshotter, "reducer.eval_receive.time_ns");
            let new_calls = get_counter_value(snapshotter, "reducer.eval_new.calls");
            let new_ns = get_counter_value(snapshotter, "reducer.eval_new.time_ns");
            let match_calls = get_counter_value(snapshotter, "reducer.eval_match.calls");
            let match_ns = get_counter_value(snapshotter, "reducer.eval_match.time_ns");
            let reducer_ns = send_ns + recv_ns + new_ns + match_ns;

            println!(
                "--- Reducer Per-Op-Type Calls ({} total dispatches) ---",
                send_calls + recv_calls + new_calls + match_calls
            );
            let pct = |ns: u64| -> f64 {
                if reducer_ns == 0 {
                    0.0
                } else {
                    ns as f64 / reducer_ns as f64 * 100.0
                }
            };
            println!(
                "  eval_send:    {:>6} calls ({:>5.0}% of reducer wall-time, includes await)",
                send_calls,
                pct(send_ns)
            );
            println!(
                "  eval_receive: {:>6} calls ({:>5.0}%)",
                recv_calls,
                pct(recv_ns)
            );
            println!("  eval_new:     {:>6} calls ({:>5.0}%)", new_calls, pct(new_ns));
            println!(
                "  eval_match:   {:>6} calls ({:>5.0}%)",
                match_calls,
                pct(match_ns)
            );
            println!("================================================================\n");

            // Bottleneck #3 epicenter — hot-store put_continuation breakdown
            let pc_calls = get_counter_value(snapshotter, "hot-store.put_continuation.calls");
            let pc_total_ns = get_counter_value(snapshotter, "hot-store.put_continuation.time_ns");
            let pc_ident_build_ns = get_counter_value(snapshotter, "hot-store.put_continuation.identity_build_ns");
            let pc_ident_cmp_ns = get_counter_value(snapshotter, "hot-store.put_continuation.identity_compare_ns");
            let pc_existing_sum = get_counter_value(snapshotter, "hot-store.put_continuation.existing_count_sum");
            let pc_dups = get_counter_value(snapshotter, "hot-store.put_continuation.duplicates");
            let pc_history_fill = get_counter_value(snapshotter, "hot-store.put_continuation.history_fill");

            let pj_calls = get_counter_value(snapshotter, "hot-store.put_join.calls");
            let pj_total_ns = get_counter_value(snapshotter, "hot-store.put_join.time_ns");
            let pj_history_fill = get_counter_value(snapshotter, "hot-store.put_join.history_fill");

            let pd_calls = get_counter_value(snapshotter, "hot-store.put_datum.calls");
            let pd_total_ns = get_counter_value(snapshotter, "hot-store.put_datum.time_ns");
            let pd_history_fill = get_counter_value(snapshotter, "hot-store.put_datum.history_fill");

            let to_ms = |ns: u64| ns as f64 / 1_000_000.0;
            let pct_of = |part: u64, whole: u64| -> f64 {
                if whole == 0 { 0.0 } else { part as f64 / whole as f64 * 100.0 }
            };

            println!("--- Hot Store Put Operations (bottleneck #3 epicenter) ---");
            println!("  put_continuation: {} calls, total {:.1}ms",
                pc_calls, to_ms(pc_total_ns));
            println!("    identity_build:    {:.1}ms ({:.1}% of put_cont)",
                to_ms(pc_ident_build_ns), pct_of(pc_ident_build_ns, pc_total_ns));
            println!("    identity_compare:  {:.1}ms ({:.1}% of put_cont)",
                to_ms(pc_ident_cmp_ns), pct_of(pc_ident_cmp_ns, pc_total_ns));
            println!("    avg existing/call: {:.2}", if pc_calls == 0 { 0.0 } else { pc_existing_sum as f64 / pc_calls as f64 });
            println!("    duplicates: {} ({:.1}%) | history_fill: {} ({:.1}%)",
                pc_dups, pct_of(pc_dups, pc_calls),
                pc_history_fill, pct_of(pc_history_fill, pc_calls));
            println!("  put_join:         {} calls, total {:.1}ms ({:.0} ns/call), history_fill: {} ({:.1}%)",
                pj_calls, to_ms(pj_total_ns),
                if pj_calls == 0 { 0.0 } else { pj_total_ns as f64 / pj_calls as f64 },
                pj_history_fill, pct_of(pj_history_fill, pj_calls));
            println!("  put_datum:        {} calls, total {:.1}ms ({:.0} ns/call), history_fill: {} ({:.1}%)",
                pd_calls, to_ms(pd_total_ns),
                if pd_calls == 0 { 0.0 } else { pd_total_ns as f64 / pd_calls as f64 },
                pd_history_fill, pct_of(pd_history_fill, pd_calls));
            println!("================================================================\n");

            // Hot store get + cache eviction
            let gc_calls = get_counter_value(snapshotter, "hot-store.get_continuations.calls");
            let gc_fill = get_counter_value(snapshotter, "hot-store.get_continuations.history_fill");
            let gd_calls = get_counter_value(snapshotter, "hot-store.get_data.calls");
            let gd_fill = get_counter_value(snapshotter, "hot-store.get_data.history_fill");
            let gj_calls = get_counter_value(snapshotter, "hot-store.get_joins.calls");
            let gj_fill = get_counter_value(snapshotter, "hot-store.get_joins.history_fill");
            let bc_cont = get_counter_value(snapshotter, "hot-store.history_cache.bulk_clear.continuations");
            let bc_data = get_counter_value(snapshotter, "hot-store.history_cache.bulk_clear.datums");
            let bc_joins = get_counter_value(snapshotter, "hot-store.history_cache.bulk_clear.joins");

            println!("--- Hot Store Reads + Cache Eviction ---");
            println!("  get_continuations: {} calls, history_fill: {} ({:.1}%)",
                gc_calls, gc_fill, pct_of(gc_fill, gc_calls));
            println!("  get_data:          {} calls, history_fill: {} ({:.1}%)",
                gd_calls, gd_fill, pct_of(gd_fill, gd_calls));
            println!("  get_joins:         {} calls, history_fill: {} ({:.1}%)",
                gj_calls, gj_fill, pct_of(gj_fill, gj_calls));
            println!("  history_cache.bulk_clear: cont={} datums={} joins={}",
                bc_cont, bc_data, bc_joins);
            println!("================================================================\n");

            // Bottleneck #2 epicenter — matcher + fold_match
            let efm_calls = get_counter_value(snapshotter, "rspace.matcher.extract_first_match.calls");
            let efm_success = get_counter_value(snapshotter, "rspace.matcher.extract_first_match.success");
            let efm_iter = get_counter_value(snapshotter, "rspace.matcher.extract_first_match.candidates_iterated");
            let efm_pair_ns = get_counter_value(snapshotter, "rspace.matcher.extract_first_match.pair_construction_ns");
            let fold_calls = get_counter_value(snapshotter, "rholang.matcher.fold_match.calls");
            let fold_depth_total = get_counter_value(snapshotter, "rholang.matcher.fold_match.recursion_depth_total");
            let fold_clone_ns = get_counter_value(snapshotter, "rholang.matcher.fold_match.tail_clone_ns");
            let matcher_get_calls = get_counter_value(snapshotter, "rspace.matcher.get_calls");
            let matcher_clone_ns = get_counter_value(snapshotter, "rspace.matcher.clone_ns");
            let matcher_fold_ns = get_counter_value(snapshotter, "rspace.matcher.fold_match_ns");

            println!("--- Matcher Path (bottleneck #2 epicenter) ---");
            println!("  extract_first_match: {} calls ({} success, {:.1}% hit rate)",
                efm_calls, efm_success, pct_of(efm_success, efm_calls));
            println!("    avg candidates iterated: {:.2}",
                if efm_calls == 0 { 0.0 } else { efm_iter as f64 / efm_calls as f64 });
            println!("    pair_construction time: {:.1}ms total ({:.0} ns/call)",
                to_ms(efm_pair_ns),
                if efm_calls == 0 { 0.0 } else { efm_pair_ns as f64 / efm_calls as f64 });
            println!("  matcher.get (rspace): {} calls, clone {:.1}ms, fold_match {:.1}ms",
                matcher_get_calls, to_ms(matcher_clone_ns), to_ms(matcher_fold_ns));
            println!("  fold_match (rholang): {} calls, depth_total {} (avg depth {:.2})",
                fold_calls, fold_depth_total,
                if fold_calls == 0 { 0.0 } else { fold_depth_total as f64 / fold_calls as f64 });
            println!("    tail_clone (to_vec/to_owned per recursion): {:.1}ms ({:.0} ns/call)",
                to_ms(fold_clone_ns),
                if fold_calls == 0 { 0.0 } else { fold_clone_ns as f64 / fold_calls as f64 });
            println!("================================================================\n");

            // Cold path
            let fd_calls = get_counter_value(snapshotter, "history.fetch_data.calls");
            let fd_total_ns = get_counter_value(snapshotter, "history.fetch_data.time_ns");
            let fd_trie_ns = get_counter_value(snapshotter, "history.fetch_data.target_history_read_ns");
            let fd_leaf_ns = get_counter_value(snapshotter, "history.fetch_data.leaf_store_get_ns");
            let fd_deser_ns = get_counter_value(snapshotter, "history.fetch_data.bincode_deserialize_ns");
            let fd_legacy = get_counter_value(snapshotter, "history.fetch_data.legacy_fallback_fired");

            println!("--- Cold Path: history.fetch_data (LMDB / radix trie) ---");
            println!("  fetch_data: {} calls, total {:.1}ms ({:.0} ns/call)",
                fd_calls, to_ms(fd_total_ns),
                if fd_calls == 0 { 0.0 } else { fd_total_ns as f64 / fd_calls as f64 });
            println!("    target_history.read (radix trie): {:.1}ms ({:.1}% of fetch)",
                to_ms(fd_trie_ns), pct_of(fd_trie_ns, fd_total_ns));
            println!("    leaf_store.get_one (LMDB):       {:.1}ms ({:.1}% of fetch)",
                to_ms(fd_leaf_ns), pct_of(fd_leaf_ns, fd_total_ns));
            println!("    bincode_deserialize:              {:.1}ms ({:.1}% of fetch)",
                to_ms(fd_deser_ns), pct_of(fd_deser_ns, fd_total_ns));
            println!("    legacy_fallback fired: {}", fd_legacy);
            println!("================================================================\n");
        },
    )
    .await
    .unwrap()
}
