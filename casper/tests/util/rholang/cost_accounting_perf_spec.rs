/// Performance benchmark for PoS cost accounting system deploys.
///
/// Measures the time taken by PreChargeDeploy and RefundDeploy in isolation
/// and counts RSpace produce/consume operations to quantify the overhead.
///
/// Run with: cargo test -p casper --test mod cost_accounting_perf -- --nocapture
/// Release:  cargo test -p casper --test mod --release cost_accounting_perf -- --nocapture
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use casper::rust::{
    errors::CasperError,
    rholang::runtime::RuntimeOps,
    util::rholang::{
        costacc::pre_charge_deploy::PreChargeDeploy,
        costacc::refund_deploy::RefundDeploy,
        runtime_manager::RuntimeManager,
        system_deploy::SystemDeployTrait,
        system_deploy_result::SystemDeployResult,
    },
};
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
use models::rust::block::state_hash::StateHash;
use rholang::rust::interpreter::{rho_runtime::RhoRuntime, system_processes::BlockData};

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

fn dump_all_metrics(snapshotter: &Snapshotter) {
    let snapshot = snapshotter.snapshot();
    let metrics = snapshot.into_hashmap();
    println!("\n--- All registered metrics ---");
    for (key, (_, _, value)) in metrics.iter() {
        let name = key.key().name();
        let labels: Vec<String> = key.key().labels().map(|l| format!("{}={}", l.key(), l.value())).collect();
        let count = match value {
            metrics_util::debugging::DebugValue::Histogram(s) => s.len(),
            metrics_util::debugging::DebugValue::Counter(c) => *c as usize,
            metrics_util::debugging::DebugValue::Gauge(_) => 0,
        };
        let kind = match value {
            metrics_util::debugging::DebugValue::Histogram(_) => "histogram",
            metrics_util::debugging::DebugValue::Counter(_) => "counter",
            metrics_util::debugging::DebugValue::Gauge(_) => "gauge",
        };
        println!("  {} [{}] labels=[{}] count={}", name, kind, labels.join(","), count);
    }
    println!("--- End metrics ---\n");
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

#[tokio::test]
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
            println!(
                "Total per deploy: avg={:.1}ms",
                precharge_avg_ms + refund_avg_ms
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
        },
    )
    .await
    .unwrap()
}
