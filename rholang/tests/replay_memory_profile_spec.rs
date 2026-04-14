use std::collections::HashMap;

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use rholang::rust::interpreter::{
    accounting::costs::Cost, rho_runtime::RhoRuntime, test_utils::resources::create_runtimes,
};
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};

fn vm_rss_kb() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<usize>().ok())
}

fn kb_to_mib(kb: usize) -> f64 {
    kb as f64 / 1024.0
}

fn delta_kb_to_mib(delta_kb: isize) -> f64 {
    delta_kb as f64 / 1024.0
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual memory profiling; run with --ignored --nocapture"]
async fn profile_debruijn_interpreter_replay_memory_usage() {
    let iterations: usize = 80;
    let sample_every: usize = 8;
    let growth_limit_kb: Option<usize> = None;

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm
        .r_space_stores()
        .await
        .expect("Failed to create in-memory rspace store");
    let mut additional_system_processes = Vec::new();
    let (mut runtime, mut replay_runtime, _) =
        create_runtimes(store, false, &mut additional_system_processes).await;

    let term = "new x in { x!(1) | for (_ <- x) { Nil } }";
    let initial_phlo = Cost::create(i64::MAX, "debruijn replay profile".to_string());

    let mut samples: Vec<(usize, usize)> = Vec::new();
    let mut last_rss_kb = vm_rss_kb();
    if let Some(rss) = last_rss_kb {
        samples.push((0, rss));
        println!(
            "replay #  0: baseline     rss={}KB ({:.2} MiB)",
            rss,
            kb_to_mib(rss)
        );
    }

    for i in 1..=iterations {
        let play_checkpoint = runtime.create_soft_checkpoint();
        let replay_checkpoint = replay_runtime.create_soft_checkpoint();
        let rand = Blake2b512Random::create_from_bytes(&[]);

        let play_result = runtime
            .evaluate(term, initial_phlo.clone(), HashMap::new(), rand.clone())
            .await
            .expect("Play evaluation failed");
        assert!(
            play_result.errors.is_empty(),
            "Play evaluation returned errors: {:?}",
            play_result.errors
        );

        let log = runtime.take_event_log();
        replay_runtime.rig(log).expect("Replay rig failed");

        let replay_result = replay_runtime
            .evaluate(term, initial_phlo.clone(), HashMap::new(), rand)
            .await
            .expect("Replay evaluation failed");
        assert!(
            replay_result.errors.is_empty(),
            "Replay evaluation returned errors: {:?}",
            replay_result.errors
        );
        replay_runtime
            .check_replay_data()
            .expect("Replay data check failed");

        runtime.revert_to_soft_checkpoint(play_checkpoint);
        replay_runtime.revert_to_soft_checkpoint(replay_checkpoint);

        if i % sample_every == 0 {
            if let Some(rss) = vm_rss_kb() {
                let baseline = samples.first().map(|(_, v)| *v).unwrap_or(rss);
                let delta_total_kb = rss as isize - baseline as isize;
                let delta_iter_kb = last_rss_kb
                    .map(|prev| rss as isize - prev as isize)
                    .unwrap_or(0);
                println!(
                    "replay #{:>3}: sampled      rss={}KB ({:.2} MiB) delta_iter={:+}KB ({:+.2} MiB) delta_total={:+}KB ({:+.2} MiB)",
                    i,
                    rss,
                    kb_to_mib(rss),
                    delta_iter_kb,
                    delta_kb_to_mib(delta_iter_kb),
                    delta_total_kb,
                    delta_kb_to_mib(delta_total_kb),
                );
                samples.push((i, rss));
                last_rss_kb = Some(rss);
            }
        }
    }

    if samples
        .last()
        .map(|(idx, _)| *idx != iterations)
        .unwrap_or(true)
    {
        if let Some(rss) = vm_rss_kb() {
            let baseline = samples.first().map(|(_, v)| *v).unwrap_or(rss);
            let delta_total_kb = rss as isize - baseline as isize;
            let delta_iter_kb = last_rss_kb
                .map(|prev| rss as isize - prev as isize)
                .unwrap_or(0);
            println!(
                "replay #{:>3}: final        rss={}KB ({:.2} MiB) delta_iter={:+}KB ({:+.2} MiB) delta_total={:+}KB ({:+.2} MiB)",
                iterations,
                rss,
                kb_to_mib(rss),
                delta_iter_kb,
                delta_kb_to_mib(delta_iter_kb),
                delta_total_kb,
                delta_kb_to_mib(delta_total_kb),
            );
            samples.push((iterations, rss));
        }
    }

    println!(
        "Debruijn replay memory profile vmrss_kb samples: {:?}",
        samples
    );

    if let (Some(limit), Some((_, first)), Some((_, last))) = (
        growth_limit_kb,
        samples.first().copied(),
        samples.last().copied(),
    ) {
        let growth = last.saturating_sub(first);
        assert!(
            growth <= limit,
            "Debruijn replay VmRSS growth {}KB exceeded limit {}KB (samples: {:?})",
            growth,
            limit,
            samples
        );
    }
}
