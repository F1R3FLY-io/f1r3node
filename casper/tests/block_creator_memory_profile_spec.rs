use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use block_storage::rust::{
    dag::block_dag_key_value_storage::{BlockDagKeyValueStorage, KeyValueDagRepresentation},
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use casper::rust::{
    blocks::proposer::{block_creator, propose_result::BlockCreatorResult},
    casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
    genesis::contracts::{proof_of_stake::ProofOfStake, validator::Validator as GenesisValidator},
    genesis::genesis::Genesis,
    util::rholang::{
        costacc::close_block_deploy::CloseBlockDeploy,
        interpreter_util::compute_parents_post_state, runtime_manager::RuntimeManager,
        system_deploy_enum::SystemDeployEnum, system_deploy_util,
    },
    validator_identity::ValidatorIdentity,
};
use crypto::rust::{
    private_key::PrivateKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg, signed::Signed},
};
use dashmap::{DashMap, DashSet};
use models::rust::casper::protocol::casper_message::{DeployData, Justification};
use models::rust::validator::Validator;
use prost::bytes::Bytes;
use rholang::rust::interpreter::external_services::ExternalServices;
use rholang::rust::interpreter::system_processes::BlockData;
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};
use tokio::time::{timeout, Duration};

const DEPLOY_LIFESPAN: i64 = 50;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

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

fn delta_kb(curr: Option<usize>, prev: Option<usize>) -> isize {
    match (curr, prev) {
        (Some(c), Some(p)) => c as isize - p as isize,
        _ => 0,
    }
}

async fn store_size_kb(kvm: &mut InMemoryStoreManager, name: &str) -> usize {
    match kvm.store(name.to_string()).await {
        Ok(store) => store.size_bytes() / 1024,
        Err(_) => 0,
    }
}

fn create_deploy(
    iteration: usize,
    validator_sk: &PrivateKey,
    shard_id: &str,
) -> Signed<DeployData> {
    let fixed_inputs = std::env::var("F1R3_BLOCK_CREATOR_PHASE_PROFILE_FIXED_INPUTS")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let timestamp = if fixed_inputs {
        0
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    };
    let deploy_data = DeployData {
        term: format!("new x in {{ x!({}) | for (_ <- x) {{ Nil }} }}", iteration),
        time_stamp: timestamp,
        phlo_price: 1,
        phlo_limit: 100000,
        valid_after_block_number: 0,
        shard_id: shard_id.to_string(),
        expiration_timestamp: None,
    };

    Signed::create(deploy_data, Box::new(Secp256k1), validator_sk.clone())
        .expect("Failed to create signed deploy")
}

fn create_snapshot_with_parent(
    dag: KeyValueDagRepresentation,
    parent: models::rust::casper::protocol::casper_message::BlockMessage,
    validator: Validator,
    shard_name: String,
) -> CasperSnapshot {
    let mut snapshot = CasperSnapshot::new(dag);
    snapshot.max_block_num = parent.body.state.block_number;
    snapshot.parents = vec![parent.clone()];
    snapshot.justifications.insert(Justification {
        validator: validator.clone(),
        latest_block_hash: parent.block_hash.clone(),
    });

    let max_seq_nums: DashMap<Validator, u64> = DashMap::new();
    max_seq_nums.insert(validator.clone(), parent.seq_num as u64);
    snapshot.max_seq_nums = max_seq_nums;

    let mut shard_conf = CasperShardConf::new();
    shard_conf.shard_name = shard_name;
    shard_conf.deploy_lifespan = DEPLOY_LIFESPAN;
    shard_conf.max_number_of_parents = 10;
    shard_conf.casper_version = 1;
    shard_conf.config_version = 1;
    shard_conf.bond_minimum = 0;
    shard_conf.bond_maximum = i64::MAX;
    shard_conf.disable_late_block_filtering = false;
    shard_conf.disable_validator_progress_check = false;

    let mut bonds_map = HashMap::new();
    bonds_map.insert(validator.clone(), 100);
    snapshot.on_chain_state = OnChainCasperState {
        shard_conf,
        bonds_map,
        active_validators: vec![validator],
    };

    snapshot.deploys_in_scope = Arc::new(DashSet::new());
    snapshot
}

#[test]
#[ignore = "manual memory profiling; run with --ignored --nocapture"]
fn profile_block_creator_create_memory_usage() {
    let stack_bytes = env_usize("F1R3_BLOCK_CREATOR_PROFILE_STACK_BYTES", 64 * 1024 * 1024);
    let handle = std::thread::Builder::new()
        .name("block-creator-memory-profile".to_string())
        .stack_size(stack_bytes)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");
            runtime.block_on(run_block_creator_create_memory_profile());
        })
        .expect("Failed to spawn profiling thread");

    handle
        .join()
        .expect("Profiling thread panicked before completing");
}

async fn run_block_creator_create_memory_profile() {
    let iterations = env_usize("F1R3_BLOCK_CREATOR_PROFILE_ITERS", 10);
    let sample_every = env_usize("F1R3_BLOCK_CREATOR_PROFILE_SAMPLE_EVERY", 5).max(1);
    let timeout_ms = env_usize("F1R3_BLOCK_CREATOR_PROFILE_TIMEOUT_MS", 2000) as u64;
    let growth_limit_kb = std::env::var("F1R3_BLOCK_CREATOR_PROFILE_MAX_GROWTH_KB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());

    let secp = Secp256k1;
    let (validator_sk, validator_pk) = secp.new_key_pair();
    let validator_identity = ValidatorIdentity::new(&validator_sk);
    let validator: Bytes = validator_pk.bytes.clone().into();
    let shard_name = "test-shard".to_string();

    let mut kvm = InMemoryStoreManager::new();
    let deploy_storage = Arc::new(Mutex::new(
        KeyValueDeployStorage::new(&mut kvm)
            .await
            .expect("Failed to create deploy storage"),
    ));
    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut kvm)
        .await
        .expect("Failed to create block store");
    let dag_storage = BlockDagKeyValueStorage::new(&mut kvm)
        .await
        .expect("Failed to create DAG storage");

    let rspace_store = kvm
        .r_space_stores()
        .await
        .expect("Failed to get rspace store");
    let mergeable_store = RuntimeManager::mergeable_store(&mut kvm)
        .await
        .expect("Failed to create mergeable store");
    let (mut runtime_manager, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        Genesis::non_negative_mergeable_tag_name(),
        ExternalServices::noop(),
    );

    let genesis = Genesis {
        shard_id: shard_name.clone(),
        timestamp: 0,
        block_number: 0,
        proof_of_stake: ProofOfStake {
            minimum_bond: 1,
            maximum_bond: i64::MAX,
            validators: vec![GenesisValidator {
                pk: validator_pk.clone(),
                stake: 100,
            }],
            epoch_length: 1000,
            quarantine_length: 50000,
            number_of_active_validators: 1,
            pos_multi_sig_public_keys: vec![
                "04db91a53a2b72fcdcb201031772da86edad1e4979eb6742928d27731b1771e0bc40c9e9c9fa6554bdec041a87cee423d6f2e09e9dfb408b78e85a4aa611aad20c".to_string(),
                "042a736b30fffcc7d5a58bb9416f7e46180818c82b15542d0a7819d1a437aa7f4b6940c50db73a67bfc5f5ec5b5fa555d24ef8339b03edaa09c096de4ded6eae14".to_string(),
                "047f0f0f5bbe1d6d1a8dac4d88a3957851940f39a57cd89d55fe25b536ab67e6d76fd3f365c83e5bfe11fe7117e549b1ae3dd39bfc867d1c725a4177692c4e7754".to_string(),
            ],
            pos_multi_sig_quorum: 2,
        },
        vaults: Vec::new(),
        supply: i64::MAX,
        version: 1,
    };
    let parent = Genesis::create_genesis_block(&mut runtime_manager, &genesis)
        .await
        .expect("Failed to create genesis block for block_creator profiling");

    block_store
        .put_block_message(&parent)
        .expect("Failed to store parent block");
    dag_storage
        .insert(&parent, false, true)
        .expect("Failed to insert parent block in DAG");

    let snapshot = create_snapshot_with_parent(
        dag_storage.get_representation(),
        parent,
        validator.clone(),
        shard_name.clone(),
    );

    let mut created_count = 0usize;
    let mut non_created_count = 0usize;
    let mut error_count = 0usize;
    let mut timeout_count = 0usize;
    let mut error_samples: Vec<String> = Vec::new();
    let mut samples: Vec<(usize, usize)> = Vec::new();
    let mut last_rss_kb = vm_rss_kb();
    if let Some(rss) = last_rss_kb {
        samples.push((0, rss));
        println!(
            "create #  0: baseline     rss={}KB ({:.2} MiB)",
            rss,
            kb_to_mib(rss)
        );
    }

    for i in 1..=iterations {
        let deploy = create_deploy(i, &validator_sk, &shard_name);
        {
            let mut ds = deploy_storage.lock().unwrap();
            ds.add(vec![deploy]).expect("Failed to add deploy");
        }

        let outcome = match timeout(
            Duration::from_millis(timeout_ms),
            block_creator::create(
                &snapshot,
                &validator_identity,
                None,
                deploy_storage.clone(),
                &mut runtime_manager,
                &mut block_store,
                false,
            ),
        )
        .await
        {
            Ok(Ok(BlockCreatorResult::Created(..))) => {
                created_count += 1;
                "created"
            }
            Ok(Ok(_)) => {
                non_created_count += 1;
                "non_created"
            }
            Ok(Err(err)) => {
                error_count += 1;
                if error_samples.len() < 5 {
                    error_samples.push(format!("{:?}", err));
                }
                "error"
            }
            Err(_) => {
                error_count += 1;
                timeout_count += 1;
                "timeout"
            }
        };

        {
            let mut ds = deploy_storage.lock().unwrap();
            let all = ds.read_all().expect("Failed to read deploy pool");
            if !all.is_empty() {
                ds.remove(all.into_iter().collect())
                    .expect("Failed to clear deploy pool");
            }
        }

        if let Some(rss) = vm_rss_kb() {
            let baseline = samples.first().map(|(_, v)| *v).unwrap_or(rss);
            let delta_total_kb = rss as isize - baseline as isize;
            let delta_iter_kb = last_rss_kb
                .map(|prev| rss as isize - prev as isize)
                .unwrap_or(0);

            println!(
                "create #{:>3}: {:<11} rss={}KB ({:.2} MiB) delta_iter={:+}KB ({:+.2} MiB) delta_total={:+}KB ({:+.2} MiB)",
                i,
                outcome,
                rss,
                kb_to_mib(rss),
                delta_iter_kb,
                delta_kb_to_mib(delta_iter_kb),
                delta_total_kb,
                delta_kb_to_mib(delta_total_kb),
            );

            last_rss_kb = Some(rss);
            if i % sample_every == 0 {
                samples.push((i, rss));
            }
        }
    }

    if samples
        .last()
        .map(|(idx, _)| *idx != iterations)
        .unwrap_or(true)
    {
        if let Some(rss) = vm_rss_kb() {
            samples.push((iterations, rss));
        }
    }

    println!(
        "block_creator::create profile created={}, non_created={}, errors={}, timeouts={}, vmrss_kb_samples={:?}, error_samples={:?}",
        created_count, non_created_count, error_count, timeout_count, samples, error_samples
    );
    assert!(
        created_count > 0,
        "profiling requires at least one successful block_creator::create; got created=0, non_created={}, errors={}, timeouts={}, samples={:?}, errors={:?}",
        non_created_count,
        error_count,
        timeout_count,
        samples,
        error_samples
    );

    if let (Some(limit), Some((_, first)), Some((_, last))) = (
        growth_limit_kb,
        samples.first().copied(),
        samples.last().copied(),
    ) {
        let growth = last.saturating_sub(first);
        assert!(
            growth <= limit,
            "block_creator::create VmRSS growth {}KB exceeded limit {}KB (samples: {:?})",
            growth,
            limit,
            samples
        );
    }
}

#[test]
#[ignore = "manual memory profiling; run with --ignored --nocapture"]
fn profile_block_creator_phase_split_memory_usage() {
    let stack_bytes = env_usize("F1R3_BLOCK_CREATOR_PROFILE_STACK_BYTES", 64 * 1024 * 1024);
    let handle = std::thread::Builder::new()
        .name("block-creator-phase-split-memory-profile".to_string())
        .stack_size(stack_bytes)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");
            runtime.block_on(run_block_creator_phase_split_memory_profile());
        })
        .expect("Failed to spawn profiling thread");

    handle
        .join()
        .expect("Phase-split profiling thread panicked before completing");
}

async fn run_block_creator_phase_split_memory_profile() {
    let iterations = env_usize("F1R3_BLOCK_CREATOR_PHASE_PROFILE_ITERS", 10);
    let timeout_ms = env_usize("F1R3_BLOCK_CREATOR_PHASE_PROFILE_TIMEOUT_MS", 4000) as u64;
    let fixed_inputs = std::env::var("F1R3_BLOCK_CREATOR_PHASE_PROFILE_FIXED_INPUTS")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let skip_user_deploy = std::env::var("F1R3_BLOCK_CREATOR_PHASE_PROFILE_SKIP_USER_DEPLOY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let skip_system_deploy = std::env::var("F1R3_BLOCK_CREATOR_PHASE_PROFILE_SKIP_SYSTEM_DEPLOY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let skip_parents_compute =
        std::env::var("F1R3_BLOCK_CREATOR_PHASE_PROFILE_SKIP_PARENTS_COMPUTE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let skip_bonds = std::env::var("F1R3_BLOCK_CREATOR_PHASE_PROFILE_SKIP_BONDS")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let secp = Secp256k1;
    let (validator_sk, validator_pk) = secp.new_key_pair();
    let validator_identity = ValidatorIdentity::new(&validator_sk);
    let validator: Bytes = validator_pk.bytes.clone().into();
    let shard_name = "test-shard".to_string();

    let mut kvm = InMemoryStoreManager::new();
    let block_store = KeyValueBlockStore::create_from_kvm(&mut kvm)
        .await
        .expect("Failed to create block store");
    let dag_storage = BlockDagKeyValueStorage::new(&mut kvm)
        .await
        .expect("Failed to create DAG storage");

    let rspace_store = kvm
        .r_space_stores()
        .await
        .expect("Failed to get rspace store");
    let mergeable_store = RuntimeManager::mergeable_store(&mut kvm)
        .await
        .expect("Failed to create mergeable store");
    let (mut runtime_manager, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        Genesis::non_negative_mergeable_tag_name(),
        ExternalServices::noop(),
    );

    let genesis = Genesis {
        shard_id: shard_name.clone(),
        timestamp: 0,
        block_number: 0,
        proof_of_stake: ProofOfStake {
            minimum_bond: 1,
            maximum_bond: i64::MAX,
            validators: vec![GenesisValidator {
                pk: validator_pk.clone(),
                stake: 100,
            }],
            epoch_length: 1000,
            quarantine_length: 50000,
            number_of_active_validators: 1,
            pos_multi_sig_public_keys: vec![
                "04db91a53a2b72fcdcb201031772da86edad1e4979eb6742928d27731b1771e0bc40c9e9c9fa6554bdec041a87cee423d6f2e09e9dfb408b78e85a4aa611aad20c".to_string(),
                "042a736b30fffcc7d5a58bb9416f7e46180818c82b15542d0a7819d1a437aa7f4b6940c50db73a67bfc5f5ec5b5fa555d24ef8339b03edaa09c096de4ded6eae14".to_string(),
                "047f0f0f5bbe1d6d1a8dac4d88a3957851940f39a57cd89d55fe25b536ab67e6d76fd3f365c83e5bfe11fe7117e549b1ae3dd39bfc867d1c725a4177692c4e7754".to_string(),
            ],
            pos_multi_sig_quorum: 2,
        },
        vaults: Vec::new(),
        supply: i64::MAX,
        version: 1,
    };
    let parent = Genesis::create_genesis_block(&mut runtime_manager, &genesis)
        .await
        .expect("Failed to create genesis block for phase-split profiling");

    block_store
        .put_block_message(&parent)
        .expect("Failed to store parent block");
    dag_storage
        .insert(&parent, false, true)
        .expect("Failed to insert parent block in DAG");

    let snapshot = create_snapshot_with_parent(
        dag_storage.get_representation(),
        parent,
        validator.clone(),
        shard_name.clone(),
    );

    let baseline_rss = vm_rss_kb();
    println!(
        "phase baseline: rss={}KB ({:.2} MiB)",
        baseline_rss.unwrap_or(0),
        kb_to_mib(baseline_rss.unwrap_or(0))
    );
    let mut prev_history_store_kb = store_size_kb(&mut kvm, "rspace-history").await;
    let mut prev_cold_store_kb = store_size_kb(&mut kvm, "rspace-cold").await;
    let mut prev_roots_store_kb = store_size_kb(&mut kvm, "rspace-roots").await;
    let mut prev_mergeable_store_kb = store_size_kb(&mut kvm, "mergeable-channel-cache").await;
    let mut prev_total_store_kb =
        prev_history_store_kb + prev_cold_store_kb + prev_roots_store_kb + prev_mergeable_store_kb;
    println!(
        "phase stores baseline: history={}KB ({:.2} MiB), cold={}KB ({:.2} MiB), roots={}KB ({:.2} MiB), mergeable={}KB ({:.2} MiB), total={}KB ({:.2} MiB)",
        prev_history_store_kb,
        kb_to_mib(prev_history_store_kb),
        prev_cold_store_kb,
        kb_to_mib(prev_cold_store_kb),
        prev_roots_store_kb,
        kb_to_mib(prev_roots_store_kb),
        prev_mergeable_store_kb,
        kb_to_mib(prev_mergeable_store_kb),
        prev_total_store_kb,
        kb_to_mib(prev_total_store_kb)
    );

    let mut success_count = 0usize;
    let mut error_count = 0usize;
    let mut timeout_count = 0usize;
    let mut error_samples: Vec<String> = Vec::new();

    for i in 1..=iterations {
        let deploy_iteration = if fixed_inputs { 1 } else { i };
        let deploys = if skip_user_deploy {
            Vec::new()
        } else {
            let deploy = create_deploy(deploy_iteration, &validator_sk, &shard_name);
            vec![deploy]
        };

        let seq_from_snapshot = snapshot
            .max_seq_nums
            .get(&validator_identity.public_key.bytes)
            .map(|seq| *seq + 1)
            .unwrap_or(1);
        let next_seq_num = seq_from_snapshot as i32;
        let next_block_num = snapshot.max_block_num + 1;
        let now = if fixed_inputs {
            0
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0)
        };

        let block_data = BlockData {
            time_stamp: now,
            block_number: next_block_num,
            sender: validator_identity.public_key.clone(),
            seq_num: next_seq_num,
        };

        let system_deploys = if skip_system_deploy {
            Vec::new()
        } else {
            vec![SystemDeployEnum::Close(CloseBlockDeploy {
                initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
                    validator_identity.public_key.clone(),
                    next_seq_num,
                ),
            })]
        };

        let rss_before = vm_rss_kb();

        let (pre_state_hash, _rejected) = if skip_parents_compute {
            match snapshot.parents.first() {
                Some(parent) => (parent.body.state.post_state_hash.clone(), Vec::new()),
                None => (runtime_manager.empty_state_hash(), Vec::new()),
            }
        } else {
            match compute_parents_post_state(
                &block_store,
                snapshot.parents.clone(),
                &snapshot,
                &runtime_manager,
                None,
                None,
            ) {
                Ok(result) => result,
                Err(err) => {
                    error_count += 1;
                    if error_samples.len() < 5 {
                        error_samples.push(format!("parents:{:?}", err));
                    }
                    println!(
                        "phase #{:>3}: parents_error rss={}KB ({:.2} MiB)",
                        i,
                        rss_before.unwrap_or(0),
                        kb_to_mib(rss_before.unwrap_or(0))
                    );
                    continue;
                }
            }
        };

        let rss_after_parents = vm_rss_kb();
        let outcome = if skip_bonds {
            match timeout(
                Duration::from_millis(timeout_ms),
                runtime_manager.compute_state(
                    &pre_state_hash,
                    deploys,
                    system_deploys,
                    block_data,
                    Some(HashMap::new()),
                ),
            )
            .await
            {
                Ok(Ok(_)) => {
                    success_count += 1;
                    "ok"
                }
                Ok(Err(err)) => {
                    error_count += 1;
                    if error_samples.len() < 5 {
                        error_samples.push(format!("state:{:?}", err));
                    }
                    "error"
                }
                Err(_) => {
                    error_count += 1;
                    timeout_count += 1;
                    "timeout"
                }
            }
        } else {
            match timeout(
                Duration::from_millis(timeout_ms),
                runtime_manager.compute_state_with_bonds(
                    &pre_state_hash,
                    deploys,
                    system_deploys,
                    block_data,
                    Some(HashMap::new()),
                ),
            )
            .await
            {
                Ok(Ok(_)) => {
                    success_count += 1;
                    "ok"
                }
                Ok(Err(err)) => {
                    error_count += 1;
                    if error_samples.len() < 5 {
                        error_samples.push(format!("state:{:?}", err));
                    }
                    "error"
                }
                Err(_) => {
                    error_count += 1;
                    timeout_count += 1;
                    "timeout"
                }
            }
        };
        RuntimeManager::trim_allocator();
        let rss_after_state = vm_rss_kb();

        let parents_delta_kb = delta_kb(rss_after_parents, rss_before);
        let state_delta_kb = delta_kb(rss_after_state, rss_after_parents);
        let total_delta_kb = delta_kb(rss_after_state, baseline_rss);
        let rss_value = rss_after_state
            .or(rss_after_parents)
            .or(rss_before)
            .unwrap_or(0);

        println!(
            "phase #{:>3}: {:<7} rss={}KB ({:.2} MiB) parents_delta={:+}KB ({:+.2} MiB) state_delta={:+}KB ({:+.2} MiB) total_delta={:+}KB ({:+.2} MiB)",
            i,
            outcome,
            rss_value,
            kb_to_mib(rss_value),
            parents_delta_kb,
            delta_kb_to_mib(parents_delta_kb),
            state_delta_kb,
            delta_kb_to_mib(state_delta_kb),
            total_delta_kb,
            delta_kb_to_mib(total_delta_kb),
        );

        let history_store_kb = store_size_kb(&mut kvm, "rspace-history").await;
        let cold_store_kb = store_size_kb(&mut kvm, "rspace-cold").await;
        let roots_store_kb = store_size_kb(&mut kvm, "rspace-roots").await;
        let mergeable_store_kb = store_size_kb(&mut kvm, "mergeable-channel-cache").await;
        let total_store_kb = history_store_kb + cold_store_kb + roots_store_kb + mergeable_store_kb;

        println!(
            "phase #{:>3} stores: history={}KB ({:+}KB), cold={}KB ({:+}KB), roots={}KB ({:+}KB), mergeable={}KB ({:+}KB), total={}KB ({:+}KB)",
            i,
            history_store_kb,
            history_store_kb as isize - prev_history_store_kb as isize,
            cold_store_kb,
            cold_store_kb as isize - prev_cold_store_kb as isize,
            roots_store_kb,
            roots_store_kb as isize - prev_roots_store_kb as isize,
            mergeable_store_kb,
            mergeable_store_kb as isize - prev_mergeable_store_kb as isize,
            total_store_kb,
            total_store_kb as isize - prev_total_store_kb as isize
        );

        prev_history_store_kb = history_store_kb;
        prev_cold_store_kb = cold_store_kb;
        prev_roots_store_kb = roots_store_kb;
        prev_mergeable_store_kb = mergeable_store_kb;
        prev_total_store_kb = total_store_kb;
    }

    println!(
        "phase summary: ok={}, errors={}, timeouts={}, error_samples={:?}",
        success_count, error_count, timeout_count, error_samples
    );

    assert!(
        success_count > 0,
        "phase-split profiling requires at least one successful compute_state_with_bonds; got ok=0, errors={}, timeouts={}, error_samples={:?}",
        error_count,
        timeout_count,
        error_samples
    );
}
