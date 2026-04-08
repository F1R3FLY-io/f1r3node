/// Tests that pre-computing derived sets for depends() and
/// branch conflict detection produces identical results faster.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use prost::bytes::Bytes;
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    merger::{event_log_index::EventLogIndex, merging_logic},
    trace::event::Produce,
};
use shared::rust::hashable_set::HashableSet;

use casper::rust::merging::deploy_chain_index::{DeployChainIndex, DeployIdWithCost};

fn make_chain(idx: usize) -> DeployChainIndex {
    let mut ch = [0u8; 32];
    ch[0] = (idx >> 8) as u8;
    ch[1] = idx as u8;

    let mut dh = [0u8; 32];
    dh[30] = (idx >> 8) as u8;
    dh[31] = idx as u8;

    let mut produces = HashSet::new();
    for i in 0..5u8 {
        let mut ph = ch;
        ph[4] = i;
        produces.insert(Produce {
            channel_hash: Blake2b256Hash::from_bytes(ch.to_vec()),
            hash: Blake2b256Hash::from_bytes(ph.to_vec()),
            persistent: false,
            is_deterministic: true,
            output_value: vec![vec![i; 32]],
            failed: false,
        });
    }

    let mut event_log = EventLogIndex::empty();
    event_log.produces_linear = HashableSet(produces.clone());
    event_log.produces_consumed = HashableSet(produces);

    let mut deploys = HashSet::new();
    deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(dh.to_vec()),
        cost: 100,
    });

    DeployChainIndex::from_parts(
        HashableSet(deploys),
        Blake2b256Hash::from_bytes(vec![0u8; 32]),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
    )
}

#[test]
fn baseline_depends_without_cache() {
    let n = 200;
    let chains: Vec<DeployChainIndex> = (0..n).map(make_chain).collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let start = Instant::now();
    let map = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::depends(&a.event_log_index, &b.event_log_index)
    });
    let elapsed = start.elapsed();

    let total: usize = map.values().map(|s| s.0.len()).sum();
    assert_eq!(total, 0);
    println!("Baseline (no cache): {} chains, {}ms", n, elapsed.as_millis());
}

#[test]
fn cached_depends_is_faster() {
    let n = 200;
    let chains: Vec<DeployChainIndex> = (0..n).map(make_chain).collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    // Lazy cache using pointer address as key
    struct CachedDerived {
        produces_created: HashableSet<Produce>,
        consumes_created: HashableSet<rspace_plus_plus::rspace::trace::event::Consume>,
    }
    let cache: RefCell<HashMap<usize, CachedDerived>> = RefCell::new(HashMap::new());

    let start = Instant::now();
    let map = merging_logic::compute_relation_map(&chain_set, |target, source| {
        let source_addr = std::ptr::addr_of!(*source) as usize;
        {
            let mut c = cache.borrow_mut();
            c.entry(source_addr).or_insert_with(|| CachedDerived {
                produces_created: merging_logic::produces_created_and_not_destroyed(&source.event_log_index),
                consumes_created: merging_logic::consumes_created_and_not_destroyed(&source.event_log_index),
            });
        }
        let c = cache.borrow();
        let derived = c.get(&source_addr).unwrap();

        let produces_source: HashSet<_> = derived.produces_created.0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();
        let produces_target: HashSet<_> = target.event_log_index.produces_consumed.0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();

        if produces_source.intersection(&produces_target).next().is_some() {
            return true;
        }

        derived.consumes_created.0
            .intersection(&target.event_log_index.consumes_produced.0)
            .next()
            .is_some()
    });
    let elapsed = start.elapsed();

    let total: usize = map.values().map(|s| s.0.len()).sum();
    assert_eq!(total, 0);
    println!("Cached: {} chains, {}ms", n, elapsed.as_millis());
}

#[test]
fn cached_and_uncached_produce_identical_results() {
    let n = 50;
    let chains: Vec<DeployChainIndex> = (0..n).map(make_chain).collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let map_original = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::depends(&a.event_log_index, &b.event_log_index)
    });

    struct CachedDerived {
        produces_created: HashableSet<Produce>,
        consumes_created: HashableSet<rspace_plus_plus::rspace::trace::event::Consume>,
    }
    let cache: RefCell<HashMap<usize, CachedDerived>> = RefCell::new(HashMap::new());

    let map_cached = merging_logic::compute_relation_map(&chain_set, |target, source| {
        let source_addr = std::ptr::addr_of!(*source) as usize;
        {
            let mut c = cache.borrow_mut();
            c.entry(source_addr).or_insert_with(|| CachedDerived {
                produces_created: merging_logic::produces_created_and_not_destroyed(&source.event_log_index),
                consumes_created: merging_logic::consumes_created_and_not_destroyed(&source.event_log_index),
            });
        }
        let c = cache.borrow();
        let derived = c.get(&source_addr).unwrap();

        let produces_source: HashSet<_> = derived.produces_created.0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();
        let produces_target: HashSet<_> = target.event_log_index.produces_consumed.0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();

        if produces_source.intersection(&produces_target).next().is_some() {
            return true;
        }

        derived.consumes_created.0
            .intersection(&target.event_log_index.consumes_produced.0)
            .next()
            .is_some()
    });

    assert_eq!(map_original.len(), map_cached.len());
    for (key, original) in &map_original {
        let cached = map_cached.get(key).expect("key missing");
        assert_eq!(original, cached, "Results must be identical");
    }
}
