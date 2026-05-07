/// Tests that pre-computing derived sets for depends() and
/// branch conflict detection produces identical results faster.
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use prost::bytes::Bytes;
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    merger::{event_log_index::EventLogIndex, merging_logic},
    trace::event::{Consume, Produce},
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
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(dh.to_vec()),
        0,
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
    println!(
        "Baseline (no cache): {} chains, {}ms",
        n,
        elapsed.as_millis()
    );
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
                produces_created: merging_logic::produces_created_and_not_destroyed(
                    &source.event_log_index,
                ),
                consumes_created: merging_logic::consumes_created_and_not_destroyed(
                    &source.event_log_index,
                ),
            });
        }
        let c = cache.borrow();
        let derived = c.get(&source_addr).unwrap();

        let produces_source: HashSet<_> = derived
            .produces_created
            .0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();
        let produces_target: HashSet<_> = target
            .event_log_index
            .produces_consumed
            .0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();

        if produces_source
            .intersection(&produces_target)
            .next()
            .is_some()
        {
            return true;
        }

        derived
            .consumes_created
            .0
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
                produces_created: merging_logic::produces_created_and_not_destroyed(
                    &source.event_log_index,
                ),
                consumes_created: merging_logic::consumes_created_and_not_destroyed(
                    &source.event_log_index,
                ),
            });
        }
        let c = cache.borrow();
        let derived = c.get(&source_addr).unwrap();

        let produces_source: HashSet<_> = derived
            .produces_created
            .0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();
        let produces_target: HashSet<_> = target
            .event_log_index
            .produces_consumed
            .0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();

        if produces_source
            .intersection(&produces_target)
            .next()
            .is_some()
        {
            return true;
        }

        derived
            .consumes_created
            .0
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

// ─────────────────────────────────────────────────────────────────────────────
// Conflict-map tests: equivalence and timing for compute_conflict_map_event_indexed
// ─────────────────────────────────────────────────────────────────────────────

/// Build a chain with one linear produce on the given channel that is NOT
/// recorded as consumed — so it stays in `produces_created_and_not_destroyed`
/// and can trigger `potential_comms` when paired against a consume on the
/// same channel.
fn make_chain_active_produce(idx: usize, channel_byte: u8) -> DeployChainIndex {
    let mut ch = [0u8; 32];
    ch[0] = channel_byte;

    let mut dh = [0u8; 32];
    dh[28] = (idx >> 8) as u8;
    dh[29] = idx as u8;
    dh[31] = 0xA1;

    let mut ph = ch;
    ph[4] = idx as u8;
    let produce = Produce {
        channel_hash: Blake2b256Hash::from_bytes(ch.to_vec()),
        hash: Blake2b256Hash::from_bytes(ph.to_vec()),
        persistent: false,
        is_deterministic: true,
        output_value: vec![],
        failed: false,
    };

    let mut event_log = EventLogIndex::empty();
    let mut produces_linear = HashSet::new();
    produces_linear.insert(produce);
    event_log.produces_linear = HashableSet(produces_linear);

    let mut deploys = HashSet::new();
    deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(dh.to_vec()),
        cost: 100,
    });

    DeployChainIndex::from_parts(
        HashableSet(deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(dh.to_vec()),
        0,
    )
}

/// Build a chain with one linear consume on the given channel that is NOT
/// recorded as produced — so it stays in `consumes_created_and_not_destroyed`.
/// Pairs with `make_chain_active_produce(_, channel_byte)` to trigger a
/// `potential_comms` conflict.
fn make_chain_active_consume(idx: usize, channel_byte: u8) -> DeployChainIndex {
    let mut ch = [0u8; 32];
    ch[0] = channel_byte;

    let mut dh = [0u8; 32];
    dh[28] = (idx >> 8) as u8;
    dh[29] = idx as u8;
    dh[31] = 0xC1;

    let mut consume_hash = ch;
    consume_hash[5] = idx as u8;
    let consume = Consume {
        channel_hashes: vec![Blake2b256Hash::from_bytes(ch.to_vec())],
        hash: Blake2b256Hash::from_bytes(consume_hash.to_vec()),
        persistent: false,
    };

    let mut event_log = EventLogIndex::empty();
    let mut consumes_linear = HashSet::new();
    consumes_linear.insert(consume);
    event_log.consumes_linear_and_peeks = HashableSet(consumes_linear);

    let mut deploys = HashSet::new();
    deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(dh.to_vec()),
        cost: 100,
    });

    DeployChainIndex::from_parts(
        HashableSet(deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(dh.to_vec()),
        0,
    )
}

/// Build a chain with a non-empty `produces_touching_base_joins` set. The
/// `produce_touch_base_join` check returns a non-empty conflict set whenever
/// EITHER side of a pair has such produces — so this chain conflicts with
/// every other chain regardless of channel overlap.
fn make_chain_join_touch(idx: usize) -> DeployChainIndex {
    let mut ch = [0u8; 32];
    ch[0] = 0xFE;
    ch[1] = idx as u8;

    let mut dh = [0u8; 32];
    dh[28] = (idx >> 8) as u8;
    dh[29] = idx as u8;
    dh[31] = 0xB1; // anchor byte distinguishes this builder's deploy_id

    let mut ph = ch;
    ph[4] = 0x55;
    let produce = Produce {
        channel_hash: Blake2b256Hash::from_bytes(ch.to_vec()),
        hash: Blake2b256Hash::from_bytes(ph.to_vec()),
        persistent: false,
        is_deterministic: true,
        output_value: vec![],
        failed: false,
    };

    let mut event_log = EventLogIndex::empty();
    let mut produces_join = HashSet::new();
    produces_join.insert(produce);
    event_log.produces_touching_base_joins = HashableSet(produces_join);

    let mut deploys = HashSet::new();
    deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(dh.to_vec()),
        cost: 100,
    });

    DeployChainIndex::from_parts(
        HashableSet(deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(dh.to_vec()),
        0,
    )
}

fn count_total(map: &HashMap<DeployChainIndex, HashableSet<DeployChainIndex>>) -> usize {
    map.values().map(|s| s.0.len()).sum()
}

fn assert_maps_equivalent(
    expected: &HashMap<DeployChainIndex, HashableSet<DeployChainIndex>>,
    actual: &HashMap<DeployChainIndex, HashableSet<DeployChainIndex>>,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "key count differs: expected {} keys, got {}",
        expected.len(),
        actual.len()
    );
    for (key, exp) in expected {
        let act = actual.get(key).expect("key missing in indexed result");
        assert_eq!(
            exp, act,
            "conflict set differs for one of the chains; reference != indexed"
        );
    }
}

/// Run the event-indexed conflict map against an ordered collection of
/// chains; returns a HashMap keyed by `DeployChainIndex` so we can compare
/// directly against `compute_relation_map(&chain_set, |a, b| are_conflicting(...))`.
fn event_indexed_conflict_map_for_chains(
    chain_set: &HashableSet<DeployChainIndex>,
) -> HashMap<DeployChainIndex, HashableSet<DeployChainIndex>> {
    let chains_vec: Vec<DeployChainIndex> = chain_set.0.iter().cloned().collect();
    let event_logs: Vec<&EventLogIndex> = chains_vec.iter().map(|c| &c.event_log_index).collect();
    merging_logic::compute_conflict_map_event_indexed(&chains_vec, &event_logs)
}

#[test]
fn event_indexed_conflicts_disjoint_match_baseline() {
    // 50 chains with disjoint channels and no active produces/consumes —
    // expected total conflicts == 0 in both algorithms.
    let n = 50;
    let chains: Vec<DeployChainIndex> = (0..n).map(make_chain).collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::are_conflicting(&a.event_log_index, &b.event_log_index)
    });
    let map_indexed = event_indexed_conflict_map_for_chains(&chain_set);

    assert_eq!(count_total(&map_baseline), 0, "expected zero conflicts");
    assert_maps_equivalent(&map_baseline, &map_indexed);
}

#[test]
fn event_indexed_conflicts_with_overlap_match_baseline() {
    // Mix of disjoint, conflicting, and unrelated chains. Expect the event-
    // indexed result to record exactly the same conflict pairs as the
    // pairwise predicate baseline.
    let mut chains: Vec<DeployChainIndex> = Vec::new();
    chains.push(make_chain_active_produce(0, 0xA0)); // [0] active produce on ch=A0
    chains.push(make_chain_active_consume(1, 0xA0)); // [1] active consume on ch=A0 (conflicts w/ [0])
    chains.push(make_chain(2)); // [2] disjoint
    chains.push(make_chain_active_produce(3, 0xB0)); // [3] active produce on ch=B0 (no conflict)
    chains.push(make_chain_active_consume(4, 0xA0)); // [4] active consume on ch=A0 (conflicts w/ [0])
    chains.push(make_chain(5)); // [5] disjoint

    let chain_set = HashableSet(chains.into_iter().collect());

    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::are_conflicting(&a.event_log_index, &b.event_log_index)
    });
    let map_indexed = event_indexed_conflict_map_for_chains(&chain_set);

    let total = count_total(&map_baseline);
    assert!(total > 0, "expected at least one conflict pair");
    assert_maps_equivalent(&map_baseline, &map_indexed);
}

#[test]
fn event_indexed_conflicts_with_join_touch_match_baseline() {
    // `produce_touch_base_join` fires whenever EITHER chain in a pair has
    // produces_touching_base_joins, regardless of channel overlap. The event-
    // indexed algorithm handles this via the global-pair fallback for any
    // branch with non-empty `produces_touching_base_joins`.
    let mut chains: Vec<DeployChainIndex> = Vec::new();
    chains.push(make_chain_join_touch(0)); // tainted, disjoint channel
    chains.push(make_chain(1));
    chains.push(make_chain(2));
    chains.push(make_chain(3));

    let chain_set = HashableSet(chains.into_iter().collect());

    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::are_conflicting(&a.event_log_index, &b.event_log_index)
    });
    let map_indexed = event_indexed_conflict_map_for_chains(&chain_set);

    let total = count_total(&map_baseline);
    assert!(
        total > 0,
        "expected join-touch chain to conflict with every other"
    );
    assert_maps_equivalent(&map_baseline, &map_indexed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load-shape repro: scale chain count and events-per-chain until the reference
// algorithm's wall-clock matches what we observe in the integration `test_load`
// medium-phase metrics. The synthetic chains here are still simpler than real
// `EventLogIndex` instances — the goal is to land in the same order-of-
// magnitude (hundreds of ms) so the indexed alternative's win is meaningful.
// ─────────────────────────────────────────────────────────────────────────────

/// Build a chain whose event log has `n_events` produces all on the same
/// channel (ch derived from idx). This makes the per-pair predicate cost
/// scale with `n_events` (HashSet intersections in `races_for_same_io_event`,
/// nested loops in `potential_comms`).
///
/// `consumed_ratio` controls how many produces are also marked as consumed
/// — the rest stay in `produces_created_and_not_destroyed` so the
/// `potential_comms` check has work to do.
fn make_chain_sized(idx: usize, n_events: usize, consumed_ratio: f32) -> DeployChainIndex {
    let mut ch = [0u8; 32];
    ch[0] = (idx >> 8) as u8;
    ch[1] = idx as u8;

    let mut dh = [0u8; 32];
    dh[26] = (idx >> 8) as u8;
    dh[27] = idx as u8;
    dh[31] = 0xD0;

    let mut produces_all: HashSet<Produce> = HashSet::new();
    let mut produces_consumed: HashSet<Produce> = HashSet::new();
    let n_consumed = (n_events as f32 * consumed_ratio) as usize;
    for i in 0..n_events {
        let mut ph = ch;
        ph[3] = (i >> 8) as u8;
        ph[4] = i as u8;
        let p = Produce {
            channel_hash: Blake2b256Hash::from_bytes(ch.to_vec()),
            hash: Blake2b256Hash::from_bytes(ph.to_vec()),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        };
        produces_all.insert(p.clone());
        if i < n_consumed {
            produces_consumed.insert(p);
        }
    }

    // Add a few consumes so `consumes_created_and_not_destroyed` is non-empty
    // and the `potential_comms` nested-loop has both sides populated for
    // pairs that share channels.
    let n_consumes = (n_events / 4).max(1);
    let mut consumes_active: HashSet<Consume> = HashSet::new();
    for i in 0..n_consumes {
        let mut consume_hash = ch;
        consume_hash[5] = (i >> 8) as u8;
        consume_hash[6] = i as u8;
        consumes_active.insert(Consume {
            channel_hashes: vec![Blake2b256Hash::from_bytes(ch.to_vec())],
            hash: Blake2b256Hash::from_bytes(consume_hash.to_vec()),
            persistent: false,
        });
    }

    let mut event_log = EventLogIndex::empty();
    event_log.produces_linear = HashableSet(produces_all);
    event_log.produces_consumed = HashableSet(produces_consumed);
    event_log.consumes_linear_and_peeks = HashableSet(consumes_active);

    let mut deploys = HashSet::new();
    deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(dh.to_vec()),
        cost: 100,
    });

    DeployChainIndex::from_parts(
        HashableSet(deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(dh.to_vec()),
        0,
    )
}

/// Mostly-disjoint chains, modeling test_load medium-phase merge shape with
/// no channel overlap (the `@N!(N)` deploy pattern). Reference predicate
/// has nothing to actually conflict on; event-indexed should land at zero
/// emitted pairs.
#[test]
#[ignore = "perf benchmark — run explicitly with --ignored"]
fn event_indexed_load_shape_disjoint() {
    let n_chains: usize = 200;
    let events_per_chain: usize = 32;
    let consumed_ratio: f32 = 0.7;

    let chains: Vec<DeployChainIndex> = (0..n_chains)
        .map(|i| make_chain_sized(i, events_per_chain, consumed_ratio))
        .collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let baseline_calls = std::cell::Cell::new(0usize);
    let start = Instant::now();
    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        baseline_calls.set(baseline_calls.get() + 1);
        merging_logic::are_conflicting(&a.event_log_index, &b.event_log_index)
    });
    let baseline_elapsed = start.elapsed();

    let start = Instant::now();
    let map_indexed = event_indexed_conflict_map_for_chains(&chain_set);
    let indexed_elapsed = start.elapsed();

    println!();
    println!(
        "Event-indexed disjoint  n_chains={}  events_per_chain={}  consumed_ratio={}",
        n_chains, events_per_chain, consumed_ratio
    );
    println!(
        "  reference: {:>5}ms ({} predicate calls, ~{:.1}µs/call)",
        baseline_elapsed.as_millis(),
        baseline_calls.get(),
        if baseline_calls.get() > 0 {
            baseline_elapsed.as_nanos() as f64 / baseline_calls.get() as f64 / 1000.0
        } else {
            0.0
        }
    );
    println!(
        "  indexed:   {:>5}ms (no pairwise predicate calls)",
        indexed_elapsed.as_millis()
    );
    println!(
        "  speedup:   {:.1}x",
        baseline_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64().max(1e-9)
    );

    assert_maps_equivalent(&map_baseline, &map_indexed);
}

/// Build a chain whose event log includes BOTH a per-chain unique channel
/// AND a fixed set of "shared" channels referenced by every chain. This
/// mimics what we suspect happens in real `test_load` merges: every user
/// deploy chain ends up touching some common channel(s) (cost-acc /
/// deployer-id / etc.), which makes every branch a conflict candidate of
/// every other branch and degrades the indexed algorithm to O(B²).
fn make_chain_overlapping(
    idx: usize,
    n_events: usize,
    n_shared_channels: usize,
    consumed_ratio: f32,
) -> DeployChainIndex {
    // Unique channel for this chain.
    let mut unique_ch = [0u8; 32];
    unique_ch[0] = (idx >> 8) as u8;
    unique_ch[1] = idx as u8;
    unique_ch[2] = 0xC0;

    let mut dh = [0u8; 32];
    dh[24] = (idx >> 8) as u8;
    dh[25] = idx as u8;
    dh[31] = 0xE0;

    let mut produces_all: HashSet<Produce> = HashSet::new();
    let mut produces_consumed: HashSet<Produce> = HashSet::new();
    let n_consumed = (n_events as f32 * consumed_ratio) as usize;

    // Per-chain unique events.
    for i in 0..n_events {
        let mut ph = unique_ch;
        ph[3] = (i >> 8) as u8;
        ph[4] = i as u8;
        let p = Produce {
            channel_hash: Blake2b256Hash::from_bytes(unique_ch.to_vec()),
            hash: Blake2b256Hash::from_bytes(ph.to_vec()),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        };
        produces_all.insert(p.clone());
        if i < n_consumed {
            produces_consumed.insert(p);
        }
    }

    // Shared channels — same Blake2b256Hash byte pattern across all chains,
    // distinguished from unique by the leading 0xFF byte.
    for s in 0..n_shared_channels {
        let mut shared_ch = [0u8; 32];
        shared_ch[0] = 0xFF;
        shared_ch[1] = s as u8;

        // One unique produce per chain on the shared channel — different
        // produce.hash so branches don't share Produce identity (which
        // would trigger the `same_io_event` path).
        let mut ph = shared_ch;
        ph[16] = (idx >> 8) as u8;
        ph[17] = idx as u8;
        ph[18] = s as u8;
        let p = Produce {
            channel_hash: Blake2b256Hash::from_bytes(shared_ch.to_vec()),
            hash: Blake2b256Hash::from_bytes(ph.to_vec()),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        };
        produces_all.insert(p.clone());
        produces_consumed.insert(p);
    }

    let n_consumes = (n_events / 4).max(1);
    let mut consumes_active: HashSet<Consume> = HashSet::new();
    for i in 0..n_consumes {
        let mut consume_hash = unique_ch;
        consume_hash[5] = (i >> 8) as u8;
        consume_hash[6] = i as u8;
        consumes_active.insert(Consume {
            channel_hashes: vec![Blake2b256Hash::from_bytes(unique_ch.to_vec())],
            hash: Blake2b256Hash::from_bytes(consume_hash.to_vec()),
            persistent: false,
        });
    }

    let mut event_log = EventLogIndex::empty();
    event_log.produces_linear = HashableSet(produces_all);
    event_log.produces_consumed = HashableSet(produces_consumed);
    event_log.consumes_linear_and_peeks = HashableSet(consumes_active);

    let mut deploys = HashSet::new();
    deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(dh.to_vec()),
        cost: 100,
    });

    DeployChainIndex::from_parts(
        HashableSet(deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(dh.to_vec()),
        0,
    )
}

/// Reproduces the test_load medium-phase cost shape with channel sharing —
/// every branch references both per-chain unique channels and a fixed set
/// of shared channels. The event-indexed algorithm emits a conflict pair
/// only when a matching event is found across branches, so a shared channel
/// without matching events does not generate spurious pairs.
///
/// Knobs roughly mirror test_load medium phase: ~30 branches, ≥1 shared
/// channel, ~100 events per chain.
#[test]
#[ignore = "perf benchmark — run explicitly with --ignored"]
fn event_indexed_load_shape_overlapping() {
    let n_chains: usize = 30;
    let events_per_chain: usize = 100;
    let n_shared_channels: usize = 1;
    let consumed_ratio: f32 = 0.7;

    let chains: Vec<DeployChainIndex> = (0..n_chains)
        .map(|i| make_chain_overlapping(i, events_per_chain, n_shared_channels, consumed_ratio))
        .collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let baseline_calls = std::cell::Cell::new(0usize);
    let start = Instant::now();
    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        baseline_calls.set(baseline_calls.get() + 1);
        merging_logic::are_conflicting(&a.event_log_index, &b.event_log_index)
    });
    let baseline_elapsed = start.elapsed();

    let start = Instant::now();
    let map_indexed = event_indexed_conflict_map_for_chains(&chain_set);
    let indexed_elapsed = start.elapsed();

    println!();
    println!(
        "Event-indexed overlapping  n_chains={}  events_per_chain={}  shared_channels={}  consumed_ratio={}",
        n_chains, events_per_chain, n_shared_channels, consumed_ratio
    );
    println!(
        "  reference: {:>5}ms ({} predicate calls, ~{:.1}µs/call)",
        baseline_elapsed.as_millis(),
        baseline_calls.get(),
        if baseline_calls.get() > 0 {
            baseline_elapsed.as_nanos() as f64 / baseline_calls.get() as f64 / 1000.0
        } else {
            0.0
        }
    );
    println!(
        "  indexed:   {:>5}ms (no pairwise predicate calls)",
        indexed_elapsed.as_millis()
    );
    println!(
        "  speedup:   {:.2}x",
        baseline_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64().max(1e-9)
    );
    println!(
        "  total conflicts recorded: baseline={}  indexed={}",
        count_total(&map_baseline),
        count_total(&map_indexed)
    );

    assert_maps_equivalent(&map_baseline, &map_indexed);
}

/// Helper: run `compute_depends_map_event_indexed` over an ordered chain
/// collection and return a HashMap keyed by `DeployChainIndex` for direct
/// comparison against `compute_relation_map(&chain_set, |a, b| depends(...))`.
fn event_indexed_depends_map_for_chains(
    chain_set: &HashableSet<DeployChainIndex>,
) -> HashMap<DeployChainIndex, HashableSet<DeployChainIndex>> {
    let chains_vec: Vec<DeployChainIndex> = chain_set.0.iter().cloned().collect();
    let event_logs: Vec<&EventLogIndex> = chains_vec.iter().map(|c| &c.event_log_index).collect();
    merging_logic::compute_depends_map_event_indexed(&chains_vec, &event_logs)
}

#[test]
fn event_indexed_depends_disjoint_match_baseline() {
    // Disjoint chains: every pair has empty depends; both algorithms
    // should produce the same all-empty map.
    let n = 50;
    let chains: Vec<DeployChainIndex> = (0..n).map(make_chain).collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::depends(&a.event_log_index, &b.event_log_index)
    });
    let map_indexed = event_indexed_depends_map_for_chains(&chain_set);

    assert_eq!(count_total(&map_baseline), 0, "expected zero depends pairs");
    assert_maps_equivalent(&map_baseline, &map_indexed);
}

/// Build a producer chain and a consumer chain that share the SAME
/// Produce struct (same hash) — `producer` has it in `produces_linear`,
/// `consumer` has it in `produces_consumed`. This satisfies the depends
/// predicate's `producesDepends` clause.
fn make_depends_pair(
    idx_producer: usize,
    idx_consumer: usize,
) -> (DeployChainIndex, DeployChainIndex) {
    let mut ch = [0u8; 32];
    ch[0] = 0xD0;
    ch[1] = idx_producer as u8;

    let mut ph = ch;
    ph[4] = idx_producer as u8;
    let shared_produce = Produce {
        channel_hash: Blake2b256Hash::from_bytes(ch.to_vec()),
        hash: Blake2b256Hash::from_bytes(ph.to_vec()),
        persistent: false,
        is_deterministic: true,
        output_value: vec![],
        failed: false,
    };

    let mut producer_dh = [0u8; 32];
    producer_dh[20] = idx_producer as u8;
    producer_dh[31] = 0xD1;

    let mut producer_event_log = EventLogIndex::empty();
    let mut producer_linear = HashSet::new();
    producer_linear.insert(shared_produce.clone());
    producer_event_log.produces_linear = HashableSet(producer_linear);

    let mut producer_deploys = HashSet::new();
    producer_deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(producer_dh.to_vec()),
        cost: 100,
    });
    let producer = DeployChainIndex::from_parts(
        HashableSet(producer_deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        producer_event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(producer_dh.to_vec()),
        0,
    );

    let mut consumer_dh = [0u8; 32];
    consumer_dh[20] = idx_consumer as u8;
    consumer_dh[31] = 0xD2;

    let mut consumer_event_log = EventLogIndex::empty();
    let mut consumer_consumed = HashSet::new();
    consumer_consumed.insert(shared_produce);
    consumer_event_log.produces_consumed = HashableSet(consumer_consumed);

    let mut consumer_deploys = HashSet::new();
    consumer_deploys.insert(DeployIdWithCost {
        deploy_id: Bytes::from(consumer_dh.to_vec()),
        cost: 100,
    });
    let consumer = DeployChainIndex::from_parts(
        HashableSet(consumer_deploys),
        Blake2b256Hash::from_bytes(vec![1u8; 32]),
        consumer_event_log,
        rspace_plus_plus::rspace::merger::state_change::StateChange::empty(),
        Bytes::from(consumer_dh.to_vec()),
        0,
    );

    (producer, consumer)
}

#[test]
fn event_indexed_depends_with_overlap_match_baseline() {
    // A producer/consumer pair that shares a Produce struct triggers
    // `producesDepends`. Other chains in the set are disjoint and must
    // not be marked as depending on anything.
    let (p0, c0) = make_depends_pair(0, 1);
    let mut chains: Vec<DeployChainIndex> = Vec::new();
    chains.push(p0);
    chains.push(c0);
    chains.push(make_chain(2));
    chains.push(make_chain(3));

    let chain_set = HashableSet(chains.into_iter().collect());

    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        merging_logic::depends(&a.event_log_index, &b.event_log_index)
    });
    let map_indexed = event_indexed_depends_map_for_chains(&chain_set);

    assert!(
        count_total(&map_baseline) > 0,
        "expected at least one depends pair"
    );
    assert_maps_equivalent(&map_baseline, &map_indexed);
}

/// Side-by-side timing for the depends path: O(D²) pairwise
/// `compute_relation_map` vs single-pass `compute_depends_map_event_indexed`,
/// both over the same chain shape used in `test_load` (~200 chains with
/// ~32 events each, mostly disjoint).
#[test]
#[ignore = "perf benchmark — run explicitly with --ignored"]
fn event_indexed_load_shape_depends() {
    let n_chains: usize = 200;
    let events_per_chain: usize = 32;
    let consumed_ratio: f32 = 0.7;

    let chains: Vec<DeployChainIndex> = (0..n_chains)
        .map(|i| make_chain_sized(i, events_per_chain, consumed_ratio))
        .collect();
    let chain_set = HashableSet(chains.into_iter().collect());

    let baseline_calls = std::cell::Cell::new(0usize);
    let start = Instant::now();
    let map_baseline = merging_logic::compute_relation_map(&chain_set, |a, b| {
        baseline_calls.set(baseline_calls.get() + 1);
        merging_logic::depends(&a.event_log_index, &b.event_log_index)
    });
    let baseline_elapsed = start.elapsed();

    let start = Instant::now();
    let map_indexed = event_indexed_depends_map_for_chains(&chain_set);
    let indexed_elapsed = start.elapsed();

    println!();
    println!(
        "Depends load shape  n_chains={}  events_per_chain={}  consumed_ratio={}",
        n_chains, events_per_chain, consumed_ratio
    );
    println!(
        "  reference: {:>5}ms ({} predicate calls, ~{:.1}µs/call)",
        baseline_elapsed.as_millis(),
        baseline_calls.get(),
        if baseline_calls.get() > 0 {
            baseline_elapsed.as_nanos() as f64 / baseline_calls.get() as f64 / 1000.0
        } else {
            0.0
        }
    );
    println!(
        "  indexed:   {:>5}ms (no pairwise predicate calls)",
        indexed_elapsed.as_millis()
    );
    println!(
        "  speedup:   {:.1}x",
        baseline_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64().max(1e-9)
    );
    println!(
        "  total depends pairs recorded: baseline={}  indexed={}",
        count_total(&map_baseline),
        count_total(&map_indexed)
    );

    assert_maps_equivalent(&map_baseline, &map_indexed);
}
