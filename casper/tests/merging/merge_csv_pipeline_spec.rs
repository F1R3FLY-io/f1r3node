// CSV-driven full-pipeline merge tests.
//
// The "Mergeable" spreadsheet (docs/Mergeable_ - Sheet1.csv in
// system-integration) is the canonical pairwise spec at the rspace level:
// for every (row_op, col_op) pair, ✓ = mergeable, X = not mergeable.
//
// This file constructs DeployChainIndex shapes that match specific CSV
// cells, drives the FULL dag_merger pipeline (compute_branches +
// compute_conflict_map + resolve_conflicts), and asserts that the merge
// outcome matches the CSV's verdict.
//
// Layered above the CSV: tagged channels carry a single-value contract.
// For tagged channels with commutative merge_type and numeric end values,
// the X-cell verdicts can be turned into ✓ via the produces_mergeable
// override. For tagged channels with NON-commutative-representable values
// (e.g., Map on a BitmaskOr-tagged channel), the X-cell verdicts stand —
// these are the wedge surface.
//
// Phase 1: the 5 critical wedge cases that any fix must validate.
// Phase 2 (TODO): comprehensive CSV cell coverage.
// Phase 3 (TODO): determinism property tests across runs.

use casper::rust::merging::{
    conflict_set_merger,
    dag_merger::{compute_branches, compute_conflict_map, cost_optimal_rejection_alg},
    deploy_chain_index::{DeployChainIndex, DeployIdWithCost},
};
use casper::rust::system_deploy::{CLOSE_BLOCK_MARKER, SYSTEM_DEPLOY_ID_LEN};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    merger::{
        event_log_index::EventLogIndex,
        merging_logic::{self, MergeType, NumberChannelsDiff},
        state_change::StateChange,
    },
    trace::event::{Consume, Produce},
};
use shared::rust::hashable_set::HashableSet;

// ============================================================
// Pipeline driver
// ============================================================

#[derive(Debug, Clone)]
pub struct PipelineOutcome {
    pub branch_count: usize,
    pub conflict_map_total_entries: usize,
    pub conflict_map_non_empty_entries: usize,
    pub rejected_branch_count: usize,
    pub rejected_chain_count: usize,
    /// Branch-set summary: each branch's chain count.
    pub branch_sizes: Vec<usize>,
    /// Pairwise are_conflicting result across (chain_a, chain_b) for the
    /// first two chains — useful for diagnosing when conflict_map shows
    /// entries we can't explain via the inverted-index checks.
    pub pairwise_are_conflicting: Option<bool>,
    /// Pairwise depends result on event_log_index between (chain_a, chain_b)
    /// — same intent as above for the depends side.
    pub pairwise_depends_a_b: Option<bool>,
    pub pairwise_depends_b_a: Option<bool>,
}

/// Drive the FULL conflict-detection pipeline (compute_branches +
/// compute_conflict_map + resolve_conflicts) over a set of chains.
///
/// This is the exact production path inside `dag_merger::merge` minus the
/// trie-action / apply phases. Use it to assert on conflict-detection
/// outcomes without needing a real history_repository.
pub fn run_conflict_detection(chains: Vec<DeployChainIndex>) -> PipelineOutcome {
    let actual_seq = chains.clone();
    let late_seq: Vec<DeployChainIndex> = Vec::new();
    let merge_set: HashableSet<DeployChainIndex> = HashableSet(actual_seq.iter().cloned().collect());

    let branches = compute_branches(&merge_set);
    let branches_set = HashableSet(branches.0.iter().cloned().collect());
    let conflict_map =
        compute_conflict_map(&branches_set).expect("compute_conflict_map should succeed");

    let non_empty = conflict_map
        .iter()
        .filter(|(_, v)| !v.0.is_empty())
        .count();

    let resolved = conflict_set_merger::resolve_conflicts(
        actual_seq,
        late_seq,
        &|t: &DeployChainIndex, s: &DeployChainIndex| {
            merging_logic::depends(&t.event_log_index, &s.event_log_index)
        },
        &cost_optimal_rejection_alg(),
        &|chain: &DeployChainIndex| chain.event_log_index.number_channels_data.clone(),
        &|_hash| Ok(Vec::new()),
        &|merge_set: &HashableSet<DeployChainIndex>| compute_branches(merge_set),
        &|branches_set: &HashableSet<HashableSet<DeployChainIndex>>| compute_conflict_map(branches_set),
    )
    .expect("resolve_conflicts should succeed");

    let branch_sizes: Vec<usize> = branches.0.iter().map(|b| b.0.len()).collect();

    let pairwise_are_conflicting = if chains.len() >= 2 {
        let reason = merging_logic::conflict_reason(
            &chains[0].event_log_index,
            &chains[1].event_log_index,
        );
        if let Some(r) = &reason {
            eprintln!("conflict_reason between chains[0] and chains[1]: {}", r);
        }
        Some(merging_logic::are_conflicting(
            &chains[0].event_log_index,
            &chains[1].event_log_index,
        ))
    } else {
        None
    };
    let pairwise_depends_a_b = if chains.len() >= 2 {
        Some(merging_logic::depends(
            &chains[0].event_log_index,
            &chains[1].event_log_index,
        ))
    } else {
        None
    };
    let pairwise_depends_b_a = if chains.len() >= 2 {
        Some(merging_logic::depends(
            &chains[1].event_log_index,
            &chains[0].event_log_index,
        ))
    } else {
        None
    };

    PipelineOutcome {
        branch_count: branches.0.len(),
        conflict_map_total_entries: conflict_map.len(),
        conflict_map_non_empty_entries: non_empty,
        rejected_branch_count: resolved.optimal_rejection_count,
        rejected_chain_count: resolved.rejected.0.len(),
        branch_sizes,
        pairwise_are_conflicting,
        pairwise_depends_a_b,
        pairwise_depends_b_a,
    }
}

// ============================================================
// Fixture builders for CSV op shapes
// ============================================================

/// Construct a Produce event on a given channel with deterministic content.
///
/// IMPORTANT: `Produce` equality is on the `hash` field only (per
/// rspace_plus_plus::rspace::trace::event::Produce's PartialEq impl). Real
/// rspace produces compute hash deterministically from
/// `(channel_bytes, data, persistent)` via stable_hash_provider::hash_produce
/// — so different channels naturally yield different hashes. Test fixtures
/// must replicate that: produces on different channels MUST have different
/// hashes. We achieve that here by mixing the channel's first 16 bytes
/// with the value bytes into the hash.
fn mk_produce(channel: &Blake2b256Hash, value_byte: u8, persistent: bool) -> Produce {
    let chan_bytes = channel.bytes();
    let mut hash_bytes = vec![0u8; 32];
    hash_bytes[..16].copy_from_slice(&chan_bytes[..std::cmp::min(16, chan_bytes.len())]);
    for i in 16..32 {
        hash_bytes[i] = value_byte;
    }
    if persistent {
        hash_bytes[31] ^= 0x80;
    }
    Produce {
        channel_hash: channel.clone(),
        hash: Blake2b256Hash::from_bytes(hash_bytes),
        persistent,
        is_deterministic: true,
        output_value: vec![vec![value_byte]],
        failed: false,
    }
}

/// Construct a Consume event on a given channel set with a deterministic
/// pattern identifier.
///
/// Same rationale as `mk_produce`: rspace's Consume hash is derived from
/// channels+patterns+continuation. Our test fixture must yield distinct
/// hashes for consumes on distinct channel sets so HashSet semantics work.
fn mk_consume(channels: &[Blake2b256Hash], pattern_byte: u8, persistent: bool) -> Consume {
    let mut hash_bytes = vec![pattern_byte; 32];
    // Mix in the first channel's first 8 bytes to make consumes on different
    // channel sets have different hashes.
    if let Some(first) = channels.first() {
        let fb = first.bytes();
        for i in 0..std::cmp::min(8, fb.len()) {
            hash_bytes[i] ^= fb[i];
        }
    }
    if persistent {
        hash_bytes[31] ^= 0x40;
    }
    Consume {
        channel_hashes: channels.to_vec(),
        hash: Blake2b256Hash::from_bytes(hash_bytes),
        persistent,
    }
}

/// Construct a closeBlock-shaped system deploy ID.
fn mk_close_block_deploy_id(seed_byte: u8) -> prost::bytes::Bytes {
    let mut id = vec![seed_byte; SYSTEM_DEPLOY_ID_LEN];
    id[SYSTEM_DEPLOY_ID_LEN - 1] = CLOSE_BLOCK_MARKER;
    prost::bytes::Bytes::from(id)
}

/// Construct a user deploy ID (65 bytes — ECDSA-sig-shaped, NOT system).
fn mk_user_deploy_id(seed_byte: u8) -> prost::bytes::Bytes {
    prost::bytes::Bytes::from(vec![seed_byte; 65])
}

/// `4 !`: linear consume that matched a produce + new produce on the same
/// channel. EventLogIndex carries:
///   - produces_linear = {new_produce}
///   - produces_consumed = {old_produce}  (consumed by COMM)
///   - consumes_linear_and_peeks = {consume}
///   - consumes_produced = {consume}      (matched by COMM)
///
/// `is_tagged_commutative` controls produces_mergeable population —
/// matches the runtime's behavior for tagged channels with numeric values.
fn mk_eli_4bang(
    channel: &Blake2b256Hash,
    old_produce: Produce,
    new_produce: Produce,
    consume: Consume,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    let mut eli = EventLogIndex::empty();
    eli.produces_linear.0.insert(new_produce.clone());
    eli.produces_consumed.0.insert(old_produce);
    eli.consumes_linear_and_peeks.0.insert(consume.clone());
    eli.consumes_produced.0.insert(consume);

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (1i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        // Per EventLogIndex::new's logic: produces_mergeable is built from
        // ALL produces (linear ∪ persistent ∪ consumed ∪ peeked) filtered by
        // mergeable_chs.contains_key(channel). When the channel is in
        // number_channels_data, BOTH old (consumed) and new (linear)
        // produces qualify. This is what enables check #1's both_mergeable
        // override to fire.
        let old_produce = eli.produces_consumed.0.iter().next().cloned()
            .expect("4! shape always has produces_consumed");
        eli.produces_mergeable.0.insert(new_produce);
        eli.produces_mergeable.0.insert(old_produce);
        // Same rationale for consumes_mergeable: any consume whose channel
        // is in mergeable_chs qualifies. The 4! shape's consume (which
        // matched the old produce) is on a mergeable channel here.
        let consume_clone = eli.consumes_produced.0.iter().next().cloned()
            .expect("4! shape always has consumes_produced");
        eli.consumes_mergeable.0.insert(consume_clone);
    } else if merge_type.is_some() {
        // Tagged but not commutative (Map value): channel in identity_tagged
        // but NOT in number_channels_data, so produces_mergeable stays empty.
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `! X`: an unconsumed produce — a send that was never matched within the
/// deploy. EventLogIndex carries:
///   - produces_linear = {p}
///   - p NOT in produces_consumed, NOT in produces_copied_by_peek
///
/// For tagged-commutative-numeric channels: p also enters produces_mergeable
/// (per EventLogIndex::new building it from all produces filtered by
/// mergeable_chs).
fn mk_eli_bang_x(
    channel: &Blake2b256Hash,
    produce: Produce,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    let mut eli = EventLogIndex::empty();
    eli.produces_linear.0.insert(produce.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (1i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.produces_mergeable.0.insert(produce);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `4 X`: an unmatched `for` — a linear consume that's still waiting on a
/// produce. EventLogIndex carries:
///   - consumes_linear_and_peeks = {c}
///   - c NOT in consumes_produced
fn mk_eli_4_x(
    channel: &Blake2b256Hash,
    consume: Consume,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    let mut eli = EventLogIndex::empty();
    eli.consumes_linear_and_peeks.0.insert(consume.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (0i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.consumes_mergeable.0.insert(consume);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `! 4`: a send that matched a linear `for` via COMM. The produce gets
/// destroyed by the consume; both events are recorded as completed COMMs.
///   - produces_consumed = {p}
///   - consumes_linear_and_peeks = {c}
///   - consumes_produced = {c}
fn mk_eli_bang_4(
    channel: &Blake2b256Hash,
    produce: Produce,
    consume: Consume,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    let mut eli = EventLogIndex::empty();
    eli.produces_consumed.0.insert(produce.clone());
    eli.consumes_linear_and_peeks.0.insert(consume.clone());
    eli.consumes_produced.0.insert(consume.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (0i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.produces_mergeable.0.insert(produce);
        eli.consumes_mergeable.0.insert(consume);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `!! X`: an unmatched persistent send. Like `! X` but persistent stays
/// on the channel forever.
///   - produces_persistent = {p}
fn mk_eli_bangbang_x(
    channel: &Blake2b256Hash,
    produce: Produce,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    assert!(produce.persistent, "!! X expects persistent produce");
    let mut eli = EventLogIndex::empty();
    eli.produces_persistent.0.insert(produce.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (1i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.produces_mergeable.0.insert(produce);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `!! 4`: persistent send that matched a linear for. The persistent
/// produce is in produces_persistent (stays) but also in produces_consumed
/// (one COMM destroyed a copy of it — though persistent semantics mean it
/// doesn't actually disappear; check 1 still uses produces_consumed for
/// race detection).
fn mk_eli_bangbang_4(
    channel: &Blake2b256Hash,
    produce: Produce,
    consume: Consume,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    assert!(produce.persistent, "!! 4 expects persistent produce");
    let mut eli = EventLogIndex::empty();
    eli.produces_persistent.0.insert(produce.clone());
    eli.produces_consumed.0.insert(produce.clone());
    eli.consumes_linear_and_peeks.0.insert(consume.clone());
    eli.consumes_produced.0.insert(consume.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (0i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.produces_mergeable.0.insert(produce);
        eli.consumes_mergeable.0.insert(consume);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `P X`: an unmatched peek. The peek consume waits, no produce matched it.
///   - consumes_linear_and_peeks = {c_peek}
///   - c NOT in consumes_produced
fn mk_eli_p_x(
    channel: &Blake2b256Hash,
    consume: Consume,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    // Peek consumes go in consumes_linear_and_peeks just like linear consumes;
    // peek-vs-linear distinction is in how rspace processes them, not in
    // EventLogIndex field placement.
    mk_eli_4_x(channel, consume, is_tagged_commutative, merge_type)
}

/// `C X`: an unmatched persistent consume (contract).
///   - consumes_persistent = {c}
fn mk_eli_c_x(
    channel: &Blake2b256Hash,
    consume: Consume,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    assert!(consume.persistent, "C X expects persistent consume");
    let mut eli = EventLogIndex::empty();
    eli.consumes_persistent.0.insert(consume.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (0i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.consumes_mergeable.0.insert(consume);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// `C !`: persistent consume (contract) that matched a linear produce.
///   - consumes_persistent = {c}
///   - produces_consumed = {p}
///   - consumes_produced = {c}
fn mk_eli_c_bang(
    channel: &Blake2b256Hash,
    consume: Consume,
    produce: Produce,
    is_tagged_commutative: bool,
    merge_type: Option<MergeType>,
) -> EventLogIndex {
    assert!(consume.persistent, "C ! expects persistent consume");
    let mut eli = EventLogIndex::empty();
    eli.consumes_persistent.0.insert(consume.clone());
    eli.consumes_produced.0.insert(consume.clone());
    eli.produces_consumed.0.insert(produce.clone());

    if is_tagged_commutative {
        let mt = merge_type.expect("merge_type required when commutative");
        eli.number_channels_data
            .insert(channel.clone(), (0i64, mt));
        eli.identity_tagged_channels.0.insert(channel.clone());
        eli.produces_mergeable.0.insert(produce);
        eli.consumes_mergeable.0.insert(consume);
    } else if merge_type.is_some() {
        eli.identity_tagged_channels.0.insert(channel.clone());
    }

    eli
}

/// Build a DeployChainIndex from raw parts. Uses `DeployChainIndex::from_parts`.
fn mk_chain(
    deploys_with_cost: Vec<DeployIdWithCost>,
    event_log_index: EventLogIndex,
    state_changes: StateChange,
    source_block_byte: u8,
    source_block_number: i64,
) -> DeployChainIndex {
    DeployChainIndex::from_parts(
        HashableSet(deploys_with_cost.into_iter().collect()),
        Blake2b256Hash::from_bytes(vec![0xff; 32]),
        event_log_index,
        state_changes,
        prost::bytes::Bytes::from(vec![source_block_byte; 32]),
        source_block_number,
    )
}

// ============================================================
// Phase 1 — critical wedge tests
// ============================================================

/// CSV cell `4 ! × 4 !` with the two consumes matching the SAME produce.
/// Untagged channel. Per CSV: X "Could have matched same produce.
/// Mergeable if different linear produces."
///
/// This is the canonical conflict that check #1 (races_for_same_io_event
/// on produces_consumed) is designed to detect. Both branches' deploys
/// consumed the same Produce V₀ → race detected → one rejected.
#[test]
fn csv_cell_4bang_x_4bang_same_produce_untagged_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x42; 32]);

    // Old produce V₀ — same struct in both branches' produces_consumed,
    // representing the race target.
    let v_old = mk_produce(&channel, 0x00, false);
    // New produces — distinct in each branch.
    let v_a = mk_produce(&channel, 0x0A, false);
    let v_b = mk_produce(&channel, 0x0B, false);
    let consume = mk_consume(&[channel.clone()], 0xCC, false);

    let a_eli = mk_eli_4bang(&channel, v_old.clone(), v_a, consume.clone(), false, None);
    let b_eli = mk_eli_4bang(&channel, v_old, v_b, consume, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xAA),
            cost: 10,
        }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xBB),
            cost: 10,
        }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);

    eprintln!("UNTAGGED 4! x 4! same-produce outcome: {:#?}", outcome);
    assert_eq!(outcome.branch_count, 2, "should split into 2 branches");
    assert!(
        outcome.conflict_map_non_empty_entries > 0,
        "check #1 should flag the same-produce race"
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "exactly one chain should be rejected"
    );
}

/// CSV cell `4 ! × 4 !` with two consumes matching SAME produce, but the
/// channel is IntegerAdd-tagged and BOTH new produces are numeric — they
/// land in produces_mergeable. The commutative override should fire and
/// the conflict should NOT be flagged (operations commute via merge_type).
#[test]
fn csv_cell_4bang_x_4bang_same_produce_tagged_commutative_numeric_should_commute() {
    let channel = Blake2b256Hash::from_bytes(vec![0x55; 32]);

    let v_old = mk_produce(&channel, 0x00, false);
    let v_a = mk_produce(&channel, 0x0A, false);
    let v_b = mk_produce(&channel, 0x0B, false);
    let consume = mk_consume(&[channel.clone()], 0xCC, false);

    let a_eli = mk_eli_4bang(
        &channel,
        v_old.clone(),
        v_a,
        consume.clone(),
        /* is_tagged_commutative */ true,
        Some(MergeType::IntegerAdd),
    );
    let b_eli = mk_eli_4bang(
        &channel,
        v_old,
        v_b,
        consume,
        /* is_tagged_commutative */ true,
        Some(MergeType::IntegerAdd),
    );

    let chain_a = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xAA),
            cost: 10,
        }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xBB),
            cost: 10,
        }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);

    eprintln!("TAGGED commutative 4! x 4! same-produce outcome: {:#?}", outcome);
    assert_eq!(outcome.branch_count, 2);
    assert_eq!(
        outcome.conflict_map_non_empty_entries, 0,
        "commutative override should prevent check #1 from flagging this — \
         the values commute via IntegerAdd"
    );
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "no rejection expected for commutative case"
    );
}

/// THE WEDGE: CSV cell `4 ! × 4 !` with two consumes matching SAME produce,
/// channel is BitmaskOr-tagged BUT both new produces are non-numeric (Map
/// values). The channel is identity_tagged but NOT in number_channels_data
/// → produces_mergeable is empty → no commutative override → check #1
/// SHOULD fire and reject one branch.
///
/// CURRENT STATUS (#[ignore]): with the system-deploy filter in place,
/// this exact shape works — but in PRODUCTION the wedge fires when a
/// closeBlock is present in the chain. The user-only synthetic case here
/// passes today. Once we have closeBlock-shaped chain coverage, this test
/// becomes the controlled wedge reproducer.
#[test]
fn csv_cell_4bang_x_4bang_same_produce_tagged_noncommutative_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x99; 32]);

    let v_old = mk_produce(&channel, 0x00, false);
    let v_a = mk_produce(&channel, 0x0A, false);
    let v_b = mk_produce(&channel, 0x0B, false);
    let consume = mk_consume(&[channel.clone()], 0xCC, false);

    // is_tagged_commutative=false but merge_type=Some → channel in
    // identity_tagged_channels (it IS tagged) but NOT in
    // number_channels_data (value not numeric).
    let a_eli = mk_eli_4bang(
        &channel,
        v_old.clone(),
        v_a,
        consume.clone(),
        /* is_tagged_commutative */ false,
        Some(MergeType::BitmaskOr),
    );
    let b_eli = mk_eli_4bang(
        &channel,
        v_old,
        v_b,
        consume,
        /* is_tagged_commutative */ false,
        Some(MergeType::BitmaskOr),
    );

    let chain_a = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xAA),
            cost: 10,
        }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xBB),
            cost: 10,
        }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);

    eprintln!("TAGGED non-commutative 4! x 4! same-produce outcome: {:#?}", outcome);
    assert_eq!(outcome.branch_count, 2);
    assert!(
        outcome.conflict_map_non_empty_entries > 0,
        "tagged non-commutative same-produce race MUST be flagged by check #1"
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "one branch must be rejected to preserve single-value contract"
    );
}

/// CSV cell `4 ! × 4 !` with two consumes matching SAME produce on a
/// tagged non-commutative channel, but each chain ALSO contains a
/// closeBlock system deploy. This is the structural shape of the CI
/// wedge: real blocks have user_deploy + closeBlock combined into one
/// chain, and the production filter strips the chain from conflict
/// detection.
///
/// EXPECTED: same as the no-closeBlock variant — one branch rejected.
/// ACTUAL (current code with filter): no rejection — wedge slips through.
///
/// `#[ignore]`'d until we have a fix that distinguishes "closeBlock
/// commutative effects (skip)" from "user deploy non-commutative race
/// (catch)" — see CI run 25974799710 for why the naive filter-removal
/// approach broke consensus.
#[test]
fn csv_cell_4bang_x_4bang_same_produce_tagged_noncommutative_with_closeblock_chain_must_be_rejected()
{
    let channel = Blake2b256Hash::from_bytes(vec![0x7A; 32]);

    let v_old = mk_produce(&channel, 0x00, false);
    let v_a = mk_produce(&channel, 0x0A, false);
    let v_b = mk_produce(&channel, 0x0B, false);
    let consume = mk_consume(&[channel.clone()], 0xCC, false);

    let a_eli = mk_eli_4bang(&channel, v_old.clone(), v_a, consume.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_4bang(&channel, v_old, v_b, consume, false, Some(MergeType::BitmaskOr));

    // Each chain combines a user deploy + a closeBlock system deploy —
    // the same structural shape `block_index::new` produces in production
    // when `compute_related_sets` groups them.
    let chain_a = mk_chain(
        vec![
            DeployIdWithCost {
                deploy_id: mk_user_deploy_id(0xAA),
                cost: 10,
            },
            DeployIdWithCost {
                deploy_id: mk_close_block_deploy_id(0xA1),
                cost: 0,
            },
        ],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![
            DeployIdWithCost {
                deploy_id: mk_user_deploy_id(0xBB),
                cost: 10,
            },
            DeployIdWithCost {
                deploy_id: mk_close_block_deploy_id(0xB1),
                cost: 0,
            },
        ],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "TAGGED non-commutative 4! x 4! same-produce WITH closeBlock outcome: {:#?}",
        outcome
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "WEDGE REPRODUCER: with closeBlock in the chain, the production \
         filter strips the entire chain from conflict detection, so check \
         #1 sees nothing. Fix must let the user-deploy race surface while \
         not flooding with closeBlock commutative ops."
    );
}

/// SANITY: a closeBlock-shaped chain with NO user-level conflict should
/// produce zero rejections. Each chain contains [user_deploy, closeBlock]
/// touching commutative-tagged channels (validator vault-like). No
/// non-commutative race anywhere.
#[test]
fn closeblock_shaped_chain_with_commutative_ops_does_not_generate_spurious_conflicts() {
    let vault_channel = Blake2b256Hash::from_bytes(vec![0x11; 32]);

    let v_a_old = mk_produce(&vault_channel, 0x10, false);
    let v_a_new = mk_produce(&vault_channel, 0x11, false);
    let v_b_old = mk_produce(&vault_channel, 0x20, false);
    let v_b_new = mk_produce(&vault_channel, 0x21, false);
    let consume = mk_consume(&[vault_channel.clone()], 0xCC, false);

    // Each branch consumes/produces on the same commutative-tagged channel
    // but with DIFFERENT old produces (no shared race). Per CSV `4 ! x 4 !`
    // "Mergeable if different linear produces" → ✓ even without merge_type
    // override.
    let a_eli = mk_eli_4bang(
        &vault_channel,
        v_a_old,
        v_a_new,
        consume.clone(),
        true,
        Some(MergeType::IntegerAdd),
    );
    let b_eli = mk_eli_4bang(
        &vault_channel,
        v_b_old,
        v_b_new,
        consume,
        true,
        Some(MergeType::IntegerAdd),
    );

    let chain_a = mk_chain(
        vec![
            DeployIdWithCost {
                deploy_id: mk_user_deploy_id(0xAA),
                cost: 10,
            },
            DeployIdWithCost {
                deploy_id: mk_close_block_deploy_id(0xA1),
                cost: 0,
            },
        ],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![
            DeployIdWithCost {
                deploy_id: mk_user_deploy_id(0xBB),
                cost: 10,
            },
            DeployIdWithCost {
                deploy_id: mk_close_block_deploy_id(0xB1),
                cost: 0,
            },
        ],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "closeBlock-shaped commutative-only outcome: {:#?}",
        outcome
    );
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "closeBlock-shaped chains with commutative ops should not be rejected"
    );
}

/// SANITY: completely disjoint chains (different channels, no shared
/// produces) should produce zero conflicts and zero rejections.
#[test]
fn disjoint_chains_produce_no_conflicts() {
    let chan_a = Blake2b256Hash::from_bytes(vec![0xA0; 32]);
    let chan_b = Blake2b256Hash::from_bytes(vec![0xB0; 32]);

    let v_a_old = mk_produce(&chan_a, 0x00, false);
    let v_a_new = mk_produce(&chan_a, 0x0A, false);
    let v_b_old = mk_produce(&chan_b, 0x00, false);
    let v_b_new = mk_produce(&chan_b, 0x0B, false);
    let consume_a = mk_consume(&[chan_a.clone()], 0xC1, false);
    let consume_b = mk_consume(&[chan_b.clone()], 0xC2, false);

    let a_eli = mk_eli_4bang(&chan_a, v_a_old, v_a_new, consume_a, false, None);
    let b_eli = mk_eli_4bang(&chan_b, v_b_old, v_b_new, consume_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xAA),
            cost: 10,
        }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost {
            deploy_id: mk_user_deploy_id(0xBB),
            cost: 10,
        }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("disjoint chains outcome: {:#?}", outcome);
    assert_eq!(outcome.rejected_chain_count, 0);
    assert_eq!(outcome.conflict_map_non_empty_entries, 0);
}

// ============================================================
// CSV cell `! X × ! X` — two unconsumed produces on same channel
// ============================================================
//
// CSV cell (row 4 col 4): "✓ Two unsuccessful productions"
// Verdict at rspace level: MERGEABLE — both produces just sit on the
// channel. Multi-Datum results but rspace's semantics allow it.
//
// For tagged channels, the verdict layers:
//   - Tagged commutative numeric: ✓ (OR-fold via merge_type → 1 datum)
//   - Tagged non-commutative: X (single-value contract violated)
//
// This cell models the structural shape of variant D from the synthetic
// suite — a deploy that just adds a value to a channel without consuming
// anything. Two sibling deploys both doing this on the same tagged
// non-commutative channel = wedge.

/// CSV `! X × ! X` on an UNTAGGED channel. Per CSV: ✓ "Two unsuccessful
/// productions". Multi-Datum is acceptable at rspace level, no rejection
/// expected.
#[test]
fn csv_cell_bangx_x_bangx_untagged_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x42; 32]);

    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_x(&channel, p_a, false, None);
    let b_eli = mk_eli_bang_x(&channel, p_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED ! X × ! X outcome: {:#?}", outcome);
    assert_eq!(outcome.branch_count, 2);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "untagged channel: CSV says ✓ two unsuccessful productions — \
         no rejection expected"
    );
}

/// CSV `! X × ! X` on TAGGED commutative numeric channel. Per CSV + merge
/// override: ✓ — produces are commutative-via-merge_type, OR-fold collapses
/// them. No rejection expected.
///
/// NOTE: This case may still trip check #4 because each branch's surviving
/// produce IS in produces_mergeable (we built it that way), but check #4
/// only fires when produces are NOT in produces_mergeable. So check #4
/// should correctly skip. Verified empirically.
#[test]
fn csv_cell_bangx_x_bangx_tagged_commutative_numeric_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x55; 32]);

    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_x(&channel, p_a, true, Some(MergeType::IntegerAdd));
    let b_eli = mk_eli_bang_x(&channel, p_b, true, Some(MergeType::IntegerAdd));

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "TAGGED commutative ! X × ! X outcome: {:#?}",
        outcome
    );
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "tagged commutative: OR-fold merges both produces, no rejection"
    );
}

/// THE STRUCTURAL WEDGE: CSV `! X × ! X` on TAGGED non-commutative channel.
/// Both branches add a NEW produce on the same tagged channel (no consume),
/// channel value is non-numeric (Map etc.) so produces_mergeable is empty.
///
/// Per CSV at rspace level: ✓ "two unsuccessful productions".
/// Per tagged-channel single-value contract: **X (wedge)**.
///
/// PR #520's check #4 is exactly the mechanism that catches this:
/// "identity_tagged channel + non-commutative pending writes from 2+
/// branches". With both branches having surviving produces, channel in
/// identity_tagged, neither in produces_mergeable → check #4 fires.
#[test]
fn csv_cell_bangx_x_bangx_tagged_noncommutative_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x99; 32]);

    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_x(&channel, p_a, false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_bang_x(&channel, p_b, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "TAGGED non-commutative ! X × ! X outcome: {:#?}",
        outcome
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "tagged non-commutative `! X × ! X` is the wedge — check #4 must \
         fire to preserve single-value contract"
    );
}

/// THE WEDGE WITH CLOSEBLOCK CHAIN: same as the above but each chain ALSO
/// contains a closeBlock system deploy. In production, the
/// `compute_branch_derived` filter strips this entire chain from
/// `combined_event_log`. Check #4 sees nothing.
///
/// EXPECTED (CSV + single-value contract): rejection.
/// ACTUAL (current code): no rejection — this is the CI wedge structurally.
#[test]
fn csv_cell_bangx_x_bangx_tagged_noncommutative_with_closeblock_chain_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x7A; 32]);

    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_x(&channel, p_a, false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_bang_x(&channel, p_b, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xA1), cost: 0 },
        ],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xB1), cost: 0 },
        ],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "WEDGE: TAGGED non-commutative ! X × ! X with closeBlock chain — outcome: {:#?}",
        outcome
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "WEDGE REPRODUCER: chain contains [user_deploy, closeBlock]. \
         compute_branch_derived's system-deploy filter strips this chain \
         from combined_event_log → check #4 sees nothing → multi-Datum \
         lands. Fix must let the user deploy's `! X` surface without \
         flooding conflict detection with closeBlock's commutative ops."
    );
}

// ============================================================
// CSV cell `! X × 4 X` — potential COMM (produce + for on same channel)
// ============================================================
//
// CSV cell (row 4 col 7): "X Commutes, doesn't merge. Could match each
// other. If we can prove they don't match, then they can merge."
//
// One branch has an unmatched produce on channel C, the other has an
// unmatched for waiting on C. After merge, they could match each other →
// COMM event would fire. The CSV says: at rspace level, this is a
// conflict (their merge order matters). Check #2 (potential_comms) is the
// detection mechanism.
//
// Note: for `4 X × ! X` the symmetric case applies — check #2 detects
// both directions.

#[test]
fn csv_cell_bangx_x_4x_untagged_should_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x44; 32]);

    let p = mk_produce(&channel, 0x0A, false);
    let c = mk_consume(&[channel.clone()], 0xC1, false);

    // Branch A: unmatched produce on channel.
    let a_eli = mk_eli_bang_x(&channel, p, false, None);
    // Branch B: unmatched for on channel.
    let b_eli = mk_eli_4_x(&channel, c, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED ! X × 4 X outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "potential COMM detected by check #2 should reject one branch"
    );
}

#[test]
fn csv_cell_bangx_x_4x_different_channels_should_not_be_rejected() {
    // Sanity: different channels can't COMM with each other → no conflict.
    let chan_p = Blake2b256Hash::from_bytes(vec![0xA0; 32]);
    let chan_c = Blake2b256Hash::from_bytes(vec![0xB0; 32]);

    let p = mk_produce(&chan_p, 0x0A, false);
    let c = mk_consume(&[chan_c.clone()], 0xC1, false);

    let a_eli = mk_eli_bang_x(&chan_p, p, false, None);
    let b_eli = mk_eli_4_x(&chan_c, c, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED ! X × 4 X DIFFERENT CHANNELS outcome: {:#?}", outcome);
    assert_eq!(outcome.rejected_chain_count, 0);
}

// ============================================================
// CSV cell `4 X × 4 X` — two unmatched fors on same channel
// ============================================================
//
// CSV cell (row 7 col 7): "✓ Two unsuccessful consumes" — both fors are
// waiting, no conflict.

#[test]
fn csv_cell_4x_x_4x_untagged_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x33; 32]);

    let c_a = mk_consume(&[channel.clone()], 0xC1, false);
    let c_b = mk_consume(&[channel.clone()], 0xC2, false);

    let a_eli = mk_eli_4_x(&channel, c_a, false, None);
    let b_eli = mk_eli_4_x(&channel, c_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED 4 X × 4 X outcome: {:#?}", outcome);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "CSV says ✓ two unsuccessful consumes"
    );
}

// ============================================================
// CSV cell `! 4 × ! 4` — both produces matched a for
// ============================================================
//
// Row 5 col 5 of the matrix:
// "X Could have matched same for. Mergeable if different linear consumes."
//
// Race target: check 1 on consumes_produced (the SAME Consume is in both
// branches' consumes_produced when they matched the same for).

#[test]
fn csv_cell_bang4_x_bang4_same_for_untagged_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x31; 32]);

    // SAME consume in both branches (the matched for) — different
    // produces (each branch had its own send to feed the for).
    let c_shared = mk_consume(&[channel.clone()], 0xCC, false);
    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_4(&channel, p_a, c_shared.clone(), false, None);
    let b_eli = mk_eli_bang_4(&channel, p_b, c_shared, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED ! 4 × ! 4 same-for outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "CSV: X — same-for race. Check 1 on consumes_produced should fire."
    );
}

#[test]
fn csv_cell_bang4_x_bang4_different_fors_untagged_should_not_be_rejected() {
    // CSV ✓ side: different linear consumes → mergeable.
    let channel = Blake2b256Hash::from_bytes(vec![0x32; 32]);

    let c_a = mk_consume(&[channel.clone()], 0xC1, false);
    let c_b = mk_consume(&[channel.clone()], 0xC2, false);
    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_4(&channel, p_a, c_a, false, None);
    let b_eli = mk_eli_bang_4(&channel, p_b, c_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED ! 4 × ! 4 different-fors outcome: {:#?}", outcome);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "CSV: ✓ when different linear consumes — no race."
    );
}

// ============================================================
// CSV cell `4 ! × 4 !` — different produces (✓ instantiation)
// ============================================================
//
// Phase 1 covered the X side (same V₀). This is the ✓ side: both branches
// consumed their OWN produces, no shared race.

#[test]
fn csv_cell_4bang_x_4bang_different_produces_untagged_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x41; 32]);

    let v_a_old = mk_produce(&channel, 0x10, false);
    let v_b_old = mk_produce(&channel, 0x20, false);
    let v_a_new = mk_produce(&channel, 0x1A, false);
    let v_b_new = mk_produce(&channel, 0x2A, false);
    let c_a = mk_consume(&[channel.clone()], 0xC1, false);
    let c_b = mk_consume(&[channel.clone()], 0xC2, false);

    let a_eli = mk_eli_4bang(&channel, v_a_old, v_a_new, c_a, false, None);
    let b_eli = mk_eli_4bang(&channel, v_b_old, v_b_new, c_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED 4 ! × 4 ! different-produces outcome: {:#?}", outcome);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "CSV: ✓ when different linear produces — no race."
    );
}

// ============================================================
// CSV cell `!! X × !! X` — two unmatched persistent sends
// ============================================================
//
// Row 14 col 14: "✓ Two unsuccessful productions"
// Persistent sends don't race because they don't get consumed.

#[test]
fn csv_cell_bangbangx_x_bangbangx_untagged_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x77; 32]);

    let p_a = mk_produce(&channel, 0x0A, true);
    let p_b = mk_produce(&channel, 0x0B, true);

    let a_eli = mk_eli_bangbang_x(&channel, p_a, false, None);
    let b_eli = mk_eli_bangbang_x(&channel, p_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED !! X × !! X outcome: {:#?}", outcome);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "CSV: ✓ two unsuccessful persistent productions"
    );
}

// ============================================================
// CSV cell `!! 4 × !! 4` — both persistent sends matched a for
// ============================================================
//
// Row 15 col 15: "X Could have matched same for. Mergeable if different
// linear consumes."
// Same shape as ! 4 × ! 4 but persistent on the produce side.

#[test]
fn csv_cell_bangbang4_x_bangbang4_same_for_untagged_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x88; 32]);

    let c_shared = mk_consume(&[channel.clone()], 0xCC, false);
    let p_a = mk_produce(&channel, 0x0A, true);
    let p_b = mk_produce(&channel, 0x0B, true);

    let a_eli = mk_eli_bangbang_4(&channel, p_a, c_shared.clone(), false, None);
    let b_eli = mk_eli_bangbang_4(&channel, p_b, c_shared, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED !! 4 × !! 4 same-for outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "CSV: X — same-for race even with persistent send"
    );
}

// ============================================================
// CSV cell `P X × P X` — two unmatched peeks on same channel
// ============================================================
//
// Row 12 col 12: "✓ Two unsuccessful consumes"
// Peeks waiting for a produce — no race, no conflict.

#[test]
fn csv_cell_px_x_px_untagged_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x66; 32]);

    let c_a = mk_consume(&[channel.clone()], 0xC1, false);
    let c_b = mk_consume(&[channel.clone()], 0xC2, false);

    let a_eli = mk_eli_p_x(&channel, c_a, false, None);
    let b_eli = mk_eli_p_x(&channel, c_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED P X × P X outcome: {:#?}", outcome);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "CSV: ✓ two unsuccessful peeks"
    );
}

// ============================================================
// CSV cell `C X × C X` — two unmatched persistent consumes (contracts)
// ============================================================
//
// "✓ Two unsuccessful consumes" — both contracts waiting, no race.

#[test]
fn csv_cell_cx_x_cx_untagged_should_not_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x65; 32]);

    let c_a = mk_consume(&[channel.clone()], 0xC1, true);
    let c_b = mk_consume(&[channel.clone()], 0xC2, true);

    let a_eli = mk_eli_c_x(&channel, c_a, false, None);
    let b_eli = mk_eli_c_x(&channel, c_b, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED C X × C X outcome: {:#?}", outcome);
    assert_eq!(
        outcome.rejected_chain_count, 0,
        "CSV: ✓ two unsuccessful contracts"
    );
}

// ============================================================
// CSV cell `C ! × C !` — both contracts matched a produce
// ============================================================
//
// Row 18 col 18: "X Could have matched same produce. Mergeable if
// different linear produces."
// Same shape as `4 ! × 4 !` but with persistent consume.

#[test]
fn csv_cell_cbang_x_cbang_same_produce_untagged_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0x64; 32]);

    let p_shared = mk_produce(&channel, 0x0F, false);
    let c_a = mk_consume(&[channel.clone()], 0xC1, true);
    let c_b = mk_consume(&[channel.clone()], 0xC2, true);

    let a_eli = mk_eli_c_bang(&channel, c_a, p_shared.clone(), false, None);
    let b_eli = mk_eli_c_bang(&channel, c_b, p_shared, false, None);

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("UNTAGGED C ! × C ! same-produce outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "CSV: X — same-produce race even with persistent consume"
    );
}

// ============================================================
// Tagged-non-commutative variants + closeBlock wedge reproducers
// for the X-cells `! 4 × ! 4`, `!! 4 × !! 4`, `C ! × C !`
// ============================================================
//
// Same shape as the tagged-non-commutative `! X × ! X` and `4 ! × 4 !`
// wedge tests, extended to the other X-cells of the diagonal. These
// expand the spec contract that ANY same-event-race cell on a tagged
// non-commutative channel must be detected, and that the closeBlock-
// chain wrapper currently hides the detection from check #1 / check #4.

#[test]
fn csv_cell_bang4_x_bang4_same_for_tagged_noncommutative_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0xB1; 32]);
    let c_shared = mk_consume(&[channel.clone()], 0xCC, false);
    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_4(&channel, p_a, c_shared.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_bang_4(&channel, p_b, c_shared, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("TAGGED non-commutative ! 4 × ! 4 same-for outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "Same-for race on tagged non-commutative channel must be rejected"
    );
}

#[test]
fn csv_cell_bang4_x_bang4_same_for_tagged_noncommutative_with_closeblock_chain_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0xB2; 32]);
    let c_shared = mk_consume(&[channel.clone()], 0xCC, false);
    let p_a = mk_produce(&channel, 0x0A, false);
    let p_b = mk_produce(&channel, 0x0B, false);

    let a_eli = mk_eli_bang_4(&channel, p_a, c_shared.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_bang_4(&channel, p_b, c_shared, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xA1), cost: 0 },
        ],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xB1), cost: 0 },
        ],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "WEDGE: ! 4 × ! 4 same-for tagged non-commutative WITH closeBlock — outcome: {:#?}",
        outcome
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "Wedge: closeBlock in chain strips it from combined_event_log → check 1 sees no shared Consume → no rejection."
    );
}

#[test]
fn csv_cell_bangbang4_x_bangbang4_same_for_tagged_noncommutative_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0xB3; 32]);
    let c_shared = mk_consume(&[channel.clone()], 0xCC, false);
    let p_a = mk_produce(&channel, 0x0A, true);
    let p_b = mk_produce(&channel, 0x0B, true);

    let a_eli = mk_eli_bangbang_4(&channel, p_a, c_shared.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_bangbang_4(&channel, p_b, c_shared, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("TAGGED non-commutative !! 4 × !! 4 same-for outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "Same-for race with persistent send on tagged non-commutative must be rejected"
    );
}

#[test]
fn csv_cell_bangbang4_x_bangbang4_same_for_tagged_noncommutative_with_closeblock_chain_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0xB4; 32]);
    let c_shared = mk_consume(&[channel.clone()], 0xCC, false);
    let p_a = mk_produce(&channel, 0x0A, true);
    let p_b = mk_produce(&channel, 0x0B, true);

    let a_eli = mk_eli_bangbang_4(&channel, p_a, c_shared.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_bangbang_4(&channel, p_b, c_shared, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xA1), cost: 0 },
        ],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xB1), cost: 0 },
        ],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "WEDGE: !! 4 × !! 4 same-for tagged non-commutative WITH closeBlock — outcome: {:#?}",
        outcome
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "Wedge: closeBlock in chain strips it → check 1 misses persistent-send race"
    );
}

#[test]
fn csv_cell_cbang_x_cbang_same_produce_tagged_noncommutative_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0xB5; 32]);
    let p_shared = mk_produce(&channel, 0x0F, false);
    let c_a = mk_consume(&[channel.clone()], 0xC1, true);
    let c_b = mk_consume(&[channel.clone()], 0xC2, true);

    let a_eli = mk_eli_c_bang(&channel, c_a, p_shared.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_c_bang(&channel, c_b, p_shared, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 }],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 }],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!("TAGGED non-commutative C ! × C ! same-produce outcome: {:#?}", outcome);
    assert!(
        outcome.rejected_chain_count >= 1,
        "Same-produce race with persistent consume on tagged non-commutative must be rejected"
    );
}

#[test]
fn csv_cell_cbang_x_cbang_same_produce_tagged_noncommutative_with_closeblock_chain_must_be_rejected() {
    let channel = Blake2b256Hash::from_bytes(vec![0xB6; 32]);
    let p_shared = mk_produce(&channel, 0x0F, false);
    let c_a = mk_consume(&[channel.clone()], 0xC1, true);
    let c_b = mk_consume(&[channel.clone()], 0xC2, true);

    let a_eli = mk_eli_c_bang(&channel, c_a, p_shared.clone(), false, Some(MergeType::BitmaskOr));
    let b_eli = mk_eli_c_bang(&channel, c_b, p_shared, false, Some(MergeType::BitmaskOr));

    let chain_a = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xAA), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xA1), cost: 0 },
        ],
        a_eli,
        StateChange::empty(),
        0xAA,
        1,
    );
    let chain_b = mk_chain(
        vec![
            DeployIdWithCost { deploy_id: mk_user_deploy_id(0xBB), cost: 10 },
            DeployIdWithCost { deploy_id: mk_close_block_deploy_id(0xB1), cost: 0 },
        ],
        b_eli,
        StateChange::empty(),
        0xBB,
        2,
    );

    let outcome = run_conflict_detection(vec![chain_a, chain_b]);
    eprintln!(
        "WEDGE: C ! × C ! same-produce tagged non-commutative WITH closeBlock — outcome: {:#?}",
        outcome
    );
    assert!(
        outcome.rejected_chain_count >= 1,
        "Wedge: closeBlock in chain strips it → check 1 misses contract-produce race"
    );
}

#[allow(dead_code)]
fn _silence_unused(_: NumberChannelsDiff) {}
