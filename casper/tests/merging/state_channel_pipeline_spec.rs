// Parallel state-channel pipeline tests.
//
// Validates that the byte-erased storage strategy is sound for
// `Datum<ListParWithRandom>` — the production payload type. If bincode
// loses fields or changes byte representation across runs, the entire
// merge layer would silently corrupt state.

use models::rhoapi::{
    expr::ExprInstance, g_unforgeable::UnfInstance, Expr, GPrivate, GUnforgeable,
    ListParWithRandom, Par,
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::Datum,
    serializers::serializers::{decode_datum, encode_datum},
    trace::event::Produce,
};

/// Bincode round-trip stability for `Datum<ListParWithRandom>`. Byte-erased
/// storage requires bincode to be a stable, reversible serialization for the
/// production payload type. Constructs a non-trivial Datum, encodes it,
/// decodes it, asserts equality, and asserts that encoding twice yields
/// identical bytes (determinism).
#[test]
fn bincode_round_trip_for_datum_list_par_with_random() {
    let payload = make_test_list_par_with_random();
    let datum = Datum {
        a: payload.clone(),
        persist: false,
        source: Produce {
            channel_hash: Blake2b256Hash::new(&[0xAB; 32]),
            hash: Blake2b256Hash::new(&[0xCD; 32]),
            persistent: false,
            is_deterministic: true,
            output_value: vec![vec![0x01, 0x02, 0x03], vec![0x04, 0x05]],
            failed: false,
        },
    };

    let bytes_1 = encode_datum(&datum);
    let bytes_2 = encode_datum(&datum);
    assert_eq!(
        bytes_1, bytes_2,
        "bincode encoding must be deterministic — same input must yield same bytes"
    );

    let decoded: Datum<ListParWithRandom> = decode_datum(&bytes_1);
    assert_eq!(
        decoded, datum,
        "bincode round-trip must preserve Datum<ListParWithRandom> exactly"
    );

    let bytes_3 = encode_datum(&decoded);
    assert_eq!(
        bytes_1, bytes_3,
        "encoding the decoded value must yield identical bytes (full round-trip stability)"
    );
}

/// Persist=true variant — bool field shouldn't behave differently from false
/// under bincode but worth confirming explicitly.
#[test]
fn bincode_round_trip_for_persistent_datum() {
    let datum = Datum {
        a: make_test_list_par_with_random(),
        persist: true,
        source: Produce {
            channel_hash: Blake2b256Hash::new(&[0x11; 32]),
            hash: Blake2b256Hash::new(&[0x22; 32]),
            persistent: true,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        },
    };

    let bytes = encode_datum(&datum);
    let decoded: Datum<ListParWithRandom> = decode_datum(&bytes);
    assert_eq!(decoded, datum);
}

/// Empty payload (no Pars, empty random_state) — confirms default-value
/// encoding is stable.
#[test]
fn bincode_round_trip_for_empty_payload() {
    let datum = Datum {
        a: ListParWithRandom::default(),
        persist: false,
        source: Produce {
            channel_hash: Blake2b256Hash::new(&[0x00; 32]),
            hash: Blake2b256Hash::new(&[0x00; 32]),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        },
    };

    let bytes = encode_datum(&datum);
    let decoded: Datum<ListParWithRandom> = decode_datum(&bytes);
    assert_eq!(decoded, datum);
}

fn make_test_list_par_with_random() -> ListParWithRandom {
    let unforgeable_par = Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
            id: vec![0xDE, 0xAD, 0xBE, 0xEF],
        })),
    }]);
    let int_par = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::GInt(42)),
    }]);

    ListParWithRandom {
        pars: vec![unforgeable_par, int_par],
        random_state: vec![0x99; 64],
    }
}

// Mid-level pipeline test: construct a `StateChannelsDiff` with
// multi-Datum content directly (bypassing runtime extraction), route it
// through `RholangMergingLogic::calculate_state_channel_merge`, and
// assert the resulting trie action contains the lowest-Blake2b256-hash
// winner. End-to-end coverage of the parallel state-channel pipeline.

use rholang::rust::interpreter::merging::rholang_merging_logic::RholangMergingLogic;
use rspace_plus_plus::rspace::{
    errors::HistoryError,
    hot_store_trie_action::{HotStoreTrieAction, TrieInsertAction},
    merger::{
        channel_change::ChannelChange,
        merging_logic::{combine_mergeable_state, MergeType},
    },
};

/// Builds a Datum with the given byte payload marker so its bincode bytes
/// hash distinctly from siblings. Mirrors the L1-test pattern.
fn make_state_datum(payload_marker: u8, payload_int: i64) -> Datum<ListParWithRandom> {
    let payload = ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![models::rhoapi::Expr {
            expr_instance: Some(models::rhoapi::expr::ExprInstance::GInt(payload_int)),
        }])],
        random_state: vec![payload_marker; 64],
    };
    Datum {
        a: payload,
        persist: false,
        source: Produce {
            channel_hash: Blake2b256Hash::new(&[payload_marker; 32]),
            hash: Blake2b256Hash::new(&[payload_marker; 32]),
            persistent: false,
            is_deterministic: true,
            output_value: vec![],
            failed: false,
        },
    }
}

/// Mid-level pipeline test: three sibling Datums on a state channel are
/// merged via `calculate_state_channel_merge`. The emitted trie action
/// must contain exactly one Datum — the lowest-hash winner.
#[test]
fn calculate_state_channel_merge_picks_lowest_hash_winner_with_empty_base() {
    let channel_hash = Blake2b256Hash::new(&[0x42; 32]);
    let datum_a = make_state_datum(0xAA, 100);
    let datum_b = make_state_datum(0xBB, 200);
    let datum_c = make_state_datum(0xCC, 300);

    // Independently compute the expected winner using the typed combine.
    let expected = combine_mergeable_state(
        &combine_mergeable_state(&datum_a, &datum_b, MergeType::MutexState),
        &datum_c,
        MergeType::MutexState,
    );

    // Build the per-deploy "diff" (this branch's contribution) and the
    // ChannelChange for sibling additions, mirroring how B5 plumbing will
    // present them.
    let diff_bytes = encode_datum(&datum_a);
    let changes = ChannelChange {
        added: vec![encode_datum(&datum_b), encode_datum(&datum_c)],
        removed: vec![],
    };

    // Empty base — no prior state on this channel.
    let get_base_data =
        |_h: &Blake2b256Hash| -> Result<Vec<Datum<ListParWithRandom>>, HistoryError> { Ok(vec![]) };

    let action = RholangMergingLogic::calculate_state_channel_merge(
        &channel_hash,
        diff_bytes,
        MergeType::MutexState,
        &changes,
        get_base_data,
    );

    match action {
        HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(insert)) => {
            assert_eq!(insert.hash, channel_hash);
            assert_eq!(
                insert.data.len(),
                1,
                "MutexState merge must collapse to exactly one Datum, got {}",
                insert.data.len(),
            );
            let written: Datum<ListParWithRandom> = decode_datum(&insert.data[0]);
            assert_eq!(
                written, expected,
                "trie action's surviving Datum must match the typed combine_mergeable_state winner"
            );
        }
        other => panic!("expected TrieInsertBinaryProduce, got {:?}", other),
    }
}

/// Same as above but with a non-empty base — the merge must consider the
/// base Datum as one of the candidates, so the winner is the lowest-hash
/// across {base, diff, added...}.
#[test]
fn calculate_state_channel_merge_includes_base_in_candidates() {
    let channel_hash = Blake2b256Hash::new(&[0x77; 32]);
    let base_datum = make_state_datum(0x01, 0);
    let datum_x = make_state_datum(0xFE, 1);
    let datum_y = make_state_datum(0xFF, 2);

    let expected = combine_mergeable_state(
        &combine_mergeable_state(&base_datum, &datum_x, MergeType::MutexState),
        &datum_y,
        MergeType::MutexState,
    );

    let diff_bytes = encode_datum(&datum_x);
    let changes = ChannelChange {
        added: vec![encode_datum(&datum_y)],
        removed: vec![],
    };

    let base_clone = base_datum.clone();
    let get_base_data =
        move |_h: &Blake2b256Hash| -> Result<Vec<Datum<ListParWithRandom>>, HistoryError> {
            Ok(vec![base_clone.clone()])
        };

    let action = RholangMergingLogic::calculate_state_channel_merge(
        &channel_hash,
        diff_bytes,
        MergeType::MutexState,
        &changes,
        get_base_data,
    );

    match action {
        HotStoreTrieAction::TrieInsertAction(TrieInsertAction::TrieInsertBinaryProduce(insert)) => {
            assert_eq!(insert.data.len(), 1);
            let written: Datum<ListParWithRandom> = decode_datum(&insert.data[0]);
            assert_eq!(
                written, expected,
                "winner must be lowest-hash across {{base, diff, added}}"
            );
        }
        other => panic!("expected TrieInsertBinaryProduce, got {:?}", other),
    }
}

/// EventLogIndex-level test: build two EventLogIndexes with conflicting
/// state-channel data, combine them, and assert the combined
/// `state_channels_data` carries the byte-level lowest-hash winner.
/// This proves the parallel State pipeline survives the EventLogIndex
/// combine path used during multi-parent merge.
#[test]
fn event_log_index_combine_picks_lowest_hash_winner_for_state_channel() {
    use rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex;

    let channel_hash = Blake2b256Hash::new(&[0xAB; 32]);
    let datum_left = make_state_datum(0x10, 111);
    let datum_right = make_state_datum(0xF0, 222);

    let left_bytes = encode_datum(&datum_left);
    let right_bytes = encode_datum(&datum_right);

    // Compute expected winner via the typed function (ground truth).
    let expected_datum = combine_mergeable_state(&datum_left, &datum_right, MergeType::MutexState);
    let expected_bytes = encode_datum(&expected_datum);

    // Construct two EventLogIndexes, each carrying one State Datum on the
    // same channel.
    let mut left = EventLogIndex::empty();
    left.state_channels_data.insert(
        channel_hash.clone(),
        (left_bytes.clone(), MergeType::MutexState),
    );
    let mut right = EventLogIndex::empty();
    right.state_channels_data.insert(
        channel_hash.clone(),
        (right_bytes.clone(), MergeType::MutexState),
    );

    let combined = EventLogIndex::combine(&left, &right)
        .expect("EventLogIndex::combine must succeed for matching merge types");

    let (combined_bytes, combined_mt) = combined
        .state_channels_data
        .get(&channel_hash)
        .expect("combined state_channels_data must contain the merged channel");
    assert_eq!(*combined_mt, MergeType::MutexState);
    assert_eq!(
        combined_bytes, &expected_bytes,
        "EventLogIndex::combine must pick the lowest-hash-of-bincode winner"
    );
}
