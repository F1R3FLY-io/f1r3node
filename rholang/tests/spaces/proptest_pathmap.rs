//! PathMap Prefix Semantics Tests
//!
//! Tests for PathMap-based RSpaces with prefix aggregation semantics.
//! These tests verify that:
//! - Data sent on child paths is visible at parent prefixes
//! - Suffix keys are correctly computed for prefix matches
//! - Par-to-path conversion works for Rholang integration
//!
//! # Rholang Syntax Correspondence
//!
//! ```rholang
//! new PathMapFactory(`rho:space:bag:pathmap:default`), space in {
//!   PathMapFactory!({}, *space) |
//!   for (s <- space) {
//!     use s {
//!       // Send to leaf path
//!       @[0, 1, 2]!("data") |
//!
//!       // Receive at prefix - gets [2, "data"] (suffix-tagged)
//!       for (@x <- @[0, 1]) {
//!         stdout!(x)  // Outputs: [2, "data"]
//!       }
//!     }
//!   }
//! }
//! ```

use std::collections::BTreeSet;

use rholang::rust::interpreter::spaces::{
    SpaceId, SpaceQualifier, GenericRSpace,
    WildcardMatch,
    par_to_path, path_to_par, is_par_path, get_path_suffix,
    RholangPathMapStore,
};
use rholang::rust::interpreter::spaces::collections::{
    BagDataCollection, BagContinuationCollection,
};
use rholang::rust::interpreter::spaces::channel_store::ChannelStore;
use rholang::rust::interpreter::spaces::agent::SpaceAgent;

use models::rhoapi::{Par, Expr, EList, BindPattern, TaggedContinuation, ListParWithRandom, ParWithRandom};
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::tagged_continuation::TaggedCont;

// =============================================================================
// Par-to-Path Conversion Tests
// =============================================================================

#[test]
fn test_par_to_path_simple_int_list() {
    // Create a Par representing @[0, 1, 2] using legacy format
    let par = path_to_par(&[0, 1, 2]);

    // Convert back to path - now returns tagged encoding
    let path = par_to_path(&par);
    assert!(path.is_some());

    // The encoded path should be 3 integers with 9 bytes each (tag + i64 LE)
    let encoded = path.unwrap();
    assert_eq!(encoded.len(), 27); // 3 * 9 bytes

    // Verify each element: tag 0x01 + i64 little-endian
    assert_eq!(encoded[0], 0x01); // Integer tag
    assert_eq!(encoded[1..9], [0, 0, 0, 0, 0, 0, 0, 0]); // i64 LE for 0

    assert_eq!(encoded[9], 0x01);
    assert_eq!(encoded[10..18], [1, 0, 0, 0, 0, 0, 0, 0]); // i64 LE for 1

    assert_eq!(encoded[18], 0x01);
    assert_eq!(encoded[19..27], [2, 0, 0, 0, 0, 0, 0, 0]); // i64 LE for 2
}

#[test]
fn test_par_to_path_empty_list() {
    // Create a Par representing @[]
    let par = path_to_par(&[]);

    // Convert back to path
    let path = par_to_path(&par);
    assert_eq!(path, Some(vec![]));
}

#[test]
fn test_par_to_path_single_element() {
    // Create a Par representing @[42] using legacy format
    let par = path_to_par(&[42]);

    // Convert back to path - now returns tagged encoding
    let path = par_to_path(&par);
    assert!(path.is_some());

    // Single integer: tag 0x01 + i64 LE = 9 bytes
    let encoded = path.unwrap();
    assert_eq!(encoded.len(), 9);
    assert_eq!(encoded[0], 0x01); // Integer tag
    assert_eq!(encoded[1..9], [42, 0, 0, 0, 0, 0, 0, 0]); // i64 LE for 42
}

#[test]
fn test_par_to_path_max_byte_values() {
    // Test with max byte values using legacy format
    let par = path_to_par(&[0, 128, 255]);
    let path = par_to_path(&par);
    assert!(path.is_some());

    // 3 integers * 9 bytes each = 27 bytes
    let encoded = path.unwrap();
    assert_eq!(encoded.len(), 27);

    // Verify values are properly encoded as i64 LE
    assert_eq!(encoded[0], 0x01);
    assert_eq!(encoded[1..9], [0, 0, 0, 0, 0, 0, 0, 0]); // 0

    assert_eq!(encoded[9], 0x01);
    assert_eq!(encoded[10..18], [128, 0, 0, 0, 0, 0, 0, 0]); // 128

    assert_eq!(encoded[18], 0x01);
    assert_eq!(encoded[19..27], [255, 0, 0, 0, 0, 0, 0, 0]); // 255
}

#[test]
fn test_is_par_path_true_for_valid_paths() {
    let par = path_to_par(&[0, 1, 2]);
    assert!(is_par_path(&par));
}

#[test]
fn test_is_par_path_true_for_string_par() {
    // With the new encoding, a Par with a string IS a valid path!
    // Strings are encoded with tag 0x02 + varint length + UTF-8 bytes
    let par = Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::GString("hello".to_string())),
    }]);
    // Now strings can be path elements
    assert!(is_par_path(&par));

    // Verify the encoding
    let path = par_to_path(&par);
    assert!(path.is_some());
    let encoded = path.unwrap();
    // tag 0x02 + varint(5) + "hello" = 1 + 1 + 5 = 7 bytes
    assert_eq!(encoded.len(), 7);
    assert_eq!(encoded[0], 0x02); // String tag
    assert_eq!(encoded[1], 5);    // Varint length
    assert_eq!(&encoded[2..7], b"hello");
}

#[test]
fn test_path_roundtrip() {
    // Test semantic roundtrip: par -> encoded bytes -> par
    // With the new encoding, the bytes change but the semantic values are preserved
    for len in 0..10 {
        let path: Vec<u8> = (0..len).collect();
        let par1 = path_to_par(&path);    // Create Par with integers 0..len
        let encoded = par_to_path(&par1); // Encode with tagged format
        assert!(encoded.is_some(), "Encoding failed for length {}", len);

        let encoded = encoded.unwrap();
        // Each integer takes 9 bytes (tag + i64 LE)
        assert_eq!(encoded.len(), len as usize * 9, "Wrong encoded length for {} elements", len);

        // Decode back to Par
        let par2 = path_to_par(&encoded);

        // Extract integers from both Pars and compare
        fn extract_ints(par: &Par) -> Vec<i64> {
            par.exprs.iter()
                .filter_map(|expr| match &expr.expr_instance {
                    Some(ExprInstance::GInt(n)) => Some(*n),
                    _ => None,
                })
                .chain(par.exprs.iter().filter_map(|expr| match &expr.expr_instance {
                    Some(ExprInstance::EListBody(elist)) => {
                        Some(elist.ps.iter().filter_map(|p| {
                            p.exprs.iter().find_map(|e| match &e.expr_instance {
                                Some(ExprInstance::GInt(n)) => Some(*n),
                                _ => None,
                            })
                        }).collect::<Vec<_>>())
                    },
                    _ => None,
                }).flatten())
                .collect()
        }

        let ints1 = extract_ints(&par1);
        let ints2 = extract_ints(&par2);
        assert_eq!(ints1, ints2, "Semantic roundtrip failed for length {}", len);
    }
}

// =============================================================================
// Suffix Key Computation Tests
// =============================================================================

/// Helper to create an encoded path for testing suffix operations
fn encoded_int_path(ints: &[i64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ints.len() * 9);
    for &n in ints {
        buf.push(0x01); // Integer tag
        buf.extend_from_slice(&n.to_le_bytes());
    }
    buf
}

#[test]
fn test_get_path_suffix_simple() {
    // With new encoding: prefix [0,1] and descendant [0,1,2] are encoded bytes
    let prefix = encoded_int_path(&[0, 1]);
    let descendant = encoded_int_path(&[0, 1, 2]);

    let suffix = get_path_suffix(&prefix, &descendant);
    // Suffix is the encoded bytes for integer 2
    assert_eq!(suffix, Some(encoded_int_path(&[2])));
}

#[test]
fn test_get_path_suffix_multi_element() {
    let prefix = encoded_int_path(&[0]);
    let descendant = encoded_int_path(&[0, 1, 2, 3]);

    let suffix = get_path_suffix(&prefix, &descendant);
    // Suffix is the encoded bytes for integers 1, 2, 3
    assert_eq!(suffix, Some(encoded_int_path(&[1, 2, 3])));
}

#[test]
fn test_get_path_suffix_exact_match_returns_empty() {
    let path = encoded_int_path(&[0, 1, 2]);

    let suffix = get_path_suffix(&path, &path);
    assert_eq!(suffix, Some(vec![]));
}

#[test]
fn test_get_path_suffix_not_prefix_returns_none() {
    let prefix = encoded_int_path(&[0, 2]);  // Not a prefix of [0, 1, 2]
    let descendant = encoded_int_path(&[0, 1, 2]);

    let suffix = get_path_suffix(&prefix, &descendant);
    assert_eq!(suffix, None);
}

#[test]
fn test_get_path_suffix_longer_prefix_returns_none() {
    let prefix = encoded_int_path(&[0, 1, 2, 3]);  // Longer than descendant
    let descendant = encoded_int_path(&[0, 1]);

    let suffix = get_path_suffix(&prefix, &descendant);
    assert_eq!(suffix, None);
}

// =============================================================================
// RholangPathMapStore Tests
// =============================================================================

fn create_test_pathmap_store() -> RholangPathMapStore<
    BindPattern,
    ListParWithRandom,
    TaggedContinuation,
    BagDataCollection<ListParWithRandom>,
    BagContinuationCollection<BindPattern, TaggedContinuation>,
> {
    RholangPathMapStore::new(
        BagDataCollection::new,
        BagContinuationCollection::new,
    )
}

fn create_test_data() -> ListParWithRandom {
    ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString("test-data".to_string())),
        }])],
        random_state: vec![],
    }
}

#[test]
fn test_pathmap_store_supports_prefix_semantics() {
    let store = create_test_pathmap_store();
    assert!(store.supports_prefix_semantics());
}

#[test]
fn test_pathmap_store_indexes_path_channels() {
    let mut store = create_test_pathmap_store();

    // Create path channels
    let ch1 = path_to_par(&[0, 1, 2]);
    let ch2 = path_to_par(&[0, 1, 3]);
    let ch3 = path_to_par(&[0, 2]);

    // Store data at each channel
    let _ = store.get_or_create_data_collection(&ch1);
    let _ = store.get_or_create_data_collection(&ch2);
    let _ = store.get_or_create_data_collection(&ch3);

    // Query with prefix @[0, 1]
    let prefix = path_to_par(&[0, 1]);
    let descendants = store.channels_with_prefix(&prefix);

    assert_eq!(descendants.len(), 2, "Should find 2 channels with prefix [0, 1]");
    assert!(descendants.contains(&ch1));
    assert!(descendants.contains(&ch2));
    assert!(!descendants.contains(&ch3)); // [0, 2] is not under [0, 1]
}

#[test]
fn test_pathmap_store_compute_suffix_key() {
    let store = create_test_pathmap_store();

    let prefix = path_to_par(&[0, 1]);
    let descendant = path_to_par(&[0, 1, 2, 3]);

    let suffix = store.compute_suffix_key(&prefix, &descendant);
    // Suffix should be the encoded bytes for integers 2 and 3
    assert_eq!(suffix, Some(encoded_int_path(&[2, 3])));
}

#[test]
fn test_pathmap_store_compute_suffix_key_exact_match() {
    let store = create_test_pathmap_store();

    let channel = path_to_par(&[0, 1, 2]);

    // Exact match should return empty suffix
    let suffix = store.compute_suffix_key(&channel, &channel);
    assert_eq!(suffix, Some(vec![]));
}

// =============================================================================
// GenericRSpace with PathMap Store Tests
// =============================================================================

type TestPathMapRSpace = GenericRSpace<
    RholangPathMapStore<
        BindPattern,
        ListParWithRandom,
        TaggedContinuation,
        BagDataCollection<ListParWithRandom>,
        BagContinuationCollection<BindPattern, TaggedContinuation>,
    >,
    WildcardMatch<BindPattern, ListParWithRandom>,
>;

fn create_test_pathmap_space() -> TestPathMapRSpace {
    let store = RholangPathMapStore::new(
        BagDataCollection::new,
        BagContinuationCollection::new,
    );
    let matcher = WildcardMatch::new();
    GenericRSpace::new(store, matcher, SpaceId::default_space(), SpaceQualifier::Default)
}

fn create_simple_pattern() -> BindPattern {
    BindPattern {
        patterns: vec![],
        remainder: None,
        free_count: 0,
    }
}

fn create_simple_continuation() -> TaggedContinuation {
    TaggedContinuation {
        tagged_cont: Some(TaggedCont::ParBody(ParWithRandom {
            body: Some(Par::default()),
            random_state: vec![],
        })),
    }
}

#[test]
fn test_pathmap_space_produce_consume_exact_match() {
    let mut space = create_test_pathmap_space();

    let channel = path_to_par(&[0, 1, 2]);
    let data = create_test_data();

    // Produce data
    let result = space.produce(channel.clone(), data.clone(), false, None);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "Produce should store data (no waiting continuation)");

    // Consume with exact match
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();
    let result = space.consume(
        vec![channel.clone()],
        vec![pattern],
        continuation,
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some(), "Consume should match data");

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].channel, channel);
    // Exact match - no suffix key
    assert!(results[0].suffix_key.is_none() || results[0].suffix_key.as_ref().map(|k| k.is_empty()).unwrap_or(false));
}

#[test]
fn test_pathmap_space_prefix_consume() {
    let mut space = create_test_pathmap_space();

    // Produce data at leaf path @[0, 1, 2]
    let leaf_channel = path_to_par(&[0, 1, 2]);
    let data = create_test_data();
    let result = space.produce(leaf_channel.clone(), data.clone(), false, None);
    assert!(result.is_ok());

    // Consume at prefix path @[0, 1]
    let prefix_channel = path_to_par(&[0, 1]);
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();
    let result = space.consume(
        vec![prefix_channel.clone()],
        vec![pattern],
        continuation,
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some(), "Prefix consume should match data from descendant");

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1);
    // The actual channel should be the leaf where data was found
    assert_eq!(results[0].channel, leaf_channel);
    // Suffix key should be the encoded bytes for integer 2
    assert!(results[0].suffix_key.is_some());
    assert_eq!(results[0].suffix_key.as_ref().unwrap(), &encoded_int_path(&[2]));
}

#[test]
fn test_pathmap_space_multiple_descendants() {
    let mut space = create_test_pathmap_space();

    // Produce data at multiple leaf paths under @[0, 1]
    let ch1 = path_to_par(&[0, 1, 2]);
    let ch2 = path_to_par(&[0, 1, 3]);
    let ch3 = path_to_par(&[0, 1, 4]);

    for (ch, suffix) in [(ch1.clone(), "a"), (ch2.clone(), "b"), (ch3.clone(), "c")] {
        let data = ListParWithRandom {
            pars: vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString(format!("data-{}", suffix))),
            }])],
            random_state: vec![],
        };
        let result = space.produce(ch, data, false, None);
        assert!(result.is_ok());
    }

    // Consume at prefix @[0, 1] - should get one of the items
    let prefix = path_to_par(&[0, 1]);
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();
    let result = space.consume(
        vec![prefix.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some());

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1);
    // Suffix should be encoded bytes for one of 2, 3, or 4
    let suffix = results[0].suffix_key.as_ref().unwrap();
    let expected_2 = encoded_int_path(&[2]);
    let expected_3 = encoded_int_path(&[3]);
    let expected_4 = encoded_int_path(&[4]);
    assert!(suffix == &expected_2 || suffix == &expected_3 || suffix == &expected_4);

    // Consume again - should get another item
    let result = space.consume(
        vec![prefix.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );
    assert!(result.is_ok());
    assert!(result.unwrap().is_some(), "Second consume should find remaining data");

    // Third consume
    let result = space.consume(
        vec![prefix.clone()],
        vec![pattern],
        continuation,
        false,
        BTreeSet::new(),
    );
    assert!(result.is_ok());
    assert!(result.unwrap().is_some(), "Third consume should find remaining data");
}

#[test]
fn test_pathmap_space_no_match_at_unrelated_prefix() {
    let mut space = create_test_pathmap_space();

    // Produce at @[0, 1, 2]
    let ch = path_to_par(&[0, 1, 2]);
    let data = create_test_data();
    let _ = space.produce(ch, data, false, None);

    // Try to consume at @[0, 2] - unrelated path, should not match
    let unrelated = path_to_par(&[0, 2]);
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();
    let result = space.consume(
        vec![unrelated],
        vec![pattern],
        continuation,
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "Consume at unrelated path should store continuation");
}

#[test]
fn test_pathmap_space_exact_match_preferred_over_prefix() {
    let mut space = create_test_pathmap_space();

    // Produce at both @[0, 1] (exact) and @[0, 1, 2] (descendant)
    let exact_ch = path_to_par(&[0, 1]);
    let descendant_ch = path_to_par(&[0, 1, 2]);

    let exact_data = ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString("exact-data".to_string())),
        }])],
        random_state: vec![],
    };
    let descendant_data = ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString("descendant-data".to_string())),
        }])],
        random_state: vec![],
    };

    let _ = space.produce(exact_ch.clone(), exact_data, false, None);
    let _ = space.produce(descendant_ch.clone(), descendant_data, false, None);

    // Consume at @[0, 1] - should get exact match first
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();
    let result = space.consume(
        vec![exact_ch.clone()],
        vec![pattern],
        continuation,
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some());

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].channel, exact_ch);
    // Exact match - no suffix key (or empty)
    let has_no_suffix = results[0].suffix_key.is_none() ||
        results[0].suffix_key.as_ref().map(|k| k.is_empty()).unwrap_or(false);
    assert!(has_no_suffix, "Exact match should have no suffix key");
}

// =============================================================================
// Property-Based Tests
// =============================================================================

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_path_roundtrip(path in prop_vec(any::<u8>(), 0..20)) {
        // With the new encoding, roundtrip changes format but preserves semantics
        // Note: path_to_par may interpret raw bytes differently depending on whether
        // they match the tagged format (first byte is a valid tag like 1=Integer)
        let par1 = path_to_par(&path);    // Parse path bytes to Par
        let encoded = par_to_path(&par1); // Encode Par back to tagged format
        prop_assert!(encoded.is_some());

        let encoded = encoded.unwrap();
        // Encoding size depends on interpretation - don't assert specific size

        // Decode and verify the values are preserved
        let par2 = path_to_par(&encoded); // Decode tagged format back to Par

        // Both Pars should represent the same path semantically
        let encoded1 = par_to_path(&par1);
        let encoded2 = par_to_path(&par2);
        prop_assert_eq!(encoded1, encoded2, "Semantic roundtrip should preserve path");
    }

    #[test]
    fn prop_suffix_key_length(
        prefix_len in 0usize..10,
        suffix_len in 1usize..10,
    ) {
        // Build encoded paths instead of raw byte paths
        let prefix_ints: Vec<i64> = (0..prefix_len as i64).collect();
        let suffix_ints: Vec<i64> = (prefix_len as i64..(prefix_len + suffix_len) as i64).collect();
        let descendant_ints: Vec<i64> = prefix_ints.iter().chain(suffix_ints.iter()).copied().collect();

        let prefix = encoded_int_path(&prefix_ints);
        let suffix = encoded_int_path(&suffix_ints);
        let descendant = encoded_int_path(&descendant_ints);

        let computed_suffix = get_path_suffix(&prefix, &descendant);
        prop_assert_eq!(computed_suffix, Some(suffix));
    }

    #[test]
    fn prop_prefix_semantics_finds_data(
        leaf_suffix in 1u8..10,
    ) {
        let mut space = create_test_pathmap_space();

        // Produce at @[0, 1, leaf_suffix]
        let leaf = path_to_par(&[0, 1, leaf_suffix]);
        let data = create_test_data();
        let _ = space.produce(leaf.clone(), data, false, None);

        // Consume at @[0, 1] - should find the data
        let prefix = path_to_par(&[0, 1]);
        let pattern = create_simple_pattern();
        let continuation = create_simple_continuation();
        let result = space.consume(
            vec![prefix],
            vec![pattern],
            continuation,
            false,
            BTreeSet::new(),
        );

        prop_assert!(result.is_ok());
        let consume_result = result.unwrap();
        prop_assert!(consume_result.is_some(), "Should find data at descendant path");

        let (_, results) = consume_result.unwrap();
        prop_assert_eq!(results.len(), 1);
        // Suffix key is now encoded bytes for the integer
        prop_assert_eq!(results[0].suffix_key.clone(), Some(encoded_int_path(&[leaf_suffix as i64])));
    }
}

// =============================================================================
// Regression Tests for Atomic Consume (TOCTOU Race Fix)
// =============================================================================

/// Regression test for the TOCTOU race condition in prefix consume.
///
/// Prior to the fix, concurrent consumers with overlapping prefixes could
/// experience a race where:
/// 1. Consumer A (at @[0]) peeks data at @[0,1,2]
/// 2. Consumer B (at @[0,1]) also peeks the same data
/// 3. Consumer A removes the data
/// 4. Consumer B's removal fails silently
/// 5. Consumer B fires with EMPTY bindings (bug!)
///
/// After the fix, consume uses atomic find_and_remove, so:
/// - Each consume either succeeds with valid data OR stores a wait pattern
/// - No consume fires with empty bindings
#[test]
fn regression_atomic_prefix_consume_no_empty_bindings() {
    // Simulates the race scenario from pathmap_prefix_aggregation.rho
    let mut space = create_test_pathmap_space();

    // Produce data at @[0,1,2] (visible to both @[0] and @[0,1] consumers)
    let ch_012 = path_to_par(&[0, 1, 2]);
    let data = create_test_data();
    let _ = space.produce(ch_012.clone(), data.clone(), false, None);

    // First consume at @[0,1] - should succeed (atomic remove)
    let prefix_01 = path_to_par(&[0, 1]);
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();
    let result = space.consume(
        vec![prefix_01.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some(), "First consume should succeed");

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1, "First consume should have exactly 1 result");
    // Verify the result has valid data, not empty
    assert!(!results[0].matched_datum.pars.is_empty(), "Result should have valid data, not empty");

    // Second consume at @[0] - should store wait pattern (no data left)
    let prefix_0 = path_to_par(&[0]);
    let result = space.consume(
        vec![prefix_0.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    // This should be None (wait pattern stored), NOT Some with empty results
    assert!(consume_result.is_none(), "Second consume should store wait pattern, not fire with empty bindings");
}

/// Test that overlapping prefix consumers correctly compete for data.
///
/// When data is available for multiple consumers with overlapping prefixes,
/// each consume should get valid data atomically. No consume should get
/// empty bindings due to race conditions.
#[test]
fn regression_overlapping_prefix_consumers_all_get_valid_data() {
    let mut space = create_test_pathmap_space();

    // Produce data at multiple descendant paths
    let ch_012 = path_to_par(&[0, 1, 2]);
    let ch_013 = path_to_par(&[0, 1, 3]);
    let ch_02 = path_to_par(&[0, 2]);

    let data_a = ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString("data-A".to_string())),
        }])],
        random_state: vec![],
    };
    let data_b = ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString("data-B".to_string())),
        }])],
        random_state: vec![],
    };
    let data_c = ListParWithRandom {
        pars: vec![Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString("data-C".to_string())),
        }])],
        random_state: vec![],
    };

    let _ = space.produce(ch_012.clone(), data_a, false, None);
    let _ = space.produce(ch_013.clone(), data_b, false, None);
    let _ = space.produce(ch_02.clone(), data_c, false, None);

    // Consume at @[0,1] - should get data from @[0,1,2] or @[0,1,3]
    let prefix_01 = path_to_par(&[0, 1]);
    let pattern = create_simple_pattern();
    let continuation = create_simple_continuation();

    let result = space.consume(
        vec![prefix_01.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some(), "First @[0,1] consume should succeed");

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0].matched_datum.pars.is_empty(), "Result must have valid data");

    // Consume at @[0] - should get remaining data
    let prefix_0 = path_to_par(&[0]);
    let result = space.consume(
        vec![prefix_0.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some(), "@[0] consume should succeed (data at @[0,2] or remaining @[0,1,*])");

    let (_, results) = consume_result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0].matched_datum.pars.is_empty(), "Result must have valid data");

    // Third consume - should get the last item
    let result = space.consume(
        vec![prefix_0.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    let consume_result = result.unwrap();
    assert!(consume_result.is_some(), "Third consume should succeed (one item left)");

    let (_, results) = consume_result.unwrap();
    assert!(!results[0].matched_datum.pars.is_empty(), "Result must have valid data");

    // Fourth consume - should store wait pattern (no data left)
    let result = space.consume(
        vec![prefix_0.clone()],
        vec![pattern.clone()],
        continuation.clone(),
        false,
        BTreeSet::new(),
    );

    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "Fourth consume should store wait pattern (no data)");
}
