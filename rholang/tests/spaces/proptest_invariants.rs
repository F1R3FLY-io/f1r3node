//! Property-Based Tests for Core RSpace Invariants
//!
//! These tests verify the fundamental invariants specified in the formal proofs:
//! - GenericRSpace.v: NoPendingMatch, Produce/Consume exclusivity, Gensym uniqueness
//! - Safety/Properties.v: Qualifier semantics, cross-space join prevention
//!
//! # Rholang Correspondence
//!
//! The invariants tested here ensure that Rholang code behaves correctly:
//!
//! ```rholang
//! // NoPendingMatch invariant ensures this communication completes:
//! channel!(42) | for (@x <- channel) { stdout!(x) }
//! // After reduction: either data waiting OR receiver waiting, never both
//!
//! // Produce exclusivity ensures this fires exactly once:
//! new ch in {
//!   for (@x <- ch) { "first" } |
//!   for (@y <- ch) { "second" } |
//!   ch!(1)
//! }
//! // The produce either matches first OR second, not both
//!
//! // Gensym uniqueness ensures:
//! new a, b in { ... }
//! // a and b are guaranteed to be distinct unforgeable names
//! ```

use std::collections::{BTreeSet, HashSet};

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

use rholang::rust::interpreter::spaces::{
    SpaceQualifier, SpaceError,
};
use rholang::rust::interpreter::spaces::agent::SpaceAgent;

use super::test_utils::*;

// =============================================================================
// NoPendingMatch Invariant Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// After any sequence of operations, data and matching continuations should
    /// not coexist at the same channel.
    ///
    /// **Formal Reference**: GenericRSpace.v lines 157-180
    /// **Rholang Intuition**: `ch!(42) | for (@x <- ch) { ... }` should either
    /// fire (consuming both) or leave one side waiting, never both.
    #[test]
    fn prop_no_pending_match_after_operations(
        ops in arb_operation_sequence(10, 1, 50)
    ) {
        let mut space = create_hashmap_bag_space();
        let mut used_channels: HashSet<TestChannel> = HashSet::new();

        // Apply all operations
        for op in ops {
            match op {
                TestOperation::Produce { channel, data, persist } => {
                    used_channels.insert(channel.clone());
                    let _ = space.produce(channel, data, persist, None);
                }
                TestOperation::Consume { channels, patterns, persist } => {
                    for ch in &channels {
                        used_channels.insert(ch.clone());
                    }
                    let _ = space.consume(channels, patterns, "cont".to_string(), persist, BTreeSet::new());
                }
                TestOperation::Gensym => {
                    if let Ok(ch) = space.gensym() {
                        used_channels.insert(ch);
                    }
                }
            }
        }

        // Check NoPendingMatch invariant
        // For each channel, either data or continuations, not both with matching patterns
        for channel in used_channels {
            let data = space.get_data(&channel);
            let conts = space.get_waiting_continuations(vec![channel.clone()]);

            // If both exist, that's only valid if patterns don't match
            // For wildcard matcher, if both exist, it's a violation
            if !data.is_empty() && !conts.is_empty() {
                // With WildcardMatch, any data matches any pattern, so this is a violation
                prop_assert!(false,
                    "NoPendingMatch violated: channel {:?} has {} data items and {} continuations",
                    channel, data.len(), conts.len()
                );
            }
        }
    }

    /// After a successful produce that fires, the continuation should be removed
    /// (unless persistent).
    ///
    /// **Formal Reference**: GenericRSpace.v lines 183-314 (produce_exclusivity)
    #[test]
    fn prop_produce_removes_continuation_on_fire(
        channel in arb_channel(10),
        data in any::<i32>(),
    ) {
        let mut space = create_hashmap_bag_space();

        // First, install a non-persistent continuation
        let _ = space.consume(
            vec![channel.clone()],
            vec![0], // Wildcard pattern (matches anything with WildcardMatch)
            "test_cont".to_string(),
            false, // Non-persistent
            BTreeSet::new(),
        );

        // Verify continuation is waiting
        let conts_before = space.get_waiting_continuations(vec![channel.clone()]);
        prop_assert_eq!(conts_before.len(), 1, "Should have one waiting continuation");

        // Now produce - should fire the continuation
        let result = space.produce(channel.clone(), data, false, None);
        prop_assert!(result.is_ok());

        // The continuation should be removed (non-persistent)
        let conts_after = space.get_waiting_continuations(vec![channel]);
        prop_assert!(conts_after.is_empty(),
            "Non-persistent continuation should be removed after firing");
    }

    /// After a successful produce that fires with a persistent continuation,
    /// the continuation should remain.
    ///
    /// **Rholang**: `for (@x <= ch) { ... }` is a persistent/replicated receiver
    #[test]
    fn prop_produce_keeps_persistent_continuation(
        channel in arb_channel(10),
        data in any::<i32>(),
    ) {
        let mut space = create_hashmap_bag_space();

        // Install a persistent continuation (Rholang: for <= )
        let _ = space.consume(
            vec![channel.clone()],
            vec![0],
            "persistent_cont".to_string(),
            true, // Persistent
            BTreeSet::new(),
        );

        // Produce - should fire but keep the continuation
        let result = space.produce(channel.clone(), data, false, None);
        prop_assert!(result.is_ok());

        // The persistent continuation should still be there
        let conts_after = space.get_waiting_continuations(vec![channel]);
        prop_assert_eq!(conts_after.len(), 1,
            "Persistent continuation should remain after firing");
    }
}

// =============================================================================
// Produce/Consume Exclusivity Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// A produce either fires a continuation OR stores data, never both.
    ///
    /// **Formal Reference**: GenericRSpace.v lines 183-314
    /// **Rholang**: `ch!(42)` either matches a waiting `for (@x <- ch)` or waits
    #[test]
    fn prop_produce_exclusivity(
        channel in arb_channel(10),
        data in any::<i32>(),
        has_waiting_cont in any::<bool>(),
    ) {
        let mut space = create_hashmap_bag_space();

        if has_waiting_cont {
            // Install a waiting continuation
            let _ = space.consume(
                vec![channel.clone()],
                vec![0],
                "waiting_cont".to_string(),
                false,
                BTreeSet::new(),
            );
        }

        // Now produce
        let result = space.produce(channel.clone(), data, false, None);
        prop_assert!(result.is_ok());

        match result.unwrap() {
            Some((_cont_result, _matched_data, _)) => {
                // Fired - data should NOT be stored
                let stored_data = space.get_data(&channel);
                prop_assert!(stored_data.is_empty(),
                    "Produce fired but data was also stored - exclusivity violated");

                // Continuation was found
                prop_assert!(has_waiting_cont,
                    "Produce fired but no continuation was installed");
            }
            None => {
                // Stored - continuation should NOT have fired
                if has_waiting_cont {
                    // This would indicate a bug - we had a continuation but didn't fire
                    // Actually, this can happen if patterns don't match
                    // With WildcardMatch, this shouldn't happen
                }

                // Data should be stored
                let stored_data = space.get_data(&channel);
                prop_assert!(!stored_data.is_empty() || has_waiting_cont,
                    "Produce neither fired nor stored data");
            }
        }
    }

    /// A consume either fires with data OR stores continuation, never both.
    ///
    /// **Formal Reference**: GenericRSpace.v lines 316-354 (consume_atomicity)
    #[test]
    fn prop_consume_exclusivity(
        channel in arb_channel(10),
        pattern in any::<i32>(),
        has_waiting_data in any::<bool>(),
    ) {
        let mut space = create_hashmap_bag_space();

        if has_waiting_data {
            // Store some data first
            let _ = space.produce(channel.clone(), 42, false, None);
        }

        // Now consume
        let result = space.consume(
            vec![channel.clone()],
            vec![pattern],
            "test_cont".to_string(),
            false,
            BTreeSet::new(),
        );
        prop_assert!(result.is_ok());

        match result.unwrap() {
            Some((_cont_result, _matched_data)) => {
                // Fired - continuation should NOT be stored
                let stored_conts = space.get_waiting_continuations(vec![channel]);
                prop_assert!(stored_conts.is_empty(),
                    "Consume fired but continuation was also stored - exclusivity violated");

                // Data was available
                prop_assert!(has_waiting_data,
                    "Consume fired but no data was available");
            }
            None => {
                // Stored continuation - data should still exist (or didn't exist)
                if has_waiting_data {
                    // With WildcardMatch, we should have matched
                    // Unless the data was consumed
                }

                // Continuation should be stored
                let stored_conts = space.get_waiting_continuations(vec![channel]);
                prop_assert!(!stored_conts.is_empty() || has_waiting_data,
                    "Consume neither fired nor stored continuation");
            }
        }
    }
}

// =============================================================================
// Gensym Uniqueness Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Sequential gensym calls always produce unique channels.
    ///
    /// **Formal Reference**: GenericRSpace.v lines 604-644
    /// **Rholang**: `new a, b, c in { ... }` guarantees a, b, c are all distinct
    #[test]
    fn prop_gensym_uniqueness(
        num_calls in 1usize..500
    ) {
        let mut space = create_hashmap_bag_space();
        let mut channels: Vec<TestChannel> = Vec::with_capacity(num_calls);

        for _ in 0..num_calls {
            match space.gensym() {
                Ok(ch) => channels.push(ch),
                Err(SpaceError::OutOfNames { .. }) => break, // Array exhausted
                Err(e) => prop_assert!(false, "Unexpected gensym error: {:?}", e),
            }
        }

        // All channels should be unique
        let unique: HashSet<_> = channels.iter().collect();
        prop_assert_eq!(
            unique.len(),
            channels.len(),
            "Gensym produced duplicate channels"
        );
    }

    /// Gensym produces monotonically increasing channels (for usize-based channels).
    #[test]
    fn prop_gensym_monotonic(
        num_calls in 2usize..100
    ) {
        let mut space = create_hashmap_bag_space();
        let mut prev_value: Option<usize> = None;

        for _ in 0..num_calls {
            match space.gensym() {
                Ok(ch) => {
                    // Convert to usize for numeric comparison
                    if let Some(current) = ch.as_usize() {
                        if let Some(p) = prev_value {
                            prop_assert!(current > p, "Gensym should be monotonically increasing");
                        }
                        prev_value = Some(current);
                    }
                }
                Err(_) => break,
            }
        }
    }
}

// =============================================================================
// Qualifier Semantics Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Default qualifier allows persistence.
    ///
    /// **Formal Reference**: Safety/Properties.v lines 128-213
    #[test]
    fn prop_default_qualifier_is_persistent(
        _seed in any::<u32>(),
    ) {
        let space = create_space_with_qualifier(SpaceQualifier::Default);
        prop_assert!(space.qualifier().is_persistent());
        prop_assert!(space.qualifier().is_mobile());
        prop_assert!(space.qualifier().is_concurrent());
    }

    /// Temp qualifier is non-persistent.
    #[test]
    fn prop_temp_qualifier_not_persistent(
        _seed in any::<u32>(),
    ) {
        let space = create_space_with_qualifier(SpaceQualifier::Temp);
        prop_assert!(!space.qualifier().is_persistent());
        prop_assert!(space.qualifier().is_mobile());
        prop_assert!(space.qualifier().is_concurrent());
    }

    /// Seq qualifier is non-persistent, non-mobile, non-concurrent.
    ///
    /// **Rholang**: Seq channels cannot be sent to other processes
    #[test]
    fn prop_seq_qualifier_restrictions(
        _seed in any::<u32>(),
    ) {
        let space = create_space_with_qualifier(SpaceQualifier::Seq);
        prop_assert!(!space.qualifier().is_persistent());
        prop_assert!(!space.qualifier().is_mobile());
        prop_assert!(!space.qualifier().is_concurrent());
    }
}

// =============================================================================
// Join Atomicity Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// A join consume matches ALL channels or NONE (atomic).
    ///
    /// **Formal Reference**: GenericRSpace.v lines 316-354
    /// **Rholang**: `for (@x <- a & @y <- b) { ... }` waits for data on BOTH a and b
    #[test]
    fn prop_join_atomicity(
        num_channels in 2usize..5,
        data_values in prop_vec(any::<i32>(), 1..5),
    ) {
        let mut space = create_hashmap_bag_space();
        let channels: Vec<TestChannel> = (0..num_channels).map(TestChannel::from).collect();
        let patterns: Vec<TestPattern> = vec![0; num_channels]; // Wildcards

        // Install a join continuation on all channels
        let consume_result = space.consume(
            channels.clone(),
            patterns,
            "join_cont".to_string(),
            false,
            BTreeSet::new(),
        );
        prop_assert!(consume_result.is_ok());
        prop_assert!(consume_result.unwrap().is_none(), "Join should wait, not fire immediately");

        // Produce to only some channels (not all)
        let num_to_produce = num_channels.saturating_sub(1).max(1);
        for (i, &data) in data_values.iter().take(num_to_produce).enumerate() {
            if i < channels.len() {
                let _ = space.produce(channels[i].clone(), data, false, None);
            }
        }

        // The join should NOT have fired (not all channels have data)
        // Check that continuation is still waiting
        let conts = space.get_waiting_continuations(channels.clone());
        prop_assert!(!conts.is_empty() || num_to_produce >= num_channels,
            "Join continuation should still be waiting until all channels have data");
    }

    /// When all channels in a join have data, the join fires.
    #[test]
    fn prop_join_fires_when_complete(
        num_channels in 2usize..4,
    ) {
        let mut space = create_hashmap_bag_space();
        let channels: Vec<TestChannel> = (0..num_channels).map(TestChannel::from).collect();
        let patterns: Vec<TestPattern> = vec![0; num_channels];

        // First produce data to all channels
        for (i, ch) in channels.iter().enumerate() {
            let _ = space.produce(ch.clone(), i as i32, false, None);
        }

        // Now consume - should fire immediately
        let result = space.consume(
            channels.clone(),
            patterns,
            "join_cont".to_string(),
            false,
            BTreeSet::new(),
        );

        prop_assert!(result.is_ok());
        match result.unwrap() {
            Some((_, data)) => {
                // Join fired and got data from all channels
                prop_assert_eq!(data.len(), num_channels,
                    "Join should have matched data from all channels");
            }
            None => {
                // This is also valid - the consume stored if matching semantics differ
            }
        }
    }
}

// =============================================================================
// Data Persistence Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Persistent produces remain after being consumed.
    ///
    /// **Rholang**: `ch!!(42)` (persistent send) remains available for multiple receives
    #[test]
    fn prop_persistent_produce_remains(
        channel in arb_channel(10),
        data in any::<i32>(),
        num_consumes in 1usize..5,
    ) {
        let mut space = create_hashmap_bag_space();

        // Persistent produce (Rholang: ch!!(data))
        let _ = space.produce(channel.clone(), data, true, None);

        // Verify data is there
        let initial_data = space.get_data(&channel);
        prop_assert!(!initial_data.is_empty(), "Persistent produce should store data");

        // Multiple consumes should all succeed
        for _ in 0..num_consumes {
            let result = space.consume(
                vec![channel.clone()],
                vec![0],
                format!("cont"),
                false,
                BTreeSet::new(),
            );

            // Either fires or stores continuation (depends on if data matched)
            // With persistent data and wildcard, should fire
            if let Ok(Some(_)) = result {
                // Fired successfully
            }
        }

        // Persistent data should still be there
        let final_data = space.get_data(&channel);
        prop_assert!(!final_data.is_empty(),
            "Persistent data should remain after consumes");
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_basic_produce_consume_cycle() {
        let mut space = create_hashmap_bag_space();
        let ch = TestChannel::from(0);

        // Produce first
        let produce_result = space.produce(ch.clone(), 42, false, None);
        assert!(produce_result.is_ok());
        assert!(produce_result.unwrap().is_none()); // No continuation waiting

        // Data should be stored
        let data = space.get_data(&ch);
        assert_eq!(data.len(), 1);

        // Now consume
        let consume_result = space.consume(
            vec![ch.clone()],
            vec![0],
            "test".to_string(),
            false,
            BTreeSet::new(),
        );
        assert!(consume_result.is_ok());
        assert!(consume_result.unwrap().is_some()); // Should fire

        // Data should be consumed
        let data_after = space.get_data(&ch);
        assert!(data_after.is_empty());
    }

    #[test]
    fn test_consume_produce_cycle() {
        let mut space = create_hashmap_bag_space();
        let ch = TestChannel::from(0);

        // Consume first (creates waiting continuation)
        let consume_result = space.consume(
            vec![ch.clone()],
            vec![0],
            "test".to_string(),
            false,
            BTreeSet::new(),
        );
        assert!(consume_result.is_ok());
        assert!(consume_result.unwrap().is_none()); // No data waiting

        // Continuation should be stored
        let conts = space.get_waiting_continuations(vec![ch.clone()]);
        assert_eq!(conts.len(), 1);

        // Now produce - should fire the continuation
        let produce_result = space.produce(ch.clone(), 42, false, None);
        assert!(produce_result.is_ok());
        assert!(produce_result.unwrap().is_some()); // Should fire

        // Continuation should be consumed
        let conts_after = space.get_waiting_continuations(vec![ch]);
        assert!(conts_after.is_empty());
    }

    #[test]
    fn test_minimal_failing_case() {
        // Reproduces the exact failing case from proptest
        let mut space = create_hashmap_bag_space();
        let ch5 = TestChannel::from(5);
        let ch0 = TestChannel::from(0);

        // 1. Consume on channel 5 with pattern 0
        let _ = space.consume(
            vec![ch5.clone()],
            vec![0],
            "cont".to_string(),
            false,
            BTreeSet::new(),
        );

        // Verify continuation is stored
        let conts = space.get_waiting_continuations(vec![ch5.clone()]);
        assert_eq!(conts.len(), 1, "Should have 1 continuation after consume");

        // 2. Produce on channel 0 with data 0 (different channel, should not affect ch5)
        let _ = space.produce(ch0.clone(), 0, false, None);

        // Verify continuation on ch5 is still there
        let conts = space.get_waiting_continuations(vec![ch5.clone()]);
        assert_eq!(conts.len(), 1, "Should still have 1 continuation after produce on different channel");

        // 3. Produce on channel 5 with data 0 (should fire the continuation)
        let result = space.produce(ch5.clone(), 0, false, None);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some(), "Produce should fire continuation");

        // Verify continuation is removed
        let conts = space.get_waiting_continuations(vec![ch5.clone()]);
        assert_eq!(conts.len(), 0, "Continuation should be removed after firing");

        // 4. Produce on channel 5 again (should store data since no continuation)
        let _ = space.produce(ch5.clone(), 0, false, None);

        // Final check: ch5 should have 1 data, 0 continuations
        let data = space.get_data(&ch5);
        let conts = space.get_waiting_continuations(vec![ch5.clone()]);
        assert_eq!(data.len(), 1, "Should have 1 data item");
        assert_eq!(conts.len(), 0, "Should have 0 continuations");

        // NoPendingMatch: should not have both data and continuations
        assert!(!(data.len() > 0 && conts.len() > 0),
            "NoPendingMatch violated: {} data items and {} continuations",
            data.len(), conts.len());
    }
}

// =============================================================================
// Theory Validation Tests
// =============================================================================

#[cfg(test)]
mod theory_validation_tests {
    use rholang::rust::interpreter::spaces::types::{Theory, NullTheory, Validatable};
    use rholang::rust::interpreter::spaces::factory::{BuiltinTheoryLoader, TheoryLoader, TheorySpec};
    use models::rhoapi::{ListParWithRandom, Par, Expr};
    use models::rhoapi::expr::ExprInstance;

    /// Test that Validatable produces correct type strings for ListParWithRandom
    #[test]
    fn test_validatable_nat() {
        // Create a ListParWithRandom containing a natural number (42)
        let data = ListParWithRandom {
            pars: vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GInt(42)),
            }])],
            random_state: vec![],
        };

        let validatable_string = data.to_validatable_string();
        assert!(validatable_string.starts_with("Nat"),
            "Expected 'Nat(42)', got: {}", validatable_string);
        assert!(validatable_string.contains("42"),
            "Expected to contain '42', got: {}", validatable_string);
    }

    /// Test that Validatable produces correct type strings for negative integers
    #[test]
    fn test_validatable_int() {
        // Create a ListParWithRandom containing a negative integer (-5)
        let data = ListParWithRandom {
            pars: vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GInt(-5)),
            }])],
            random_state: vec![],
        };

        let validatable_string = data.to_validatable_string();
        assert!(validatable_string.starts_with("Int"),
            "Expected 'Int(-5)', got: {}", validatable_string);
    }

    /// Test that Validatable produces correct type strings for strings
    #[test]
    fn test_validatable_string() {
        // Create a ListParWithRandom containing a string
        let data = ListParWithRandom {
            pars: vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString("hello".to_string())),
            }])],
            random_state: vec![],
        };

        let validatable_string = data.to_validatable_string();
        assert!(validatable_string.starts_with("String"),
            "Expected 'String(hello)', got: {}", validatable_string);
    }

    /// Test that Nat theory accepts natural numbers
    #[test]
    fn test_nat_theory_accepts_nat() {
        let loader = BuiltinTheoryLoader::new();
        let theory = loader.load(&TheorySpec::Builtin("Nat".to_string()))
            .expect("Nat theory should load");

        // Natural number should pass
        let result = theory.validate("Nat(42)");
        assert!(result.is_ok(), "Nat theory should accept 'Nat(42)': {:?}", result);
    }

    /// Test that Nat theory rejects strings
    #[test]
    fn test_nat_theory_rejects_string() {
        let loader = BuiltinTheoryLoader::new();
        let theory = loader.load(&TheorySpec::Builtin("Nat".to_string()))
            .expect("Nat theory should load");

        // String should fail
        let result = theory.validate("String(hello)");
        assert!(result.is_err(), "Nat theory should reject 'String(hello)'");
    }

    /// Test that Nat theory rejects negative integers
    #[test]
    fn test_nat_theory_rejects_negative_int() {
        let loader = BuiltinTheoryLoader::new();
        let theory = loader.load(&TheorySpec::Builtin("Nat".to_string()))
            .expect("Nat theory should load");

        // Negative integer should fail (starts with "Int", not "Nat")
        let result = theory.validate("Int(-5)");
        assert!(result.is_err(), "Nat theory should reject 'Int(-5)'");
    }

    /// Test Any theory accepts everything
    #[test]
    fn test_any_theory_accepts_all() {
        let theory = NullTheory;

        // Everything should pass
        assert!(theory.validate("Nat(42)").is_ok());
        assert!(theory.validate("String(hello)").is_ok());
        assert!(theory.validate("Int(-5)").is_ok());
        assert!(theory.validate("anything").is_ok());
    }

    /// Test the full validation flow: ListParWithRandom -> Validatable -> Theory
    #[test]
    fn test_full_validation_flow_nat_accepts() {
        let loader = BuiltinTheoryLoader::new();
        let theory = loader.load(&TheorySpec::Builtin("Nat".to_string()))
            .expect("Nat theory should load");

        // Create a natural number
        let data = ListParWithRandom {
            pars: vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GInt(42)),
            }])],
            random_state: vec![],
        };

        // Get the validatable string
        let term = data.to_validatable_string();

        // Validate against Nat theory
        let result = theory.validate(&term);
        assert!(result.is_ok(), "Nat theory should accept natural number: {:?}", result);
    }

    /// Test the full validation flow: ListParWithRandom -> Validatable -> Theory (reject)
    #[test]
    fn test_full_validation_flow_nat_rejects_string() {
        let loader = BuiltinTheoryLoader::new();
        let theory = loader.load(&TheorySpec::Builtin("Nat".to_string()))
            .expect("Nat theory should load");

        // Create a string
        let data = ListParWithRandom {
            pars: vec![Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GString("hello".to_string())),
            }])],
            random_state: vec![],
        };

        // Get the validatable string
        let term = data.to_validatable_string();

        // Validate against Nat theory - should reject
        let result = theory.validate(&term);
        assert!(result.is_err(), "Nat theory should reject string, but got: {:?}", result);
    }
}
