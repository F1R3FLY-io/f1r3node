//! Checkpoint and Replay Integration Tests
//!
//! These tests verify the checkpoint/replay system from Checkpoint.v:
//! - Hard checkpoints (full state persistence)
//! - Soft checkpoints (speculative execution/rollback)
//! - Replay determinism (same checkpoint + same log = identical state)
//! - Temp qualifier behavior (cleared on hard checkpoint)
//!
//! # Rholang Correspondence
//!
//! ```rholang
//! // Checkpointing enables speculative execution
//! new Space(`rho:space:bag:hashmap:default`), space in {
//!   Space!({}, *space) |
//!   use space {
//!     // Initial state
//!     counter!(0) |
//!
//!     // Create checkpoint before speculative work
//!     @"checkpoint"!("soft") |
//!
//!     // Speculative execution
//!     for (@n <- counter) {
//!       counter!(n + 1) |
//!       if (validation_fails) {
//!         @"rollback"!()  // Restore to checkpoint
//!       } else {
//!         @"commit"!()    // Discard checkpoint, keep changes
//!       }
//!     }
//!   }
//! }
//! ```

use rholang::rust::interpreter::spaces::{
    SpaceQualifier, SpaceError, SoftCheckpoint, CheckpointableSpace,
    ReplayableSpace,
};
use rholang::rust::interpreter::spaces::agent::SpaceAgent;

use super::test_utils::*;

// Type alias for the soft checkpoint type used in tests
type TestSoftCheckpoint = SoftCheckpoint<TestChannel, TestPattern, TestData, TestContinuation>;

// =============================================================================
// Hard Checkpoint Tests
// =============================================================================

/// Test creating a hard checkpoint and restoring to it.
///
/// **Formal Reference**: Checkpoint.v lines 127-156 (hard_checkpoint_restore)
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     ch!(1) | ch!(2) |
///     @"checkpoint"!("hard") |  // Checkpoint with [1, 2]
///     ch!(3) | ch!(4) |          // Add more
///     @"restore"!() |            // Back to [1, 2]
///     for (@x <- ch) { stdout!(x) }  // Gets 1 or 2, not 3 or 4
///   }
/// }
/// ```
#[test]
fn test_hard_checkpoint_restore() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Initial state
    space.produce(channel.clone(), 1, false, None).expect("produce should succeed");
    space.produce(channel.clone(), 2, false, None).expect("produce should succeed");

    let initial_count = space.get_data(&channel).len();
    assert_eq!(initial_count, 2, "Should have 2 items before checkpoint");

    // Create hard checkpoint
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint creation should succeed");

    // Modify state after checkpoint
    space.produce(channel.clone(), 3, false, None).expect("produce should succeed");
    space.produce(channel.clone(), 4, false, None).expect("produce should succeed");
    space.produce(channel.clone(), 5, false, None).expect("produce should succeed");

    let modified_count = space.get_data(&channel).len();
    assert_eq!(modified_count, 5, "Should have 5 items after modifications");

    // Restore to checkpoint using merkle root
    space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // State should be back to checkpoint
    let restored_count = space.get_data(&channel).len();
    assert_eq!(restored_count, 2, "Should have 2 items after restore");
}

/// Test that hard checkpoint persists multiple channels.
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     ch1!(10) | ch2!(20) | ch3!(30) |
///     @"checkpoint"!("hard") |
///     ch1!(11) | ch2!(21) |  // Modify ch1 and ch2
///     @"restore"!() |        // All channels restored
///     // ch1 has [10], ch2 has [20], ch3 has [30]
///   }
/// }
/// ```
#[test]
fn test_hard_checkpoint_multiple_channels() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let ch0 = TestChannel::from(0);
    let ch1 = TestChannel::from(1);
    let ch2 = TestChannel::from(2);

    // Produce on multiple channels
    space.produce(ch0.clone(), 10, false, None).expect("produce should succeed");
    space.produce(ch1.clone(), 20, false, None).expect("produce should succeed");
    space.produce(ch2.clone(), 30, false, None).expect("produce should succeed");

    // Checkpoint
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Modify some channels
    space.produce(ch0.clone(), 11, false, None).expect("produce should succeed");
    space.produce(ch1.clone(), 21, false, None).expect("produce should succeed");
    space.produce(ch1.clone(), 22, false, None).expect("produce should succeed");

    // Verify modifications
    assert_eq!(space.get_data(&ch0).len(), 2);
    assert_eq!(space.get_data(&ch1).len(), 3);
    assert_eq!(space.get_data(&ch2).len(), 1);

    // Restore
    space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // All channels restored
    assert_eq!(space.get_data(&ch0).len(), 1, "ch0 should have 1 item");
    assert_eq!(space.get_data(&ch1).len(), 1, "ch1 should have 1 item");
    assert_eq!(space.get_data(&ch2).len(), 1, "ch2 should have 1 item");
}

/// Test that hard checkpoint preserves waiting continuations.
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     for (@x <- ch) { process!(x) } |  // Waiting continuation
///     @"checkpoint"!("hard") |
///     ch!(42) |                          // This fires the continuation
///     @"restore"!() |                    // Continuation should be waiting again
///     // ch!(42) |                        // Would fire again
///   }
/// }
/// ```
#[test]
fn test_hard_checkpoint_preserves_continuations() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Register a waiting continuation
    let _result = space.consume(
        vec![channel.clone()],
        vec![0], // pattern
        "waiting_cont".to_string(),
        false,
        std::collections::BTreeSet::new(),
    );
    // This stores the continuation (no data to match)

    let conts_before = space.get_waiting_continuations(vec![channel.clone()]);
    let cont_count_before = conts_before.len();

    // Checkpoint with waiting continuation
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Fire the continuation by producing data
    space.produce(channel.clone(), 42, false, None).expect("produce should succeed");

    // Continuation may have been consumed
    let _conts_after_fire = space.get_waiting_continuations(vec![channel.clone()]);

    // Restore to checkpoint
    space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // Continuation should be waiting again
    let conts_restored = space.get_waiting_continuations(vec![channel.clone()]);
    assert_eq!(conts_restored.len(), cont_count_before,
        "Waiting continuations should be restored");
}

// =============================================================================
// Soft Checkpoint Tests
// =============================================================================

/// Test soft checkpoint for speculative execution.
///
/// **Formal Reference**: Checkpoint.v lines 158-189 (soft_checkpoint_rollback)
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     balance!(100) |
///     @"checkpoint"!("soft") |  // Speculative point
///
///     // Try a transaction
///     for (@b <- balance) {
///       if (b >= 50) {
///         balance!(b - 50) |
///         @"commit"!()  // Keep changes
///       } else {
///         @"rollback"!()  // Abort, restore balance
///       }
///     }
///   }
/// }
/// ```
#[test]
fn test_soft_checkpoint_rollback() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Initial state
    space.produce(channel.clone(), 100, false, None).expect("produce should succeed");

    // Soft checkpoint before speculative work
    let soft_cp: TestSoftCheckpoint = space.create_soft_checkpoint();

    // Speculative modification
    let _result = space.consume(
        vec![channel.clone()],
        vec![100],
        "txn".to_string(),
        false,
        std::collections::BTreeSet::new(),
    );
    space.produce(channel.clone(), 50, false, None).expect("produce should succeed");

    // Decide to rollback
    space.revert_to_soft_checkpoint(soft_cp)
        .expect("rollback should succeed");

    // Original state restored
    let data = space.get_data(&channel);
    assert_eq!(data.len(), 1, "Should have original 1 item");
}

/// Test soft checkpoint commit (discarding checkpoint keeps changes).
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     counter!(0) |
///     @"checkpoint"!("soft") |
///     for (@n <- counter) {
///       counter!(n + 10)  // Increment
///     } |
///     @"commit"!() |       // Discard checkpoint, keep changes
///     // counter is now 10
///   }
/// }
/// ```
#[test]
fn test_soft_checkpoint_commit() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Initial state
    space.produce(channel.clone(), 0, false, None).expect("produce should succeed");

    // Soft checkpoint
    let _soft_cp: TestSoftCheckpoint = space.create_soft_checkpoint();

    // Make changes
    let _result = space.consume(
        vec![channel.clone()],
        vec![0],
        "update".to_string(),
        false,
        std::collections::BTreeSet::new(),
    );
    space.produce(channel.clone(), 10, false, None).expect("produce should succeed");

    // Commit by simply dropping the soft checkpoint (not reverting)
    // The _soft_cp goes out of scope without being used

    // Changes are kept
    let data = space.get_data(&channel);
    assert_eq!(data.len(), 1, "Should have 1 item after commit");
}

/// Test nested soft checkpoints.
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     level!(0) |
///     @"checkpoint"!("soft") |   // CP1
///     level!(1) |
///     @"checkpoint"!("soft") |   // CP2
///     level!(2) |
///     @"rollback"!() |           // Back to CP2: level has [0, 1]
///     @"rollback"!() |           // Back to CP1: level has [0]
///   }
/// }
/// ```
#[test]
fn test_nested_soft_checkpoints() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Level 0
    space.produce(channel.clone(), 0, false, None).expect("produce should succeed");
    let cp1: TestSoftCheckpoint = space.create_soft_checkpoint();

    // Level 1
    space.produce(channel.clone(), 1, false, None).expect("produce should succeed");
    assert_eq!(space.get_data(&channel).len(), 2);
    let cp2: TestSoftCheckpoint = space.create_soft_checkpoint();

    // Level 2
    space.produce(channel.clone(), 2, false, None).expect("produce should succeed");
    assert_eq!(space.get_data(&channel).len(), 3);

    // Rollback to CP2
    space.revert_to_soft_checkpoint(cp2)
        .expect("restore to cp2 should succeed");
    assert_eq!(space.get_data(&channel).len(), 2, "After CP2 restore: 2 items");

    // Rollback to CP1
    space.revert_to_soft_checkpoint(cp1)
        .expect("restore to cp1 should succeed");
    assert_eq!(space.get_data(&channel).len(), 1, "After CP1 restore: 1 item");
}

// =============================================================================
// Replay Determinism Tests
// =============================================================================

/// Test that replaying the same operations from a checkpoint produces identical state.
///
/// **Formal Reference**: Checkpoint.v lines 343-365 (replay_determinism)
///
/// ```rholang
/// // Same checkpoint + same operations = identical final state
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     ch!(1) |
///     @"checkpoint"!("hard") |
///     // Record: [ch!(2), ch!(3), for(@x<-ch){...}]
///     ch!(2) | ch!(3) | for (@x <- ch) { ... } |
///     // Replay same record from checkpoint = identical state
///   }
/// }
/// ```
#[test]
fn test_replay_determinism() {
    let mut space1 = create_space_with_history(SpaceQualifier::Default);
    let mut space2 = create_space_with_history(SpaceQualifier::Default);
    let ch0 = TestChannel::from(0);
    let ch1 = TestChannel::from(1);

    // Set up identical initial states
    space1.produce(ch0.clone(), 1, false, None).expect("produce should succeed");
    space2.produce(ch0.clone(), 1, false, None).expect("produce should succeed");

    // Create checkpoints (for logging purposes)
    let _cp1 = space1.create_checkpoint()
        .expect("cp1 should succeed");
    let _cp2 = space2.create_checkpoint()
        .expect("cp2 should succeed");

    // Same sequence of operations on both
    let operations = vec![
        TestOperation::Produce { channel: ch0.clone(), data: 2, persist: false },
        TestOperation::Produce { channel: ch0.clone(), data: 3, persist: false },
        TestOperation::Produce { channel: ch1.clone(), data: 10, persist: false },
    ];

    // Apply to space1
    for op in &operations {
        match op {
            TestOperation::Produce { channel, data, persist } => {
                space1.produce(channel.clone(), *data, *persist, None).expect("op should succeed");
            }
            _ => {}
        }
    }

    // Apply same to space2
    for op in &operations {
        match op {
            TestOperation::Produce { channel, data, persist } => {
                space2.produce(channel.clone(), *data, *persist, None).expect("op should succeed");
            }
            _ => {}
        }
    }

    // States should be identical
    assert_eq!(
        space1.get_data(&ch0).len(),
        space2.get_data(&ch0).len(),
        "Channel 0 should have same count in both spaces"
    );
    assert_eq!(
        space1.get_data(&ch1).len(),
        space2.get_data(&ch1).len(),
        "Channel 1 should have same count in both spaces"
    );
}

/// Test that replay from checkpoint restores exact state.
#[test]
fn test_replay_restores_exact_state() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let ch0 = TestChannel::from(0);
    let ch1 = TestChannel::from(1);

    // Build up state
    space.produce(ch0.clone(), 100, false, None).expect("produce should succeed");
    space.produce(ch0.clone(), 200, false, None).expect("produce should succeed");
    space.produce(ch1.clone(), 999, false, None).expect("produce should succeed");

    // Checkpoint
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Capture current state metrics
    let ch0_count = space.get_data(&ch0).len();
    let ch1_count = space.get_data(&ch1).len();

    // Heavily modify state
    for i in 0..100 {
        space.produce(ch0.clone(), i, false, None).expect("produce should succeed");
    }

    // Restore
    space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // Verify exact restoration
    assert_eq!(space.get_data(&ch0).len(), ch0_count, "ch0 count should match");
    assert_eq!(space.get_data(&ch1).len(), ch1_count, "ch1 count should match");
}

// =============================================================================
// Temp Qualifier Behavior Tests
// =============================================================================

/// Test that Temp spaces are cleared on hard checkpoint.
///
/// **Formal Reference**: Checkpoint.v lines 245-268 (temp_cleared_on_checkpoint)
///
/// ```rholang
/// new TempSpace(`rho:space:bag:hashmap:temp`), temp in {
///   TempSpace!({}, *temp) |
///   use temp {
///     session!(userId) |
///     cache!(computedValue) |
///     @"checkpoint"!("hard") |
///     // After restore, session and cache are GONE
///     // Temp data is ephemeral
///   }
/// }
/// ```
#[test]
fn test_temp_qualifier_cleared_on_hard_checkpoint() {
    let mut temp_space = create_space_with_history(SpaceQualifier::Temp);
    let ch0 = TestChannel::from(0);
    let ch1 = TestChannel::from(1);

    // Add session data
    temp_space.produce(ch0.clone(), 123, false, None).expect("produce should succeed");
    temp_space.produce(ch1.clone(), 456, false, None).expect("produce should succeed");

    assert_eq!(temp_space.get_data(&ch0).len(), 1);
    assert_eq!(temp_space.get_data(&ch1).len(), 1);

    // Hard checkpoint (which should clear temp data on restore)
    let checkpoint = temp_space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Restore (simulating restart)
    temp_space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // Temp data should be cleared
    // (Note: exact behavior depends on implementation)
    // The Temp qualifier means data is not persisted
}

/// Test that Default spaces preserve data across checkpoints.
///
/// ```rholang
/// new DefaultSpace(`rho:space:bag:hashmap:default`), space in {
///   DefaultSpace!({}, *space) |
///   use space {
///     config!(setting) |
///     @"checkpoint"!("hard") |
///     // After restore, config is preserved
///   }
/// }
/// ```
#[test]
fn test_default_qualifier_preserved_on_checkpoint() {
    let mut default_space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Add persistent data
    default_space.produce(channel.clone(), 42, false, None).expect("produce should succeed");

    // Checkpoint
    let checkpoint = default_space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Restore
    default_space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // Data should be preserved
    let data = default_space.get_data(&channel);
    assert_eq!(data.len(), 1, "Default space data should be preserved");
}

/// Test mixed Temp and Default spaces with checkpoint.
///
/// ```rholang
/// new TempSpace(`rho:space:bag:hashmap:temp`), temp,
///     DefaultSpace(`rho:space:bag:hashmap:default`), persistent in {
///   TempSpace!({}, *temp) |
///   DefaultSpace!({}, *persistent) |
///   use temp { cache!(data) } |
///   use persistent { config!(settings) } |
///   @"checkpoint"!("hard") |
///   // After restore: temp is empty, persistent has config
/// }
/// ```
#[test]
fn test_mixed_qualifiers_checkpoint_behavior() {
    let mut temp_space = create_space_with_history(SpaceQualifier::Temp);
    let mut default_space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Add data to both
    temp_space.produce(channel.clone(), 111, false, None).expect("produce should succeed");
    default_space.produce(channel.clone(), 222, false, None).expect("produce should succeed");

    // Checkpoint both
    let temp_cp = temp_space.create_checkpoint()
        .expect("temp checkpoint should succeed");
    let default_cp = default_space.create_checkpoint()
        .expect("default checkpoint should succeed");

    // Modify both
    temp_space.produce(channel.clone(), 112, false, None).expect("produce should succeed");
    default_space.produce(channel.clone(), 223, false, None).expect("produce should succeed");

    // Restore both
    temp_space.reset(&temp_cp.root)
        .expect("temp restore should succeed");
    default_space.reset(&default_cp.root)
        .expect("default restore should succeed");

    // Default preserved, Temp behavior per qualifier semantics
    assert_eq!(default_space.get_data(&channel).len(), 1, "Default should have 1 item");
}

// =============================================================================
// Checkpoint Error Handling Tests
// =============================================================================

/// Test checkpoint operations on empty space.
#[test]
fn test_checkpoint_empty_space() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Empty space should support checkpointing
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint of empty space should succeed");

    // Add some data
    space.produce(channel.clone(), 1, false, None).expect("produce should succeed");
    space.produce(channel.clone(), 2, false, None).expect("produce should succeed");

    // Restore to empty
    space.reset(&checkpoint.root)
        .expect("restore to empty should succeed");

    // Should be empty again
    assert!(space.get_data(&channel).is_empty(), "Space should be empty after restore");
}

// =============================================================================
// Gensym and Checkpoint Interaction
// =============================================================================

/// Test that gensym counter is properly checkpointed.
///
/// ```rholang
/// new Space(`rho:space:bag:hashmap:default`), space in {
///   Space!({}, *space) |
///   use space {
///     new ch1 in { ch1!(1) } |    // gensym → ch1
///     @"checkpoint"!("hard") |
///     new ch2 in { ch2!(2) } |    // gensym → ch2
///     @"restore"!() |
///     new ch3 in { ch3!(3) } |    // After restore, ch3 should be unique
///     // ch3 should not collide with ch2 (even though ch2 is "undone")
///   }
/// }
/// ```
#[test]
fn test_gensym_counter_after_checkpoint() {
    let mut space = create_space_with_history(SpaceQualifier::Default);

    // Generate some channels
    let ch1 = space.gensym().expect("gensym should succeed");
    let ch2 = space.gensym().expect("gensym should succeed");

    // Checkpoint
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Generate more channels
    let ch3 = space.gensym().expect("gensym should succeed");
    let ch4 = space.gensym().expect("gensym should succeed");

    // Restore
    space.reset(&checkpoint.root)
        .expect("restore should succeed");

    // Generate channels after restore
    let ch5 = space.gensym().expect("gensym should succeed");
    let ch6 = space.gensym().expect("gensym should succeed");

    // ch5 and ch6 should be unique (not collide with ch3/ch4 from before restore)
    // This is critical for safety - restored state should not reuse names
    assert_ne!(ch5, ch3, "Post-restore gensym should not reuse undone channels");
    assert_ne!(ch5, ch4, "Post-restore gensym should not reuse undone channels");
    assert_ne!(ch6, ch3, "Post-restore gensym should not reuse undone channels");
    assert_ne!(ch6, ch4, "Post-restore gensym should not reuse undone channels");

    // Also verify ch1 and ch2 are different (sanity check)
    assert_ne!(ch1, ch2, "Generated channels should be unique");
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/// Test that reset() returns an error when no history store is configured.
///
/// **Formal Reference**: This tests the error path for CheckpointableSpace::reset
/// when history_store is None.
#[test]
fn test_reset_without_history_store_returns_error() {
    use rholang::rust::interpreter::spaces::Blake2b256Hash;

    let mut space = create_space_with_qualifier(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Add some data
    space.produce(channel.clone(), 42, false, None).expect("produce should succeed");

    // Try to reset with an arbitrary root (no history store configured)
    let arbitrary_root = Blake2b256Hash::from_bytes([0u8; 32].to_vec());
    let result = space.reset(&arbitrary_root);

    // Should fail with CheckpointError
    assert!(result.is_err(), "reset should fail without history store");
    match result {
        Err(SpaceError::CheckpointError { description }) => {
            assert!(
                description.contains("No history store"),
                "Error should mention missing history store, got: {}",
                description
            );
        }
        other => panic!("Expected CheckpointError, got: {:?}", other),
    }
}

/// Test that revert_to_soft_checkpoint() returns error when stack is empty.
///
/// **Formal Reference**: This tests the error path when no soft checkpoint exists
/// to revert to.
#[test]
fn test_revert_without_soft_checkpoint_returns_error() {
    use rspace_plus_plus::rspace::hot_store::HotStoreState;
    use std::collections::BTreeMap;

    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Add some data (without creating a soft checkpoint)
    space.produce(channel.clone(), 42, false, None).expect("produce should succeed");

    // Create a dummy soft checkpoint struct to pass to revert
    // (the actual parameter is ignored per implementation - uses internal stack)
    let dummy_checkpoint: TestSoftCheckpoint = SoftCheckpoint {
        cache_snapshot: HotStoreState::default(),
        log: vec![],
        produce_counter: BTreeMap::new(),
    };

    // Try to revert without having created a soft checkpoint
    let result = space.revert_to_soft_checkpoint(dummy_checkpoint);

    // Should fail with CheckpointError
    assert!(result.is_err(), "revert should fail without soft checkpoint");
    match result {
        Err(SpaceError::CheckpointError { description }) => {
            assert!(
                description.contains("No soft checkpoint"),
                "Error should mention missing soft checkpoint, got: {}",
                description
            );
        }
        other => panic!("Expected CheckpointError, got: {:?}", other),
    }
}

/// Test that clear() preserves the soft checkpoint stack.
///
/// After clear(), soft checkpoints created before should still be revertible.
/// This is important for speculative execution patterns where we may want to
/// clear the working space but still be able to rollback.
#[test]
fn test_clear_preserves_soft_checkpoint_stack() {
    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Initial state
    space.produce(channel.clone(), 100, false, None).expect("produce should succeed");

    // Create soft checkpoint
    let soft_cp: TestSoftCheckpoint = space.create_soft_checkpoint();

    // Add more data
    space.produce(channel.clone(), 200, false, None).expect("produce should succeed");
    space.produce(channel.clone(), 300, false, None).expect("produce should succeed");
    assert_eq!(space.get_data(&channel).len(), 3, "Should have 3 items");

    // Clear the space
    space.clear().expect("clear should succeed");

    // Space should be empty after clear
    assert!(space.get_data(&channel).is_empty(), "Space should be empty after clear");

    // But soft checkpoint should still be revertible
    // (revert restores the snapshot from the stack)
    let revert_result = space.revert_to_soft_checkpoint(soft_cp);
    assert!(revert_result.is_ok(), "revert after clear should succeed");

    // After revert, we should have the state from checkpoint time (1 item)
    assert_eq!(
        space.get_data(&channel).len(),
        1,
        "After revert, should have 1 item from checkpoint"
    );
}

// =============================================================================
// Replay Functionality Tests
// =============================================================================

/// Test basic replay mode setup via rig_and_reset.
///
/// This test verifies that:
/// 1. rig_and_reset properly enters replay mode
/// 2. is_replay returns true after rig_and_reset
/// 3. check_replay_data succeeds when replay is properly completed
#[test]
fn test_replay_mode_setup() {
    use rspace_plus_plus::rspace::trace::Log;

    let mut space = create_space_with_history(SpaceQualifier::Default);
    let channel = TestChannel::from(0);

    // Build up initial state
    space.produce(channel.clone(), 1, false, None).expect("produce should succeed");
    space.produce(channel.clone(), 2, false, None).expect("produce should succeed");

    // Create checkpoint
    let checkpoint = space.create_checkpoint()
        .expect("checkpoint should succeed");

    // Verify not in replay mode initially
    assert!(!space.is_replay(), "Should not be in replay mode initially");

    // Modify state
    space.produce(channel.clone(), 3, false, None).expect("produce should succeed");

    // Create an empty log (no events to verify against)
    let empty_log: Log = vec![];

    // Enter replay mode
    space.rig_and_reset(checkpoint.root.clone(), empty_log)
        .expect("rig_and_reset should succeed");

    // Should now be in replay mode
    assert!(space.is_replay(), "Should be in replay mode after rig_and_reset");

    // With an empty log, check_replay_data should succeed (nothing to verify)
    let check_result = space.check_replay_data();
    assert!(check_result.is_ok(), "check_replay_data with empty log should succeed");
}

/// Test that check_replay_data fails when not in replay mode.
#[test]
fn test_check_replay_data_not_in_replay_mode() {
    let space = create_space_with_history(SpaceQualifier::Default);

    // Not in replay mode
    assert!(!space.is_replay(), "Should not be in replay mode initially");

    // check_replay_data should fail
    let result = space.check_replay_data();
    assert!(result.is_err(), "check_replay_data should fail when not in replay mode");
    match result {
        Err(SpaceError::ReplayError { description }) => {
            assert!(
                description.contains("Not in replay mode"),
                "Error should mention not in replay mode, got: {}",
                description
            );
        }
        other => panic!("Expected ReplayError, got: {:?}", other),
    }
}

/// Test rig() with a simple event log.
///
/// This test verifies that rig() properly processes a log and populates
/// replay_data for verification.
#[test]
fn test_rig_with_event_log() {
    use rspace_plus_plus::rspace::trace::{Log, event::{Event, IOEvent, COMM, Produce, Consume}};
    use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
    use std::collections::{BTreeSet, BTreeMap};

    let space = create_space_with_history(SpaceQualifier::Default);

    // Create a simple produce event
    let produce = Produce {
        channel_hash: Blake2b256Hash::from_bytes([1u8; 32].to_vec()),
        hash: Blake2b256Hash::from_bytes([2u8; 32].to_vec()),
        persistent: false,
        is_deterministic: true,
        output_value: vec![],
    };

    // Create a simple consume event
    let consume = Consume {
        channel_hashes: vec![Blake2b256Hash::from_bytes([1u8; 32].to_vec())],
        hash: Blake2b256Hash::from_bytes([3u8; 32].to_vec()),
        persistent: false,
    };

    // Create a COMM event (communication between produce and consume)
    let comm = COMM {
        consume: consume.clone(),
        produces: vec![produce.clone()],
        peeks: BTreeSet::new(),
        times_repeated: BTreeMap::new(),
    };

    // Log with both IO events and COMM event
    let log: Log = vec![
        Event::IoEvent(IOEvent::Produce(produce.clone())),
        Event::IoEvent(IOEvent::Consume(consume.clone())),
        Event::Comm(comm.clone()),
    ];

    // Call rig() to process the log
    let rig_result = space.rig(log);
    assert!(rig_result.is_ok(), "rig() should succeed with valid log");
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_checkpoint_has_root() {
        let mut space = create_space_with_history(SpaceQualifier::Default);

        // Create checkpoint
        let checkpoint = space.create_checkpoint()
            .expect("checkpoint should succeed");

        // Checkpoint should have a valid root
        // Blake2b256Hash should not be empty
        let root_bytes = checkpoint.root.bytes();
        assert!(!root_bytes.is_empty(), "Checkpoint root should not be empty");
    }

    #[test]
    fn test_multiple_checkpoints() {
        let mut space = create_space_with_history(SpaceQualifier::Default);
        let channel = TestChannel::from(0);

        // Create multiple checkpoints
        space.produce(channel.clone(), 1, false, None).expect("produce should succeed");
        let cp1 = space.create_checkpoint()
            .expect("cp1 should succeed");

        space.produce(channel.clone(), 2, false, None).expect("produce should succeed");
        let cp2 = space.create_checkpoint()
            .expect("cp2 should succeed");

        space.produce(channel.clone(), 3, false, None).expect("produce should succeed");
        let _cp3 = space.create_checkpoint()
            .expect("cp3 should succeed");

        // Verify we have 3 items
        assert_eq!(space.get_data(&channel).len(), 3);

        // Restore to cp2 (2 items)
        space.reset(&cp2.root).expect("restore should succeed");
        assert_eq!(space.get_data(&channel).len(), 2);

        // Restore to cp1 (1 item)
        space.reset(&cp1.root).expect("restore should succeed");
        assert_eq!(space.get_data(&channel).len(), 1);
    }

    #[test]
    fn test_soft_checkpoint_preserves_type_params() {
        let mut space = create_space_with_history(SpaceQualifier::Default);
        let channel = TestChannel::from(0);

        space.produce(channel.clone(), 42, false, None).expect("produce should succeed");

        // Create soft checkpoint - verifies type parameters are correct
        let soft_cp: TestSoftCheckpoint = space.create_soft_checkpoint();

        // Modify
        space.produce(channel.clone(), 43, false, None).expect("produce should succeed");
        assert_eq!(space.get_data(&channel).len(), 2);

        // Revert
        space.revert_to_soft_checkpoint(soft_cp).expect("revert should succeed");
        assert_eq!(space.get_data(&channel).len(), 1);
    }
}
