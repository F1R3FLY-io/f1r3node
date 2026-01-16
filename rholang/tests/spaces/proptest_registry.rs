//! Property-Based Tests for Space Registry Module
//!
//! This module contains property-based tests for the SpaceRegistry, which is the
//! central component for managing multiple tuple spaces in a Rholang runtime.
//!
//! # Rocq Correspondence
//!
//! These tests correspond to formal proofs in:
//! - `theories/Registry/Invariants.v` - Registry state invariants
//! - `formal/tla/CheckpointReplay.tla` - Checkpoint replay protocol
//! - `formal/tla/SpaceCoordination.tla` - Space coordination
//!
//! # Properties Tested
//!
//! 1. **Space Registration**: Unique IDs, idempotent failure
//! 2. **Channel Ownership**: Single space per channel, lookup consistency
//! 3. **Use Block Stack**: Push/pop symmetry, LIFO ordering
//! 4. **Checkpoint/Restore**: State preservation, replay correctness
//! 5. **Seq Mobility**: Non-mobile channels cannot be sent
//! 6. **Operation Logging**: Log append-only, replay order preserved
//!
//! # Rholang Usage
//!
//! ```rholang
//! // Use block changes default space for channel creation
//! new MySpace(`rho:space:queue:hashmap:default`), ms in {
//!   MySpace!({}, *ms) |
//!   for (space <- ms) {
//!     use space {
//!       new ch in { ch!(42) }  // ch created in MySpace
//!     }
//!   }
//! }
//! ```

use proptest::prelude::*;
use proptest::collection::{vec as prop_vec, hash_set};

use rholang::rust::interpreter::spaces::{
    SpaceId, SpaceConfig, SpaceQualifier, SpaceError,
};
use rholang::rust::interpreter::spaces::registry::{
    SpaceRegistry, UseBlockStack, OperationLog, OperationType, ReplayState,
};

// =============================================================================
// Arbitrary Generators
// =============================================================================

fn arb_space_id() -> impl Strategy<Value = SpaceId> {
    prop_vec(any::<u8>(), 1..=16).prop_map(SpaceId::new)
}

fn arb_unique_space_ids(count: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<SpaceId>> {
    hash_set(prop_vec(any::<u8>(), 4..=8), count)
        .prop_map(|set| set.into_iter().map(SpaceId::new).collect())
}

fn arb_channel_hash() -> impl Strategy<Value = Vec<u8>> {
    prop_vec(any::<u8>(), 8..=32)
}

fn arb_unique_channel_hashes(count: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<Vec<u8>>> {
    hash_set(prop_vec(any::<u8>(), 8..=16), count)
        .prop_map(|set| set.into_iter().collect())
}

fn arb_task_id() -> impl Strategy<Value = u64> {
    any::<u64>()
}

fn arb_qualifier() -> impl Strategy<Value = SpaceQualifier> {
    prop_oneof![
        Just(SpaceQualifier::Default),
        Just(SpaceQualifier::Temp),
        Just(SpaceQualifier::Seq),
    ]
}

fn arb_space_config() -> impl Strategy<Value = SpaceConfig> {
    prop_oneof![
        Just(SpaceConfig::default()),
        Just(SpaceConfig::queue()),
        Just(SpaceConfig::stack()),
        Just(SpaceConfig::set()),
        Just(SpaceConfig::cell()),
    ]
}

#[allow(dead_code)]
fn arb_operation(space_ids: Vec<SpaceId>) -> impl Strategy<Value = OperationType> {
    let space_ids_clone = space_ids.clone();
    prop_oneof![
        // Produce operation
        (0..space_ids.len(), arb_channel_hash(), prop_vec(any::<u8>(), 1..=32), any::<bool>())
            .prop_map(move |(idx, channel, data, persist)| {
                let space_id = space_ids[idx % space_ids.len()].clone();
                OperationType::Produce { space_id, channel, data, persist }
            }),
        // Consume operation
        (0..space_ids_clone.len(), prop_vec(arb_channel_hash(), 1..=3), any::<bool>())
            .prop_map(move |(idx, channels, persist)| {
                let space_id = space_ids_clone[idx % space_ids_clone.len()].clone();
                let patterns = channels.iter().map(|_| Vec::new()).collect();
                OperationType::Consume {
                    space_id,
                    channels,
                    patterns,
                    persist,
                    peeks: std::collections::BTreeSet::new()
                }
            }),
    ]
}

// =============================================================================
// Space Registration Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: Registering a space makes it exist.
    ///
    /// register_space(id, config) → space_exists(id)
    ///
    /// Rocq: `register_makes_exist` in Registry/Invariants.v
    #[test]
    fn prop_register_makes_exist(
        space_id in arb_space_id(),
        config in arb_space_config()
    ) {
        let registry = SpaceRegistry::new();

        // Skip if it's the default space
        if space_id == *registry.default_space_id() {
            return Ok(());
        }

        prop_assert!(!registry.space_exists(&space_id), "Space should not exist before registration");

        let result = registry.register_space(space_id.clone(), config);
        prop_assert!(result.is_ok(), "Registration should succeed");

        prop_assert!(registry.space_exists(&space_id), "Space should exist after registration");
    }

    /// Property: Registering the same space twice fails.
    ///
    /// register(id) → register(id) → Error
    ///
    /// Rocq: `register_idempotent_failure` in Registry/Invariants.v
    #[test]
    fn prop_duplicate_registration_fails(
        space_id in arb_space_id(),
        config1 in arb_space_config(),
        config2 in arb_space_config()
    ) {
        let registry = SpaceRegistry::new();

        // Skip if it's the default space
        if space_id == *registry.default_space_id() {
            return Ok(());
        }

        registry.register_space(space_id.clone(), config1).expect("first registration");

        let result = registry.register_space(space_id, config2);
        prop_assert!(result.is_err(), "Duplicate registration should fail");
    }

    /// Property: All registered spaces are listed.
    ///
    /// register(ids) → all_space_ids() ⊇ ids
    ///
    /// Rocq: `all_space_ids_complete` in Registry/Invariants.v
    #[test]
    fn prop_all_space_ids_complete(
        space_ids in arb_unique_space_ids(1..=5)
    ) {
        let registry = SpaceRegistry::new();
        let default_id = registry.default_space_id().clone();

        for id in &space_ids {
            if *id != default_id {
                registry.register_space(id.clone(), SpaceConfig::default()).expect("register");
            }
        }

        let all_ids = registry.all_space_ids();

        for id in &space_ids {
            if *id != default_id {
                prop_assert!(all_ids.contains(id), "Registered space should be in all_space_ids");
            }
        }
    }

    /// Property: Get space returns the correct config.
    ///
    /// Rocq: `get_space_returns_config` in Registry/Invariants.v
    #[test]
    fn prop_get_space_returns_config(
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();

        // Skip if it's the default space
        if space_id == *registry.default_space_id() {
            return Ok(());
        }

        let config = SpaceConfig::queue();
        registry.register_space(space_id.clone(), config.clone()).expect("register");

        let entry = registry.get_space(&space_id).expect("should exist");
        prop_assert_eq!(entry.config.data_collection, config.data_collection);
    }
}

// =============================================================================
// Channel Ownership Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: Registering a channel makes it findable.
    ///
    /// register_channel(hash, space) → get_channel_space(hash) == Some(space)
    ///
    /// Rocq: `register_channel_findable` in Registry/Invariants.v
    #[test]
    fn prop_register_channel_findable(
        channel_hash in arb_channel_hash(),
        space_id in arb_space_id(),
        qualifier in arb_qualifier()
    ) {
        let registry = SpaceRegistry::new();

        registry.register_channel(channel_hash.clone(), space_id.clone(), qualifier);

        let info = registry.get_channel_space(&channel_hash);
        prop_assert!(info.is_some(), "Channel should be findable after registration");

        let info = info.unwrap();
        prop_assert_eq!(info.space_id, space_id);
        prop_assert_eq!(info.qualifier, qualifier);
    }

    /// Property: verify_same_space succeeds when all channels are in the same space.
    ///
    /// Rocq: `verify_same_space_homogeneous` in Registry/Invariants.v
    #[test]
    fn prop_verify_same_space_homogeneous(
        channel_hashes in arb_unique_channel_hashes(2..=5),
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();

        for hash in &channel_hashes {
            registry.register_channel(hash.clone(), space_id.clone(), SpaceQualifier::Default);
        }

        let result = registry.verify_same_space(&channel_hashes);
        prop_assert!(result.is_ok(), "All channels in same space should pass");
        prop_assert_eq!(result.unwrap(), space_id);
    }

    /// Property: verify_same_space fails when channels are in different spaces.
    ///
    /// Rocq: `verify_same_space_heterogeneous_fails` in Registry/Invariants.v
    #[test]
    fn prop_verify_same_space_heterogeneous_fails(
        channel_hashes in arb_unique_channel_hashes(2..=4),
        space_ids in arb_unique_space_ids(2..=3)
    ) {
        prop_assume!(channel_hashes.len() >= 2);
        prop_assume!(space_ids.len() >= 2);

        let registry = SpaceRegistry::new();

        // Register first channel in first space
        registry.register_channel(
            channel_hashes[0].clone(),
            space_ids[0].clone(),
            SpaceQualifier::Default,
        );

        // Register second channel in second space
        registry.register_channel(
            channel_hashes[1].clone(),
            space_ids[1].clone(),
            SpaceQualifier::Default,
        );

        let result = registry.verify_same_space(&[channel_hashes[0].clone(), channel_hashes[1].clone()]);
        let is_cross_space_error = matches!(result, Err(SpaceError::CrossSpaceJoinNotAllowed { .. }));
        prop_assert!(is_cross_space_error, "Expected CrossSpaceJoinNotAllowed error");
    }

    /// Property: resolve_space returns the channel's space if known.
    #[test]
    fn prop_resolve_known_channel(
        channel_hash in arb_channel_hash(),
        space_id in arb_space_id(),
        task_id in arb_task_id()
    ) {
        let registry = SpaceRegistry::new();

        registry.register_channel(channel_hash.clone(), space_id.clone(), SpaceQualifier::Default);

        let resolved = registry.resolve_space(Some(&channel_hash), task_id);
        prop_assert_eq!(resolved, space_id);
    }

    /// Property: resolve_space returns default space for unknown channels.
    #[test]
    fn prop_resolve_unknown_channel_defaults(
        channel_hash in arb_channel_hash(),
        task_id in arb_task_id()
    ) {
        let registry = SpaceRegistry::new();

        // Don't register the channel
        let resolved = registry.resolve_space(Some(&channel_hash), task_id);
        prop_assert_eq!(resolved, registry.default_space_id().clone());
    }
}

// =============================================================================
// Use Block Stack Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: Push/pop symmetry - pushing then popping returns the same value.
    ///
    /// push(id) → pop() == Some(id)
    ///
    /// Rocq: `push_pop_symmetry` in Registry/Invariants.v
    #[test]
    fn prop_push_pop_symmetry(space_id in arb_space_id()) {
        let mut stack = UseBlockStack::new();

        stack.push(space_id.clone());
        let popped = stack.pop();

        prop_assert_eq!(popped, Some(space_id));
    }

    /// Property: Use block stack is LIFO.
    ///
    /// push(a) → push(b) → pop() == b ∧ pop() == a
    ///
    /// Rocq: `use_block_lifo` in Registry/Invariants.v
    #[test]
    fn prop_use_block_lifo(
        space_ids in prop_vec(arb_space_id(), 1..=10)
    ) {
        let mut stack = UseBlockStack::new();

        for id in &space_ids {
            stack.push(id.clone());
        }

        // Pop in reverse order
        for id in space_ids.iter().rev() {
            let popped = stack.pop();
            prop_assert_eq!(popped, Some(id.clone()));
        }

        // Stack should be empty now
        prop_assert!(stack.is_empty());
    }

    /// Property: Stack depth increases with push.
    #[test]
    fn prop_stack_depth_increases(
        space_ids in prop_vec(arb_space_id(), 1..=5)
    ) {
        let mut stack = UseBlockStack::new();

        for (i, id) in space_ids.iter().enumerate() {
            prop_assert_eq!(stack.depth(), i);
            stack.push(id.clone());
            prop_assert_eq!(stack.depth(), i + 1);
        }
    }

    /// Property: current() returns the most recently pushed space.
    #[test]
    fn prop_current_returns_last_pushed(
        space_ids in prop_vec(arb_space_id(), 1..=5)
    ) {
        let mut stack = UseBlockStack::new();

        for id in &space_ids {
            stack.push(id.clone());
            prop_assert_eq!(stack.current(), Some(id));
        }
    }

    /// Property: Pop on empty stack returns None.
    #[test]
    fn prop_pop_empty_returns_none(_dummy in 0..1) {
        let mut stack = UseBlockStack::new();
        prop_assert!(stack.pop().is_none());
        prop_assert!(stack.current().is_none());
    }

    /// Property: Registry use block stack per task isolation.
    #[test]
    fn prop_use_block_task_isolation(
        task1 in arb_task_id(),
        task2 in arb_task_id(),
        space1 in arb_space_id(),
        space2 in arb_space_id()
    ) {
        prop_assume!(task1 != task2);

        let registry = SpaceRegistry::new();

        registry.push_use_block(task1, space1.clone());
        registry.push_use_block(task2, space2.clone());

        // Each task should see its own space
        prop_assert_eq!(registry.current_default_space(task1), Some(space1.clone()));
        prop_assert_eq!(registry.current_default_space(task2), Some(space2.clone()));

        // Popping from one shouldn't affect the other
        registry.pop_use_block(task1);
        prop_assert!(registry.current_default_space(task1).is_none());
        prop_assert_eq!(registry.current_default_space(task2), Some(space2));
    }
}

// =============================================================================
// Checkpoint/Restore Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: Checkpoint captures current state.
    ///
    /// Rocq: `checkpoint_captures_state` in Checkpoint.v
    #[test]
    fn prop_checkpoint_captures_state(
        space_id in arb_space_id(),
        config in arb_space_config()
    ) {
        let registry = SpaceRegistry::new();

        // Skip default space
        if space_id == *registry.default_space_id() {
            return Ok(());
        }

        registry.register_space(space_id.clone(), config.clone()).expect("register");

        let checkpoint = registry.create_checkpoint_default();

        prop_assert!(checkpoint.spaces().contains_key(&space_id));
    }

    /// Property: Restore brings back checkpointed state.
    ///
    /// Rocq: `restore_reverts_state` in Checkpoint.v
    #[test]
    fn prop_restore_reverts_state(
        space1 in arb_space_id(),
        space2 in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();
        let default_id = registry.default_space_id().clone();

        prop_assume!(space1 != default_id);
        prop_assume!(space2 != default_id);
        prop_assume!(space1 != space2);

        // Register space1
        registry.register_space(space1.clone(), SpaceConfig::default()).expect("register 1");

        // Checkpoint
        let checkpoint = registry.create_checkpoint_default();

        // Register space2
        registry.register_space(space2.clone(), SpaceConfig::default()).expect("register 2");

        prop_assert!(registry.space_exists(&space2));

        // Restore
        registry.restore_checkpoint(checkpoint);

        // space2 should be gone
        prop_assert!(!registry.space_exists(&space2));
        prop_assert!(registry.space_exists(&space1));
    }

    /// Property: Block height increases with each checkpoint.
    #[test]
    fn prop_block_height_increases(count in 1usize..=10) {
        let registry = SpaceRegistry::new();

        let mut prev_height = registry.block_height();

        for _ in 0..count {
            let checkpoint = registry.create_checkpoint_default();
            let current_height = checkpoint.block_height();
            prop_assert!(current_height >= prev_height);
            prev_height = registry.block_height();
        }
    }

    /// Property: Soft checkpoint revert restores state.
    #[test]
    fn prop_soft_checkpoint_revert(
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();
        let default_id = registry.default_space_id().clone();

        prop_assume!(space_id != default_id);

        // Create soft checkpoint
        registry.create_soft_checkpoint().expect("create soft checkpoint");
        prop_assert!(registry.has_soft_checkpoint());

        // Register new space
        registry.register_space(space_id.clone(), SpaceConfig::default()).expect("register");
        prop_assert!(registry.space_exists(&space_id));

        // Revert
        registry.revert_to_soft_checkpoint().expect("revert");

        // Space should be gone
        prop_assert!(!registry.space_exists(&space_id));
        prop_assert!(!registry.has_soft_checkpoint());
    }

    /// Property: Soft checkpoint commit preserves changes.
    #[test]
    fn prop_soft_checkpoint_commit(
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();
        let default_id = registry.default_space_id().clone();

        prop_assume!(space_id != default_id);

        registry.create_soft_checkpoint().expect("create");
        registry.register_space(space_id.clone(), SpaceConfig::default()).expect("register");
        registry.commit_soft_checkpoint().expect("commit");

        // Space should still exist
        prop_assert!(registry.space_exists(&space_id));
        prop_assert!(!registry.has_soft_checkpoint());
    }
}

// =============================================================================
// Operation Logging Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: Operation log grows with each logged operation.
    #[test]
    fn prop_operation_log_grows(count in 1usize..=20) {
        let mut log = OperationLog::new();

        for i in 0..count {
            prop_assert_eq!(log.len(), i);

            log.append(OperationType::Produce {
                space_id: SpaceId::default_space(),
                channel: vec![i as u8],
                data: vec![],
                persist: false,
            });

            prop_assert_eq!(log.len(), i + 1);
        }
    }

    /// Property: Operation log preserves insertion order.
    #[test]
    fn prop_operation_log_order_preserved(
        count in 1usize..=10
    ) {
        let mut log = OperationLog::new();

        for i in 0..count {
            log.append(OperationType::Produce {
                space_id: SpaceId::default_space(),
                channel: vec![i as u8],
                data: vec![],
                persist: false,
            });
        }

        for i in 0..count {
            if let Some(OperationType::Produce { channel, .. }) = log.get(i) {
                prop_assert_eq!(channel, &vec![i as u8]);
            } else {
                prop_assert!(false, "Expected Produce operation");
            }
        }
    }

    /// Property: Clear empties the log.
    #[test]
    fn prop_operation_log_clear(count in 1usize..=10) {
        let mut log = OperationLog::new();

        for i in 0..count {
            log.append(OperationType::Install {
                space_id: SpaceId::default_space(),
                channels: vec![vec![i as u8]],
                patterns: vec![],
            });
        }

        prop_assert_eq!(log.len(), count);

        log.clear();

        prop_assert!(log.is_empty());
        prop_assert_eq!(log.len(), 0);
    }
}

// =============================================================================
// Replay Mode Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: Replay mode produces operations in order.
    #[test]
    fn prop_replay_order_preserved(count in 1usize..=10) {
        let registry = SpaceRegistry::new();
        let mut log = OperationLog::new();

        for i in 0..count {
            log.append(OperationType::Produce {
                space_id: SpaceId::default_space(),
                channel: vec![i as u8],
                data: vec![],
                persist: false,
            });
        }

        registry.enter_replay_mode(log).expect("enter replay");

        for i in 0..count {
            let op = registry.replay_next_operation();
            prop_assert!(op.is_some());

            if let Some(OperationType::Produce { channel, .. }) = op {
                prop_assert_eq!(channel, vec![i as u8]);
            }
        }

        // No more operations
        prop_assert!(registry.replay_next_operation().is_none());
    }

    /// Property: Replay state transitions correctly.
    #[test]
    fn prop_replay_state_transitions(count in 1usize..=5) {
        let registry = SpaceRegistry::new();
        let mut log = OperationLog::new();

        for i in 0..count {
            log.append(OperationType::Produce {
                space_id: SpaceId::default_space(),
                channel: vec![i as u8],
                data: vec![],
                persist: false,
            });
        }

        // Initial state
        prop_assert_eq!(registry.replay_state(), ReplayState::Normal);

        // Enter replay
        registry.enter_replay_mode(log).expect("enter");
        prop_assert!(registry.is_replay_mode());

        // Consume all operations
        for _ in 0..count {
            let is_replaying = matches!(registry.replay_state(), ReplayState::Replaying { .. });
            prop_assert!(is_replaying, "Expected Replaying state");
            registry.replay_next_operation();
        }

        // Completed
        prop_assert!(matches!(registry.replay_state(), ReplayState::Completed));

        // Exit replay
        registry.exit_replay_mode().expect("exit");
        prop_assert_eq!(registry.replay_state(), ReplayState::Normal);
    }

    /// Property: Cannot enter replay mode when already replaying.
    #[test]
    fn prop_cannot_reenter_replay(_dummy in 0..1) {
        let registry = SpaceRegistry::new();

        registry.enter_replay_mode(OperationLog::new()).expect("first enter");

        let result = registry.enter_replay_mode(OperationLog::new());
        prop_assert!(result.is_err());
    }
}

// =============================================================================
// Seq Mobility Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: Seq channels are not mobile.
    ///
    /// register(ch, Seq) → is_seq_channel(ch) ∧ ¬is_mobile_channel(ch)
    ///
    /// Rocq: `seq_not_mobile` in SpaceCoordination.v
    #[test]
    fn prop_seq_not_mobile(
        channel_hash in arb_channel_hash(),
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();

        registry.register_channel(channel_hash.clone(), space_id, SpaceQualifier::Seq);

        prop_assert!(registry.is_seq_channel(&channel_hash));
        prop_assert!(!registry.is_mobile_channel(&channel_hash));
    }

    /// Property: Default and Temp channels are mobile.
    #[test]
    fn prop_default_temp_mobile(
        channel_hash in arb_channel_hash(),
        space_id in arb_space_id(),
        qualifier_idx in 0u8..2  // 0 = Default, 1 = Temp
    ) {
        let registry = SpaceRegistry::new();
        let qualifier = if qualifier_idx == 0 { SpaceQualifier::Default } else { SpaceQualifier::Temp };

        registry.register_channel(channel_hash.clone(), space_id, qualifier);

        prop_assert!(!registry.is_seq_channel(&channel_hash));
        prop_assert!(registry.is_mobile_channel(&channel_hash));
    }

    /// Property: validate_send_channels fails for Seq channels.
    #[test]
    fn prop_validate_send_fails_for_seq(
        channel_hash in arb_channel_hash(),
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();

        registry.register_channel(channel_hash.clone(), space_id, SpaceQualifier::Seq);

        let result = registry.validate_send_channels(&[channel_hash]);
        let is_seq_error = matches!(result, Err(SpaceError::SeqChannelNotMobile { .. }));
        prop_assert!(is_seq_error, "Expected SeqChannelNotMobile error");
    }

    /// Property: validate_send_channels succeeds for mobile channels.
    #[test]
    fn prop_validate_send_succeeds_for_mobile(
        channel_hashes in arb_unique_channel_hashes(1..=5),
        space_id in arb_space_id()
    ) {
        let registry = SpaceRegistry::new();

        for hash in &channel_hashes {
            registry.register_channel(hash.clone(), space_id.clone(), SpaceQualifier::Default);
        }

        let result = registry.validate_send_channels(&channel_hashes);
        prop_assert!(result.is_ok());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_default_space_always_exists() {
        let registry = SpaceRegistry::new();
        assert!(registry.space_exists(registry.default_space_id()));
    }

    #[test]
    fn test_empty_channel_list_verify() {
        let registry = SpaceRegistry::new();
        let result = registry.verify_same_space(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_channel_defaults_to_mobile() {
        let registry = SpaceRegistry::new();
        let unknown_hash = vec![0xFF, 0xFE, 0xFD];
        assert!(registry.is_mobile_channel(&unknown_hash));
    }

    #[test]
    fn test_cleanup_task_removes_stack() {
        let registry = SpaceRegistry::new();
        let task_id = 999;
        let space_id = SpaceId::new(vec![1, 2, 3]);

        registry.push_use_block(task_id, space_id.clone());
        assert_eq!(registry.use_block_depth(task_id), 1);

        registry.cleanup_task(task_id);
        assert_eq!(registry.use_block_depth(task_id), 0);
    }

    #[test]
    fn test_revert_without_soft_checkpoint_fails() {
        let registry = SpaceRegistry::new();
        let result = registry.revert_to_soft_checkpoint();
        assert!(result.is_err());
    }

    #[test]
    fn test_commit_without_soft_checkpoint_fails() {
        let registry = SpaceRegistry::new();
        let result = registry.commit_soft_checkpoint();
        assert!(result.is_err());
    }

    #[test]
    fn test_exit_replay_when_not_replaying_fails() {
        let registry = SpaceRegistry::new();
        let result = registry.exit_replay_mode();
        assert!(result.is_err());
    }

    #[test]
    fn test_operation_type_space_id() {
        let space_id = SpaceId::new(vec![42]);

        let produce = OperationType::Produce {
            space_id: space_id.clone(),
            channel: vec![],
            data: vec![],
            persist: false,
        };
        assert_eq!(produce.space_id(), &space_id);

        let consume = OperationType::Consume {
            space_id: space_id.clone(),
            channels: vec![],
            patterns: vec![],
            persist: false,
            peeks: std::collections::BTreeSet::new(),
        };
        assert_eq!(consume.space_id(), &space_id);

        let install = OperationType::Install {
            space_id: space_id.clone(),
            channels: vec![],
            patterns: vec![],
        };
        assert_eq!(install.space_id(), &space_id);
    }

    #[test]
    fn test_operation_type_names() {
        let produce = OperationType::Produce {
            space_id: SpaceId::default_space(),
            channel: vec![],
            data: vec![],
            persist: false,
        };
        assert_eq!(produce.type_name(), "Produce");

        let consume = OperationType::Consume {
            space_id: SpaceId::default_space(),
            channels: vec![],
            patterns: vec![],
            persist: false,
            peeks: std::collections::BTreeSet::new(),
        };
        assert_eq!(consume.type_name(), "Consume");

        let install = OperationType::Install {
            space_id: SpaceId::default_space(),
            channels: vec![],
            patterns: vec![],
        };
        assert_eq!(install.type_name(), "Install");
    }

    #[test]
    fn test_replay_state_methods() {
        let normal = ReplayState::Normal;
        assert!(!normal.is_replaying());
        assert!(!normal.is_finished());

        let replaying = ReplayState::Replaying { index: 5, total: 10 };
        assert!(replaying.is_replaying());
        assert!(!replaying.is_finished());

        let completed = ReplayState::Completed;
        assert!(!completed.is_replaying());
        assert!(completed.is_finished());

        let failed = ReplayState::Failed;
        assert!(!failed.is_replaying());
        assert!(failed.is_finished());
    }
}
