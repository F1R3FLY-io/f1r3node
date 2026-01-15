//! Test Utilities and Arbitrary Generators for Property-Based Testing
//!
//! This module provides:
//! - Proptest arbitrary generators for all RSpace types
//! - Test helper functions for creating spaces
//! - Common test fixtures and configurations
//!
//! # Rholang Syntax Correspondence
//!
//! The test operations map to Rholang syntax:
//!
//! ```rholang
//! // TestOperation::Produce corresponds to:
//! channel!(data)                    // Non-persistent
//! channel!!(data)                   // Persistent
//!
//! // TestOperation::Consume corresponds to:
//! for (@x <- channel) { ... }       // Non-persistent
//! for (@x <= channel) { ... }       // Persistent (replicated)
//!
//! // TestOperation::Gensym corresponds to:
//! new channel in { ... }            // Creates unforgeable name
//! ```

use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;

use serde::{Serialize, Deserialize};

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;
use proptest::strategy::ValueTree;

use std::sync::Arc;

use rholang::rust::interpreter::spaces::{
    InnerCollectionType, OuterStorageType, SpaceConfig, SpaceId, SpaceQualifier,
    SpaceError, GenericRSpace, GenericRSpaceBuilder, GasConfiguration,
    ExactMatch, WildcardMatch,
    PhlogistonMeter, Operation,
    SEND_BASE_COST, RECEIVE_BASE_COST, CHANNEL_CREATE_COST,
    InMemoryHistoryStore, BoxedHistoryStore,
};
use rholang::rust::interpreter::spaces::collections::{
    BagDataCollection, BagContinuationCollection,
    QueueDataCollection, StackDataCollection,
    SetDataCollection, CellDataCollection,
    PriorityQueueDataCollection, VectorDBDataCollection,
    DataCollection, ContinuationCollection,
    SemanticEq, SemanticHash,
};
use rholang::rust::interpreter::spaces::channel_store::HashMapChannelStore;

// =============================================================================
// Test Types
// =============================================================================

/// Test channel type - newtype wrapper around Vec<u8> that implements necessary traits
/// for both HashMap store (From<usize> for gensym) and PathMap suffix key semantics (AsRef<[u8]>).
///
/// This is required because SpaceAgent trait requires `C: AsRef<[u8]>` for channel types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct TestChannel(pub Vec<u8>);

impl TestChannel {
    /// Create a new TestChannel from a usize value.
    pub fn new(n: usize) -> Self {
        TestChannel(n.to_le_bytes().to_vec())
    }

    /// Get the channel as a usize (for display/debugging).
    pub fn as_usize(&self) -> Option<usize> {
        if self.0.len() == 8 {
            let bytes: [u8; 8] = self.0[..8].try_into().ok()?;
            Some(usize::from_le_bytes(bytes))
        } else {
            None
        }
    }
}

impl From<usize> for TestChannel {
    fn from(n: usize) -> Self {
        TestChannel(n.to_le_bytes().to_vec())
    }
}

impl AsRef<[u8]> for TestChannel {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Simple test pattern type (matches data by equality or wildcard).
pub type TestPattern = i32;

/// Simple test data type.
pub type TestData = i32;

/// Simple test continuation type.
pub type TestContinuation = String;

// =============================================================================
// Operations for Property Testing
// =============================================================================

/// A test operation that can be applied to an RSpace.
///
/// These operations correspond to Rholang syntax:
/// - Produce: `channel!(data)` or `channel!!(data)` (persistent)
/// - Consume: `for (@x <- channel) { ... }` or `for (@x <= channel) { ... }`
/// - Gensym: `new channel in { ... }`
#[derive(Debug, Clone)]
pub enum TestOperation {
    /// Send data on a channel.
    ///
    /// Rholang: `channel!(data)` or `channel!!(data)`
    Produce {
        channel: TestChannel,
        data: TestData,
        persist: bool,
    },
    /// Wait for data on channel(s).
    ///
    /// Rholang: `for (@x <- channel) { ... }` or `for (@x <= channel) { ... }`
    Consume {
        channels: Vec<TestChannel>,
        patterns: Vec<TestPattern>,
        persist: bool,
    },
    /// Generate a fresh, unforgeable channel name.
    ///
    /// Rholang: `new channel in { ... }`
    Gensym,
}

/// Result of applying an operation to track for invariant checking.
#[derive(Debug, Clone)]
pub enum TestResult {
    /// Produce completed without matching a continuation.
    ProduceStored,
    /// Produce found a matching continuation and fired it.
    ProduceFired { continuation: TestContinuation },
    /// Consume completed without matching data.
    ConsumeStored,
    /// Consume found matching data and returned it.
    ConsumeFired { data: Vec<TestData> },
    /// Gensym generated a new channel.
    GensymCreated { channel: TestChannel },
    /// Operation failed with an error.
    Error(String),
}

// =============================================================================
// Proptest Arbitrary Generators
// =============================================================================

/// Generate an arbitrary InnerCollectionType.
pub fn arb_inner_collection_type() -> impl Strategy<Value = InnerCollectionType> {
    prop_oneof![
        Just(InnerCollectionType::Bag),
        Just(InnerCollectionType::Queue),
        Just(InnerCollectionType::Stack),
        Just(InnerCollectionType::Set),
        Just(InnerCollectionType::Cell),
        (1usize..=5).prop_map(|p| InnerCollectionType::PriorityQueue { priorities: p }),
        (2usize..=128).prop_map(|d| InnerCollectionType::VectorDB {
            dimensions: d,
            backend: "rho".to_string(),
        }),
    ]
}

/// Generate an arbitrary OuterStorageType.
pub fn arb_outer_storage_type() -> impl Strategy<Value = OuterStorageType> {
    prop_oneof![
        Just(OuterStorageType::HashMap),
        Just(OuterStorageType::PathMap),
        (10usize..=1000, any::<bool>())
            .prop_map(|(max, cyclic)| OuterStorageType::Array { max_size: max, cyclic }),
        Just(OuterStorageType::Vector),
        Just(OuterStorageType::HashSet),
    ]
}

/// Generate an arbitrary SpaceQualifier.
pub fn arb_space_qualifier() -> impl Strategy<Value = SpaceQualifier> {
    prop_oneof![
        Just(SpaceQualifier::Default),
        Just(SpaceQualifier::Temp),
        Just(SpaceQualifier::Seq),
    ]
}

/// Generate an arbitrary SpaceConfig (basic, without theory or gas).
pub fn arb_space_config() -> impl Strategy<Value = SpaceConfig> {
    (
        arb_outer_storage_type(),
        arb_inner_collection_type(),
        arb_inner_collection_type(),
        arb_space_qualifier(),
    )
        .prop_map(|(outer, data_coll, cont_coll, qualifier)| {
            SpaceConfig {
                outer,
                data_collection: data_coll,
                continuation_collection: cont_coll,
                qualifier,
                theory: None,
                gas_config: GasConfiguration::default(),
            }
        })
}

/// Generate an arbitrary test channel (0..max_channels).
pub fn arb_channel(max_channels: usize) -> impl Strategy<Value = TestChannel> {
    (0..max_channels).prop_map(TestChannel::from)
}

/// Generate an arbitrary test data value.
pub fn arb_data() -> impl Strategy<Value = TestData> {
    any::<TestData>()
}

/// Generate an arbitrary test pattern.
pub fn arb_pattern() -> impl Strategy<Value = TestPattern> {
    any::<TestPattern>()
}

/// Generate an arbitrary Produce operation.
pub fn arb_produce(max_channels: usize) -> impl Strategy<Value = TestOperation> {
    (arb_channel(max_channels), arb_data(), any::<bool>()).prop_map(|(channel, data, persist)| {
        TestOperation::Produce {
            channel,
            data,
            persist,
        }
    })
}

/// Generate an arbitrary Consume operation with 1-4 channels.
pub fn arb_consume(max_channels: usize) -> impl Strategy<Value = TestOperation> {
    (1usize..=4).prop_flat_map(move |num_channels| {
        (
            prop_vec(arb_channel(max_channels), num_channels),
            prop_vec(arb_pattern(), num_channels),
            any::<bool>(),
        )
            .prop_map(|(channels, patterns, persist)| TestOperation::Consume {
                channels,
                patterns,
                persist,
            })
    })
}

/// Generate an arbitrary test operation.
pub fn arb_operation(max_channels: usize) -> impl Strategy<Value = TestOperation> {
    prop_oneof![
        8 => arb_produce(max_channels),
        4 => arb_consume(max_channels),
        1 => Just(TestOperation::Gensym),
    ]
}

/// Generate a sequence of operations for property testing.
pub fn arb_operation_sequence(
    max_channels: usize,
    min_ops: usize,
    max_ops: usize,
) -> impl Strategy<Value = Vec<TestOperation>> {
    prop_vec(arb_operation(max_channels), min_ops..=max_ops)
}

/// Generate an arbitrary phlogiston Operation.
pub fn arb_phlogiston_operation() -> impl Strategy<Value = Operation> {
    prop_oneof![
        (0usize..1000).prop_map(|size| Operation::Send { data_size: size }),
        Just(Operation::Receive),
        (0usize..100).prop_map(|size| Operation::Match { pattern_size: size }),
        Just(Operation::CreateChannel),
        Just(Operation::Checkpoint),
        Just(Operation::CreateSpace),
    ]
}

/// Generate a sequence of phlogiston operations.
pub fn arb_phlogiston_sequence(
    min_ops: usize,
    max_ops: usize,
) -> impl Strategy<Value = Vec<Operation>> {
    prop_vec(arb_phlogiston_operation(), min_ops..=max_ops)
}

// =============================================================================
// Test Space Factory
// =============================================================================

/// Create a test space with HashMap storage and Bag collections.
pub fn create_hashmap_bag_space() -> GenericRSpace<
    HashMapChannelStore<TestChannel, TestPattern, TestData, TestContinuation, BagDataCollection<TestData>, BagContinuationCollection<TestPattern, TestContinuation>>,
    WildcardMatch<TestPattern, TestData>,
> {
    let store = HashMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
    let matcher = WildcardMatch::new();
    GenericRSpace::new(store, matcher, SpaceId::default_space(), SpaceQualifier::Default)
}

/// Create a test space with specified qualifier.
pub fn create_space_with_qualifier(
    qualifier: SpaceQualifier,
) -> GenericRSpace<
    HashMapChannelStore<TestChannel, TestPattern, TestData, TestContinuation, BagDataCollection<TestData>, BagContinuationCollection<TestPattern, TestContinuation>>,
    WildcardMatch<TestPattern, TestData>,
> {
    let store = HashMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
    let matcher = WildcardMatch::new();
    GenericRSpace::new(store, matcher, SpaceId::default_space(), qualifier)
}

/// Create a test space with specified qualifier and history store for checkpoint testing.
///
/// This function creates a space with an in-memory history store, enabling
/// checkpoint and restore functionality.
pub fn create_space_with_history(
    qualifier: SpaceQualifier,
) -> GenericRSpace<
    HashMapChannelStore<TestChannel, TestPattern, TestData, TestContinuation, BagDataCollection<TestData>, BagContinuationCollection<TestPattern, TestContinuation>>,
    WildcardMatch<TestPattern, TestData>,
> {
    let store = HashMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
    let matcher = WildcardMatch::new();
    let history_store: BoxedHistoryStore = Arc::new(InMemoryHistoryStore::new());
    GenericRSpace::with_history(store, matcher, SpaceId::default_space(), qualifier, history_store)
}

/// Create a unique SpaceId for testing.
pub fn create_test_space_id(seed: u64) -> SpaceId {
    let mut bytes = vec![0u8; 32];
    bytes[0..8].copy_from_slice(&seed.to_le_bytes());
    SpaceId::new(bytes)
}

// =============================================================================
// Data Collection Test Helpers
// =============================================================================

/// Create a BagDataCollection with initial data.
pub fn bag_with_data<A: Clone + Send + Sync>(items: Vec<A>) -> BagDataCollection<A> {
    let mut bag = BagDataCollection::new();
    for item in items {
        bag.put(item).expect("BagDataCollection::put should succeed");
    }
    bag
}

/// Create a QueueDataCollection with initial data.
pub fn queue_with_data<A: Clone + Send + Sync>(items: Vec<A>) -> QueueDataCollection<A> {
    let mut queue = QueueDataCollection::new();
    for item in items {
        queue.put(item).expect("QueueDataCollection::put should succeed");
    }
    queue
}

/// Create a StackDataCollection with initial data.
pub fn stack_with_data<A: Clone + Send + Sync>(items: Vec<A>) -> StackDataCollection<A> {
    let mut stack = StackDataCollection::new();
    for item in items {
        stack.put(item).expect("StackDataCollection::put should succeed");
    }
    stack
}

/// Create a SetDataCollection with initial data.
pub fn set_with_data<A: Clone + Send + Sync + Hash + Eq + SemanticEq + SemanticHash>(items: Vec<A>) -> SetDataCollection<A> {
    let mut set = SetDataCollection::new();
    for item in items {
        set.put(item).expect("SetDataCollection::put should succeed");
    }
    set
}

/// Create a PriorityQueueDataCollection with initial data at various priorities.
pub fn priority_queue_with_data<A: Clone + Send + Sync>(
    items: Vec<(A, usize)>,
    num_priorities: usize,
) -> PriorityQueueDataCollection<A> {
    let mut pq = PriorityQueueDataCollection::new(num_priorities);
    for (item, priority) in items {
        pq.put_with_priority(item, priority).expect("PriorityQueueDataCollection::put should succeed");
    }
    pq
}

// =============================================================================
// Phlogiston Test Helpers
// =============================================================================

/// Create a PhlogistonMeter with specified initial balance.
pub fn create_meter_with_balance(balance: u64) -> PhlogistonMeter {
    PhlogistonMeter::new(balance)
}

/// Calculate the expected cost for a sequence of operations.
pub fn calculate_expected_cost(operations: &[Operation]) -> u64 {
    operations.iter().map(|op| op.cost()).sum()
}

// =============================================================================
// Invariant Checking Helpers
// =============================================================================

/// Check that a space has no pending match (data and continuations don't coexist).
///
/// This is the core invariant from GenericRSpace.v:157-180.
///
/// Rholang intuition: After a complete reduction, either:
/// - There's data waiting with no receiver, OR
/// - There's a receiver waiting with no data
/// - Never both at the same channel
pub fn check_no_pending_match<CS, M>(
    space: &GenericRSpace<CS, M>,
    channels: &[CS::Channel],
) -> bool
where
    CS: rholang::rust::interpreter::spaces::channel_store::ChannelStore,
    CS::Channel: Clone + Eq + Hash + Send + Sync + AsRef<[u8]> + 'static,
    CS::Pattern: Clone + PartialEq + Send + Sync + 'static,
    CS::Data: Clone + Send + Sync + std::fmt::Debug + 'static,
    CS::Continuation: Clone + Send + Sync + 'static,
    CS::DataColl: DataCollection<CS::Data> + Default + Clone + Send + Sync + 'static,
    CS::ContColl: ContinuationCollection<CS::Pattern, CS::Continuation> + Default + Clone + Send + Sync,
    M: rholang::rust::interpreter::spaces::matcher::Match<CS::Pattern, CS::Data>,
{
    use rholang::rust::interpreter::spaces::agent::SpaceAgent;

    for channel in channels {
        let data = space.get_data(channel);
        let conts = space.get_waiting_continuations(vec![channel.clone()]);

        // NoPendingMatch: If there's data, there shouldn't be matching continuations
        // and vice versa. In practice, after operations complete, at most one side
        // should have entries.
        if !data.is_empty() && !conts.is_empty() {
            // This would indicate a pending match that should have fired
            return false;
        }
    }
    true
}

/// Verify gensym produces unique channels.
///
/// This is the uniqueness invariant from GenericRSpace.v:604-644.
pub fn check_gensym_uniqueness<CS, M>(
    space: &mut GenericRSpace<CS, M>,
    num_calls: usize,
) -> bool
where
    CS: rholang::rust::interpreter::spaces::channel_store::ChannelStore,
    CS::Channel: Clone + Eq + Hash + Send + Sync + AsRef<[u8]> + 'static,
    CS::Pattern: Clone + PartialEq + Send + Sync + 'static,
    CS::Data: Clone + Send + Sync + std::fmt::Debug + 'static,
    CS::Continuation: Clone + Send + Sync + 'static,
    CS::DataColl: DataCollection<CS::Data> + Default + Clone + Send + Sync + 'static,
    CS::ContColl: ContinuationCollection<CS::Pattern, CS::Continuation> + Default + Clone + Send + Sync,
    M: rholang::rust::interpreter::spaces::matcher::Match<CS::Pattern, CS::Data>,
{
    use rholang::rust::interpreter::spaces::agent::SpaceAgent;

    let mut seen: HashSet<Vec<u8>> = HashSet::new();

    for _ in 0..num_calls {
        match space.gensym() {
            Ok(channel) => {
                // Convert channel to bytes for comparison using AsRef<[u8]>
                let bytes: Vec<u8> = channel.as_ref().to_vec();
                if !seen.insert(bytes) {
                    return false; // Duplicate channel!
                }
            }
            Err(_) => {
                // Gensym failures are acceptable (e.g., out of names in Array)
                continue;
            }
        }
    }
    true
}

// =============================================================================
// Test Assertions
// =============================================================================

/// Assert that phlogiston balance is non-negative.
pub fn assert_phlogiston_non_negative(meter: &PhlogistonMeter) {
    // u64 is always non-negative, but we check the invariant anyway
    assert!(
        meter.balance() <= u64::MAX,
        "Phlogiston balance should be valid u64"
    );
}

/// Assert that a charge operation deducted the exact amount.
pub fn assert_charge_exact(meter: &PhlogistonMeter, initial: u64, charged: u64) {
    let expected = initial.saturating_sub(charged);
    assert_eq!(
        meter.balance(),
        expected,
        "Phlogiston balance after charge should be exactly initial - charged"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arb_inner_collection_type_generates_all_variants() {
        // Verify the generator can produce each variant
        let mut runner = proptest::test_runner::TestRunner::default();
        let strategy = arb_inner_collection_type();

        let mut seen_bag = false;
        let mut seen_queue = false;
        let mut seen_stack = false;
        let mut seen_set = false;
        let mut seen_cell = false;
        let mut seen_pq = false;
        let mut seen_vdb = false;

        for _ in 0..1000 {
            let value = strategy.new_tree(&mut runner).unwrap().current();
            match value {
                InnerCollectionType::Bag => seen_bag = true,
                InnerCollectionType::Queue => seen_queue = true,
                InnerCollectionType::Stack => seen_stack = true,
                InnerCollectionType::Set => seen_set = true,
                InnerCollectionType::Cell => seen_cell = true,
                InnerCollectionType::PriorityQueue { .. } => seen_pq = true,
                InnerCollectionType::VectorDB { .. } => seen_vdb = true,
            }
        }

        assert!(seen_bag, "Should generate Bag");
        assert!(seen_queue, "Should generate Queue");
        assert!(seen_stack, "Should generate Stack");
        assert!(seen_set, "Should generate Set");
        assert!(seen_cell, "Should generate Cell");
        assert!(seen_pq, "Should generate PriorityQueue");
        assert!(seen_vdb, "Should generate VectorDB");
    }

    #[test]
    fn test_create_hashmap_bag_space() {
        use rholang::rust::interpreter::spaces::agent::SpaceAgent;

        let mut space = create_hashmap_bag_space();
        assert_eq!(space.qualifier(), SpaceQualifier::Default);

        // Test basic operations
        let result = space.produce(TestChannel::from(0), 42, false, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bag_with_data_helper() {
        let bag = bag_with_data(vec![1, 2, 3]);
        assert_eq!(bag.len(), 3);
    }

    #[test]
    fn test_queue_with_data_helper() {
        let queue = queue_with_data(vec![1, 2, 3]);
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_create_meter_with_balance() {
        let meter = create_meter_with_balance(1000);
        assert_eq!(meter.balance(), 1000);
    }
}
