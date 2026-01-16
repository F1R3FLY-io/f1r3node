//! Property-Based Tests for Charging Space Agent Module
//!
//! This module provides comprehensive property-based testing for gas metering
//! and phlogiston accounting in the ChargingSpaceAgent wrapper.
//!
//! # Formal Correspondence
//! - `Phlogiston.v`: Charge preservation, non-negativity, and accumulation invariants
//! - `GenericRSpace.v`: Integration of gas accounting with space operations
//! - `Safety/Properties.v`: Resource exhaustion safety properties
//!
//! # Test Coverage
//! - Balance non-negativity invariant
//! - Exact charge deduction
//! - Out-of-gas failure handling
//! - Unlimited agent never fails gas
//! - Sequential charge accumulation
//! - Operation cost formulas
//!
//! # Rholang Syntax Examples
//!
//! The charging agent meters gas for all operations:
//! ```rholang
//! // Each operation deducts phlogiston from the meter
//! new ch in {           // CHANNEL_CREATE_COST
//!   ch!(42) |           // SEND_BASE_COST + data_size
//!   for (@x <- ch) {    // RECEIVE_BASE_COST + MATCH_COST
//!     stdout!(x)
//!   }
//! }
//! ```

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;
use std::collections::BTreeSet;
use std::sync::Arc;

use rholang::rust::interpreter::spaces::{
    PhlogistonMeter, Operation, SpaceAgent, SpaceError,
    ChargingSpaceAgent, ChargingAgentBuilder,
    SpaceId, SpaceQualifier,
    SEND_BASE_COST, SEND_PER_BYTE_COST, RECEIVE_BASE_COST,
    MATCH_BASE_COST, MATCH_PER_ELEMENT_COST, CHANNEL_CREATE_COST,
    CHECKPOINT_COST, SPACE_CREATE_COST,
};

use rspace_plus_plus::rspace::{
    internal::{Datum, WaitingContinuation},
    rspace_interface::{ContResult, RSpaceResult},
    trace::event::Produce,
};

// =============================================================================
// Proptest Configuration
// =============================================================================

const PROPTEST_CASES: u32 = 500;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(PROPTEST_CASES)
}

// =============================================================================
// Mock Space Agent for Testing
// =============================================================================

/// A mock space agent that tracks operations without real storage.
/// Used to test ChargingSpaceAgent's metering behavior in isolation.
#[derive(Clone)]
struct MockSpaceAgent {
    id: SpaceId,
    gensym_counter: usize,
}

impl MockSpaceAgent {
    fn new() -> Self {
        MockSpaceAgent {
            id: SpaceId::default_space(),
            gensym_counter: 0,
        }
    }
}

impl SpaceAgent<u64, String, String, String> for MockSpaceAgent {
    fn space_id(&self) -> &SpaceId {
        &self.id
    }

    fn qualifier(&self) -> SpaceQualifier {
        SpaceQualifier::Default
    }

    fn gensym(&mut self) -> Result<u64, SpaceError> {
        self.gensym_counter += 1;
        Ok(self.gensym_counter as u64)
    }

    fn produce(
        &mut self,
        _channel: u64,
        _data: String,
        _persist: bool,
        _priority: Option<usize>,
    ) -> Result<Option<(ContResult<u64, String, String>, Vec<RSpaceResult<u64, String>>, Produce)>, SpaceError> {
        Ok(None)
    }

    fn consume(
        &mut self,
        _channels: Vec<u64>,
        _patterns: Vec<String>,
        _continuation: String,
        _persist: bool,
        _peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<u64, String, String>, Vec<RSpaceResult<u64, String>>)>, SpaceError> {
        Ok(None)
    }

    fn install(
        &mut self,
        _channels: Vec<u64>,
        _patterns: Vec<String>,
        _continuation: String,
    ) -> Result<Option<(String, Vec<String>)>, SpaceError> {
        Ok(None)
    }

    fn get_data(&self, _channel: &u64) -> Vec<Datum<String>> {
        vec![]
    }

    fn get_waiting_continuations(&self, _channels: Vec<u64>) -> Vec<WaitingContinuation<String, String>> {
        vec![]
    }

    fn get_joins(&self, _channel: u64) -> Vec<Vec<u64>> {
        vec![]
    }
}

// =============================================================================
// Arbitrary Generators
// =============================================================================

/// Generate an arbitrary initial balance (reasonable range for testing).
fn arb_initial_balance() -> impl Strategy<Value = u64> {
    0u64..10_000_000u64
}

/// Generate a balance that will always be sufficient for operations.
fn arb_sufficient_balance() -> impl Strategy<Value = u64> {
    1_000_000u64..100_000_000u64
}

/// Generate a very small balance that will cause out-of-gas.
fn arb_insufficient_balance() -> impl Strategy<Value = u64> {
    0u64..CHANNEL_CREATE_COST
}

/// Generate an arbitrary phlogiston operation.
#[allow(dead_code)]
fn arb_operation() -> impl Strategy<Value = Operation> {
    prop_oneof![
        (0usize..1000).prop_map(|size| Operation::Send { data_size: size }),
        Just(Operation::Receive),
        (0usize..100).prop_map(|size| Operation::Match { pattern_size: size }),
        Just(Operation::CreateChannel),
        Just(Operation::Checkpoint),
        Just(Operation::CreateSpace),
    ]
}

/// Generate a sequence of operations.
#[allow(dead_code)]
fn arb_operation_sequence(min: usize, max: usize) -> impl Strategy<Value = Vec<Operation>> {
    prop_vec(arb_operation(), min..=max)
}

/// Generate arbitrary number of patterns for consume tests.
#[allow(dead_code)]
fn arb_pattern_count() -> impl Strategy<Value = usize> {
    0usize..10
}

/// Generate arbitrary data size for produce tests.
#[allow(dead_code)]
fn arb_data_size() -> impl Strategy<Value = usize> {
    0usize..1000
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate expected cost for an operation.
#[allow(dead_code)]
fn expected_cost(op: &Operation) -> u64 {
    op.cost()
}

/// Calculate total expected cost for a sequence of operations.
#[allow(dead_code)]
fn total_expected_cost(ops: &[Operation]) -> u64 {
    ops.iter().map(expected_cost).sum()
}

/// Create a charging agent with the given balance.
fn create_charging_agent(balance: u64) -> ChargingSpaceAgent<MockSpaceAgent, u64, String, String, String> {
    let mock = MockSpaceAgent::new();
    let meter = Arc::new(PhlogistonMeter::new(balance));
    ChargingSpaceAgent::new(mock, meter)
}

/// Create an unlimited charging agent.
fn create_unlimited_agent() -> ChargingSpaceAgent<MockSpaceAgent, u64, String, String, String> {
    ChargingSpaceAgent::unlimited(MockSpaceAgent::new())
}

// =============================================================================
// Balance Non-Negativity Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Balance is always non-negative (u64 guarantees this).
    ///
    /// ∀ balance, ops. meter.balance() >= 0 after any operations
    ///
    /// Note: u64 inherently cannot be negative, but we verify the meter
    /// doesn't underflow through proper charge rejection.
    ///
    /// Rocq: `balance_never_negative` in Phlogiston.v
    #[test]
    fn prop_balance_never_negative(
        initial in arb_sufficient_balance(),
        op_count in 1usize..100
    ) {
        let mut agent = create_charging_agent(initial);

        // Perform operations until we run out of gas or hit the limit
        for _ in 0..op_count {
            let _ = agent.gensym();
            // Balance is always valid u64
            prop_assert!(agent.balance() <= initial);
        }
    }

    /// Property: PhlogistonMeter balance() returns consistent values.
    ///
    /// Rocq: `meter_balance_consistent` in Phlogiston.v
    #[test]
    fn prop_meter_balance_consistent(balance in arb_initial_balance()) {
        let meter = PhlogistonMeter::new(balance);
        prop_assert_eq!(meter.balance(), balance);
        prop_assert_eq!(meter.initial_limit(), balance);
        prop_assert_eq!(meter.total_consumed(), 0);
    }
}

// =============================================================================
// Exact Charge Deduction Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Charge deduction is exact.
    ///
    /// ∀ initial, cost. if initial >= cost then
    ///   balance_after = initial - cost
    ///
    /// Rocq: `charge_deduction_exact` in Phlogiston.v
    #[test]
    fn prop_charge_deduction_exact(
        initial in arb_sufficient_balance()
    ) {
        let mut agent = create_charging_agent(initial);
        let balance_before = agent.balance();

        // Gensym costs CHANNEL_CREATE_COST
        let result = agent.gensym();
        prop_assert!(result.is_ok());

        let balance_after = agent.balance();
        prop_assert_eq!(
            balance_after,
            balance_before - CHANNEL_CREATE_COST,
            "Balance should decrease by exactly CHANNEL_CREATE_COST"
        );
    }

    /// Property: Produce charges SEND_BASE_COST + data overhead.
    ///
    /// Rocq: `produce_cost_formula` in Phlogiston.v
    #[test]
    fn prop_produce_cost_formula(
        initial in arb_sufficient_balance()
    ) {
        let mut agent = create_charging_agent(initial);
        let balance_before = agent.balance();

        // Produce with String data
        let data = "test data".to_string();
        let result = agent.produce(1, data, false, None);
        prop_assert!(result.is_ok());

        let balance_after = agent.balance();
        let charged = balance_before - balance_after;

        // Should charge at least SEND_BASE_COST
        prop_assert!(
            charged >= SEND_BASE_COST,
            "Produce should charge at least SEND_BASE_COST"
        );
    }

    /// Property: Consume charges RECEIVE_BASE_COST + pattern matching cost.
    ///
    /// Rocq: `consume_cost_formula` in Phlogiston.v
    #[test]
    fn prop_consume_cost_formula(
        initial in arb_sufficient_balance(),
        pattern_count in 1usize..5
    ) {
        let mut agent = create_charging_agent(initial);
        let balance_before = agent.balance();

        let channels: Vec<u64> = (0..pattern_count as u64).collect();
        let patterns: Vec<String> = (0..pattern_count).map(|i| format!("pattern{}", i)).collect();

        let result = agent.consume(channels, patterns, "continuation".to_string(), false, BTreeSet::new());
        prop_assert!(result.is_ok());

        let balance_after = agent.balance();
        let charged = balance_before - balance_after;

        // Should charge RECEIVE_BASE_COST + pattern matching cost
        let expected_min = RECEIVE_BASE_COST + MATCH_BASE_COST + pattern_count as u64 * MATCH_PER_ELEMENT_COST;
        prop_assert!(
            charged >= expected_min,
            "Consume should charge at least RECEIVE_BASE_COST + match cost, got {} expected >= {}",
            charged,
            expected_min
        );
    }

    /// Property: Gensym charges exactly CHANNEL_CREATE_COST.
    ///
    /// Rocq: `gensym_charges_fixed_cost` in Phlogiston.v
    #[test]
    fn prop_gensym_charges_fixed_cost(
        initial in arb_sufficient_balance(),
        gensym_count in 1usize..10
    ) {
        let mut agent = create_charging_agent(initial);

        for _ in 0..gensym_count {
            let balance_before = agent.balance();
            let result = agent.gensym();
            prop_assert!(result.is_ok());
            let balance_after = agent.balance();

            prop_assert_eq!(
                balance_after,
                balance_before - CHANNEL_CREATE_COST,
                "Each gensym should cost exactly CHANNEL_CREATE_COST"
            );
        }
    }
}

// =============================================================================
// Out-of-Gas Failure Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Operations fail gracefully when out of gas.
    ///
    /// ∀ balance < cost. operation fails with OutOfPhlogiston error
    ///
    /// Rocq: `out_of_gas_fails_gracefully` in Phlogiston.v
    #[test]
    fn prop_out_of_gas_fails_gracefully(
        balance in arb_insufficient_balance()
    ) {
        let mut agent = create_charging_agent(balance);

        // Gensym requires CHANNEL_CREATE_COST which should exceed our balance
        let result = agent.gensym();

        match result {
            Err(SpaceError::OutOfPhlogiston { required, available, .. }) => {
                prop_assert_eq!(available, balance);
                prop_assert!(required > available);
            }
            Ok(_) if balance >= CHANNEL_CREATE_COST => {
                // If balance was sufficient, operation should succeed
            }
            _ => {
                prop_assert!(
                    balance >= CHANNEL_CREATE_COST,
                    "Operation should fail with OutOfPhlogiston when balance is insufficient"
                );
            }
        }
    }

    /// Property: Failed charges don't modify balance.
    ///
    /// ∀ balance, cost where balance < cost. balance is unchanged after failed charge
    ///
    /// Rocq: `failed_charge_no_side_effect` in Phlogiston.v
    #[test]
    fn prop_failed_charge_no_side_effect(
        balance in arb_insufficient_balance()
    ) {
        let mut agent = create_charging_agent(balance);
        let balance_before = agent.balance();

        let _ = agent.gensym(); // May fail

        // If it failed, balance should be unchanged
        if balance < CHANNEL_CREATE_COST {
            prop_assert_eq!(
                agent.balance(),
                balance_before,
                "Failed charge should not modify balance"
            );
        }
    }
}

// =============================================================================
// Unlimited Agent Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Unlimited agent never fails due to gas.
    ///
    /// ∀ ops. unlimited_agent.operation() succeeds (gas-wise)
    ///
    /// Rocq: `unlimited_never_fails_gas` in Phlogiston.v
    #[test]
    fn prop_unlimited_never_fails_gas(
        op_count in 1usize..1000
    ) {
        let mut agent = create_unlimited_agent();

        for _ in 0..op_count {
            let result = agent.gensym();
            prop_assert!(
                result.is_ok(),
                "Unlimited agent should never fail due to gas"
            );
        }
    }

    /// Property: Unlimited agent has maximum balance.
    ///
    /// Rocq: `unlimited_balance_max` in Phlogiston.v
    #[test]
    fn prop_unlimited_balance_max(_seed in any::<u64>()) {
        let agent = create_unlimited_agent();
        prop_assert_eq!(agent.balance(), u64::MAX);
    }
}

// =============================================================================
// Sequential Charge Accumulation Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Sequential charges accumulate correctly.
    ///
    /// ∀ ops. total_consumed = sum(cost(op) for op in ops)
    ///
    /// Rocq: `sequential_charges_accumulate` in Phlogiston.v
    #[test]
    fn prop_sequential_charges_accumulate(
        initial in arb_sufficient_balance(),
        op_count in 1usize..50
    ) {
        let mut agent = create_charging_agent(initial);
        let mut expected_consumed: u64 = 0;

        for _ in 0..op_count {
            if agent.balance() >= CHANNEL_CREATE_COST {
                let result = agent.gensym();
                if result.is_ok() {
                    expected_consumed += CHANNEL_CREATE_COST;
                }
            } else {
                break;
            }
        }

        prop_assert_eq!(
            agent.total_consumed(),
            expected_consumed,
            "Total consumed should equal sum of all successful charges"
        );
    }

    /// Property: Initial limit minus consumed equals current balance.
    ///
    /// ∀ ops. initial_limit - total_consumed = balance
    ///
    /// Rocq: `balance_consumed_invariant` in Phlogiston.v
    #[test]
    fn prop_balance_consumed_invariant(
        initial in arb_sufficient_balance(),
        op_count in 1usize..50
    ) {
        let mut agent = create_charging_agent(initial);

        for _ in 0..op_count {
            if agent.balance() >= CHANNEL_CREATE_COST {
                let _ = agent.gensym();
            } else {
                break;
            }
        }

        let meter = agent.meter();
        prop_assert_eq!(
            meter.initial_limit() - meter.total_consumed(),
            meter.balance(),
            "initial_limit - total_consumed should equal balance"
        );
    }
}

// =============================================================================
// Builder Pattern Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Builder creates agent with specified limit.
    ///
    /// Rocq: `builder_limit_honored` in Phlogiston.v
    #[test]
    fn prop_builder_limit_honored(limit in arb_initial_balance()) {
        let mock = MockSpaceAgent::new();
        let agent: ChargingSpaceAgent<_, u64, String, String, String> = ChargingAgentBuilder::new()
            .with_space(mock)
            .with_limit(limit)
            .build()
            .expect("Builder should succeed");

        prop_assert_eq!(agent.balance(), limit);
    }

    /// Property: Builder with shared meter preserves meter reference.
    ///
    /// Rocq: `builder_shared_meter` in Phlogiston.v
    #[test]
    fn prop_builder_shared_meter(limit in arb_initial_balance()) {
        let meter = Arc::new(PhlogistonMeter::new(limit));
        let mock = MockSpaceAgent::new();

        let agent: ChargingSpaceAgent<_, u64, String, String, String> = ChargingAgentBuilder::new()
            .with_space(mock)
            .with_meter(meter.clone())
            .build()
            .expect("Builder should succeed");

        prop_assert_eq!(agent.balance(), limit);
        prop_assert_eq!(meter.balance(), limit);
    }
}

// =============================================================================
// Cost Formula Verification Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Send cost formula is correct.
    ///
    /// cost(Send { data_size }) = SEND_BASE_COST + data_size * SEND_PER_BYTE_COST
    ///
    /// Rocq: `send_cost_formula` in Phlogiston.v
    #[test]
    fn prop_send_cost_formula(data_size in 0usize..1000) {
        let op = Operation::Send { data_size };
        let expected = SEND_BASE_COST + (data_size as u64) * SEND_PER_BYTE_COST;
        prop_assert_eq!(op.cost(), expected);
    }

    /// Property: Receive cost is constant.
    ///
    /// Rocq: `receive_cost_constant` in Phlogiston.v
    #[test]
    fn prop_receive_cost_constant(_seed in any::<u64>()) {
        let op = Operation::Receive;
        prop_assert_eq!(op.cost(), RECEIVE_BASE_COST);
    }

    /// Property: Match cost formula is correct.
    ///
    /// cost(Match { pattern_size }) = MATCH_BASE_COST + pattern_size * MATCH_PER_ELEMENT_COST
    ///
    /// Rocq: `match_cost_formula` in Phlogiston.v
    #[test]
    fn prop_match_cost_formula(pattern_size in 0usize..100) {
        let op = Operation::Match { pattern_size };
        let expected = MATCH_BASE_COST + (pattern_size as u64) * MATCH_PER_ELEMENT_COST;
        prop_assert_eq!(op.cost(), expected);
    }

    /// Property: CreateChannel cost is constant.
    ///
    /// Rocq: `create_channel_cost_constant` in Phlogiston.v
    #[test]
    fn prop_create_channel_cost_constant(_seed in any::<u64>()) {
        let op = Operation::CreateChannel;
        prop_assert_eq!(op.cost(), CHANNEL_CREATE_COST);
    }

    /// Property: Checkpoint cost is constant.
    ///
    /// Rocq: `checkpoint_cost_constant` in Phlogiston.v
    #[test]
    fn prop_checkpoint_cost_constant(_seed in any::<u64>()) {
        let op = Operation::Checkpoint;
        prop_assert_eq!(op.cost(), CHECKPOINT_COST);
    }

    /// Property: CreateSpace cost is constant.
    ///
    /// Rocq: `create_space_cost_constant` in Phlogiston.v
    #[test]
    fn prop_create_space_cost_constant(_seed in any::<u64>()) {
        let op = Operation::CreateSpace;
        prop_assert_eq!(op.cost(), SPACE_CREATE_COST);
    }
}

// =============================================================================
// Static Invariant Tests (Non-Proptest)
// =============================================================================

#[test]
fn test_cost_constants_positive() {
    assert!(SEND_BASE_COST > 0);
    assert!(RECEIVE_BASE_COST > 0);
    assert!(MATCH_BASE_COST > 0);
    assert!(CHANNEL_CREATE_COST > 0);
    assert!(CHECKPOINT_COST > 0);
    assert!(SPACE_CREATE_COST > 0);
}

#[test]
fn test_unlimited_meter() {
    let meter = PhlogistonMeter::unlimited();
    assert_eq!(meter.balance(), u64::MAX);
    assert_eq!(meter.initial_limit(), u64::MAX);
    // Unlimited meter is characterized by having u64::MAX balance
    assert_eq!(meter.balance(), u64::MAX);
}

#[test]
fn test_meter_charge_reduces_balance() {
    let meter = PhlogistonMeter::new(1000);
    let op = Operation::CreateChannel;
    let result = meter.charge(&op);
    assert!(result.is_ok());
    assert_eq!(meter.balance(), 1000 - CHANNEL_CREATE_COST);
}

#[test]
fn test_meter_insufficient_balance_error() {
    let meter = PhlogistonMeter::new(10);
    let op = Operation::CreateChannel;
    let result = meter.charge(&op);
    assert!(result.is_err());
    assert_eq!(meter.balance(), 10); // Unchanged
}

#[test]
fn test_inner_accessor() {
    let mock = MockSpaceAgent::new();
    let agent: ChargingSpaceAgent<_, u64, String, String, String> = ChargingSpaceAgent::unlimited(mock);
    assert_eq!(agent.inner().qualifier(), SpaceQualifier::Default);
}

#[test]
fn test_meter_accessor() {
    let meter = Arc::new(PhlogistonMeter::new(5000));
    let mock = MockSpaceAgent::new();
    let agent: ChargingSpaceAgent<_, u64, String, String, String> = ChargingSpaceAgent::new(mock, meter.clone());
    assert_eq!(agent.meter().balance(), 5000);
}
