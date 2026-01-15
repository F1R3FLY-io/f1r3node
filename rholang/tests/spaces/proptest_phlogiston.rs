//! Property-Based Tests for Phlogiston (Gas) Accounting
//!
//! These tests verify the gas metering invariants from Phlogiston.v:
//! - Balance non-negativity
//! - Exact charge deduction
//! - Sequential charge accumulation
//! - Out-of-gas error handling
//!
//! # Rholang Correspondence
//!
//! ```rholang
//! // Gas-limited execution with phlogiston
//! new GasLimitedSpace(`rho:space:hashmap:bag:default`), space in {
//!   GasLimitedSpace!({ "gas_limit": 1000 }, *space) |
//!   use space {
//!     // Each operation consumes gas
//!     ch!(42) |              // Costs: SEND_BASE + data_size
//!     for (@x <- ch) { ... } // Costs: RECEIVE_BASE
//!     // If gas runs out: OutOfPhlogiston error
//!   }
//! }
//! ```

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

use rholang::rust::interpreter::spaces::{
    PhlogistonMeter, Operation, SpaceError,
    SEND_BASE_COST, SEND_PER_BYTE_COST, RECEIVE_BASE_COST,
    MATCH_BASE_COST, MATCH_PER_ELEMENT_COST, CHANNEL_CREATE_COST,
    CHECKPOINT_COST, SPACE_CREATE_COST,
};

use super::test_utils::*;

// =============================================================================
// Balance Non-Negativity Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Phlogiston balance is always non-negative (u64 enforces this).
    ///
    /// **Formal Reference**: Phlogiston.v lines 341-361 (phlogiston_non_negative)
    /// **Rholang**: Gas cannot go below zero; operations fail before that happens
    #[test]
    fn prop_phlogiston_non_negative(
        initial in 0u64..10_000_000,
        operations in arb_phlogiston_sequence(1, 50),
    ) {
        let meter = PhlogistonMeter::new(initial);

        for op in operations {
            let cost = op.cost();

            if cost <= meter.balance() {
                // Should succeed
                let result = meter.charge(&op);
                prop_assert!(result.is_ok(), "Charge should succeed when balance >= cost");

                // Balance should still be valid (non-negative by u64)
                prop_assert!(meter.balance() <= initial,
                    "Balance should not increase after charge");
            } else {
                // Should fail with OutOfPhlogiston
                let result = meter.charge(&op);
                prop_assert!(result.is_err(), "Charge should fail when balance < cost");

                // Balance should be unchanged
                let balance_before = meter.balance();
                let _ = meter.charge(&op);
                prop_assert_eq!(meter.balance(), balance_before,
                    "Failed charge should not change balance");
            }
        }

        // Final balance is always valid
        assert_phlogiston_non_negative(&meter);
    }

    /// Starting with any balance, the meter balance is always <= initial.
    #[test]
    fn prop_balance_never_exceeds_initial(
        initial in 1u64..10_000_000,
        operations in arb_phlogiston_sequence(1, 100),
    ) {
        let meter = PhlogistonMeter::new(initial);

        for op in operations {
            let _ = meter.charge(&op);
            prop_assert!(meter.balance() <= initial,
                "Balance {} should never exceed initial {}", meter.balance(), initial);
        }
    }
}

// =============================================================================
// Exact Charge Deduction Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// A successful charge deducts exactly the operation's cost.
    ///
    /// **Formal Reference**: Phlogiston.v lines 152-164 (charge_exact)
    #[test]
    fn prop_charge_exact_deduction(
        initial in 1000u64..10_000_000,
        op in arb_phlogiston_operation(),
    ) {
        let meter = PhlogistonMeter::new(initial);
        let cost = op.cost();

        let balance_before = meter.balance();

        if cost <= balance_before {
            let result = meter.charge(&op);
            prop_assert!(result.is_ok());

            let balance_after = meter.balance();
            let actual_deduction = balance_before - balance_after;

            prop_assert_eq!(actual_deduction, cost,
                "Charge should deduct exactly {} but deducted {}", cost, actual_deduction);
        }
    }

    /// Send operation cost is SEND_BASE + SEND_PER_BYTE * data_size.
    #[test]
    fn prop_send_cost_formula(
        data_size in 0usize..10000,
    ) {
        let op = Operation::Send { data_size };

        let expected_cost = SEND_BASE_COST + (SEND_PER_BYTE_COST * data_size as u64);
        let actual_cost = op.cost();

        prop_assert_eq!(actual_cost, expected_cost,
            "Send cost formula mismatch");
    }

    /// Match operation cost is MATCH_BASE + MATCH_PER_ELEMENT * pattern_size.
    #[test]
    fn prop_match_cost_formula(
        pattern_size in 0usize..1000,
    ) {
        let op = Operation::Match { pattern_size };

        let expected_cost = MATCH_BASE_COST + (MATCH_PER_ELEMENT_COST * pattern_size as u64);
        let actual_cost = op.cost();

        prop_assert_eq!(actual_cost, expected_cost,
            "Match cost formula mismatch");
    }
}

// =============================================================================
// Sequential Charge Accumulation Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Sequential charges accumulate correctly.
    ///
    /// **Formal Reference**: Phlogiston.v lines 239-252 (sequential_charges)
    #[test]
    fn prop_sequential_charges_accumulate(
        initial in 100_000u64..10_000_000,
        operations in arb_phlogiston_sequence(1, 20),
    ) {
        let meter = PhlogistonMeter::new(initial);

        let mut total_charged: u64 = 0;

        for op in &operations {
            let cost = op.cost();

            if total_charged + cost <= initial {
                let result = meter.charge(op);
                if result.is_ok() {
                    total_charged += cost;
                }
            }
        }

        let expected_balance = initial - total_charged;
        prop_assert_eq!(meter.balance(), expected_balance,
            "Sequential charges should accumulate to {} but got {}",
            expected_balance, meter.balance());
    }

    /// Order of charges doesn't affect final balance (for same operations).
    #[test]
    fn prop_charge_order_independence(
        initial in 100_000u64..10_000_000,
        operations in arb_phlogiston_sequence(2, 10),
    ) {
        // Calculate total cost (assuming all succeed)
        let total_cost: u64 = operations.iter().map(|op| op.cost()).sum();

        if total_cost <= initial {
            // Forward order
            let meter1 = PhlogistonMeter::new(initial);
            for op in &operations {
                let _ = meter1.charge(op);
            }

            // Reverse order
            let meter2 = PhlogistonMeter::new(initial);
            for op in operations.iter().rev() {
                let _ = meter2.charge(op);
            }

            prop_assert_eq!(meter1.balance(), meter2.balance(),
                "Order of charges should not affect final balance");
        }
    }
}

// =============================================================================
// Out-of-Gas Error Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Operations that exceed the balance fail with OutOfPhlogiston.
    ///
    /// **Rholang**: When gas runs out, operations fail safely
    #[test]
    fn prop_out_of_gas_error(
        initial in 1u64..1000,
        data_size in 100usize..10000, // Large enough to likely exceed
    ) {
        let meter = PhlogistonMeter::new(initial);
        let op = Operation::Send { data_size };

        let cost = op.cost();

        if cost > initial {
            let result = meter.charge(&op);
            prop_assert!(result.is_err(), "Should fail when cost {} > balance {}", cost, initial);

            match result {
                Err(SpaceError::OutOfPhlogiston { required, available, .. }) => {
                    prop_assert_eq!(required, cost);
                    prop_assert_eq!(available, initial);
                }
                Err(other) => prop_assert!(false, "Expected OutOfPhlogiston, got {:?}", other),
                Ok(_) => prop_assert!(false, "Should have failed"),
            }
        }
    }

    /// Balance is unchanged after a failed charge.
    #[test]
    fn prop_failed_charge_unchanged_balance(
        initial in 1u64..100,
        data_size in 1000usize..10000,
    ) {
        let meter = PhlogistonMeter::new(initial);
        let op = Operation::Send { data_size };

        let cost = op.cost();

        if cost > initial {
            let balance_before = meter.balance();
            let _ = meter.charge(&op);
            prop_assert_eq!(meter.balance(), balance_before,
                "Failed charge should not change balance");
        }
    }
}

// =============================================================================
// Gas Configuration Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Operation costs are always positive.
    #[test]
    fn prop_operation_cost_positive(
        data_size in 0usize..1000,
        pattern_size in 0usize..100,
    ) {
        let send_op = Operation::Send { data_size };
        let match_op = Operation::Match { pattern_size };
        let receive_op = Operation::Receive;
        let create_op = Operation::CreateChannel;

        prop_assert!(send_op.cost() > 0, "Send cost should be positive");
        prop_assert!(match_op.cost() > 0, "Match cost should be positive");
        prop_assert!(receive_op.cost() > 0, "Receive cost should be positive");
        prop_assert!(create_op.cost() > 0, "CreateChannel cost should be positive");
    }

    /// Unlimited meter allows all operations.
    #[test]
    fn prop_unlimited_meter_allows_all(
        operations in arb_phlogiston_sequence(1, 100),
    ) {
        let meter = PhlogistonMeter::unlimited();

        for op in operations {
            let result = meter.charge(&op);
            prop_assert!(result.is_ok(), "Unlimited meter should allow all operations");
        }
    }
}

// =============================================================================
// Invariant Preservation Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Total consumed + remaining balance = initial balance (conservation).
    #[test]
    fn prop_balance_conservation(
        initial in 1000u64..10_000_000,
        operations in arb_phlogiston_sequence(1, 50),
    ) {
        let meter = PhlogistonMeter::new(initial);
        let mut total_consumed: u64 = 0;

        for op in operations {
            let balance_before = meter.balance();

            let result = meter.charge(&op);

            if result.is_ok() {
                let balance_after = meter.balance();
                let consumed_this_op = balance_before - balance_after;
                total_consumed += consumed_this_op;
            }
        }

        // Conservation: initial = consumed + remaining
        prop_assert_eq!(
            initial,
            total_consumed + meter.balance(),
            "Balance conservation violated: {} != {} + {}",
            initial, total_consumed, meter.balance()
        );
    }
}

// =============================================================================
// Fixed Cost Tests
// =============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_receive_cost_is_fixed() {
        let op = Operation::Receive;
        assert_eq!(op.cost(), RECEIVE_BASE_COST);
    }

    #[test]
    fn test_create_channel_cost_is_fixed() {
        let op = Operation::CreateChannel;
        assert_eq!(op.cost(), CHANNEL_CREATE_COST);
    }

    #[test]
    fn test_checkpoint_cost_is_fixed() {
        let op = Operation::Checkpoint;
        assert_eq!(op.cost(), CHECKPOINT_COST);
    }

    #[test]
    fn test_space_create_cost_is_fixed() {
        let op = Operation::CreateSpace;
        assert_eq!(op.cost(), SPACE_CREATE_COST);
    }

    #[test]
    fn test_zero_size_send() {
        let op = Operation::Send { data_size: 0 };
        assert_eq!(op.cost(), SEND_BASE_COST);
    }

    #[test]
    fn test_zero_size_match() {
        let op = Operation::Match { pattern_size: 0 };
        assert_eq!(op.cost(), MATCH_BASE_COST);
    }

    #[test]
    fn test_meter_initial_balance() {
        let meter = PhlogistonMeter::new(12345);
        assert_eq!(meter.balance(), 12345);
    }

    #[test]
    fn test_successful_charge() {
        let meter = PhlogistonMeter::new(10000);
        let op = Operation::Receive;

        let result = meter.charge(&op);
        assert!(result.is_ok());
        assert_eq!(meter.balance(), 10000 - RECEIVE_BASE_COST);
    }

    #[test]
    fn test_failed_charge_insufficient_balance() {
        let meter = PhlogistonMeter::new(10);
        let op = Operation::Send { data_size: 1000 }; // Expensive

        let cost = op.cost();
        assert!(cost > 10);

        let result = meter.charge(&op);
        assert!(result.is_err());
        assert_eq!(meter.balance(), 10); // Unchanged
    }

    #[test]
    fn test_multiple_charges() {
        let meter = PhlogistonMeter::new(1000);

        // Three receives
        for _ in 0..3 {
            let result = meter.charge(&Operation::Receive);
            assert!(result.is_ok());
        }

        assert_eq!(meter.balance(), 1000 - (3 * RECEIVE_BASE_COST));
    }
}
